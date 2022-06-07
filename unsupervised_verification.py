import argparse
import copy
import random
import sys
import time
import gc
import json
import numpy as np
import pandas as pd
from collections import OrderedDict
import torch.nn.functional as F

import rocl.data_loader as data_loader
from rocl.utils import progress_bar
from rocl.attack_lib import FastGradientSignUntargeted,RepresentationAdv

from beta_crown.model_beta_CROWN import LiRPAConvNet
from beta_crown.relu_conv_parallel import relu_bab_parallel
from beta_crown.utils import *

from beta_crown.auto_LiRPA import BoundedModule, BoundedTensor
from beta_crown.auto_LiRPA.perturbations import *

parser = argparse.ArgumentParser(description='unsupervised verification')

##### arguments for beta CROWN #####
parser.add_argument('--no_solve_slope', action='store_false', dest='solve_slope', help='do not optimize slope/alpha in compute bounds')
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help='use cpu or cuda')
parser.add_argument('--gpuno', default='0', type=str)

parser.add_argument("--norm", type=float, default='inf', help='p norm for epsilon perturbation')
parser.add_argument("--bound_type", type=str, default="CROWN-IBP",
                    choices=["IBP", "CROWN-IBP", "CROWN"], help='method of bound analysis')
parser.add_argument("--model", type=str, default="cnn_4layer_b", help='model name (cifar_model, cifar_model_deep, cifar_model_wide, cnn_4layer, cnn_4layer_b, mnist_cnn_4layer)')
parser.add_argument("--batch_size", type=int, default=256, help='batch size')
parser.add_argument("--bound_opts", type=str, default="same-slope", choices=["same-slope", "zero-lb", "one-lb"],
                    help='bound options for relu')
parser.add_argument('--no_warm', action='store_true', default=False, help='using warm up for lp solver, true by default')
parser.add_argument('--no_beta', action='store_true', default=False, help='using beta splits, true by default')
parser.add_argument("--max_subproblems_list", type=int, default=200000, help='max length of sub-problems list')
parser.add_argument("--decision_thresh", type=float, default=0, help='decision threshold of lower bounds')
parser.add_argument("--timeout", type=int, default=180, help='timeout for one property')
parser.add_argument("--mode", type=str, default="incomplete", choices=["complete", "incomplete", "verified-acc"], help='which mode to use')
parser.add_argument("--ver_total", type=int, default=100, help='number of img to verify')

##### arguments for model #####
parser.add_argument('--train_type', default='contrastive', type=str, help='contrastive/linear eval/test/supervised')
parser.add_argument('--dataset', default='cifar-10', type=str, help='cifar-10/mnist')
parser.add_argument('--load_checkpoint', default='', type=str, help='PATH TO CHECKPOINT')
parser.add_argument('--seed', default=1, type=int, help='random seed')

##### arguments for data augmentation #####
parser.add_argument('--color_jitter_strength', default=0.5, type=float, help='0.5 for CIFAR, 1.0 for ImageNet')
parser.add_argument('--temperature', default=0.5, type=float, help='temperature for pairwise-similarity')

##### arguments for PGD attack & Adversarial Training #####
parser.add_argument('--attack_type', type=str, default='linf', help='adversarial l_p')
parser.add_argument('--epsilon', type=float, default=4.0/255, help='maximum perturbation of adversaries (8/255 0.0314 for cifar-10)')
parser.add_argument('--target_eps', type=float, default=16.0/255, help='maximum perturbation of adversaries (8/255 0.0314 for cifar-10)')
parser.add_argument('--alpha', type=float, default=0.001, help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
parser.add_argument('--k', type=int, default=200, help='maximum iteration when generating adversarial examples')
parser.add_argument('--random_start', type=bool, default=True, help='True for PGD')
parser.add_argument('--loss_type', type=str, default='mse', help='loss type for Rep: mse/sim/l1/cos')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuno
print_args(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
img_clip = min_max_value(args)

if args.mode == "incomplete":
    print("incomplete verification, set decision_thresh to be inf") 
    args.decision_thresh = float('inf')
elif args.mode == "verified-acc":
    print("complete verification for verified accuracy, set decision_thresh to be 0") 
    args.decision_thresh = 0

# Model
print('==> Building model..')
model_ori = torch.load(args.load_checkpoint) 
model_ori.to(args.device)
print(model_ori)
model_output_size = list(model_ori.children())[-1].weight.data.shape[0]
print('output_size:' + str(model_output_size))

def generate_attack(args, ori, target):
    pgd_target = RepresentationAdv(model_ori, None, epsilon=args.target_eps, alpha=args.alpha, min_val=img_clip['min'].to(args.device), max_val=img_clip['max'].to(args.device), max_iters=args.k, _type=args.attack_type, loss_type=args.loss_type)
    adv_target = pgd_target.attack_pgd(original_images=ori.to(args.device), target=target.to(args.device), type='attack', random_start=True)

    pgd = RepresentationAdv(model_ori, None, epsilon=args.epsilon, alpha=args.alpha, min_val=img_clip['min'].to(args.device), max_val=img_clip['max'].to(args.device), max_iters=args.k, _type=args.attack_type, loss_type=args.loss_type)
    adv_img = pgd.attack_pgd(original_images=ori.to(args.device), target=adv_target.to(args.device), type='sim', random_start=True)
    return adv_target, adv_img

def generate_ver_data(loader, total, class_num):
    count = [0 for _ in range(class_num)]
    per_class = total//class_num
    data_loader = iter(loader)
    ans_image= []
    adv_target = []
    adv_eps = []
    ans_label = []
    while sum(count)<total:
        (ori, aug_img, _, label) = next(data_loader)
        i = int(label)
        if count[i] < per_class:
            progress_bar(sum(count), args.ver_total)
            ans_image.append(ori)
            ans_label.append(i)
            i1, i2 = generate_attack(args, ori, aug_img)
            adv_target.append(i1)
            adv_eps.append(i2)
            count[i] += 1
    return ans_image, adv_target, adv_eps, ans_label

def unsupervised_bab(model_ori, data, ori, target, norm, eps, args, output_size, data_max=None, data_min=None):
    if norm == np.inf:
        if data_max is None:
            data_ub = data + eps  # eps is already normalized
            data_lb = data - eps  
        else:
            data_ub = torch.min(data + eps, data_max)
            data_lb = torch.max(data - eps, data_min)
    else:
        data_ub = data_lb = data

    # LiRPA wrapper
    model = LiRPAConvNet(model_ori, ori, target, output_size=output_size, contrastive=True, simplify=True, solve_slope=args.solve_slope, device=args.device, in_size=data.shape)
    
    print('cuda:'+str(list(model.net.parameters())[0].is_cuda))
    if list(model.net.parameters())[0].is_cuda:
        data = data.cuda()
        data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

    ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb).to(data_lb.device)
    domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1) 

    # with torch.autograd.set_detect_anomaly(True):
    print('beta splits:', not args.no_beta)
    min_lb, min_ub, ub_point, nb_states = relu_bab_parallel(model, domain, x, batch=args.batch_size, no_LP=True,
                                                            decision_thresh=args.decision_thresh, beta=not args.no_beta,
                                                            max_subproblems_list=args.max_subproblems_list,
                                                            timeout=args.timeout)

    if isinstance(min_lb, torch.Tensor):
        min_lb = min_lb.item()
    return min_lb, nb_states, model.modify_net, x

# Data
print('==> Preparing adversial data..')
trainloader, traindst, testloader, testdst = data_loader.get_dataset(args)
image, adv_target, adv_eps, label = generate_ver_data(testloader, args.ver_total, class_num=10)

torch.cuda.empty_cache()
gc.collect()

log  = pd.DataFrame(columns=['pgd', 'pgd_acc', 'cbc', 'cbc_acc', 'crown', 'crown_acc', 'ibp', 'ibp_acc', 'crown-ibp', 'crown-ibp_acc', 'ori', 
                    'target', 'label', 'cbc_time', 'eps_target='+str(args.target_eps), 'eps='+str(args.epsilon), args.load_checkpoint])


save_name = time.strftime("%H%M", time.localtime()) + '_timeout' + str(args.timeout) + '_mode_' + args.mode + '_vertotal' + str(args.ver_total) + '_eps' + str(args.epsilon) + '_targeteps' + str(args.target_eps) + '_losstype_' + args.loss_type + '_seed' + str(args.seed)
print(save_name)
save_path = f'./result/unsupervised_verification/{args.dataset}/{args.model}/{time.strftime("%m%d")}'
if not os.path.exists(save_path):
    os.makedirs(save_path) 
log_path = os.path.join(save_path, save_name + '.csv')

pgd_acc = cbc_acc = ibp_acc = crown_acc = crown_ibp_acc = 0

for iter in range(args.ver_total):
    try:
        img_ori = image[iter]
        img_target = adv_target[iter]
        model_ori.to('cpu')
        f_ori = F.normalize(model_ori(img_ori.to('cpu').detach()), p=2, dim=1)
        f_target = F.normalize(model_ori(img_target.to('cpu').detach()), p=2, dim=1)

        log.loc[iter, 'label'] = label[iter]
        # Main function to run verification
        start =  time.time()
        l, nodes, modify_net, bound_img = unsupervised_bab(model_ori, img_ori, f_ori, f_target, args.norm, args.epsilon, args, model_output_size, img_clip['max'], img_clip['min'])
        log.loc[iter, 'cbc_time'] = time.time() - start
        log.loc[iter, 'cbc'] = l
        if l > 0:
            cbc_acc += 1
        log.loc[iter, 'cbc_acc'] = cbc_acc/(iter+1)*100.

        # pgd bound
        modify_net = modify_net.to(args.device)
        pgd_bound = modify_net(adv_eps[iter].to(args.device)).item()
        if pgd_bound > 0:
            pgd_acc += 1
        log.loc[iter, 'pgd'] = pgd_bound
        log.loc[iter, 'pgd_acc'] = pgd_acc/(iter+1)*100.
        log.loc[iter, 'ori'] = modify_net(img_ori.to(args.device)).item()
        log.loc[iter, 'target'] = modify_net(img_target.to(args.device)).item()

        # compare with other verification algorithm
        bound_modify_net = BoundedModule(modify_net, torch.empty_like(img_ori.to(args.device)), bound_opts={"conv_mode": "patches"})
        for ii, method in enumerate(['IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)']):
            lb, ub = bound_modify_net.compute_bounds(x=(bound_img,), method=method.split()[0])
            lb = lb.detach().cpu().numpy()
            ub = ub.detach().cpu().numpy()
            print("Bounding method:", method)
            print("f_0(x_0): {l:8.3f} <= f_0(x_0+delta) <= {u:8.3f}".format(l=lb[0][0], u=ub[0][0]))

            if ii == 0:
                log.loc[iter, 'ibp'] = lb[0][0]
                if lb[0][0] > 0:
                    ibp_acc += 1
                log.loc[iter, 'ibp_acc'] = ibp_acc/(iter+1)*100.
            elif ii == 1:
                log.loc[iter, 'crown-ibp'] = lb[0][0]
                if lb[0][0] > 0:
                    crown_ibp_acc += 1
                log.loc[iter, 'crown-ibp_acc'] = crown_ibp_acc/(iter+1)*100.
            else:
                log.loc[iter, 'crown'] = lb[0][0]
                if lb[0][0] > 0:
                    crown_acc += 1
                log.loc[iter, 'crown_acc'] = crown_acc/(iter+1)*100.
        
        log.to_csv(log_path)
    except:
        pass