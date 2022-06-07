import argparse
import copy
import random
import sys
import time
import gc
import json
import numpy as np
import pandas as pd
import copy
import torch.nn.functional as F

import rocl.data_loader as data_loader
from rocl.attack_lib import FastGradientSignUntargeted,RepresentationAdv

from beta_crown.model_beta_CROWN import LiRPAConvNet, return_modify_model
from beta_crown.relu_conv_parallel import relu_bab_parallel
from beta_crown.utils import *

from beta_crown.auto_LiRPA import BoundedModule, BoundedTensor
from beta_crown.auto_LiRPA.perturbations import *

parser = argparse.ArgumentParser(description='unsupervised binary search')

##### arguments for CROWN #####
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help='use cpu or cuda')
parser.add_argument('--gpuno', default='0', type=str)

parser.add_argument("--norm", type=float, default='inf', help='p norm for epsilon perturbation')
parser.add_argument("--model", type=str, default="cnn_4layer_b", help='model name (cifar_model, cifar_model_deep, cifar_model_wide, cnn_4layer, cnn_4layer_b, mnist_cnn_4layer)')
parser.add_argument("--batch_size", type=int, default=256, help='batch size')

##### arguments for model #####
parser.add_argument('--train_type', default='contrastive', type=str, help='contrastive/linear eval/test/supervised')
parser.add_argument('--dataset', default='cifar-10', type=str, help='cifar-10/mnist')
parser.add_argument('--load_checkpoint', default='', type=str, help='PATH TO CHECKPOINT')
parser.add_argument('--name', default='', type=str, help='name of run')
parser.add_argument('--seed', default=1, type=int, help='random seed')

##### arguments for data augmentation #####
parser.add_argument('--color_jitter_strength', default=0.5, type=float, help='0.5 for CIFAR, 1.0 for ImageNet')
parser.add_argument('--temperature', default=0.5, type=float, help='temperature for pairwise-similarity')

##### arguments for PGD attack & Adversarial Training #####
parser.add_argument('--attack_type', type=str, default='linf', help='adversarial l_p')
parser.add_argument('--target_eps', type=float, default=16.0/255, help='maximum perturbation of adversaries (8/255 0.0314 for cifar-10)')
parser.add_argument('--alpha', type=float, default=0.001, help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
parser.add_argument('--k', type=int, default=150, help='maximum iteration when generating adversarial examples')
parser.add_argument('--random_start', type=bool, default=True, help='True for PGD')
parser.add_argument('--loss_type', type=str, default='mse', help='loss type for Rep: mse/sim/l1/cos')

##### arguments for binary_search #####
parser.add_argument('--mini_batch', type=int, default=10, help='mini batch for PGD')
parser.add_argument("--ver_total", type=int, default=100, help='number of img to verify')
parser.add_argument('--max_steps', type=int, default=200, help='max steps for search')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuno
print_args(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

img_clip = min_max_value(args)

# Model
print('==> Building model..')
model_ori = torch.load(args.load_checkpoint) 
model_ori.to(args.device)
print(model_ori)
output_size = list(model_ori.children())[-1].weight.data.shape[0]

def generate_attack(args, ori, target):
    pgd_target = RepresentationAdv(model_ori, None, epsilon=args.target_eps, alpha=args.alpha, min_val=img_clip['min'].to(args.device), max_val=img_clip['max'].to(args.device), max_iters=args.k, _type=args.attack_type, loss_type=args.loss_type)
    adv_target = pgd_target.attack_pgd(original_images=ori.to(args.device), target=target.to(args.device), type='attack', random_start=True)

    pgd = RepresentationAdv(model_ori, None, epsilon=args.epsilon, alpha=args.alpha, min_val=img_clip['min'].to(args.device), max_val=img_clip['max'].to(args.device), max_iters=args.k, _type=args.attack_type, loss_type=args.loss_type)
    adv_img = pgd.attack_pgd(original_images=ori.to(args.device), target=adv_target.to(args.device), type='sim', random_start=True)
    return adv_target, adv_img

def generate_ver_data(loader, total, class_num, adv=True):
    count = [0 for _ in range(class_num)]
    per_class = total//class_num
    data_loader = iter(loader)
    ans_image= []
    if adv:
        adv_target = []
        adv_eps = []
    ans_label = []
    while sum(count)<total:
        (ori, aug_img, _, label) = next(data_loader)
        i = int(label)
        if count[i] < per_class:
            ans_image.append(ori)
            ans_label.append(i)
            if adv:
                i1, i2 = generate_attack(args, ori, aug_img)
                adv_target.append(i1)
                adv_eps.append(i2)
            count[i] += 1
    if adv:
        return ans_image, adv_target, adv_eps, ans_label
    else:
        return ans_image, ans_label

def unsupervised_search(model_ori, data, ori, target, norm, args, output_size, data_max=None, data_min=None, upper = 1.0, lower = 0.0, tol = 0.000001, max_steps = 100):
    model = return_modify_model(model_ori, ori, target, output_size, contrastive=True, simplify=True)
    model.to(args.device)
    bound_modify_net = BoundedModule(model, torch.empty_like(data.to(args.device)), bound_opts={"conv_mode": "patches"}, device=args.device)

    step = 0
    while upper-lower > tol:
        eps = 0.5 * (lower + upper)
        if norm == np.inf:
            if data_max is None:
                data_ub = data + eps  # torch.min(data + eps, data_max)  # eps is already normalized
                data_lb = data - eps  # torch.max(data - eps, data_min)
            else:
                data_ub = torch.min(data + eps, data_max)
                data_lb = torch.max(data - eps, data_min)
        else:
            data_ub = data_lb = data
        ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb.to(args.device), x_U=data_ub.to(args.device))
        image = BoundedTensor(data, ptb).to(args.device)
        lb, _ = copy.deepcopy(bound_modify_net).compute_bounds(x=(image,), method='backward')
        lb = lb.item()
        print("[binary search] step = {}, current = {:.6f}, success = {}, val = {:.2f}".format(step,eps,lb > 0,lb))

        if lb > 0: # success at current value
            lower = eps
        else:
            upper = eps
        step += 1
        if step >= max_steps:
            break
    return lower, step, model

#save log
save_name = ''
if args.name != '':
    save_name = args.name + '_'
save_name += (args.model + '_minibatch' + str(args.mini_batch) + '_ver_total' + str(args.ver_total))

save_dir = f'./result/unsupervised_binary/{args.dataset}/{time.strftime("%m%d")}'
save_path = f'{save_dir}/{time.strftime("%H%M", time.localtime())}_{save_name}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir) 

# Data
print('==> Preparing data..')
_, _, testloader, testdst = data_loader.get_dataset(args)
image, label = generate_ver_data(testloader, args.ver_total, class_num=10, adv=False)
if args.mini_batch == 50:
    image_savenp = np.array([item.cpu().detach().numpy() for item in image])
    np.save(save_path+'.npy', image_savenp)

log = pd.DataFrame(columns=['ori', 'target', 'veri_lower', 'steps', 'value_upper', 'value_lower', 'time', 'avg_veri_lower', 'min_veri_lower', 'max_veri_lower', 'avg_time', 'total_time', 'min_time', 'max_time', 'avg_steps', 'min_steps', 'max_steps', 'total_avg_lower', 'avg_total_time', args.load_checkpoint])

line_iter = 0
upper_eps = (torch.max(img_clip['max']) - torch.min(img_clip['min'])).item()
total_avg = []
total_time = []
for batch_iter in range(args.ver_total//args.mini_batch):
    for ori_iter in range(args.mini_batch):
        total_ori_iter = batch_iter * args.mini_batch + ori_iter
        print("verifying {}-th image".format(total_ori_iter))
        avg_veri_lower = []
        avg_time = []
        avg_steps = []
        for target_iter in range(args.mini_batch):
            if ori_iter == target_iter:
                continue
            total_target_iter = batch_iter * args.mini_batch + target_iter
            print("verifying {}-th image, against {}-th image".format(total_ori_iter, total_target_iter))
            img_ori = image[total_ori_iter]
            img_target = image[total_target_iter]

            model_ori.to('cpu')
            f_ori = F.normalize(model_ori(img_ori.to('cpu').detach()), p=2, dim=1)
            f_target = F.normalize(model_ori(img_target.to('cpu').detach()), p=2, dim=1)

            start = time.time()
            log.loc[line_iter, 'veri_lower'], log.loc[line_iter, 'steps'], modify_net = unsupervised_search(model_ori, img_ori, f_ori, f_target, args.norm, args, output_size, img_clip['max'], img_clip['min'], upper=upper_eps, lower=0.0, max_steps=args.max_steps)
            log.loc[line_iter, 'time'] = time.time() - start

            log.loc[line_iter, 'value_upper'] = modify_net(img_ori.to(args.device)).item()
            log.loc[line_iter, 'value_lower'] = modify_net(img_target.to(args.device)).item()

            log.loc[line_iter, 'ori'] = total_ori_iter
            log.loc[line_iter, 'target'] = total_target_iter

            avg_veri_lower.append(log.loc[line_iter, 'veri_lower'])
            avg_time.append(log.loc[line_iter, 'time'])
            avg_steps.append(log.loc[line_iter, 'steps'])
            line_iter += 1
        
        log.loc[line_iter, 'avg_veri_lower'] = np.mean(avg_veri_lower)
        total_avg.append(np.mean(avg_veri_lower))
        log.loc[line_iter, 'avg_time'] = np.mean(avg_time)
        log.loc[line_iter, 'total_time'] = np.sum(avg_time)
        total_time.append(np.sum(avg_time))
        log.loc[line_iter, 'avg_steps'] = np.mean(avg_steps)
        log.loc[line_iter, 'min_veri_lower'] = min(avg_veri_lower)
        log.loc[line_iter, 'min_time'] = min(avg_time)
        log.loc[line_iter, 'min_steps'] = min(avg_steps)
        log.loc[line_iter, 'max_veri_lower'] = max(avg_veri_lower)
        log.loc[line_iter, 'max_time'] = max(avg_time)
        log.loc[line_iter, 'max_steps'] = max(avg_steps)
        log.loc[line_iter, 'ori'] = total_ori_iter
        line_iter += 1
        log.to_csv(save_path+'.csv')

log.loc[0, 'total_avg_lower'] = np.mean(total_avg)
log.loc[0, 'avg_total_time'] = np.mean(total_time)
log.to_csv(save_path+'.csv')
