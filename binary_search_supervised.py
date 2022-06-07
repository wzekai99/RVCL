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
from collections import OrderedDict
import torch.nn.functional as F

import rocl.data_loader as data_loader
from beta_crown.utils import *

from beta_crown.auto_LiRPA import BoundedModule, BoundedTensor
from beta_crown.auto_LiRPA.perturbations import *

parser = argparse.ArgumentParser(description='supervised binary search')
##### arguments for CROWN #####
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help='use cpu or cuda')
parser.add_argument('--gpuno', default='0', type=str)

parser.add_argument("--norm", type=float, default='inf', help='p norm for epsilon perturbation')
parser.add_argument("--model", type=str, default="cnn_4layer_b", help='model name (cifar_model, cifar_model_deep, cifar_model_wide, cnn_4layer, cnn_4layer_b, mnist_cnn_4layer)')
parser.add_argument("--batch_size", type=int, default=256, help='batch size')

##### arguments for model #####
parser.add_argument('--train_type', default='linear_eval', type=str, help='contrastive/linear eval/test/supervised')
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
upper_eps = (torch.max(img_clip['max']) - torch.min(img_clip['min'])).item()

# Model
print('==> Building model..')
model = torch.load(args.load_checkpoint) 
model.to(args.device)

def generate_ver_data(loader, total, class_num):
    count = [0 for _ in range(class_num)]
    per_class = total//class_num
    data_loader = iter(loader)
    ans_image= []
    ans_label = []
    while sum(count)<total:
        (img, label) = next(data_loader)
        i = int(label)
        img = img.to(args.device)
        label = label.to(args.device)
        out = model(img)
        _, predx = torch.max(out.data, 1)
        if count[i] < per_class and predx.item() == label.item():
            ans_image.append(img)
            ans_label.append(i)
            count[i] += 1
    return ans_image, ans_label

def supervised_search(data, label, norm, args, data_max=None, data_min=None, upper = 1.0, lower = 0.0, tol = 0.000001, max_steps = 100):
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
        lb, ub = bound_net.compute_bounds(x=(image,), method='backward')
        true_label_low = lb[0][label].item()
        target_label_upper = np.delete(ub.detach().cpu().numpy()[0], label).max()
        success = target_label_upper < true_label_low
        print("[binary search] step = {}, current = {:.6f}, success = {}, val = {:.2f}".format(step,eps,success,true_label_low-target_label_upper))

        if success: # success at current value
            lower = eps
        else:
            upper = eps
        step += 1
        if step >= max_steps:
            break
    return lower, step, np.argmax(np.delete(ub.detach().cpu().numpy()[0], label))

#save log
save_name = ''
if args.name != '':
    save_name = args.name + '_'
save_name += (args.model + '_ver_total' + str(args.ver_total))

save_dir = f'./result/supervised_binary/{args.dataset}/{time.strftime("%m%d")}'
save_path = f'{save_dir}/{time.strftime("%H%M", time.localtime())}_{save_name}.csv'
if not os.path.exists(save_dir):
    os.makedirs(save_dir) 

# Data
print('==> Preparing data..')
_, _, testloader, testdst = data_loader.get_dataset(args)
total_image, total_label = generate_ver_data(testloader, args.ver_total, class_num=10)
bound_net = BoundedModule(model, torch.empty_like(total_image[0].to(args.device)), bound_opts={"conv_mode": "patches"}, device=args.device)

log = pd.DataFrame(columns=['veri_lower', 'true_label', 'target_label', 'steps', 'time', 'avg_veri_lower', 'min_veri_lower', 'max_veri_lower', 'avg_time', 'avg_steps', args.load_checkpoint])
avg_veri_lower = []
avg_time = []
avg_steps = []

for iter in range(args.ver_total):
    print("verifying {}-th image".format(iter))
    img = total_image[iter]
    label = total_label[iter]
    start = time.time()
    low, step, target_label = supervised_search(img.to('cpu'), label, args.norm, args, img_clip['max'], img_clip['min'], upper=upper_eps, lower=0.0, max_steps=args.max_steps)
    per_time = time.time() - start

    log.loc[iter, 'veri_lower'] = low
    log.loc[iter, 'true_label'] = label
    log.loc[iter, 'target_label'] = target_label
    log.loc[iter, 'steps'] = step
    log.loc[iter, 'time'] = per_time

    avg_veri_lower.append(low)
    avg_time.append(per_time)
    avg_steps.append(step)
    log.to_csv(save_path)

log.loc[0, 'avg_veri_lower'] = np.mean(avg_veri_lower)
log.loc[0, 'min_veri_lower'] = min(avg_veri_lower)
log.loc[0, 'max_veri_lower'] = max(avg_veri_lower)
log.loc[0, 'avg_time'] = np.mean(avg_time)
log.loc[0, 'avg_steps'] = np.mean(avg_steps)
log.to_csv(save_path)
