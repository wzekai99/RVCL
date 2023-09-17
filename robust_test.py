from __future__ import print_function
import csv
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import random
import argparse
import time

import rocl.data_loader as data_loader
from rocl.utils import progress_bar
from collections import OrderedDict
from rocl.attack_lib import FastGradientSignUntargeted
from beta_crown.utils import *

parser = argparse.ArgumentParser(description='linear eval test')
parser.add_argument('--train_type', default='linear_eval', type=str, help='standard')
parser.add_argument('--dataset', default='cifar-10', type=str, help='cifar-10/mnist')
parser.add_argument('--load_checkpoint', default='', type=str, help='PATH TO CHECKPOINT')
parser.add_argument("--model", type=str, default="cifar_model",
                    help='model name (cifar_model, cifar_model_deep, cifar_model_wide, cnn_4layer, cnn_4layer_b, mnist_cnn_4layer)')
parser.add_argument('--name', default='', type=str, help='name of run')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--batch-size', default=256, type=int, help='batch size / multi-gpu setting: batch per gpu')

parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help='use cpu or cuda')
parser.add_argument('--gpuno', default='0', type=str)
parser.add_argument('--test_mode', default='supervised', type=str, help='supervised/unsupervised')

##### arguments for data augmentation #####
parser.add_argument('--color_jitter_strength', default=0.5, type=float, help='0.5 for CIFAR, 1.0 for ImageNet')
parser.add_argument('--temperature', default=0.5, type=float, help='temperature for pairwise-similarity')

##### arguments for PGD attack & Adversarial Training #####
parser.add_argument('--attack_type', type=str, default='linf', help='adversarial l_p')
parser.add_argument('--epsilon', type=float, default=4.0/255, help='maximum perturbation of adversaries (8/255 for cifar-10)')
parser.add_argument('--alpha', type=float, default=0.006,
    help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
parser.add_argument('--k', type=int, default=50,
    help='maximum iteration when generating adversarial examples')
parser.add_argument('--random_start', type=bool, default=True, help='True for PGD')

args = parser.parse_args()

save_name = ''
if args.name != '':
    save_name = args.name + '_'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuno
print_args(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
img_clip = min_max_value(args)

# Data
print('==> Preparing data..')
if not (args.train_type=='linear_eval'):
    assert('wrong train phase...')
else:
    trainloader, traindst, testloader, testdst = data_loader.get_dataset(args)

# Model
print('==> Building model..')
model = torch.load(args.load_checkpoint)
model.to(args.device)
print(model)

attack_info = 'Adv_epsilon_'+str(args.epsilon)+'_alpha_'+ str(args.alpha) + '_max_iters_' + str(args.k) + '_type_' + str(args.attack_type) + '_randomstart_' + str(args.random_start)
save_name += (attack_info + '_')
print("Adversarial info...")
print(attack_info)
attacker = FastGradientSignUntargeted(model, linear='None', epsilon=args.epsilon, alpha=args.alpha, min_val=img_clip['min'].to(args.device), max_val=img_clip['max'].to(args.device), max_iters=args.k, _type=args.attack_type)

criterion = nn.CrossEntropyLoss()

# robust testing
model.eval()
test_clean_loss = 0
test_adv_loss = 0
clean_correct = 0
adv_correct = 0
clean_acc = 0
total = 0

for idx, (image, label) in enumerate(testloader):
    img = image.cuda()
    y = label.cuda()
    total += y.size(0)

    out = model(img)
    _, predx = torch.max(out.data, 1)
    clean_loss = criterion(out, y)
    clean_correct += predx.eq(y.data).cpu().sum().item()
    clean_acc = 100.*clean_correct/total
    test_clean_loss += clean_loss.data
    adv_inputs = attacker.perturb(original_images=img, labels=y, random_start=args.random_start)
    out = model(adv_inputs)
    
    _, predx = torch.max(out.data, 1)
    adv_loss = criterion(out, y)

    adv_correct += predx.eq(y.data).cpu().sum().item()
    adv_acc = 100.*adv_correct/total

    test_adv_loss += adv_loss.data
    progress_bar(idx, len(testloader),'Testing Loss {:.3f}, acc {:.3f} , adv Loss {:.3f}, adv acc {:.3f}'.format(test_clean_loss/(idx+1), clean_acc, test_adv_loss/(idx+1), adv_acc))

print ("Test accuracy: {0}/{1}".format(clean_acc, adv_acc))

save_name += (args.model + '_train_type_' + str(args.train_type) + '_seed' + str(args.seed))

if args.test_mode == 'unsupervised':
    save_dir = f'./result/unsupervised_robust_test/{args.dataset}/{time.strftime("%m%d")}'
else:
    save_dir = f'./result/supervised_robust_test/{args.dataset}/{time.strftime("%m%d")}'

save_path = f'{save_dir}/{time.strftime("%H%M", time.localtime())}_{save_name}.csv'
if not os.path.exists(save_dir):
    os.makedirs(save_dir) 

with open(save_path, 'w') as logfile:
    logwriter = csv.writer(logfile, delimiter=',')
    logwriter.writerow(['random_start', 'attack_type', 'epsilon', 'k', 'clean_acc', 'adv_acc', args.load_checkpoint])

with open(save_path, 'a') as logfile:
    logwriter = csv.writer(logfile, delimiter=',')
    logwriter.writerow([args.random_start, args.attack_type, args.epsilon, args.k, clean_acc, adv_acc])