from __future__ import print_function

import argparse
import csv
import os
import json
import copy
import time
import random
from collections import OrderedDict
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import rocl.data_loader as data_loader
from rocl.utils import progress_bar, checkpoint
from rocl.attack_lib import FastGradientSignUntargeted
from rocl.loss import pairwise_similarity, NT_xent
from beta_crown.utils import *

parser = argparse.ArgumentParser(description='RoCL linear training')
parser.add_argument('--module',action='store_true')

##### arguments for RoCL Linear eval #####
parser.add_argument('--trans', default=False, type=bool, help='use transformed sample')
parser.add_argument('--clean', default=False, type=bool, help='use clean sample')
parser.add_argument('--adv_img', default=False, type=bool, help='use adversarial sample')
parser.add_argument('--finetune', default=False, type=bool, help='finetune the model')
parser.add_argument('--ss', default=False, type=bool, help='using self-supervised learning loss')

##### arguments for Training Self-Sup #####
parser.add_argument('--train_type', default='linear_eval', type=str, help='contrastive/linear_eval/test/supervised')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--step_size', default=50, type=int, help='scheduler step size')

parser.add_argument('--decay', default=1e-6, type=float, help='weight decay')
parser.add_argument('--dataset', default='cifar-10', type=str, help='cifar-10/mnist')

parser.add_argument('--name', default='', type=str, help='name of run')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--batch-size', default=256, type=int, help='batch size / multi-gpu setting: batch per gpu')
parser.add_argument('--epoch', default=100, type=int, help='total epochs to run')

parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help='use cpu or cuda')
parser.add_argument('--gpuno', default='0', type=str)

##### arguments for data augmentation #####
parser.add_argument('--color_jitter_strength', default=0.5, type=float, help='0.5 for CIFAR')
parser.add_argument('--temperature', default=0.5, type=float, help='temperature for pairwise-similarity')

##### arguments for PGD attack & Adversarial Training #####
parser.add_argument('--attack_type', type=str, default='linf', help='adversarial l_p')
parser.add_argument('--epsilon', type=float, default=8.0/255, 
    help='maximum perturbation of adversaries (8/255(0.0314) for cifar-10)')
parser.add_argument('--alpha', type=float, default=0.007, help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
parser.add_argument('--k', type=int, default=10, help='maximum iteration when generating adversarial examples')
parser.add_argument('--random_start', type=bool, default=True, help='True for PGD')

##### model select #####
parser.add_argument("--model", type=str, default="cifar_model",
                    help='model name (cifar_model, cifar_model_deep, cifar_model_wide, cnn_4layer, cnn_4layer_b, mnist_cnn_4layer)')
parser.add_argument('--no_load_weight', action='store_true', default=False, help='load the weight')
parser.add_argument('--load_checkpoint', default='', type=str, help='PATH TO CHECKPOINT')
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

# Data
print('==> Preparing data..')
if not (args.train_type=='linear_eval'):
    assert('wrong train phase...')
else:
    trainloader, traindst, testloader, testdst = data_loader.get_dataset(args)

# Model
print('==> Building model..')
# the path of simclr's projector
projector_path = ''

model = torch.load(args.load_checkpoint)
print(model)

projector = None
if args.ss:
    projector = torch.load(projector_path)
    projector.to(args.device)

model_output_size = list(model.children())[-1].weight.data.shape[0]
linear = nn.Sequential(nn.Linear(model_output_size, 10))

model.to(args.device)
linear.to(args.device)

model_params = []
if args.finetune:
    model_params += model.parameters()
    if args.ss:
        model_params += projector.parameters()
model_params += linear.parameters()
#loptim = torch.optim.SGD(model_params, lr = args.lr, momentum=0.9, weight_decay=args.decay)
loptim = torch.optim.Adam(model_params, lr = args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(loptim, args.step_size, gamma=0.1)

attacker = None
if args.adv_img:
    attack_info = 'Adv_train_epsilon_'+str(args.epsilon)+'_alpha_'+ str(args.alpha) + '_max_iters_' + str(args.k) + '_type_' + str(args.attack_type) + '_randomstart_' + str(args.random_start)
    save_name += (attack_info + '_')
    print("Adversarial training info...")
    print(attack_info)
    img_clip = min_max_value(args)
    attacker = FastGradientSignUntargeted(model, linear='None', epsilon=args.epsilon, alpha=args.alpha, min_val=img_clip['min'].to(args.device), max_val=img_clip['max'].to(args.device), max_iters=args.k, _type=args.attack_type)

criterion = nn.CrossEntropyLoss()

def linear_train(epoch, model, Linear, projector, loptim, attacker=None):
    Linear.train()
    if args.finetune:
        model.train()
        if args.ss:
            projector.train()
    else:
        model.eval()

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (ori, inputs, inputs_2, target) in enumerate(trainloader):
        ori, inputs_1, inputs_2, target = ori.cuda(), inputs.cuda(), inputs_2.cuda(), target.cuda()
        input_flag = False
        if args.trans:
            inputs = inputs_1
        else:
            inputs = ori

        if args.adv_img:
            advinputs      = attacker.perturb(original_images=inputs, labels=target, random_start=args.random_start)
        
        if args.clean:
            total_inputs = inputs
            total_targets = target
            input_flag = True
        
        if args.ss:
            total_inputs = torch.cat((inputs, inputs_2))
            total_targets = torch.cat((target, target))

        if args.adv_img:
            if input_flag:
                total_inputs = torch.cat((total_inputs, advinputs))
                total_targets = torch.cat((total_targets, target))
            else:
                total_inputs = advinputs
                total_targets = target
                input_flag = True

        if not input_flag:
            assert('choose the linear evaluation data type (clean, adv_img)')

        feat   = model(total_inputs)
        if args.ss:
            output_p = projector(feat)
            B = ori.size(0)

            similarity, _ = pairwise_similarity(output_p[:2*B,:2*B], temperature=args.temperature, multi_gpu=False, adv_type = 'None')
            simloss  = NT_xent(similarity, 'None')
        
        output = Linear(feat)

        _, predx = torch.max(output.data, 1)
        loss = criterion(output, total_targets)
        
        if args.ss:
            loss += simloss

        correct += predx.eq(total_targets.data).cpu().sum().item()
        total += total_targets.size(0)
        acc = 100.*correct/total

        total_loss += loss.data

        loptim.zero_grad()
        loss.backward()
        loptim.step()
        
        progress_bar(batch_idx, len(trainloader),
                    'Loss: {:.4f} | Acc: {:.2f}'.format(total_loss/(batch_idx+1), acc))

    print ("Epoch: {}, train accuracy: {}".format(epoch, acc))
    return acc, model, Linear, projector, loptim

def test(model, Linear):
    model.eval()
    Linear.eval()

    test_loss = 0
    correct = 0
    total = 0

    for idx, (image, label) in enumerate(testloader):
        img = image.cuda()
        y = label.cuda()

        out = Linear(model(img))

        _, predx = torch.max(out.data, 1)
        loss = criterion(out, y)

        correct += predx.eq(y.data).cpu().sum().item()
        total += y.size(0)
        acc = 100.*correct/total

        test_loss += loss.data
        progress_bar(idx, len(testloader),'Testing Loss {:.3f}, acc {:.3f}'.format(test_loss/(idx+1), acc))

    print ("Test accuracy: {0}".format(acc))
    return (acc, model, Linear)

save_name += (args.train_type + '_lr'+str(args.lr) + '_' +args.model + '_' + args.dataset + '_epoch'+ str(args.epoch) + '_batch' + str(args.batch_size) 
                + '_loadweight_'+ str(not args.no_load_weight) + '_trans_'+ str(args.trans) + '_adv_'+ str(args.adv_img) + '_finetune_'+ str(args.finetune) + '_ss_'+ str(args.ss) + '_seed' + str(args.seed))
save_path = f'./result/linear_evaluate/{args.dataset}/{args.model}/{time.strftime("%m%d")}/{time.strftime("%H%M", time.localtime())}_{save_name}'
if not os.path.exists(save_path):
    os.makedirs(save_path) 

logname = (os.path.join(save_path, 'log.csv'))

with open(logname, 'w') as logfile:
    logwriter = csv.writer(logfile, delimiter=',')
    logwriter.writerow(['epoch', 'train acc','test acc', args.load_checkpoint])

##### Linear evaluation #####
for epoch in range(args.epoch):
    print('Epoch ', epoch)

    train_acc, model, linear, projector, loptim = linear_train(epoch, model=model, Linear=linear, projector=projector, loptim=loptim, attacker=attacker)
    test_acc, model, linear = test(model, linear)
    scheduler.step()

    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_acc, test_acc])

checkpoint(model, test_acc, args.epoch, args, loptim, save_path)
checkpoint(linear, test_acc, args.epoch, args, loptim, save_path, save_name_add='_linear')
supervised_model = nn.Sequential(*(list(model.children()) + list(linear.children())))
print(supervised_model)
torch.save(supervised_model, os.path.join(save_path, 'supervised.pkl')) 
