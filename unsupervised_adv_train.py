from __future__ import print_function

import csv
import os

import torch
import torch.backends.cudnn as cudnn

import torch.optim as optim
import rocl.data_loader as data_loader

from rocl.attack_lib import FastGradientSignUntargeted,RepresentationAdv
from rocl.models.projector import Projector
from rocl.utils import progress_bar, checkpoint, AverageMeter, accuracy
from rocl.loss import pairwise_similarity, NT_xent
from beta_crown.utils import *

import time
import argparse
import random
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch RoCL training')
parser.add_argument('--module',action='store_true')

##### arguments for RoCL #####
parser.add_argument('--lamda', default=256, type=float, help='1.0/lamda')
parser.add_argument('--regularize_to', default='other', type=str, help='original/other')
parser.add_argument('--attack_to', default='other', type=str, help='original/other')
parser.add_argument('--loss_type', type=str, default='sim', help='loss type for Rep')
parser.add_argument('--advtrain_type', default='Rep', type=str, help='Rep/None')

##### arguments for Training Self-Sup #####
parser.add_argument('--train_type', default='contrastive', type=str, help='contrastive/linear eval/test')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--step_size', default=100, type=int, help='scheduler step size')
parser.add_argument('--decay', default=1e-6, type=float, help='weight decay')
parser.add_argument('--dataset', default='cifar-10', type=str, help='cifar-10/mnist')

parser.add_argument('--name', default='', type=str, help='name of run')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--batch-size', default=256, type=int, help='batch size / multi-gpu setting: batch per gpu')
parser.add_argument('--epoch', default=500, type=int, help='total epochs to run')

parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help='use cpu or cuda')
parser.add_argument('--gpuno', default='0', type=str)

##### arguments for data augmentation #####
parser.add_argument('--color_jitter_strength', default=0.5, type=float, help='0.5 for CIFAR')
parser.add_argument('--temperature', default=0.5, type=float, help='temperature for pairwise-similarity')

##### arguments for PGD attack & Adversarial Training #####
parser.add_argument('--attack_type', type=str, default='linf', help='adversarial l_p')
parser.add_argument('--epsilon', type=float, default=8.8/255, 
    help='maximum perturbation of adversaries (8/255(0.0314) for cifar-10)')
parser.add_argument('--alpha', type=float, default=0.007, help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
parser.add_argument('--k', type=int, default=10, help='maximum iteration when generating adversarial examples')
parser.add_argument('--random_start', type=bool, default=True, help='True for PGD')

##### model select #####
parser.add_argument("--model", type=str, default="cifar_model",
                    help='model name (cifar_model, cifar_model_deep, cifar_model_wide, cnn_4layer, cnn_4layer_b, mnist_cnn_4layer)')
parser.add_argument('--no_load_weight', action='store_true', default=False, help='load the weight')

args = parser.parse_args()


save_name = ''
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuno
print_args(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# Data
print('==> Preparing data..')
if not (args.train_type=='contrastive'):
    assert('wrong train phase...')
else:
    trainloader, traindst, testloader, testdst = data_loader.get_dataset(args)

# Model
print('==> Building model..')
model = load_model_contrastive(args, weights_loaded=not args.no_load_weight)
print(model)

model_output_size = list(model.children())[-1].weight.data.shape[0]
projector = Projector(input_size=model_output_size, medium=model_output_size+200)

if args.name != '':
    save_name = args.name + '_'

if 'Rep' in args.advtrain_type:
    Rep_info = 'Rep_attack_ep_'+str(args.epsilon)+'_alpha_'+ str(args.alpha) + '_max_iters_' + str(args.k) + '_losstype_' + args.loss_type + '_type_' + str(args.attack_type) + '_randomstart_' + str(args.random_start) + '_'
    save_name += Rep_info
    
    print("Representation attack info ...")
    print(Rep_info)
    img_clip = min_max_value(args)
    Rep = RepresentationAdv(model, projector, epsilon=args.epsilon, alpha=args.alpha, min_val=img_clip['min'].to(args.device), max_val=img_clip['max'].to(args.device), max_iters=args.k, _type=args.attack_type, loss_type=args.loss_type, regularize = args.regularize_to)
else:
    assert('wrong adversarial train type')

# Model upload to GPU # 
model.to(args.device)
projector.to(args.device)

# Aggregating model parameter & projection parameter #
model_params = []
model_params += model.parameters()
model_params += projector.parameters()

#optimizer  = optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=args.decay)
optimizer  = optim.Adam(model_params, lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=0.1)

save_name += (args.train_type + '_lr'+str(args.lr) + '_' + args.model + '_' + args.dataset + '_epoch'+ str(args.epoch) + '_batch' + str(args.batch_size) + '_l'+str(args.lamda) + '_loadweight_'+ str(not args.no_load_weight) +  '_seed' + str(args.seed))
save_path = f'./result/unsupervised/{args.dataset}/{args.model}/{time.strftime("%m%d")}/{time.strftime("%H%M", time.localtime())}_{save_name}'
if not os.path.exists(save_path):
    os.makedirs(save_path) 

def train(epoch):
    print('\nEpoch: %d' % epoch)

    model.train()
    projector.train()

    total_loss = 0
    reg_simloss = 0
    reg_loss = 0

    for batch_idx, (ori, inputs_1, inputs_2, label) in enumerate(trainloader):
        ori, inputs_1, inputs_2 = ori.to(args.device), inputs_1.to(args.device), inputs_2.to(args.device)

        if args.attack_to=='original':
            attack_target = inputs_1
        else:   # other
            attack_target = inputs_2

        if 'Rep' in args.advtrain_type :
            advinputs, adv_loss = Rep.get_loss(original_images=inputs_1, target=attack_target, optimizer=optimizer, weight=args.lamda, random_start=args.random_start)
            reg_loss += adv_loss.data

        if not (args.advtrain_type == 'None'):
            inputs = torch.cat((inputs_1, inputs_2, advinputs))
        else:
            inputs = torch.cat((inputs_1, inputs_2))
        
        outputs = projector(model(inputs))
        similarity, gathered_outputs = pairwise_similarity(outputs, temperature=args.temperature, multi_gpu=False, adv_type=args.advtrain_type) 
        
        simloss  = NT_xent(similarity, args.advtrain_type)
        
        if not (args.advtrain_type=='None'):
            loss = simloss + adv_loss
        else:   # rep
            loss = simloss
        
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.data
        reg_simloss += simloss.data
        
        optimizer.step()

        if 'Rep' in args.advtrain_type:
            progress_bar(batch_idx, len(trainloader),
                            'Loss: %.3f | SimLoss: %.3f | Adv: %.2f'
                            % (total_loss / (batch_idx + 1), reg_simloss / (batch_idx + 1), reg_loss / (batch_idx + 1)))
        else:
            progress_bar(batch_idx, len(trainloader),
                        'Loss: %.3f | Adv: %.3f'
                        % (total_loss/(batch_idx+1), reg_simloss/(batch_idx+1)))
        
    return (total_loss/batch_idx, reg_simloss/batch_idx)

def test(epoch, train_loss):
    model.eval()
    projector.eval()

    # Save at the last epoch #       
    if epoch == args.epoch - 1:
        checkpoint(model, train_loss, epoch, args, optimizer, save_path)
        checkpoint(projector, train_loss, epoch, args, optimizer, save_path, save_name_add='_projector')
       
    # Save at every 100 epoch #
    elif epoch % 100 == 0:
        checkpoint(model, train_loss, epoch, args, optimizer, save_path, save_name_add='_epoch_'+str(epoch))
        checkpoint(projector, train_loss, epoch, args, optimizer, save_path, save_name_add=('_projector_epoch_' + str(epoch)))

# Log and saving checkpoint information #
logname = (os.path.join(save_path, 'log.csv'))
print('Training info...')
print(save_name)

##### Log file #####
with open(logname, 'w') as logfile:
    logwriter = csv.writer(logfile, delimiter=',')
    logwriter.writerow(['epoch', 'train loss', 'reg loss'])

##### Training #####
for epoch in range(args.epoch):
    train_loss, reg_loss = train(epoch)
    test(epoch, train_loss)
    scheduler.step()

    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss.item(), reg_loss.item()])

