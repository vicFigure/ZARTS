import sys
import os

import time
import copy
import argparse
import json
import csv

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
from scipy import interpolate
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dset
from torch.autograd import Variable
import torchvision.datasets as dset

sys.path.append('./')
import utils
from spaces import spaces_dict
from model_search import Network
from loss_landscape.plot_2D import plot_2D

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--space', type=str, default='s6', help='space index')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--val_batch_size', type=int, default=4096, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

# visualization
parser.add_argument('--x', type=str, default='-1:1:51', help='A string with format xmin:x_max:xnum')
parser.add_argument('--y', type=str, default='-1:1:51', help='A string with format ymin:y_max:ynum')
parser.add_argument('--test_infer', action='store_true', default=False, help='run inference to test whether the model is loaded correctly')
parser.add_argument('--show', action='store_true', default=False, help='show graph before saving')
parser.add_argument('--azim', type=float, default=-60, help='azimuthal angle for 3d landscape')
parser.add_argument('--elev', type=float, default=30, help='elevation angle for 3d landscape')

args = parser.parse_args()
args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
args.xnum, args.ynum = int(args.xnum), int(args.ynum)

CIFAR_CLASSES = 10
SINCE = time.time()

def load_alpha(alpha_file):
    print("Loading alpha from %s"%alpha_file)
    alphas_value = json.load(open(alpha_file,'r'))
    return [torch.from_numpy(np.array(alphas_value['alphas_normal'])), torch.from_numpy(np.array(alphas_value['alphas_reduce']))]

def infer(valid_queue, model, criterion, verbose=False, early_stop=-1):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = Variable(input).cuda()
      target = Variable(target).cuda()
  
      logits = model(input)
      loss = criterion(logits, target)
  
      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)
  
      if verbose and step % args.report_freq == 0:
        print('valid %03d %.3e %.3f %.3f' % (step, objs.avg, top1.avg, top5.avg))
      if early_stop > 0 and step > early_stop: break

  return top1.avg, objs.avg

def train_weight(train_queue, model, iter_w, criterion, optimizer, verbose=False):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()

  model.train()
  for step, (input, target) in enumerate(train_queue):
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda()
#    target = Variable(target, requires_grad=False).cuda()

    # update weight
    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if verbose and step % args.report_freq == 0:
      print('train %03d %e %f %f' % (step, objs.avg, top1.avg, top5.avg))
    if step >= iter_w: break

  return top1.avg, objs.avg


def main():
  save_dir = 'loss_landscape/save-model49-noES'
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  base_dir = 'ckpt/search-EXP-20210804-093301-task0'
  model_file = os.path.join(base_dir, 'ckpt/49.pt')
  weight_file = os.path.join(base_dir, 'ckpt/weights_49.pt')
  alpha_file = os.path.join(base_dir, 'results_of_7q/alpha/49.txt')
#  base_dir = model_file = weight_file = alpha_file = ''

  PRIMITIVES = spaces_dict[args.space]
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  # construct model & load weights and alpha
  if os.path.exists(model_file):
    print(">>> Loading model")
    model = torch.load(model_file)
    model = model.cuda()
    model.PRIMITIVES = PRIMITIVES
    model.drop_path_prob = 0.
  else:
    print(">>> Constructing model & load weights and alphas")
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, PRIMITIVES)
    model = model.cuda()
    if os.path.exists(weight_file):
      model.load_state_dict(torch.load_state(weight_file), strict=True)
    else: 
      print("No such file in path %s, so we randomly intial the weights"%weight_file)
    if os.path.exists(alpha_file):
      arch_params = model.arch_parameters()
      alpha_value = load_alpha(alpha_file)
      for i, alpha in enumerate(arch_params):
        alpha.data.copy_(alpha_value[i])
    else: 
      print("No such file in path %s, so we randomly intial the alpha"%alpha_file)

  # get data_loader
  print(">>> Constructing dataloader")
  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.val_batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=8)

  # verify on the validation set
  if args.test_infer:
    print(">>> First test whether the weight and alpha are loaded correctly")
    acc, loss = infer(valid_queue, model, criterion, verbose=True)
    print("Acc: %f, Loss: %f"%(acc, loss))

  iter_w = 10
  print(">>> Deal with iter_w=%d"%iter_w)
  cond_dir = os.path.join(save_dir, 'cond-iterw-%d'%iter_w)
  if not os.path.exists(cond_dir):
    os.makedirs(cond_dir)
  loss_file = os.path.join(cond_dir, 'cond_loss.npy')
  cond_file = os.path.join(cond_dir, 'cond.npy')

  # compute condition number
  if os.path.exists(cond_file) and os.path.exists(loss_file):
    print(">>> Loading loss from file")
    conds = np.load(cond_file)
  else:
    print(">>> Computing loss & condition number")
    losses, conds = [], []
    bk_state_dict = copy.deepcopy(model.state_dict())
    base_acc, base_loss = infer(valid_queue, model, criterion, early_stop=10)
    epsilon = 0.01
    with torch.no_grad():
      alpha_norm = torch.stack(model.arch_parameters(), dim=0).norm()
    for k in range(50):
      direction =  torch.stack([torch.randn(a.size()) for a in model.arch_parameters()], dim=0).cuda()
      direction = direction / (direction.norm()+1e-10) * epsilon
      for alpha, d in zip(model.arch_parameters(), direction):
        alpha.data.add_(d)
      model.load_state_dict(bk_state_dict)
      optimizer = torch.optim.SGD(
          model.parameters(),
          0.001,
          momentum=args.momentum,
          weight_decay=args.weight_decay)
      train_weight(train_queue, model, iter_w, criterion, optimizer, verbose=False)
      del optimizer
      acc, loss = infer(valid_queue, model, criterion, early_stop=10)
      cond = ((loss - base_loss)/epsilon * alpha_norm/base_loss).item()
      conds.append(abs(cond))
      print("%d-th cond:%f, max_cond:%f"%(k, cond, max(conds)))
      for alpha, d in zip(model.arch_parameters(), direction):
        alpha.data.sub_(d)
    conds = np.array(conds)

  print("Max cond: %f" % conds.max())

#    with torch.no_grad():
#      alpha_norm = torch.stack(model.arch_parameters(), dim=0).norm()
#    for alpha in model.arch_parameters():
#      tmp_losses, tmp_conds = np.zeros(alpha.shape), np.zeros(alpha.shape)
#      for i in range(alpha.shape[0]):
#        for j in range(alpha.shape[1]):
#          epsilon = 0.1 * alpha[i,j].item()
#          alpha[i,j].data.add_(epsilon)
#          model.load_state_dict(bk_state_dict)
#          optimizer = torch.optim.SGD(
#              model.parameters(),
#              0.001,
#              momentum=args.momentum,
#              weight_decay=args.weight_decay)
#          train_weight(train_queue, model, iter_w, criterion, optimizer, verbose=False)
#          del optimizer
#          acc, loss = infer(valid_queue, model, criterion, early_stop=10)
#          cond = ((loss - base_loss)/epsilon * base_loss/alpha[i,j].data).item()
#          cond = ((loss - base_loss)/epsilon * alpha_norm/base_loss).item()
#          tmp_conds[i,j] = abs(cond)
#          tmp_losses[i,j] = loss
#          print("%d-th cond:%f, max_cond:%f"%(i, cond, tmp_conds.max()))
#          alpha[i,j].data.sub_(epsilon)
#          torch.cuda.empty_cache()
#      losses.append(tmp_losses)
#      conds.append(tmp_conds)
#    losses = np.array(losses)
#    np.save(loss_file, losses)
#    conds = np.array(conds)
#    np.save(cond_file, conds)
#
#  print("Max cond: %f" % conds.max())
    


if __name__ == '__main__':
  main()

