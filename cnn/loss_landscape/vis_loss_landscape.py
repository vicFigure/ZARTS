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

def create_random_direction(alphas):
    """
        Setup a random (normalized) direction with the same dimension as
        the weights or states.

        Args:
          net: the given trained model
          dir_type: 'weights' or 'states', type of directions.
          ignore: 'biasbn', ignore biases and BN parameters.
          norm: direction normalization method, including
                'filter" | 'layer' | 'weight' | 'dlayer' | 'dfilter'

        Returns:
          direction: a random direction with the same dimension as weights or states.
    """
    # random direction
    direction =  [torch.randn(a.size()) for a in alphas]
    return direction

def obtain_grad_direction(model, valid_queue):
  direction = [0. for a in model.arch_parameters()]
  alphas = model.arch_parameters()
  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda()

    logits = model(input)
    loss = criterion(logits, target)
    grads_alphas = torch.autograd.grad(loss, alphas, grad_outputs=None,
                       allow_unused=True,
                       retain_graph=None,
                       create_graph=False)
    direction = [d+grad for d, grad in zip(direction, grads_alpha)]

#    if step % args.report_freq == 0:
#      logging.info('valid %03d %.3e %.3f %.3f', step, objs.avg, top1.avg, top5.avg)
  direction = [d/len(valid_queue) for d in direction]
  return direction

def norm_direction(direction, norm, alphas):
  if norm == 'rowwise':
      for d, alpha in zip(direction, alphas):
          for r in range(d.shape[0]):
              d[r,:].mul_(alpha[r,:].norm()/(d[r,:].norm() + 1e-10))
  elif norm == 'cellwise':
      # Rescale the entries in the direction so that each cell direction
      # has the unit norm.
      for d, alpha in zip(direction, alphas):
          d.mul_(alpha.norm()/(d.norm() + 1e-10))
  elif norm == 'modelwise':
      # Rescale the entries in the direction so that the model direction has
      # the unit norm.
      norm_d = 0.
      norm_a = 0.
      for d, a in zip(direction, alphas):
        norm_d += (d*d).sum()
        norm_a += (a*a).sum()
      norm_d.sqrt_()
      norm_a.sqrt_()
      for d in direction:
        d.mul_(alpha/(norm+1e-10))
  else:
      raise(ValueError("Not implemented norm named as %s"%norm))
  return direction


def obtain_direction(model, valid_queue, direction_file=None, method='random', norm_type='cellwise'):
  if method == 'fromfile' and direction_file is None:
      raise(ValueError("For fromfile method, you should give a direction file name"))
  if method == 'fromfile' and not os.path.exists(direction_file):
      print("Since no direction file, we use default random method, and then save the direction to the file: %s"%direction_file)
      method = 'random'

  if method == 'fromfile':
    print("file exists, loading direction file from %s"%direction_file)
    d_dict = json.load(open(direction_file,'r'))
    d1 = d_dict['d1']
    d1 = [torch.from_numpy(np.array(d)) for d in d1]
    d2 = d_dict['d2']
    d2 = [torch.from_numpy(np.array(d)) for d in d2]
    return d1, d2

  elif method == 'random':
      print("direction as random Gaussian vector")
      d1 = create_random_direction(model.arch_parameters())    
      d2 = create_random_direction(model.arch_parameters())    

  elif method == 'grad':
      print("direction as the gradient of alpha")
      d1 = obtain_grad_direction(model, valid_queue)
      d2 = create_random_direction(model.arch_parameters())    

  else:
      raise(ValueError("Not implemented method named as %s"%method))

  # to get orthogonal direction
  norm = 0.
  inner_product = 0.
  for i, d in enumerate(d1): 
      norm += (d*d).sum()
      inner_product += (d*d2[i]).sum()
  norm.sqrt_()
  for i, d in enumerate(d1): d2[i] = d2[i] - d*inner_product/norm

  # norm
  alphas = [alpha.data.cpu() for alpha in model.arch_parameters()]
  d1 = norm_direction(d1, norm=norm_type, alphas=alphas)
  d2 = norm_direction(d2, norm=norm_type, alphas=alphas)

  # save direction to json file
  if direction_file is not None and (not os.path.exists(direction_file)):
    print("Save direction to file %s"%direction_file)
    d_dict = {'d1': [d.tolist() for d in d1], 'd2':[d.tolist() for d in d2]}
    with open(direction_file, 'w') as f:
      json.dump(d_dict, f)
  return d1, d2


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

def compute_loss_landscape(d1, d2, x, y, valid_queue, model, criterion, es_infer=-1, iter_w=0, train_queue=None):
  losses = np.zeros((len(x), len(y)))
  accs = np.zeros((len(x), len(y)))

  state_dict = model.state_dict()
  model_new = model.new()
  arch_params = model_new.arch_parameters()
  alpha_value = copy.deepcopy([a.data.cpu().numpy() for a in arch_params])
  step = 0
  for i, delta1 in enumerate(x):
    for j, delta2 in enumerate(y):
        model_new.load_state_dict(state_dict, strict=True)
        arch_params[0].data.copy_(torch.from_numpy(alpha_value[0]) + d1[0]*delta1+d2[0]*delta2)
        arch_params[1].data.copy_(torch.from_numpy(alpha_value[1]) + d1[1]*delta1+d2[1]*delta2)
        if iter_w > 0:
          torch.cuda.empty_cache()
          assert (train_queue is not None)
          optimizer = torch.optim.SGD(
              model_new.parameters(),
              0.001,
              momentum=args.momentum,
              weight_decay=args.weight_decay)
          train_weight(train_queue, model_new, iter_w, criterion, optimizer, verbose=False)
          torch.cuda.empty_cache()
        acc, loss = infer(valid_queue, model_new, criterion, early_stop=es_infer)
        if step % 10 == 0:
          print("Pos: %f, %f; Acc: %f; Loss: %f <Cost %f h>"%(delta1, delta2, acc, loss, (time.time()-SINCE)*1./3600))
        accs[i,j] = acc
        losses[i,j] = loss
        step += 1
  return accs, losses

def my_interpolate(X, Y, Z, x_new, y_new):
  f = interpolate.interp2d(X, Y, Z, kind='cubic')
  z_new = f(x_new, y_new)
  x_new, y_new = np.meshgrid(x_new, y_new)
  return x_new, y_new, z_new
    

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

  # obtain directions of pertubation
  print(">>> Obtaining direction of pertubation")
  direction_file = os.path.join(save_dir, 'direction.json')
  d1, d2 = obtain_direction(model, valid_queue, direction_file, method='fromfile', norm_type='cellwise')
  x = np.linspace(args.xmin, args.xmax, num=args.xnum)
  y = np.linspace(args.ymin, args.ymax, num=args.ynum)

#            '0': [np.concatenate((np.arange(0.2, 0.5, 0.1), np.arange(0.5, 4.0, 0.5), np.arange(4.0, 10, 2)), axis=-1), np.concatenate((np.arange(0.1, 0.5, 0.1), np.arange(0.5, 0.7, 0.05), np.arange(0.7, 1.0, 0.02)), axis=-1)],
#            '1': [np.concatenate((np.arange(0.2, 0.5, 0.1), np.arange(0.5, 4.0, 0.5), np.arange(4.0, 10, 2)), axis=-1), np.concatenate((np.arange(0.1, 0.3, 0.1), np.arange(0.3, 0.6, 0.05), np.arange(0.6, 1.0, 0.02)), axis=-1)],
  iters_w = [0, 1, 10]
  all_levels = {
            '0': [np.concatenate((np.arange(0.52, 0.53, 0.002), np.arange(0.54, 0.6, 0.01), np.arange(0.6, 0.8, 0.05), np.arange(0.85, 1.5, 0.1)), axis=-1), np.concatenate((np.arange(0.4, 0.6, 0.05), np.arange(0.6, 0.7, 0.02), np.arange(0.7, 0.8, 0.01), np.arange(0.81, 0.9, 0.005), np.arange(0.9, 1.0, 0.01)), axis=-1)],
            '1': [np.concatenate((np.arange(0.52, 0.53, 0.002), np.arange(0.54, 0.6, 0.01), np.arange(0.6, 0.8, 0.05), np.arange(0.85, 1.5, 0.1)), axis=-1), np.concatenate((np.arange(0.4, 0.6, 0.05), np.arange(0.6, 0.7, 0.02), np.arange(0.7, 0.8, 0.01), np.arange(0.81, 0.9, 0.005), np.arange(0.9, 1.0, 0.01)), axis=-1)],
            '10': [np.concatenate((np.arange(0.52, 0.53, 0.002), np.arange(0.54, 0.6, 0.01), np.arange(0.6, 0.8, 0.05), np.arange(0.85, 1.5, 0.1)), axis=-1), np.concatenate((np.arange(0.4, 0.6, 0.05), np.arange(0.6, 0.7, 0.02), np.arange(0.7, 0.8, 0.01), np.arange(0.81, 0.9, 0.005), np.arange(0.9, 1.0, 0.01)), axis=-1)],

           }
  for iter_w in iters_w:
    print(">>> Deal with iter_w=%d"%iter_w)
    landscape_dir = os.path.join(save_dir, 'iterw-%d'%iter_w)
    if not os.path.exists(landscape_dir):
      os.makedirs(landscape_dir)
    acc_file = os.path.join(landscape_dir, 'acc.npy')
    loss_file = os.path.join(landscape_dir, 'loss.npy')
    # compute landscape
    if os.path.exists(acc_file) and os.path.exists(loss_file):
      print(">>> Loading acc an loss from file")
      acc = np.load(acc_file)
      loss = np.load(loss_file)
    else:
      print(">>> Computing acc an loss")
      if not os.path.exists(landscape_dir):
        os.makedirs(landscape_dir)
      acc, loss = compute_loss_landscape(d1, d2, x, y, valid_queue, model, criterion, es_infer=-1, iter_w=iter_w, train_queue=train_queue)
      # save landscape
      landscape_file = os.path.join(landscape_dir, 'acc.npy')
      np.save(landscape_file, acc) # load: np.load(file)
      landscape_file = os.path.join(landscape_dir, 'loss.npy')
      np.save(landscape_file, loss) # load: np.load(file)

    # plot loss landscape
    print(">>> Plotting landscape")
    x_new = np.linspace(args.xmin, args.xmax, num=100)
    y_new = np.linspace(args.ymin, args.ymax, num=100)

    X, Y = np.meshgrid(x, y)
    Z = loss
#    X,Y,Z = my_interpolate(X, Y, Z, x_new, y_new)
#    vmin, vmax, vlevel = 0.0, 1.2, 0.03
#    levels = np.arange(vmin, vmax, vlevel)
    levels = all_levels['%d'%iter_w][0]
    plot_2D(X, Y, Z, landscape_dir, 'valid_loss', levels, args.show, args.azim, args.elev, scatter_func=np.argmin)
    
    # plot acc landscape
    print(">>> Plotting landscape")
    X, Y = np.meshgrid(x, y)
    Z = acc / 100.
#    X,Y,Z = my_interpolate(X, Y, Z, x_new, y_new)
#    vmin, vmax, vlevel = 0.4, 1.0, 0.01
#    levels = np.arange(vmin, vmax, vlevel)
    levels = all_levels['%d'%iter_w][1]
    plot_2D(X, Y, Z, landscape_dir, 'valid_acc', levels, args.show, args.azim, args.elev, scatter_func=np.argmax)


if __name__ == '__main__':
  main()
