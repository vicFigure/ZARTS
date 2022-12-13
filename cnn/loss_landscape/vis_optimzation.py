import math
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
from loss_landscape.plot_2D import plot_2D, plot_contour
from architect import Architect

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--space', type=str, default='s6', help='space index')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--val_batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.0, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
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
parser.add_argument('--arch_learning_rate', type=float, default=1.5e-1, help='learning rate for arch encoding') # 0.5 for darts
parser.add_argument('--arch_weight_decay', type=float, default=2e-3, help='weight decay for arch encoding')

# visualization
parser.add_argument('--x', type=str, default='-1:1:51', help='A string with format xmin:x_max:xnum')
parser.add_argument('--y', type=str, default='-1:1:51', help='A string with format ymin:y_max:ynum')
parser.add_argument('--test_infer', action='store_true', default=False, help='run inference to test whether the model is loaded correctly')
parser.add_argument('--show', action='store_true', default=False, help='show graph before saving')
parser.add_argument('--azim', type=float, default=-60, help='azimuthal angle for 3d landscape')
parser.add_argument('--elev', type=float, default=30, help='elevation angle for 3d landscape')

# For Important Sampling
parser.add_argument('--num_sample', type=int, default=4, help='number of samples in import sampling')
parser.add_argument('--iter_w', type=int, default=10, help='number of iterations to train weight for each sample')
parser.add_argument('--std', type=float, default=1.0e-3, help='variance of Gaussian Distrubution to sample delta alpha, square of standard deviation') # 1e-3
parser.add_argument('--tau', type=float, default=1., help='temprature of Exp Distribution')
parser.add_argument('--lamb', type=float, default=1e-1, help='Coefficient of Gaussian Distribution')

parser.add_argument('--rs_lr', type=float, default=1e-4, help='learning rate of arch u    pdate')
parser.add_argument('--rs_delta', type=float, default=0.1, help='smoothing parameter'    )
parser.add_argument('--rs_error', type=float, default=0.01, help='error bound for rs')
parser.add_argument('--rs_maxr', type=float, default=0.13, help='max sample step')
parser.add_argument('--rs_dist', type=str, default='uniform', help='sampling distribut    ion for alpha')
parser.add_argument('--rs_anti', type=bool, default=True, help='use antithetic samples     to decrease variance')
parser.add_argument('--rs_multipoint', action='store_true', default=False, help='use m    ulti-point gradient estimation')

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

  device = model.arch_parameters()[0].device

  if method == 'fromfile':
    print("file exists, loading direction file from %s"%direction_file)
    d_dict = json.load(open(direction_file,'r'))
    d1 = d_dict['d1']
    d1 = [torch.from_numpy(np.array(d)).to(device) for d in d1]
    d2 = d_dict['d2']
    d2 = [torch.from_numpy(np.array(d)).to(device) for d in d2]
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

def valid_generator(queue):
  while True:
    for x, t in queue:
      yield x,t

def optimize_darts(train_queue, valid_queue, model, architect, criterion, optimizer, poses, losses, iter_num, d1, d2):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()

  valid_gen = valid_generator(valid_queue)

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda()

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(valid_gen)
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda()
  
    loss_search, grad_alpha = architect.obtain_loss_grad(input_search, target_search, unrolled=args.unrolled)
    g1 = sum([torch.mul(d, g).sum() for d, g in zip(d1, grad_alpha)])
    g2 = sum([torch.mul(d, g).sum() for d, g in zip(d2, grad_alpha)])
    new_pos = [p-g.item()*args.arch_learning_rate for p, g in zip(poses[-1], [g1, g2])]
    poses.append(new_pos)
    print("pos @ step-{}: {}".format(step+1, new_pos))
    d_update = [g1*d1_t + g2*d2_t for d1_t,d2_t in zip(d1,d2)]
    # update alpha
    for a, d in zip(model.arch_parameters(), d_update):
      a.data.sub_(d*args.arch_learning_rate)
    
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

    acc, loss = infer(valid_queue, model, criterion, verbose=True, early_stop=10)
    losses.append(loss)

#    if step % args.report_freq == 0:
#      print('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    if step >= iter_num: break
  return poses, losses


def optimize_zarts(train_queue, train_weight_queue, valid_queue, model, architect, criterion, optimizer, std, poses, losses, iter_num, d1, d2):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  ess = utils.AverageMeter()
  ESS = 0.

  valid_gen = valid_generator(valid_queue)
  train_weight_gen = valid_generator(train_weight_queue)

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda()

    # update alpha
    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(valid_gen)
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda()
  
    base_loss, grads = architect.obtain_loss_grad(input_search, target_search, unrolled=args.unrolled)
    all_losses = []
    all_dalphas = []
    new_model = model.new()
    model_dict = model.state_dict()
    
#    train_weight_inputs = train_weight_queue
    train_weight_inputs = []
    for _ in range(args.iter_w):
      train_weight_inputs.append(next(train_weight_gen))
    # sample for dalpha
    torch.cuda.empty_cache()
    grads = torch.stack(grads, dim=0)
    for i in range(args.num_sample):
        # fine-tune for iter_w iterations
#        new_model.load_state_dict(model_dict)
        for name, p in new_model.named_parameters():
            p.data.copy_(model_dict[name])

#        while True:
#          rand1 = torch.normal(torch.zeros_like(grads), std=std)
#          rand2 = torch.normal(torch.zeros_like(grads), std=std)
#          if rand1.norm()>1 and rand2.norm()>1:
#            rand1 = rand1 - grads
#            break

#        rand1 = torch.normal(torch.zeros_like(grads), std=std) - grads
#        rand2 = torch.normal(torch.zeros_like(grads), std=std)
#        dalphas = (1-args.lamb)*rand1+args.lamb*rand2
#        dalphas = (1-args.lamb)*torch.normal(-grads, std=std)+args.lamb*torch.normal(torch.zeros_like(grads), std=std)
#        dalphas = args.arch_learning_rate * ( (1-args.lamb)*torch.normal(-grads, std=std)+args.lamb*torch.normal(torch.zeros_like(grads), std=std) )
        dalphas = args.arch_learning_rate * ( (1-args.lamb)*torch.normal(-grads, std=std)+args.lamb*torch.normal(torch.zeros_like(grads), std=std) )

        dalphas = dalphas.chunk(dalphas.shape[0], dim=0)
        dalphas = [d.squeeze() for d in dalphas]
        all_dalphas.append(dalphas)
        for x, y, d in zip(new_model.arch_parameters(), model.arch_parameters(), dalphas):
            x.data.copy_(y.data + d)

        # deepcopy optimizer
#        tmp_opt = optimizer
#        tmp_opt = copy.deepcopy(optimizer)
        tmp_opt = torch.optim.SGD(
              new_model.parameters(),
              0.001,
              momentum=args.momentum,
              weight_decay=args.weight_decay)
        train_weight(train_weight_inputs, new_model, args.iter_w, criterion, tmp_opt)
        del tmp_opt
        # compute valid loss
        with torch.no_grad():
          loss = new_model._loss(input_search, target_search)
          all_losses.append(loss)
    del new_model
    torch.cuda.empty_cache()

    # obtain dalpha Expectation
    all_coeffs = torch.zeros([args.num_sample], device=input_search.device)
    for i in range(args.num_sample):
        prob = torch.exp(-(all_losses[i]-base_loss)/args.tau)
        dalphas = torch.stack(all_dalphas[i], dim=0)
#        gauss_prob = (1-args.lamb) * stats.multivariate_normal.pdf(dalphas.cpu().data.numpy().reshape(-1), mean=(-grads).cpu().data.numpy().reshape(-1), cov=std*std) + args.lamb * stats.multivariate_normal.pdf(dalphas.cpu().data.numpy().reshape(-1), mean=torch.zeros_like(-grads).cpu().data.numpy().reshape(-1), cov=std*std)
        tmp = (dalphas + grads).view(-1)
        var = std*std
#        gauss_prob = torch.exp(-torch.dot(tmp, tmp) / (2.*var)) 
        gauss_prob = (1-args.lamb) * torch.exp(-torch.dot(tmp, tmp) / (2.*var)) + args.lamb * torch.exp(-torch.dot(dalphas.view(-1), dalphas.view(-1)) / 2.*var)
#        D = tmp.numel()*1.
#        gauss_prob = gauss_prob / math.pow(2*math.pi*var, D/2)
        prob = prob / (gauss_prob + 1e-4)
        all_coeffs[i] = prob
    all_coeffs = all_coeffs / all_coeffs.sum()

    # compute ESS for bias analysis
    ESS = (all_coeffs*args.num_sample - 1.).pow(2).sum()
    ESS = (ESS / args.num_sample).sqrt()
    ess.update(ESS.item(), 1)

    # get the update value for alpha
    alphas = model.arch_parameters()
    update_alpha = [torch.zeros_like(a) for a in alphas]
    for j in range(args.num_sample):
        for i, alpha in enumerate(update_alpha):
            alpha.data.add_(all_coeffs[j]*all_dalphas[j][i])

    g1 = sum([torch.mul(d, g).sum() for d, g in zip(d1, update_alpha)])
    g2 = sum([torch.mul(d, g).sum() for d, g in zip(d2, update_alpha)])
    new_pos = [p+g.item() for p, g in zip(poses[-1], [g1, g2])]
    poses.append(new_pos)
    print("pos @ step-{}: {}".format(step+1, new_pos))
    d_update = [g1*d1_t + g2*d2_t for d1_t,d2_t in zip(d1,d2)]
    # update alpha
    for a, d in zip(model.arch_parameters(), d_update):
      a.data.add_(d)


    # update weight
#    train_weight_inputs = []
#    for _ in range(args.iter_w):
#      train_weight_inputs.append(next(train_queue))
    acc, loss = train_weight(train_queue, model, args.iter_w, criterion, optimizer)

    acc, loss = infer(valid_queue, model, criterion, verbose=True, early_stop=10)
    losses.append(loss)

    if step >= iter_num: break
  return poses, losses

def optimize_gld(train_queue, train_weight_queue, valid_queue, model, architect, criterion, optimizer, std, poses, losses, iter_num, d1, d2):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  ess = utils.AverageMeter()
  ESS = 0.

  valid_gen = valid_generator(valid_queue)
  train_weight_gen = valid_generator(train_weight_queue)

  # update weight
  acc, loss = train_weight(train_queue, model, args.iter_w, criterion, optimizer)
  acc, loss = infer(valid_queue, model, criterion, verbose=True, early_stop=10)
  losses.append(loss)

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda()

    # update alpha
    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(valid_gen)
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda()

#    train_weight_inputs = train_weight_queue
    train_weight_inputs = []
    for _ in range(args.iter_w):
      train_weight_inputs.append(next(train_weight_gen))

    base_loss, grads = architect.obtain_loss_grad(input_search, target_search, unrolled=args.unrolled)
    new_model = model.new()
    model_dict = model.state_dict()

    def sample_spherical(npoints, ndim):
      vec = torch.randn(ndim, npoints)
      vec /= torch.linalg.norm(vec, axis=0)
      return torch.reshape(vec, (vec.shape[1], vec.shape[0]))
    alphas = model.arch_parameters()
    alphas = torch.stack(alphas, dim=0)  # [2, 14, 7]
    ndim = alphas.shape[1] * alphas.shape[2]
    R = 2 * alphas.shape[1]
    r = args.rs_error / math.sqrt(ndim)
    if args.rs_maxr is not None:
      x = R / args.rs_maxr
      R /= x
      r /= x
    K = math.log2(R/r)
    best_loss = base_loss
    update_alpha = torch.zeros_like(alphas)
    new_pos = None

    for i in range(int(K+1)):
      radius = R / (2 ** i)
      samples = sample_spherical(alphas.shape[0], ndim)
      samples = radius * torch.reshape(samples, alphas.shape).cuda()
#      update_pos = (torch.rand(2) * radius).data.numpy().tolist()
#      samples = [update_pos[0]*d1_t + update_pos[1]*d2_t for d1_t,d2_t in zip(d1,d2)]
      g1 = sum([torch.mul(d, g).sum() for d, g in zip(d1, samples)])
      g2 = sum([torch.mul(d, g).sum() for d, g in zip(d2, samples)])
      samples = [g1*d1_t + g2*d2_t for d1_t,d2_t in zip(d1,d2)]
      for name, p in new_model.named_parameters():
        p.data.copy_(model_dict[name])
      for x, y, d in zip(new_model.arch_parameters(), model.arch_parameters(), samples    ):
        x.data.copy_(y.data + d)
#      tmp_opt = optimizer
      tmp_opt = torch.optim.SGD(
            new_model.parameters(),
            0.001,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
      train_weight(train_weight_inputs, new_model, args.iter_w, criterion, tmp_opt)
      del tmp_opt
      with torch.no_grad():
        new_loss = new_model._loss(input_search, target_search)
#        _, new_loss = infer(valid_queue, model, criterion, verbose=True, early_stop=10)
      if new_loss < best_loss:
        best_loss = new_loss
        update_alpha = samples
        new_pos = [p+g.item() for p, g in zip(poses[-1], [g1, g2])]
#        new_pos = [base+u for base, u in zip(poses[-1], update_pos)]
    del new_model
    torch.cuda.empty_cache()
    alphas = model.arch_parameters()
    for i, alpha in enumerate(alphas):
      alpha.data.add_(update_alpha[i])
  
    if new_pos is not None:
      poses.append(new_pos)
    print("pos @ step-{}: {}".format(step+1, new_pos))

    # update weight
#    train_weight_inputs = []
#    for _ in range(args.iter_w):
#      train_weight_inputs.append(next(train_queue))
    acc, loss = train_weight(train_queue, model, args.iter_w, criterion, optimizer)

    acc, loss = infer(valid_queue, model, criterion, verbose=True, early_stop=10)
    losses.append(loss)

    if step >= iter_num: break
  return poses, losses

def optimize_rs(train_queue, train_weight_queue, valid_queue, model, architect, criterion, optimizer, std, poses, losses, iter_num, d1, d2):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  ess = utils.AverageMeter()
  ESS = 0.

  valid_gen = valid_generator(valid_queue)
  train_weight_gen = valid_generator(train_weight_queue)

  # update weight
  acc, loss = train_weight(train_queue, model, args.iter_w, criterion, optimizer)
  acc, loss = infer(valid_queue, model, criterion, verbose=True, early_stop=10)
  losses.append(loss)

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda()

    # update alpha
    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(valid_gen)
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda()

#    train_weight_inputs = train_weight_queue
    train_weight_inputs = []
    for _ in range(args.iter_w):
      train_weight_inputs.append(next(train_weight_gen))

    base_loss, grads = architect.obtain_loss_grad(input_search, target_search, unrolled=args.unrolled)
    new_model = model.new()
    model_dict = model.state_dict()

    def sample_normal(npoints, ndim):
      return torch.randn(npoints, ndim)
    def sample_spherical(npoints, ndim):
      vec = torch.randn(ndim, npoints)
      vec /= torch.linalg.norm(vec, axis=0)
      return torch.reshape(vec, (vec.shape[1], vec.shape[0]))
 
    alphas = model.arch_parameters()
    alphas = torch.stack(alphas, dim=0)  # [2, 14, 7]
    ndim = alphas.shape[1] * alphas.shape[2]
    grad_ests = []
 
    for i in range(args.num_sample):
      if args.rs_dist == 'uniform':
        samples = sample_spherical(alphas.shape[0], ndim)
        phi = ndim
      elif args.rs_dist == 'normal':  # lr should be smaller?
        samples = sample_normal(alphas.shape[0], ndim)
        phi = 1
      else:
        raise ValueError
      samples = torch.reshape(samples, alphas.shape).cuda()
#      update_pos = torch.rand(2).data.numpy().tolist()
#      samples = [update_pos[0]*d1_t + update_pos[1]*d2_t for d1_t,d2_t in zip(d1,d2)]
#      phi = ndim
      g1 = sum([torch.mul(d, g).sum() for d, g in zip(d1, samples)])
      g2 = sum([torch.mul(d, g).sum() for d, g in zip(d2, samples)])
      print(g1, g2)
      samples = [g1*d1_t + g2*d2_t for d1_t,d2_t in zip(d1,d2)]
      samples = torch.stack(samples, dim=0)
 
      for name, p in new_model.named_parameters():
        p.data.copy_(model_dict[name])
      for x, y, d in zip(new_model.arch_parameters(), model.arch_parameters(), samples):
        x.data.copy_(y.data + args.rs_delta * d)
 #     tmp_opt = optimizer
      tmp_opt = torch.optim.SGD(
            new_model.parameters(),
            0.001,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
      train_weight(train_weight_inputs, new_model, args.iter_w, criterion, tmp_opt)
      del tmp_opt
      with torch.no_grad():
        loss_pos = new_model._loss(input_search, target_search)
 
      if args.rs_multipoint:
        grad_ests.append((loss_pos - base_loss) * samples)
      elif args.rs_anti:
        phi /= 2
        for name, p in new_model.named_parameters():
          p.data.copy_(model_dict[name])
        for x, y, d in zip(new_model.arch_parameters(), model.arch_parameters(), samples):
          x.data.copy_(y.data - args.rs_delta * d)
#        tmp_opt = optimizer
        tmp_opt = torch.optim.SGD(
              new_model.parameters(),
              0.001,
              momentum=args.momentum,
              weight_decay=args.weight_decay)
        train_weight(train_weight_inputs, new_model, args.iter_w, criterion, tmp_opt)
        del tmp_opt
        with torch.no_grad():
          loss_neg = new_model._loss(input_search, target_search)
        grad_ests.append((loss_pos - loss_neg) * samples)
      else:
        grad_ests.append(loss_pos * samples)
    del new_model
    # torch.cuda.empty_cache()
 
    grad_est = 1.0*phi / args.rs_delta * sum(grad_ests) / args.num_sample
    g1 = sum([torch.mul(d, g).sum() for d, g in zip(d1, grad_est)])
    g2 = sum([torch.mul(d, g).sum() for d, g in zip(d2, grad_est)])
    print(g1, g2)
    grad_est = [g1*d1_t + g2*d2_t for d1_t,d2_t in zip(d1,d2)]
    alphas = model.arch_parameters()
    for i, alpha in enumerate(alphas):
      alpha.data.add_(-args.rs_lr * grad_est[i])

    new_pos = [p-args.rs_lr*g.item() for p, g in zip(poses[-1], [g1, g2])]
    poses.append(new_pos)
    print("pos @ step-{}: {}".format(step+1, new_pos))

    # update weight
#    train_weight_inputs = []
#    for _ in range(args.iter_w):
#      train_weight_inputs.append(next(train_queue))
    acc, loss = train_weight(train_queue, model, args.iter_w, criterion, optimizer)

    acc, loss = infer(valid_queue, model, criterion, verbose=True, early_stop=10)
    losses.append(loss)

    if step >= iter_num: break
  return poses, losses
    

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
  train_weight_queue = torch.utils.data.DataLoader(
      train_data, batch_size=48,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=4)

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
  

  ##############################
  # compute path of optimization
  ##############################
  # Get initial point
  max_iter = 10
  p_init = [-0.5, 0.5]
  for i, a in enumerate(model.arch_parameters()):
    a.data.add_(p_init[0]*d1[i]+p_init[1]*d2[i])
  poses = [p_init]

#  d_update = [p_init[0]*d1_t + p_init[1]*d2_t for d1_t,d2_t in zip(d1,d2)]
#  # initialize alpha to initial position
#  for a, d in zip(model.arch_parameters(), d_update):
#    a.data.add_(d)
  acc, loss = infer(valid_queue, model, criterion, verbose=True, early_stop=10)
  losses = [loss]

  bk_state_dict = copy.deepcopy(model.state_dict())
  bk_arch_parameters = copy.deepcopy(model.arch_parameters())

  # darts-1st
  print(">>> Compute path for darts")
  args.arch_learning_rate = 3e-4; args.arch_weight_decay=1e-3
  path_file = os.path.join(save_dir, 'darts_path.npy')
  losses_file = os.path.join(save_dir, 'darts_loss.npy')
  if os.path.exists(path_file):
    poses_darts = np.load(path_file)
    losses_darts = np.load(losses_file)
  else:
    model_new = model.new()
    model_new.load_state_dict(model.state_dict(), strict=True)
    optimizer = torch.optim.SGD(
        model_new.parameters(),
        0.001,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    architect = Architect(model_new, args)
    poses_darts, losses_darts = optimize_darts(train_queue, valid_queue, model_new, architect, criterion, optimizer, poses, losses, max_iter, d1, d2)
    poses_darts = np.array(poses_darts)
    np.save(path_file, poses_darts)
    losses_darts = np.array(losses_darts)
    np.save(losses_file, losses_darts)

  # zarts-MGS
  poses = [p_init]
  model.load_state_dict(bk_state_dict)
  for a, a_bk in zip(model.arch_parameters(), bk_arch_parameters):
    a.data.copy_(a_bk.data)
  print(">>> Compute path for zarts")
  args.arch_learning_rate = 0.5; args.arch_weight_decay=1e-3
  acc, loss = infer(valid_queue, model, criterion, verbose=True, early_stop=10)
  path_file = os.path.join(save_dir, 'zarts_path.npy')
  losses_file = os.path.join(save_dir, 'zarts_loss.npy')
  if os.path.exists(path_file):
    poses_zarts = np.load(path_file)
    losses_zarts = np.load(losses_file)
  else:
    model_new = model.new()
    model_new.load_state_dict(model.state_dict(), strict=True)
    optimizer = torch.optim.SGD(
        model_new.parameters(),
        0.001,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    architect = Architect(model_new, args)
    poses_zarts, losses_zarts = optimize_zarts(train_queue, train_weight_queue, valid_queue, model_new, architect, criterion, optimizer, args.std, poses, losses, max_iter, d1, d2)
    poses_zarts = np.array(poses_zarts)
    np.save(path_file, poses_zarts)
    losses_zarts = np.array(losses_zarts)
    np.save(losses_file, losses_zarts)

  # zarts-GLD
  poses = [p_init]
  model.load_state_dict(bk_state_dict)
  for a, a_bk in zip(model.arch_parameters(), bk_arch_parameters):
    a.data.copy_(a_bk.data)
  print(">>> Compute path for zarts-GLD")
  acc, loss = infer(valid_queue, model, criterion, verbose=True, early_stop=10)
  path_file = os.path.join(save_dir, 'zarts_gld_path.npy')
  losses_file = os.path.join(save_dir, 'zarts_gld_loss.npy')
  if os.path.exists(path_file):
    poses_zarts_gld = np.load(path_file)
    losses_zarts_gld = np.load(losses_file)
  else:
    model_new = model.new()
    model_new.load_state_dict(model.state_dict(), strict=True)
    optimizer = torch.optim.SGD(
        model_new.parameters(),
        0.001,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    architect = Architect(model_new, args)
    poses_zarts_gld, losses_zarts_gld = optimize_gld(train_queue, train_weight_queue, valid_queue, model_new, architect, criterion, optimizer, args.std, poses, losses, max_iter, d1, d2)
    poses_zarts_gld = np.array(poses_zarts_gld)
    np.save(path_file, poses_zarts_gld)
    losses_zarts_gld = np.array(losses_zarts_gld)
    np.save(losses_file, losses_zarts_gld)

  # zarts-RS
  poses = [p_init]
  model.load_state_dict(bk_state_dict)
  for a, a_bk in zip(model.arch_parameters(), bk_arch_parameters):
    a.data.copy_(a_bk.data)
  print(">>> Compute path for zarts-RS")
  args.num_sample = 8
  acc, loss = infer(valid_queue, model, criterion, verbose=True, early_stop=10)
  path_file = os.path.join(save_dir, 'zarts_rs_path.npy')
  losses_file = os.path.join(save_dir, 'zarts_rs_loss.npy')
  if os.path.exists(path_file):
    poses_zarts_rs = np.load(path_file)
    losses_zarts_rs = np.load(losses_file)
  else:
    model_new = model.new()
    model_new.load_state_dict(model.state_dict(), strict=True)
    optimizer = torch.optim.SGD(
        model_new.parameters(),
        0.001,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    architect = Architect(model_new, args)
    poses_zarts_rs, losses_zarts_rs = optimize_rs(train_queue, train_weight_queue, valid_queue, model_new, architect, criterion, optimizer, args.std, poses, losses, max_iter, d1, d2)
    poses_zarts_rs = np.array(poses_zarts_rs)
    np.save(path_file, poses_zarts_rs)
    losses_zarts_rs = np.array(losses_zarts_rs)
    np.save(losses_file, losses_zarts_rs)

  ##############################
  # plot path of optimization
  ##############################
  # obtain landscape
  landscape_dir = os.path.join(save_dir, 'iterw-10')
  assert(os.path.exists(landscape_dir))
  acc_file = os.path.join(landscape_dir, 'acc.npy')
  loss_file = os.path.join(landscape_dir, 'loss.npy')
  assert(os.path.exists(acc_file) and os.path.exists(loss_file))
  acc = np.load(acc_file)
  loss = np.load(loss_file)

  # plot landscape
  x = np.linspace(args.xmin, args.xmax, num=args.xnum)
  y = np.linspace(args.ymin, args.ymax, num=args.ynum)
  X, Y = np.meshgrid(x, y)
  Z = loss
  levels = np.concatenate((np.arange(0.52, 0.53, 0.002), np.arange(0.54, 0.6, 0.01), np.arange(0.6, 0.8, 0.05), np.arange(0.85, 1.5, 0.1)), axis=-1)
  fig = plot_contour(X, Y, Z, levels, scatter_func=np.argmin)

  # plot path of optimization for darts
#  plt_scatter = plt.scatter(poses_darts[:,0], poses_darts[:,1], s=5 , c='orange', marker='^', alpha=0.8)
  plt_line = plt.plot(poses_darts[:,0], poses_darts[:,1], linewidth=2, color='deepskyblue', marker='^', markersize=5, markerfacecolor="blue", markeredgewidth=0.6, alpha=0.8, label='darts')
#  h1, l1 = plt_line.get_legend_handles_labels()
#  for x,y,z in zip(poses_darts[:,0], poses_darts[:,1], losses_darts):
#      plt.text(x,y+0.03,'%.2f'%z,ha='center',va='bottom',fontsize=3, color='blue')  
  plt.setp(plt_line, 'zorder', 4)
  for i in range(len(poses_darts)-1):
    start = poses_darts[i]
    end = poses_darts[i+1]
    plt_arrow = plt.annotate("",
                xy=(end[0], end[1]),
                xytext=(start[0], start[1]),
                # xycoords="figure points",
                arrowprops=dict(arrowstyle="->", color="deepskyblue"), alpha=0.8)
    plt.setp(plt_arrow, 'zorder', 3)


  # plot path of optimization for zarts-RS
  plt_line = plt.plot(poses_zarts_rs[:,0], poses_zarts_rs[:,1], linewidth=2, color='lightgreen', marker='D', markersize=5, markerfacecolor="green", markeredgewidth=0.6, alpha=0.8, label='zarts-gld')
  plt.setp(plt_line, 'zorder', 4)
  for i in range(len(poses_zarts_rs)-1):
    start = poses_zarts_rs[i]
    end = poses_zarts_rs[i+1]
    dist = sum((a-b)*(a-b) for a,b in zip(end, start))
    if dist < 0.001: 
      continue
    plt_arrow = plt.annotate("",
                xy=(end[0], end[1]),
                xytext=(start[0], start[1]),
                # xycoords="figure points",
                arrowprops=dict(arrowstyle="->", color="lightgreen"), alpha=0.8)
    plt.setp(plt_arrow, 'zorder', 5)

  # plot path of optimization for zarts-GLD
  plt_line = plt.plot(poses_zarts_gld[:-1,0], poses_zarts_gld[:-1,1], linewidth=2, color='gold', marker='s', markersize=5, markerfacecolor="darkgoldenrod", markeredgewidth=0.6, alpha=0.8, label='zarts-gld')
  plt.setp(plt_line, 'zorder', 6)
  for i in range(len(poses_zarts_gld)-2):
    start = poses_zarts_gld[i]
    end = poses_zarts_gld[i+1]
    dist = sum((a-b)*(a-b) for a,b in zip(end, start))
    if dist < 0.001: 
      continue
    plt_arrow = plt.annotate("",
                xy=(end[0], end[1]),
                xytext=(start[0], start[1]),
                # xycoords="figure points",
                arrowprops=dict(arrowstyle="->", color="gold"), alpha=0.8)
    plt.setp(plt_arrow, 'zorder', 7)

  # plot path of optimization for zarts
  plt_line = plt.plot(poses_zarts[:-1,0], poses_zarts[:-1,1], linewidth=2, color='violet', marker='o', markersize=5, markerfacecolor="purple", markeredgewidth=0.6, alpha=0.8, label='zarts')
#  h = [h1, h2]; l = [l1, l2];
#  plt.legend(h, l, loc=2)
#  for x,y,z in zip(poses_zarts[:,0], poses_zarts[:,1], losses_zarts):
#      plt.text(x,y-0.03,'%.2f'%z,ha='center',va='bottom',fontsize=3, color='red')  
  plt.setp(plt_line, 'zorder', 8)
  for i in range(len(poses_zarts)-2):
    start = poses_zarts[i]
    end = poses_zarts[i+1]
    dist = sum((a-b)*(a-b) for a,b in zip(end, start))
#    print(dist)
    if dist < 0.001: 
      continue
      plt_arrow = plt.annotate("",
                xy=(end[0], end[1]),
                xytext=(start[0], start[1]),
                # xycoords="figure points",
                arrowprops=dict(arrowstyle="->", color="violet"), alpha=0.8)
    else:
      plt_arrow = plt.annotate("",
                xy=(end[0], end[1]),
                xytext=(start[0], start[1]),
                # xycoords="figure points",
                arrowprops=dict(arrowstyle="->", color="violet"), alpha=0.8)
    plt.setp(plt_arrow, 'zorder', 9)

  save_name = os.path.join(landscape_dir, 'path_optimize.pdf')
  fig.savefig(save_name, dpi=300,
              bbox_inches='tight', format='pdf')
  plt.close(fig)

  color1 = ['deepskyblue', 'cornflowerblue', 'rosybrown', 'tomato', 'darkred']
  color2 = ['lightyellow', 'khaki', 'peachpuff', 'sandybrown', 'darkorange']

if __name__ == '__main__':
  main()

