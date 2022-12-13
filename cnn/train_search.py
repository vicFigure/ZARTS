import os
import sys
import math
import time
import glob
import numpy as np
from scipy import stats
import json
import csv
import copy

import torch
import utils
from utils import DecayScheduler
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network_cifar
from model_search import Network_imagenet
from architect import Architect
from spaces import spaces_dict


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--num_workers', type=int, default=2, help='batch size')
parser.add_argument('--space', type=str, default='s1', help='space index')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
#parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=-1, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

# For Important Sampling
parser.add_argument('--warmup', type=int, default=10, help='epochs of warm up before pruning')
parser.add_argument('--num_sample', type=int, default=4, help='number of samples in import sampling')
parser.add_argument('--iter_w', type=int, default=10, help='number of iterations to train weight for each sample')
parser.add_argument('--std', type=float, default=2.5e-2, help='variance of Gaussian Distrubution to sample delta alpha, square of standard deviation')
parser.add_argument('--std_decay_type', type=str, default='stable', help='decay_type for std')
parser.add_argument('--tau', type=float, default=1., help='temprature of Exp Distribution')
parser.add_argument('--lamb', type=float, default=1e-3, help='Coefficient of Gaussian Distribution')
parser.add_argument('--require_cond', action='store_true', default=False, help='use one-step unrolled validation loss')

parser.add_argument('--use_merge', action='store_true', default=False, help='use cutout')

parser.add_argument('--task', type=int, default=0, help='task ID')

args = parser.parse_args()

args.save = 'ckpt/search-{}-{}-task{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"), args.task)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10
torch.backends.cudnn.deterministic = True


def main(primitives):
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  if args.seed is None: args.seed = -1
  if args.seed < 0:
    args.seed = np.random.randint(low=0, high=10000)
  np.random.seed(args.seed)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info("seed = %d", args.seed)
  logging.info("args = %s", args)

  # dataLoader
  if args.dataset == 'cifar10':
      train_transform, valid_transform = utils._data_transforms_cifar10(args)
      train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
      args.n_classes = 10
      Network = Network_cifar
  elif args.dataset == 'cifar100':
      train_transform, valid_transform = utils._data_transforms_cifar100(args)
      train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
      args.n_classes = 100
      Network = Network_cifar
  elif args.dataset == 'svhn':
      train_transform, valid_transform = utils._data_transforms_svhn(args)
      train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
      args.n_classes = 10
      Network = Network_cifar
  elif 'imagenet' in args.dataset:
      normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      train_data = dset.ImageFolder(
        args.data,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
      search_ratio = 1./4
      num_search_data = int(search_ratio * len(train_data))
      print(f"search using {num_search_data}/{len(train_data)} samples")
      args.n_classes = 1000
      Network = Network_imagenet

  num_train = len(train_data)
  indices = list(range(num_train))
  if num_search_data: 
      indices = np.random.choice(indices, num_search_data, replace=False)
      num_train = len(indices)
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=args.num_workers)
#  train_weight_queue = torch.utils.data.DataLoader(
#      train_data, batch_size=48,
#      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
#      pin_memory=True, num_workers=4)
  train_weight_queue = torch.utils.data.DataLoader(
      train_data, batch_size=48,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=args.num_workers)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=args.num_workers)

  # Construct Network
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, args.n_classes, args.layers, criterion, primitives, use_merge=args.use_merge)

  num_gpus = torch.cuda.device_count()   
  if num_gpus > 1:
    assert 0
    model_infer = nn.DataParallel(model).cuda()
    model = model_infer.module
  else:
    model_infer = model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)
  std_scheduler = DecayScheduler(base_lr=args.std, T_max=args.epochs-args.warmup, T_stop=args.epochs-args.warmup, decay_type=args.std_decay_type)

  architect = Architect(model, args)

  val_loss = [] # record the loss of val for each epoch  - q1
  val_acc = [] # recod the acc of val for each epoch  - q2
  results = {}
  results['val_loss'] = []
  results['val_acc'] = []

  ckpt_dir = os.path.join(args.save, 'ckpt')
  result_dir = os.path.join(args.save, 'results_of_7q') # preserve the results
  genotype_dir = os.path.join(result_dir, 'genotype') # preserve the argmax genotype for each epoch  - q3,5,7
  alpha_dir = os.path.join(result_dir, 'alpha')
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)
  if not os.path.exists(genotype_dir):
    os.makedirs(genotype_dir)
  if not os.path.exists(alpha_dir):
    os.makedirs(alpha_dir)

  conds = []
  for epoch in range(args.epochs):
    lr = scheduler.get_lr()[0]
    std = std_scheduler.get_lr()
    logging.info('epoch %d lr %e std %f', epoch, lr, std)

    print(F.softmax(model.alphas_normal, dim=-1))
    print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj = train(train_queue, train_weight_queue, valid_queue, model, architect, criterion, optimizer, lr, std, epoch)
    logging.info('train_acc %f', train_acc)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

#    utils.save_supernet(model, os.path.join(ckpt_dir, 'weights_%d.pt'%epoch))

    # for seven questions
    #q1 & q2
    results['val_loss'].append(valid_obj)
    results['val_acc'].append(valid_acc)
    #q3,5,7
    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    genotype_file = os.path.join(genotype_dir, '%d.txt'%epoch)
    with open(genotype_file, 'w') as f:
      json.dump(genotype._asdict(), f)
      # to recover: genotype = genotype(**dict)
    #q6: save the alpha weights
    alpha_file = os.path.join(alpha_dir, '%d.txt'%epoch)
    alpha_weights = model.arch_parameters()
    alphas = {}
    alphas['alphas_normal'] = F.softmax(alpha_weights[0], dim=-1).data.cpu().numpy().tolist()
    alphas['alphas_reduce'] = F.softmax(alpha_weights[1], dim=-1).data.cpu().numpy().tolist()
    with open(alpha_file, 'w') as f:
      json.dump(alphas, f)
    # compute condition number
    if args.require_cond:
      cond = compute_cond(model, train_weight_queue, args.iter_w, criterion, epsilon=0.01, K=50)
      conds.append(cond)

    scheduler.step()
    if epoch >= args.warmup:
      std_scheduler.step(epoch-args.warmup)

  # save the results:
  result_file = os.path.join(result_dir, 'results.csv')
  with open(result_file, 'w') as f:
    writer = csv.writer(f)
    title = ['epoch', 'val_loss', 'val_acc']
    writer.writerow(title)
    for epoch, val_loss in enumerate(results['val_loss']):
      a = [epoch, val_loss, results['val_acc'][epoch]]
      writer.writerow(a)
  if args.require_cond:
    cond_file = os.path.join(result_dir, 'cond.csv')
    with open(cond_file, 'w') as f:
      writer = csv.writer(f)
      title = ['epoch', 'cond']
      writer.writerow(title)
      for epoch, cond in enumerate(conds):
        a = [epoch, cond]
        writer.writerow(a)


def compute_cond(model, train_queue, iter_w, criterion, epsilon=0.01, K=50):
    losses, conds = [], []
    bk_state_dict = copy.deepcopy(model.state_dict())
    base_acc, base_loss = infer(valid_queue, model, criterion, early_stop=10)
    epsilon = 0.01
    with torch.no_grad():
      alpha_norm = torch.stack(model.arch_parameters(), dim=0).norm()
    for k in range(K):
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
    model.load_state_dict(bk_state_dict)
    return conds.max()


def valid_generator(queue):
  while True:
    for x, t in queue:
      yield x,t

def train(train_queue, train_weight_queue, valid_queue, model, architect, criterion, optimizer, lr, std, epoch):
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
#    target = Variable(target, requires_grad=False).cuda(async=True)

    # update alpha
    if epoch >=args.warmup and step % args.iter_w == 0:
      # get a random minibatch from the search queue with replacement
#      input_search, target_search = next(iter(valid_queue))
      input_search, target_search = next(valid_gen)
      input_search = Variable(input_search, requires_grad=False).cuda()
      target_search = Variable(target_search, requires_grad=False).cuda()
#      target_search = Variable(target_search, requires_grad=False).cuda(async=True)
  
      base_loss, grads = architect.obtain_loss_grad(input_search, target_search, unrolled=args.unrolled)
      all_losses = []
      all_dalphas = []
      new_model = model.new()
      model_dict = model.state_dict()
      
#      train_weight_inputs = train_weight_queue
      train_weight_inputs = []
      for _ in range(args.iter_w):
        train_weight_inputs.append(next(train_weight_gen))
      # sample for dalpha
      torch.cuda.empty_cache()
      grads = torch.stack(grads, dim=0)
      for i in range(args.num_sample):
          # fine-tune for iter_w iterations
#          new_model.load_state_dict(model_dict)
          for name, p in new_model.named_parameters():
              p.data.copy_(model_dict[name])

#          dalphas = args.arch_learning_rate * ( (1-args.lamb)*torch.normal(-grads, std=std)+args.lamb*torch.normal(torch.zeros_like(grads), std=std) )
          dalphas = (1-args.lamb)*torch.normal(-grads, std=std)+args.lamb*torch.normal(torch.zeros_like(grads), std=std)
          dalphas = dalphas.chunk(dalphas.shape[0], dim=0)
          dalphas = [d.squeeze() for d in dalphas]
          all_dalphas.append(dalphas)
          for x, y, d in zip(new_model.arch_parameters(), model.arch_parameters(), dalphas):
              x.data.copy_(y.data + d)

          # deepcopy optimizer
          tmp_opt = optimizer
#          tmp_opt = copy.deepcopy(optimizer)
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
#          gauss_prob = (1-args.lamb) * stats.multivariate_normal.pdf(dalphas.cpu().data.numpy().reshape(-1), mean=(-grads).cpu().data.numpy().reshape(-1), cov=std*std) + args.lamb * stats.multivariate_normal.pdf(dalphas.cpu().data.numpy().reshape(-1), mean=torch.zeros_like(-grads).cpu().data.numpy().reshape(-1), cov=std*std)
          tmp = (dalphas + grads).view(-1)
          var = std*std
#          gauss_prob = torch.exp(-torch.dot(tmp, tmp) / (2.*var)) 
          gauss_prob = (1-args.lamb) * torch.exp(-torch.dot(tmp, tmp) / (2.*var)) + args.lamb * torch.exp(-torch.dot(dalphas.view(-1), dalphas.view(-1)) / 2.*var)
#          D = tmp.numel()*1.
#          gauss_prob = gauss_prob / math.pow(2*math.pi*var, D/2)
          prob = prob / (gauss_prob + 1e-4)
          all_coeffs[i] = prob
      all_coeffs = all_coeffs / all_coeffs.sum()

      # compute ESS for bias analysis
      ESS = (all_coeffs*args.num_sample - 1.).pow(2).sum()
      ESS = (ESS / args.num_sample).sqrt()
      ess.update(ESS.item(), 1)

      # update alpha
      alphas = model.arch_parameters()
      for j in range(args.num_sample):
          for i, alpha in enumerate(alphas):
              alpha.data.add_(all_coeffs[j]*all_dalphas[j][i])
    else:
#      print("Warm up: Donot update architecture parameters")
      pass

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

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
      logging.info('ESS %f %f', ess.avg, ESS)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = Variable(input, volatile=True).cuda()
      target = Variable(target, volatile=True).cuda()
#      target = Variable(target, volatile=True).cuda(async=True)
  
      logits = model(input)
      loss = criterion(logits, target)
  
      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)
  
      if step % args.report_freq == 0:
        logging.info('valid %03d %.3e %.3f %.3f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

def train_weight(train_queue, model, iter_w, criterion, optimizer, verbose=False):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda()
#    target = Variable(target, requires_grad=False).cuda(async=True)

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
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    if step >= iter_w: break

  return top1.avg, objs.avg


if __name__ == '__main__':
  space = spaces_dict[args.space]
  main(space)

