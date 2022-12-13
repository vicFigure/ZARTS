import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import json
import csv

import torch.nn as nn
import genotypes
from tools import compare_genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network
import model_no_restrict

sys.path.append('../')
import utils

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='tmp', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--base_dir',type=str,help='model dir')
parser.add_argument('--genotype_names',nargs='*', type=int, default=None, help='model dir')

parser.add_argument('--no_restrict', action='store_true', default=False, help='use auxiliary tower')
args = parser.parse_args()

args.save = args.base_dir
#args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
#utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
if not os.path.exists(args.save):
  os.makedirs(args.save)
fh = logging.FileHandler(os.path.join(args.base_dir, 'eval_log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)



def train(train_queue, model, criterion, optimizer):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
#    input = Variable(input).cuda()
#    target = Variable(target).cuda()
    target = target.cuda(non_blocking=True)
    input = input.cuda(non_blocking=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

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
  
      logits, _ = model(input)
      loss = criterion(logits, target)
  
      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)
  
      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def train_from_scratch(args, train_queue, valid_queue, genotype, ckpt_dir, model_idx, init_model=None):
    if args.no_restrict:
      model = model_no_restrict.NetworkCIFAR(args.init_channels, args.n_claases, args.layers, args.auxiliary, genotype)

    else:
      model = Network(args.init_channels, args.n_classes, args.layers, args.auxiliary, genotype)
    model = model.cuda()
    if init_model is not None:
      model.load_state_dict(torch.load(init_model), strict=False)
#    # clear unused gpu memory
#    torch.cuda.empty_cache() 
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  
    best_acc = 0;best_loss = 0
    for epoch in range(args.epochs):
      scheduler.step()
      logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
      model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
  
#      torch.cuda.empty_cache()
      train_acc, train_obj = train(train_queue, model, criterion, optimizer)
      logging.info('train_acc %f', train_acc)
#      torch.cuda.empty_cache()
  
      valid_acc, valid_obj = infer(valid_queue, model, criterion)
      logging.info('valid_acc %f', valid_acc)
#      torch.cuda.empty_cache()
  
      if valid_acc > best_acc:
        try:
            last_model = os.path.join(ckpt_dir, 'weights_%s_%.3f.pt'%(model_idx, best_acc))
            os.remove(last_model)
        except:
            pass
        utils.save(model, os.path.join(ckpt_dir, 'weights_%s_%.3f.pt'%(model_idx, valid_acc)))
        best_acc = valid_acc
        best_loss = valid_obj
    return best_acc, best_loss


def search_optim_arch(base_dir, genotype_names, epochs, choose_type='fromfile'):
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  # dataLoader
  if args.dataset == 'cifar10':
      train_transform, valid_transform = utils._data_transforms_cifar10(args)
      train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
      valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
      args.n_classes = 10
  elif args.dataset == 'cifar100':
      train_transform, valid_transform = utils._data_transforms_cifar100(args)
      train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
      valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
      args.n_classes = 100
  elif args.dataset == 'svhn':
      train_transform, valid_transform = utils._data_transforms_svhn(args)
      train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
      valid_data = dset.SVHN(root=args.data, split='test', download=True, transform=valid_transform)
      args.n_classes = 10

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  genotype_path = os.path.join(base_dir, 'results_of_7q/genotype')
#  genotype_path = os.path.join(base_dir, 'results_of_7q/genotype_sal')
  alpha_path = os.path.join(base_dir, 'results_of_7q/alpha')
  print(choose_type)
  if choose_type == 'fromfile':
    eval_dir = os.path.join(base_dir, 'hyperband/hyperband%d-%depochs'%(len(genotype_names), epochs))
  else:
    eval_dir = os.path.join(base_dir, 'searchArch_sample_withoutNoneOp')
  if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)
  val_acc = []
  val_loss = []
  num_sample = 5 if choose_type=='sample' else 1
  for name in genotype_names:
    if choose_type == 'fromfile':
      genotype_file = os.path.join(genotype_path, '%s.txt'%name)
      tmp_dict = json.load(open(genotype_file,'r'))
      genotype = genotypes.Genotype(**tmp_dict)
      print(genotype)
      best_acc, best_loss = train_from_scratch(args, train_queue, valid_queue, genotype, eval_dir, name)
      val_acc.append(best_acc)
      val_loss.append(best_loss)
      print('Best: %f / %f'%(best_loss, best_acc))
    elif choose_type == 'sample':
#      alpha_file = None
      alpha_file = os.path.join(alpha_path, '%s.txt'%name)
      ckpt_dir = os.path.join(eval_dir, '%s'%name)
      if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
      for idx in range(num_sample):
        model_idx = '%s-%d'%(name, idx)
        genotype = sample_genotype(alpha_file)
        print(genotype)
        # save genotype
        genotype_file = os.path.join(ckpt_dir, 'genotype_%s.txt'%model_idx)
        with open(genotype_file, 'w') as f:
          json.dump(genotype._asdict(), f)
        best_acc, best_loss = train_from_scratch(args, train_queue, valid_queue, genotype, ckpt_dir, model_idx)
        val_acc.append(best_acc)
        val_loss.append(best_loss)
        print('Best: %f / %f'%(best_loss, best_acc))
    else:
      raise(ValueError('No such choose_type: %s'%choose_type))

  # print best results
  for idx, res in enumerate(val_acc):
    print(genotype_names[int(idx/num_sample)], res)
  res = np.array(val_acc)
  sort_idx = np.argsort(res)[::-1]
  sort_names = []
  for idx in sort_idx:
    sort_names.append(genotype_names[int(idx/num_sample)])
    print(idx, genotype_names[int(idx/num_sample)], val_acc[idx])

  # save the results
  result_file = os.path.join(eval_dir, 'results.csv')
  with open(result_file, 'w') as f:
    writer = csv.writer(f)
    title = ['genotype_name', 'val_loss', 'val_acc']
    writer.writerow(title)
    for idx, loss in enumerate(val_loss):
        a = [genotype_names[int(idx/num_sample)], loss, val_acc[idx]]
        writer.writerow(a)
  return sort_names

if __name__ == '__main__':
#  base_dir = 'search-EXP-20200625-091629/'
  base_dir = args.base_dir
  if args.genotype_names is None:
    genotype_names = [-1]
  else: genotype_names = args.genotype_names
  
  # deal with negative names
  genotype_path = os.path.join(base_dir, 'results_of_7q/genotype')
  files = os.listdir(genotype_path)
  max_name = 0
  for f in files:
    tmp = int(f.split('.')[0])
    if tmp > max_name: max_name = tmp
  for i, name in enumerate(genotype_names):
    if name < 0: genotype_names[i] = max_name+1 + name

  epochs = args.epochs
  print("\n############")
  print("genotype names: ", genotype_names)
  print("search for %d epochs "%args.epochs)
  print("############\n")
  sorted_names = search_optim_arch(base_dir, genotype_names, epochs, choose_type='fromfile')

  '''

  #####################
  # Hyperband
  #####################
#  base_dir = 'search-EXP-20200625-091629/'
  base_dir = args.base_dir
  genotype_path = os.path.join(base_dir, 'results_of_7q/genotype')
  genotype_names = compare_genotypes.get_distinct_genotype(genotype_path, None)
  print(genotype_names)
  num_search = [len(genotype_names), 6, 1]
  hyperband_epochs = [20, 60, 600]
  assert(len(num_search) == len(hyperband_epochs))
  sorted_names = genotype_names
  for i, epochs in enumerate(hyperband_epochs):
      sorted_names = sorted_names[:num_search[i]]
      args.epochs = epochs
      print("\n############")
      print("hyperband search genotype names: ", sorted_names)
      print("search for %d epochs "%args.epochs)
      print("############\n")
      sorted_names = search_optim_arch(base_dir, sorted_names, epochs, choose_type='fromfile')
  '''

