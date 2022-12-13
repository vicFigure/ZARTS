import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import Genotype

import numpy as np


class MixedOp(nn.Module):

  def __init__(self, C, stride, PRIMITIVES, use_merge=False):
    super(MixedOp, self).__init__()
    if use_merge:
      self._PRIMITIVES = [p for p in PRIMITIVES if 'conv' not in p]
      self._PRIMITIVES.append('merge_conv_5x5')
    else: self._PRIMITIVES = PRIMITIVES

    self._ops = nn.ModuleList()
    for primitive in self._PRIMITIVES:
      if 'merge_conv_5x5' in primitive:
        op = OPS[primitive](C, stride, False, single_sepConv=True, with_d5=True)
      else:
        op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
#    return sum(w * op(x) for w, op in zip(weights, self._ops))
    results = 0
    for idx, op in enumerate(self._ops):
       if 'merge' not in self._PRIMITIVES[idx]:
           results += weights[idx]*op(x) 
       else:
#           tmp = op(x, weights[idx:])
#           print(results.shape, x.shape, tmp.shape)
#           results += tmp
           results += op(x, weights[idx:idx+4])
    return results


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, use_merge=False):
    super(Cell, self).__init__()
    self.reduction = reduction
    self.primitives = self.PRIMITIVES['primitives_reduct' if reduction else 'primitives_normal']

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()

    edge_index = 0
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride, self.primitives[edge_index], use_merge=use_merge)
        self._ops.append(op)
        edge_index += 1

  def forward(self, s0, s1, weights, drop_prob=0.):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      if drop_prob > 0. and self.training:
        s = sum(drop_path(self._ops[offset+j](h, weights[offset+j]), drop_prob) for j, h in enumerate(states))
      else:
        s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)

class Network(nn.Module):
  def __init__(self, C, num_classes, layers, criterion, primitives, steps=4, multiplier=4, stem_multiplier=3, alpha_weights=None, drop_path_prob=0.0, use_merge=False):
      super(Network, self).__init__()
      self.alpha_weights = alpha_weights
      self._C = C
      self._num_classes = num_classes
      self._layers = layers
      self._criterion = criterion
      self._steps = steps
      self._multiplier = multiplier
      self.drop_path_prob = drop_path_prob
      self.use_merge = use_merge

      nn.Module.PRIMITIVES = primitives
      self._initialize_alphas()

  def new(self):
    model_new = self.__class__(self._C, self._num_classes, self._layers, self._criterion, self.PRIMITIVES, drop_path_prob=self.drop_path_prob, use_merge=self.use_merge).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(self.PRIMITIVES['primitives_normal'][0])

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights, normal=True):
      PRIMITIVES = self.PRIMITIVES['primitives_normal' if normal else 'primitives_reduct']

      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()

        try:
          edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES[x].index('none')))[:2]
        except ValueError: # This error happens when the 'none' op is not present in the ops
          edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]

        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if 'none' in PRIMITIVES[j]:
              if k != PRIMITIVES[j].index('none'):
                if k_best is None or W[j][k] > W[j][k_best]:
                  k_best = k
            else:
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[start+j][k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), True)
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), False)

    concat = list(range(2+self._steps-self._multiplier, self._steps+2))
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype



class Network_cifar(Network):

  def __init__(self, C, num_classes, layers, criterion, primitives, steps=4, multiplier=4, stem_multiplier=3, alpha_weights=None, drop_path_prob=0.0, use_merge=False):
    super(Network_cifar, self).__init__(C, num_classes, layers, criterion, primitives, steps, multiplier, stem_multiplier, alpha_weights, drop_path_prob, use_merge)

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, use_merge)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def new(self):
    model_new = Network_cifar(self._C, self._num_classes, self._layers, self._criterion, self.PRIMITIVES, drop_path_prob=self.drop_path_prob, use_merge=self.use_merge).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new


  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if self.alpha_weights is None:
        if cell.reduction:
          weights = F.softmax(self.alphas_reduce, dim=-1)
        else:
          weights = F.softmax(self.alphas_normal, dim=-1)
      else:
        raise(ValueError("Why you want to set alphas manually?"))
        print(self.alpha_weights['alphas_normal'])
        print(self.alpha_weights['alphas_reduce'])
        if cell.reduction:
          weights = self.alpha_weights['alphas_reduce']
        else:
          weights = self.alpha_weights['alphas_normal']
      
      s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits


class Network_imagenet(Network):

  def __init__(self, C, num_classes, layers, criterion, primitives, steps=4, multiplier=4, stem_multiplier=3, alpha_weights=None, drop_path_prob=0.0, use_merge=False):
    super(Network_imagenet, self).__init__(C, num_classes, layers, criterion, primitives, steps, multiplier, stem_multiplier, alpha_weights, drop_path_prob, use_merge)

    C_curr = stem_multiplier*C
    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C_curr // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_curr // 2, C_curr, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C_curr, C_curr, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr),
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = True
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, use_merge)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def new(self):
    model_new = Network_imagenet(self._C, self._num_classes, self._layers, self._criterion, self.PRIMITIVES, drop_path_prob=self.drop_path_prob, use_merge=self.use_merge).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      if self.alpha_weights is None:
        if cell.reduction:
          weights = F.softmax(self.alphas_reduce, dim=-1)
        else:
          weights = F.softmax(self.alphas_normal, dim=-1)
      else:
        raise(ValueError("Why you want to set alphas manually?"))
        print(self.alpha_weights['alphas_normal'])
        print(self.alpha_weights['alphas_reduce'])
        if cell.reduction:
          weights = self.alpha_weights['alphas_reduce']
        else:
          weights = self.alpha_weights['alphas_normal']
      
      s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits
