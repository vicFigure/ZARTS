import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
#from genotypes import PRIMITIVES
from genotypes import Genotype, ops_names

import numpy as np
from utils import drop_path

#######################
# For NAS
#######################
class MixedOp(nn.Module):

  def __init__(self, C, stride, PRIMITIVES):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))

  def forward_single(self, x, weights):
    index = weights.max(-1, keepdim=True)[1].item()
    return weights[index] * self._ops[index](x)


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
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
        op = MixedOp(C, stride, self.primitives[edge_index])
        self._ops.append(op)
        edge_index += 1

  def forward(self, s0, s1, weights, weights2, drop_prob=0.):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0

    for i in range(self._steps):
      if drop_prob > 0. and self.training:
        s = sum(drop_path(weights2[offset+j]*self._ops[offset+j](h, weights[offset+j]), drop_prob) for j, h in enumerate(states))
      else:
        s = sum(weights2[offset+j]*self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)

  def forward_single(self, s0, s1, hardwts, hardwts2, drop_prob=0.):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0

    for i in range(self._steps):
      s = 0.
      for j, h in enumerate(states):
        if hardwts2[offset+j] > 0.:
          if drop_prob > 0. and self.training:
            s += drop_path(hardwts2[offset+j]*self._ops[offset+j].forward_single(h, hardwts[offset+j]), drop_prob)
          else:
            s += hardwts2[offset+j]*self._ops[offset+j].forward_single(h, hardwts[offset+j])
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network_imagenet(nn.Module):

  def __init__(self, args, C, num_classes, op_num, sub_policy_num, layers, criterion, primitives, steps=4, multiplier=4, stem_multiplier=3, alpha_weights=None, drop_path_prob=0., augmenting=True, init_policy=True):
    super(Network_imagenet, self).__init__()
    self.args = args
    self.alpha_weights = alpha_weights
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self.drop_path_prob = drop_path_prob

    self.op_num = op_num
    self.sub_policy_num = sub_policy_num
    self.policy_num = int(math.pow(op_num, sub_policy_num))

    nn.Module.PRIMITIVES = primitives

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
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()
    if init_policy:
      self._initialize_policy()
    self._initialize_augment_parameters()
    self.augmenting = augmenting

  def set_tau(self, tau):
    self.tau = tau

  def set_temperature(self, value):
    self.temperature = value

  def get_tau(self):
    return self.tau

  def get_temperature(self):
    return self.temperature

  def set_augmenting(self, value):
      assert value in [False, True]
      self.augmenting = value

  def new(self):
    model_new = Network_imagenet(self.args, self._C, self._num_classes, self.op_num, self.sub_policy_num, self._layers, self._criterion, self.PRIMITIVES, drop_path_prob=self.drop_path_prob, init_policy=False).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    for x, y in zip(model_new.augment_parameters(), self.augment_parameters()):
        x.data.copy_(y.data)
    model_new.gumbel = self.gumbel
    model_new.set_tau(self.tau)
    model_new.set_temperature(self.temperature)
    model_new.policy = self.policy
    model_new.policy_idx = self.policy_idx
    model_new.ops_weights_b = self.ops_weights_b
    model_new.probabilities_b = self.probabilities_b
    model_new.ops_weights_softmax_z = self.ops_weights_softmax_z
    model_new.sample_policy_idx = self.sample_policy_idx
    return model_new

  def forward_joint(self, input):
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      if self.alpha_weights is None:
        if cell.reduction:
          weights = F.softmax(self.alphas_reduce, dim=-1)
          weights2 = F.softmax(self.betas_reduce[0:2], dim=-1)
          n = 3
          start = 2
          for i in range(self._steps-1):
            end = start + n
            tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
            start = end
            n += 1
            weights2 = torch.cat([weights2,tw2],dim=0)
        else:
          weights = F.softmax(self.alphas_normal, dim=-1)
          weights2 = F.softmax(self.betas_normal[0:2], dim=-1)
          n = 3
          start = 2
          for i in range(self._steps-1):
            end = start + n
            tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
            start = end
            n += 1
            weights2 = torch.cat([weights2,tw2],dim=0)
      else:
        raise(ValueError("Why you want to set alphas manually?"))
      
      s0, s1 = s1, cell.forward(s0, s1, weights, weights2, self.drop_path_prob)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def obtain_gumbel(self, weights, gumbels=None):
    if gumbels is None:
      while True:
        gumbels = -torch.empty_like(weights).exponential_().log()
        logits  = (weights.log_softmax(dim=-1) + gumbels) / self.tau
        probs   = nn.functional.softmax(logits, dim=-1)
        if (torch.isinf(gumbels).any()) or (torch.isinf(probs).any()) or (torch.isnan(probs).any()): continue
        else: break
    else:
        logits  = (weights.log_softmax(dim=-1) + gumbels) / self.tau
        probs   = nn.functional.softmax(logits, dim=-1)
    return gumbels, probs

  def forward_aug(self, origin_images):
      return origin_images

  def forward(self, origin_images, joint=False, joint_beta=False, use_max=False, sample_arch=True):
      if self.augmenting:
          img = self.forward_aug(origin_images)
      else:
          img = origin_images
      return self.forward_nas(img, joint, joint_beta, use_max, sample_arch=sample_arch)

  def sample_arch(self, use_max=False, sample=True):
    if use_max:
      # for alphas
      reduce_probs = F.softmax(self.alphas_reduce, dim=-1)
      index   = reduce_probs.max(-1, keepdim=True)[1]
      one_h_r   = torch.zeros_like(reduce_probs).scatter_(-1, index, 1.0)
      reduce_hardwts = one_h_r - reduce_probs.detach() + reduce_probs
      normal_probs = F.softmax(self.alphas_normal, dim=-1)
      index   = normal_probs.max(-1, keepdim=True)[1]
      one_h_n   = torch.zeros_like(normal_probs).scatter_(-1, index, 1.0)
      normal_hardwts = one_h_n - normal_probs.detach() + normal_probs
      # for betas, keep 2 inputs
      ## for reduce
      reduce_beta = F.softmax(self.betas_reduce[0:2], dim=-1)
      index_beta   = reduce_beta.sort(-1, descending=True)[1][:2]
      one_h_r_beta   = torch.zeros_like(reduce_beta).scatter_(-1, index_beta, 1.0)
      reduce_hardwts_beta = one_h_r_beta - reduce_beta.detach() + reduce_beta
      assert reduce_hardwts_beta.sum() == 2
      n = 3
      start = 2
      for i in range(self._steps-1):
        end = start + n
        tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
        index_tw2   = tw2.sort(-1, descending=True)[1][:2]
        one_h_r_tw2   = torch.zeros_like(tw2).scatter_(-1, index_tw2, 1.0)
        reduce_hardwts_tw2 = one_h_r_tw2 - tw2.detach() + tw2
        assert reduce_hardwts_tw2.sum() == 2
        start = end
        n += 1
        reduce_beta = torch.cat([reduce_beta,tw2],dim=0)
        reduce_hardwts_beta = torch.cat([reduce_hardwts_beta,reduce_hardwts_tw2],dim=0)
      ## for normal
      normal_beta = F.softmax(self.betas_normal[0:2], dim=-1)
      index_beta   = normal_beta.sort(-1, descending=True)[1][:2]
      one_h_r_beta   = torch.zeros_like(normal_beta).scatter_(-1, index_beta, 1.0)
      normal_hardwts_beta = one_h_r_beta - normal_beta.detach() + normal_beta
      assert normal_hardwts_beta.sum() == 2
      n = 3
      start = 2
      for i in range(self._steps-1):
        end = start + n
        tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
        index_tw2   = tw2.sort(-1, descending=True)[1][:2]
        one_h_r_tw2   = torch.zeros_like(tw2).scatter_(-1, index_tw2, 1.0)
        normal_hardwts_tw2 = one_h_r_tw2 - tw2.detach() + tw2
        assert normal_hardwts_tw2.sum() == 2
        start = end
        n += 1
        normal_beta = torch.cat([normal_beta,tw2],dim=0)
        normal_hardwts_beta = torch.cat([normal_hardwts_beta,normal_hardwts_tw2],dim=0)

    else:
      if sample:
        self.gumbel = {'reduce': None,
                       'normal': None,
                       'reduce_beta': torch.zeros_like(self.betas_reduce),
                       'normal_beta': torch.zeros_like(self.betas_normal)}

      self.gumbel['reduce'], reduce_probs = self.obtain_gumbel(self.alphas_reduce, self.gumbel['reduce'])
      index   = reduce_probs.max(-1, keepdim=True)[1]
      one_h_r   = torch.zeros_like(reduce_probs).scatter_(-1, index, 1.0)
      reduce_hardwts = one_h_r - reduce_probs.detach() + reduce_probs
      self.gumbel['normal'], normal_probs = self.obtain_gumbel(self.alphas_normal, self.gumbel['normal'])
      index   = normal_probs.max(-1, keepdim=True)[1]
      one_h_n   = torch.zeros_like(normal_probs).scatter_(-1, index, 1.0)
      normal_hardwts = one_h_n - normal_probs.detach() + normal_probs
      # for betas, keep 2 inputs
      ## for reduce
      if sample:
        self.gumbel['reduce_beta'][0:2], reduce_beta = self.obtain_gumbel(self.betas_reduce[0:2], None)
      else:
        self.gumbel['reduce_beta'][0:2], reduce_beta = self.obtain_gumbel(self.betas_reduce[0:2], self.gumbel['reduce_beta'][0:2])
      index_beta   = reduce_beta.sort(-1, descending=True)[1][:2]
      one_h_r_beta   = torch.zeros_like(reduce_beta).scatter_(-1, index_beta, 1.0)
      reduce_hardwts_beta = one_h_r_beta - reduce_beta.detach() + reduce_beta
      n = 3
      start = 2
      for i in range(self._steps-1):
        end = start + n
        if sample:
          self.gumbel['reduce_beta'][start:end], tw2 = self.obtain_gumbel(self.betas_reduce[start:end], None)
        else:
          self.gumbel['reduce_beta'][start:end], tw2 = self.obtain_gumbel(self.betas_reduce[start:end], self.gumbel['reduce_beta'][start:end])
        index_tw2   = tw2.sort(-1, descending=True)[1][:2]
        one_h_r_tw2   = torch.zeros_like(tw2).scatter_(-1, index_tw2, 1.0)
        reduce_hardwts_tw2 = one_h_r_tw2 - tw2.detach() + tw2
        start = end
        n += 1
        reduce_beta = torch.cat([reduce_beta,tw2],dim=0)
        reduce_hardwts_beta = torch.cat([reduce_hardwts_beta,reduce_hardwts_tw2],dim=0)
      ## for normal
      if sample:
        self.gumbel['normal_beta'][0:2], normal_beta = self.obtain_gumbel(self.betas_normal[0:2], None)
      else:
        self.gumbel['normal_beta'][0:2], normal_beta = self.obtain_gumbel(self.betas_normal[0:2], self.gumbel['normal_beta'][0:2])
      index_beta   = normal_beta.sort(-1, descending=True)[1][:2]
      one_h_r_beta   = torch.zeros_like(normal_beta).scatter_(-1, index_beta, 1.0)
      normal_hardwts_beta = one_h_r_beta - normal_beta.detach() + normal_beta
      n = 3
      start = 2
      for i in range(self._steps-1):
        end = start + n
        if sample:
          self.gumbel['normal_beta'][start:end], tw2 = self.obtain_gumbel(self.betas_normal[start:end], None)
        else:
          self.gumbel['normal_beta'][start:end], tw2 = self.obtain_gumbel(self.betas_normal[start:end], self.gumbel['normal_beta'][start:end])
#        _, tw2 = self.obtain_gumbel(self.betas_normal[start:end])
        index_tw2   = tw2.sort(-1, descending=True)[1][:2]
        one_h_r_tw2   = torch.zeros_like(tw2).scatter_(-1, index_tw2, 1.0)
        normal_hardwts_tw2 = one_h_r_tw2 - tw2.detach() + tw2
        start = end
        n += 1
        normal_beta = torch.cat([normal_beta,tw2],dim=0)
        normal_hardwts_beta = torch.cat([normal_hardwts_beta,normal_hardwts_tw2],dim=0)
    return reduce_probs, reduce_beta, reduce_hardwts, reduce_hardwts_beta, normal_probs, normal_beta, normal_hardwts, normal_hardwts_beta

  def forward_nas(self, input, joint=False, joint_beta=False, use_max=False, sample_arch=True):
    reduce_probs, reduce_beta, reduce_hardwts, reduce_hardwts_beta, normal_probs, normal_beta, normal_hardwts, normal_hardwts_beta = self.sample_arch(use_max, sample_arch)

    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      if self.alpha_weights is None:
        if cell.reduction:
          weights = reduce_hardwts if not joint else reduce_probs
          weights2 = reduce_hardwts_beta if not joint_beta else reduce_beta
        else:
          weights = normal_hardwts if not joint else normal_probs
          weights2 = normal_hardwts_beta if not joint_beta else normal_beta
      else:
        raise(ValueError("Why you want to set alphas manually?"))
      
      if joint:
        s0, s1 = s1, cell.forward(s0, s1, weights, weights2, self.drop_path_prob)
      else:
        s0, s1 = s1, cell.forward_single(s0, s1, weights, weights2, self.drop_path_prob)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target, joint=False, joint_beta=False, sample_arch=True):
    logits = self.forward(input, joint, joint_beta, sample_arch=sample_arch)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(self.PRIMITIVES['primitives_normal'][0])

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.betas_normal = Variable(1e-3*torch.randn(k).cuda(), requires_grad=True)
    self.betas_reduce = Variable(1e-3*torch.randn(k).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
      self.betas_normal,
      self.betas_reduce,
    ]

  def _initialize_augment_parameters(self):
      num_sub_policies = self.sub_policy_num
      num_ops = self.op_num
      num_policies = self.policy_num
      self.probabilities = Variable(0.0*torch.ones(num_policies, num_sub_policies).cuda(), requires_grad=True)
      self.ops_weights = Variable(1e-3*torch.ones(num_sub_policies, num_ops).cuda(), requires_grad=True)
      self.magnitudes = Variable(1e-3*torch.ones(num_policies, num_sub_policies, 10).cuda(), requires_grad=True)

      self._augment_parameters = [
          self.probabilities,
          self.ops_weights,
          self.magnitudes,
      ]
      # self.probabilities_dist = torch.distributions.RelaxedBernoulli(
      #     self.temperature, self.probabilities)
      # self.ops_weights_dist = torch.distributions.RelaxedOneHotCategorical(
      #     self.temperature, self.ops_weights)

  def _initialize_policy(self):
      self.policy = []
      self.policy_idx = []
      def _dfs(index=0, sub_policy=[], sub_policy_idx=[], depth=0):
         if depth == self.sub_policy_num:
            self.policy += [tuple(sub_policy)]
            self.policy_idx += [tuple(sub_policy_idx)]
            return
         for i, ops_name in enumerate(ops_names):
            _dfs(i+1, sub_policy + [ops_name], sub_policy_idx + [i], depth+1)
       
      _dfs(index=0, sub_policy=[], sub_policy_idx=[], depth=0)
      assert(len(self.policy) == self.policy_num)

  def arch_parameters(self):
    return self._arch_parameters

  def augment_parameters(self):
    return self._augment_parameters

  def genotype(self):

    def _parse(weights, weights2, normal=True):
      PRIMITIVES = self.PRIMITIVES['primitives_normal' if normal else 'primitives_reduct']

      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        W2 = weights2[start:end].copy()
        for j in range(n):
          W[j,:] = W[j,:] * W2[j]

        try:
           edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES[x].index('none')))[:2]
        except ValueError: # This error happens when the 'none' op is not present in t    he ops
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

    n = 3
    start = 2
    weightsr2 = F.softmax(self.betas_reduce[0:2], dim=-1)
    weightsn2 = F.softmax(self.betas_normal[0:2], dim=-1)
    for i in range(self._steps-1):
      end = start + n
      tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
      tn2 = F.softmax(self.betas_normal[start:end], dim=-1)
      start = end
      n += 1
      weightsr2 = torch.cat([weightsr2,tw2],dim=0)
      weightsn2 = torch.cat([weightsn2,tn2],dim=0)

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), weightsn2.data.cpu().numpy(), True)
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), weightsr2.data.cpu().numpy(), False)

    concat = list(range(2+self._steps-self._multiplier, self._steps+2))
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

  def genotype_aug(self, geno_type='t1'):
    if geno_type == 't1':
      # construct policy_sal
      ops_weights = torch.nn.functional.softmax(self.ops_weights, dim=-1)
      policy_sal = torch.ones(self.policy_num, device=ops_weights.device)
      for i in range(self.policy_num):
          policy_idx = self.policy_idx[i] # one policy with K sub-policies
          for k in range(len(policy_idx)):
              policy_sal[i] *= ops_weights[k, policy_idx[k]] 
      index = torch.argsort(policy_sal, descending=True)
      probabilities = torch.sigmoid(self.probabilities)
      magnitudes = self.magnitudes.clamp(0, 1)
      magnitudes = (torch.argmax(self.magnitudes, dim=-1).float() + 1)*0.1
      gene = []
      for idx in index:
          policy = self.policy[idx] # one policy with K sub-policies
          policy_idx = self.policy_idx[idx]
          gene += [tuple([(policy[k],
                 probabilities[idx][k].data.detach().item(),
                 magnitudes[idx][k].data.detach().item(),
                 policy_sal[idx].data.detach().item()) for k in range(len(policy))])]
    else:
      raise(ValueError("No geno_type as %s"%geno_type))
        
    return gene


  def sample(self):
      EPS = 1e-6
      num_policies = self.policy_num
      num_sub_policies = self.sub_policy_num
      num_ops = self.op_num

      # sample for probability for all policies
      probabilities = torch.sigmoid(self.probabilities)
      probabilities_logits = torch.log(probabilities - torch.log1p(-probabilities))
      while True:
        probabilities_u = torch.rand(num_policies, num_sub_policies).cuda()
        probabilities_v = torch.rand(num_policies, num_sub_policies).cuda()
        if (probabilities_u<EPS).any() or (probabilities_v<EPS).any(): continue
        else: break
      probabilities_z = probabilities_logits + torch.log(probabilities_u) - torch.log1p(-probabilities_u)
      probabilities_b = probabilities_z.gt(0.0).type_as(probabilities_z)
      def _get_probabilities_z_tilde(theta, logits, b, v):
#          theta = torch.sigmoid(logits)
          v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
          z_tilde = logits + torch.log(v_prime) - torch.log1p(-v_prime)
          return z_tilde
      probabilities_z_tilde = _get_probabilities_z_tilde(probabilities, probabilities_logits, probabilities_b, probabilities_v)
      self.probabilities_b = probabilities_b
      self.probabilities_sig = probabilities
      self.probabilities_sig_z = torch.sigmoid(probabilities_z/self.temperature)
      self.probabilities_sig_z_tilde = torch.sigmoid(probabilities_z_tilde/self.temperature)

      # sample magnitude for all policies
      magnitude_p = torch.nn.functional.softmax(self.magnitudes, dim=-1)
      magnitude_logits = torch.log(magnitude_p)
      while True:
        magnitude_u = -torch.empty_like(magnitude_logits).exponential_().log()
        if torch.isinf(magnitude_u).any(): continue
        else: break
      magnitude_z = magnitude_logits + magnitude_u
      magnitude_b = torch.argmax(magnitude_z, dim=-1)
      self.magnitude_softmax = magnitude_p
      self.magnitude_b = magnitude_b
      self.magnitude_softmax_z = torch.nn.functional.softmax(magnitude_z/self.temperature, dim=-1)

      # sample policy
      ops_weights_p = torch.nn.functional.softmax(self.ops_weights, dim=-1)
      ops_weights_logits = torch.log(ops_weights_p)
      while True:
        ops_weights_u = -torch.empty_like(ops_weights_logits).exponential_().log()
        if torch.isinf(ops_weights_u).any(): continue
        else: break
      ops_weights_z = ops_weights_logits + ops_weights_u
      ops_weights_b = torch.argmax(ops_weights_z, dim=-1)
      def _get_ops_weights_z_tilde(theta, b):
        while True:
          v = torch.rand(num_sub_policies, num_ops).cuda()
          vb = torch.gather(v, -1, b.view(-1,1)).view(-1, 1)
          z_tilde = -torch.log(-torch.log(v)/theta-torch.log(vb).expand_as(v))
          src = -torch.log(-torch.log(vb))
          if (torch.isinf(z_tilde).any() or torch.isinf(src).any()): continue
          else: break
        z_tilde = z_tilde.scatter(dim=-1, index=b.view(-1,1), src=src)
        return z_tilde
      ops_weights_z_tilde = _get_ops_weights_z_tilde(ops_weights_p, ops_weights_b)
      self.ops_weights_b = ops_weights_b
      self.ops_weights_softmax_z = torch.nn.functional.softmax(ops_weights_z/self.temperature, dim=-1)
      self.ops_weights_softmax_z_tilde = torch.nn.functional.softmax(ops_weights_z_tilde/self.temperature, dim=-1)
      self.sample_policy_idx = self.get_policy_idx(ops_weights_b)

  def get_policy_idx(self, ops_weights_b):
    for i, idxes in enumerate(self.policy_idx):
      same = True
      for k, k_idx in enumerate(ops_weights_b):
        if idxes[k] != k_idx.item(): same=False; break
      if same: 
        return i
    raise(ValueError("Something goes wrong!"))


  def relax(self, f_b):
    probabilities_b = self.probabilities_b[self.sample_policy_idx]
#    probabilities_sig_z = self.probabilities_sig_z[self.sample_policy_idx]
    probabilities_sig_z = self.probabilities_sig[self.sample_policy_idx]
    probabilities_sig_z = probabilities_sig_z * probabilities_b + (1.- probabilities_sig_z) * (1. - probabilities_b)

    magnitude_b = self.magnitude_b[self.sample_policy_idx]
    magnitude_softmax = self.magnitude_softmax[self.sample_policy_idx]
    magnitude_softmax_z = torch.gather(magnitude_softmax, dim=-1, index=magnitude_b.view(-1,1)).view(-1)

    ops_weights_max_z_idx = self.ops_weights_softmax_z.max(dim=-1, keepdim=True)[1]
    ops_weights_max_z = torch.gather(torch.nn.functional.softmax(self.ops_weights, dim=-1), dim=-1, index=ops_weights_max_z_idx)
    log_prob = torch.log(probabilities_sig_z).sum() + torch.log(ops_weights_max_z).sum() + (torch.log(magnitude_softmax_z)*probabilities_b.detach()).sum()
    d_log_prob_list = torch.autograd.grad(
          log_prob, [self.probabilities, self.ops_weights, self.magnitudes], grad_outputs=torch.ones_like(log_prob), retain_graph=True)

    d_logits_list = [d*f_b for d in d_log_prob_list]

    return [d_logits.detach() for d_logits in d_logits_list]


