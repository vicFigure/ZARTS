import torch
import torch.nn as nn
from torch.autograd import Variable

OPS = {
  'noise': lambda C, stride, affine: NoiseOp(stride, 0., 1.),
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'merge_conv_5x5' : lambda C, stride, affine, single_sepConv, with_d5: MergeSepConv(C, C, 5, stride, 4, affine=affine, single=single_sepConv, with_d5=with_d5),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_3x3' : lambda C, stride, affine: ReLUConvBN(C, C, 3, stride, 1, affine=affine, dilation=1),
  'conv_5x5' : lambda C, stride, affine: ReLUConvBN(C, C, 5, stride, 2, affine=affine, dilation=1),
  'conv_3x3_dil' : lambda C, stride, affine: ReLUConvBN(C, C, 3, stride, 2, affine=affine, dilation=2),
  'conv_5x5_dil' : lambda C, stride, affine: ReLUConvBN(C, C, 5, stride, 4, affine=affine, dilation=2),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
}

class NoiseOp(nn.Module):
    def __init__(self, stride, mean, std):
        super(NoiseOp, self).__init__()
        self.stride = stride
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.stride != 1:
          x_new = x[:,:,::self.stride,::self.stride]
        else:
          x_new = x
        noise = Variable(x_new.data.new(x_new.size()).normal_(self.mean, self.std))
#        if self.training:
#          noise = Variable(x_new.data.new(x_new.size()).normal_(self.mean, self.std))
#        else:
#          noise = torch.zeros_like(x_new)
        return noise


class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, dilation=1):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out


class MergeSepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, single=False, with_d5=True):
    super(MergeSepConv, self).__init__()
    self.stride = stride
    self.padding = padding
    self.affine = affine
    self.single = single
    self.with_d5 = with_d5
    if self.with_d5:
      self.kernel_size = kernel_size*2-1
    else:
      self.kernel_size = kernel_size

    self.relu1 = nn.ReLU(inplace=False)
#    self.depth_conv1 = nn.Conv2d(C_in, C_in, kernel_size=self.kernel_size, stride=stride, padding=padding, groups=C_in, bias=False)
    tmp1 = torch.Tensor(C_in, 1, self.kernel_size, self.kernel_size)
    torch.nn.init.kaiming_normal(tmp1,  mode='fan_in')
    self.depth_weight1 = nn.Parameter(tmp1)
    C_tmp = C_out if single else C_in
    self.point_conv1 = nn.Conv2d(C_in, C_tmp, kernel_size=1, padding=0, bias=False)
    self.bn1 = nn.BatchNorm2d(C_tmp, affine=affine)

    '''
    self.Mask5_tmp = torch.ones(self.kernel_size, requires_grad=True)
    self.Mask3_tmp = torch.zeros(self.kernel_size, requires_grad=True)
    self.Maskd3_tmp = torch.zeros(self.kernel_size, requires_grad=True)
    self.Mask3[1:4,1:4] = 1.
    self.Maskd3[0:5:2,0:5:2] = 1.
    self.Mask5_tmp = self.Mask5_tmp.unsqueeze(0).unsqueeze(0)
    self.Mask3_tmp = self.Mask5_tmp.unsqueeze(0).unsqueeze(0)
    self.Maskd3_tmp = self.Mask5_tmp.unsqueeze(0).unsqueeze(0)
    self.register_buffer("Mask5", self.Mask5_tmp)
    self.register_buffer("Mask3", self.Mask3_tmp)
    self.register_buffer("Maskd3", self.Maskd3_tmp)
    if self.with_d5:
      self.Maskd5_tmp = torch.ones(self.kernel_size)
    '''

    if not single:
      self.relu2 = nn.ReLU(inplace=False)
#      self.depth_conv2 = nn.Conv2d(C_in, C_in, kernel_size=self.kernel_size, stride=1, padding=padding, groups=C_in, bias=False)
      tmp2 = torch.Tensor(C_in, 1, self.kernel_size, self.kernel_size)
      torch.nn.init.kaiming_normal(tmp2,  mode='fan_in')
      self.depth_weight2 = nn.Parameter(tmp2)
      self.point_conv2 = nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)
      self.bn2 = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x, weight):
    x = self.relu1(x)
    padding = 4 if self.with_d5 else 2
    C_in = x.size(1)
    w5_1 = self.depth_weight1
    merge_kernel_1 = self.get_merge_kernel(w5_1, weight, with_d5=self.with_d5)
    x = torch.nn.functional.conv2d(x, merge_kernel_1, stride=self.stride, padding=padding, dilation=1, groups=C_in)
    x = self.point_conv1(x)
    x = self.bn1(x)

    if not self.single:
      x = self.relu2(x)
      w5_2 = self.depth_weight2
      merge_kernel_2 = self.get_merge_kernel(w5_2, weight, with_d5=self.with_d5)
      x = torch.nn.functional.conv2d(x, merge_kernel_2, stride=1, padding=padding, dilation=1, groups=C_in)
      x = self.point_conv2(x)
      x = self.bn2(x)
    
    return x

  def get_merge_kernel(self, w5, alphas, with_d5=True):
    if with_d5:
        Cout,C,_,_ = w5.shape
        alpha3, alpha5, alphad3, alphad5 = alphas
        w5_pad = w5[:,:,2:7,2:7]
        w5_pad = torch.nn.functional.pad(w5_pad, (2,2,2,2), "constant", value=0)
    
        w3 = w5[:,:,3:6,3:6]
        w3_pad = torch.nn.functional.pad(w3, (3,3,3,3), "constant", value=0)
    
#        dw3 = Variable(torch.zeros(Cout, C, 5, 5)).cuda()
        dw3 = torch.zeros_like(w5)
        dw3[:,:,2:7:2,2:7:2] = w5[:,:,2:7:2,2:7:2]
    
#        dw5 = Variable(torch.zeros(Cout, C, 9, 9)).cuda()
#        dw5 = torch.zeros_like(w5, requires_grad=True)
        dw5 = torch.zeros_like(w5)
        dw5[:,:,0:9:2,0:9:2] = w5[:,:,0:9:2,0:9:2]
        merge_kernel = w5_pad*alpha5 + w3_pad*alpha3 + dw3*alphad3 + dw5*alphad5
    else:
        Cout,C,_,_ = w5.shape
        alpha3, alpha5, alphad3 = alphas
    
        w3 = w5[:,:,1:4,1:4]
        w3_pad = torch.nn.functional.pad(w3, (1,1,1,1), "constant", value=0)
    
#        dw3 = Variable(torch.zeros(Cout, C, 5, 5)).cuda()
        dw3 = torch.zeros_like(w5)
        dw3[:,:,0:5:2,0:5:2] = w5[:,:,0:5:2,0:5:2]
    
        merge_kernel = w5*alpha5 + w3_pad*alpha3 + dw3*alphad3
    return merge_kernel
    
  def get_merge_kernel_mask(self, w5, alphas, with_d5=True):
    if with_d5:
        raise(ValueError("for mask forward, with_d5 should be False"))
        Cout,C,_,_ = w5.shape
        alpha3, alpha5, alphad3, alphad5 = alphas
        w5_pad = torch.nn.functional.pad(w5, (2,2,2,2), "constant", value=0)
        merge_kernel = w5_pad*(alpha5*self.Mask5 + alpha3*self.Mask3 + alphad3*self.Maskd3)
    else:
        Cout,C,_,_ = w5.shape
        alpha3, alpha5, alphad3 = alphas

        self.Mask5_tmp = torch.ones_like(w5, requires_grad=True)
        self.Mask3_tmp = torch.zeros_like(w5, requires_grad=True)
        self.Maskd3_tmp = torch.zeros_like(w5, requires_grad=True)
        self.Mask3[1:4,1:4] = 1.
        self.Maskd3[0:5:2,0:5:2] = 1.
    
        merge_kernel = w5*(alpha5*self.Mask5 + alpha3*self.Mask3 + alphad3*self.Maskd3)
    return merge_kernel
