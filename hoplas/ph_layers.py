
##########################################################
# Code is from E. Grassucci et al., HyperNets repo
# Hypercomplex Neural Networks with PyTorch, https://github.com/eleGAN23/HyperNets
##########################################################
"""
MIT License

Copyright (c) 2021 Eleonora Grassucci

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

## New code added by S.H. Hawley, June 2026, is at the end of this file, after the original code from E. Grassucci et al.

import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, init
from torch.nn.parameter import Parameter



########################
## STANDARD PHM LAYER ##
########################

class PHMLinear(nn.Module):

  def __init__(self, n, in_features, out_features, cuda=True, rand_init_a=True):
    super(PHMLinear, self).__init__()
    self.n = n
    self.in_features = in_features
    self.out_features = out_features
    self.cuda = cuda

    self.bias = nn.Parameter(torch.Tensor(out_features))

    # rand_init_a=False: zero-init (no RNG draw) for the frozen-algebra case (e.g. quat),
    # so the RNG stream reaching s/bias matches a layer that never draws a (KingdonQuaternion).
    if rand_init_a:
        self.a = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))
    else:
        self.a = nn.Parameter(torch.zeros((n, n, n)))

    self.s = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, self.out_features//n, self.in_features//n))))

    self.weight = torch.zeros((self.out_features, self.in_features))

    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(self.bias, -bound, bound)


  def kronecker_product1(self, a, b):
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    out = res.reshape(siz0 + siz1)
    return out

  def kronecker_product2(self):
    H = torch.zeros((self.out_features, self.in_features))
    for i in range(self.n):
        H = H + torch.kron(self.a[i], self.s[i])
    return H

  def forward(self, input):
    self.weight = torch.sum(self.kronecker_product1(self.a, self.s), dim=0)
#     self.weight = self.kronecker_product2()
    input = input.to(dtype=self.weight.dtype)
    return F.linear(input, weight=self.weight, bias=self.bias)

  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}'.format(
      self.in_features, self.out_features, self.bias is not None)
    
  def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.a, a=math.sqrt(5))
    init.kaiming_uniform_(self.s, a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(self.placeholder)
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(self.bias, -bound, bound)

################################
## PHC LAYER: 2D convolutions ##
################################

class PHConv2d(Module):

  def __init__(self, n, in_features, out_features, kernel_size, padding=0, stride=1, cuda=True):
    super(PHConv2d, self).__init__()
    self.n = n
    self.in_features = in_features
    self.out_features = out_features
    self.padding = padding
    self.stride = stride
    self.cuda = cuda

    self.bias = nn.Parameter(torch.Tensor(out_features))
    self.A = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))
    self.F = nn.Parameter(torch.nn.init.xavier_uniform_(
        torch.zeros((n, self.out_features//n, self.in_features//n, kernel_size, kernel_size))))
    self.weight = torch.zeros((self.out_features, self.in_features))
    self.kernel_size = kernel_size

    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(self.bias, -bound, bound)

  def kronecker_product1(self, A, F):
    siz1 = torch.Size(torch.tensor(A.shape[-2:]) * torch.tensor(F.shape[-4:-2]))
    siz2 = torch.Size(torch.tensor(F.shape[-2:]))
    res = A.unsqueeze(-1).unsqueeze(-3).unsqueeze(-1).unsqueeze(-1) * F.unsqueeze(-4).unsqueeze(-6)
    siz0 = res.shape[:1]
    out = res.reshape(siz0 + siz1 + siz2)
    return out

  def kronecker_product2(self):
    H = torch.zeros((self.out_features, self.in_features, self.kernel_size, self.kernel_size))
    if self.cuda:
        H = H.cuda()
    for i in range(self.n):
        kron_prod = torch.kron(self.A[i], self.F[i]).view(self.out_features, self.in_features, self.kernel_size, self.kernel_size)
        H = H + kron_prod
    return H

  def forward(self, input):
    self.weight = torch.sum(self.kronecker_product1(self.A, self.F), dim=0)
    # self.weight = self.kronecker_product2()
    if self.cuda:
        self.weight = self.weight.cuda()

    input = input.to(dtype=self.weight.dtype)
        
    return F.conv2d(input, weight=self.weight, stride=self.stride, padding=self.padding)

  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}'.format(
      self.in_features, self.out_features, self.bias is not None)
    
  def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.A, a=math.sqrt(5))
    init.kaiming_uniform_(self.F, a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(self.placeholder)
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(self.bias, -bound, bound)

################################
## PHC LAYER: 1D convolutions ##
################################
    
class PHConv1D(Module):

  def __init__(self, n, in_features, out_features, kernel_size, padding=0, stride=1, dilation=1, cuda=True):
    super(PHConv1D, self).__init__()
    self.n = n
    self.in_features = in_features
    self.out_features = out_features
    self.padding = padding
    self.stride = stride
    self.dilation=dilation
    self.cuda = cuda

    self.bias = nn.Parameter(torch.Tensor(out_features))

    self.A = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))
    self.F = nn.Parameter(torch.nn.init.xavier_uniform_(
        torch.zeros((n, self.out_features//n, self.in_features//n, kernel_size))))
    self.weight = torch.zeros((self.out_features, self.in_features))
    self.kernel_size = kernel_size

    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(self.bias, -bound, bound)

  def kronecker_product1(self, A, F):
    siz1 = torch.Size(torch.tensor(A.shape[-2:]) * torch.tensor(F.shape[-3:-1]))
    siz2 = torch.Size(torch.tensor(F.shape[-1:]))
    res = A.unsqueeze(-1).unsqueeze(-3).unsqueeze(-1) * F.unsqueeze(-3).unsqueeze(-5)
    siz0 = res.shape[:1]
    out = res.reshape(siz0 + siz1 + siz2)
    return out

  def kronecker_product2(self):
    H = torch.zeros((self.out_features, self.in_features, self.kernel_size, self.kernel_size))
    if self.cuda:
        H = H.cuda()
    for i in range(self.n):
        kron_prod = torch.kron(self.A[i], self.F[i]).view(self.out_features, self.in_features, self.kernel_size, self.kernel_size)
        H = H + kron_prod
    return H

  def forward(self, input):
    self.weight = torch.sum(self.kronecker_product1(self.A, self.F), dim=0)
    # self.weight = self.kronecker_product2()
    if self.cuda:
        self.weight = self.weight.cuda()

    input = input.to(dtype=self.weight.dtype)
    return F.conv1d(input, weight=self.weight, stride=self.stride, padding=self.padding,dilation=self.dilation)
    
    def extra_repr(self) -> str:
      return 'in_features={}, out_features={}, bias={}'.format(
        self.in_features, self.out_features, self.bias is not None)
    
  def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.A, a=math.sqrt(5))
    init.kaiming_uniform_(self.F, a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(self.placeholder)
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(self.bias, -bound, bound)
        
################################
## PHC LAYER: 3D convolutions ##
################################
  
  
class PHConv3D(Module):

  def __init__(self, n, in_features, out_features, kernel_size, padding=0, stride=1, dilation=1, cuda=True):
    super(PHConv3D, self).__init__()
    self.n = n
    self.in_features = in_features
    self.out_features = out_features
    self.padding = padding
    self.stride = stride
    self.dilation=dilation
    self.cuda = cuda

    self.bias = nn.Parameter(torch.Tensor(out_features))

    self.A = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))
    self.F = nn.Parameter(torch.nn.init.xavier_uniform_(
        torch.zeros((n, self.out_features//n, self.in_features//n, kernel_size, kernel_size, kernel_size))))
    self.weight = torch.zeros((self.out_features, self.in_features))
    self.kernel_size = kernel_size

    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(self.bias, -bound, bound)

  def kronecker_product1(self, A, F):
    siz1 = torch.Size(torch.tensor(A.shape[-2:]) * torch.tensor(F.shape[-5:-3]))
    siz2 = torch.Size(torch.tensor(F.shape[-3:]))
    res = A.unsqueeze(-1).unsqueeze(-3).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * F.unsqueeze(-5).unsqueeze(-7)
    siz0 = res.shape[:1]
    out = res.reshape(siz0 + siz1 + siz2)
    return out

  def kronecker_product2(self):
    H = torch.zeros((self.out_features, self.in_features, self.kernel_size, self.kernel_size,self.kernel_size))
    if self.cuda:
        H = H.cuda()
    for i in range(self.n):
        kron_prod = torch.kron(self.A[i], self.F[i]).view(self.out_features, self.in_features, self.kernel_size, self.kernel_size, self.kernel_size)
        H = H + kron_prod
    return H

  def forward(self, input):
    self.weight = torch.sum(self.kronecker_product1(self.A, self.F), dim=0)
    # self.weight = self.kronecker_product2()
#     if self.cuda:
#         self.weight = self.weight.cuda()

    input = input.to(dtype=self.weight.dtype)
    return F.conv3d(input, weight=self.weight, stride=self.stride, padding=self.padding,dilation=self.dilation)    
  
    def extra_repr(self) -> str:
      return 'in_features={}, out_features={}, bias={}'.format(
        self.in_features, self.out_features, self.bias is not None)
    
  def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.A, a=math.sqrt(5))
    init.kaiming_uniform_(self.F, a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(self.placeholder)
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(self.bias, -bound, bound)
   




"""
New code below added by S.H. Hawley, June 2026
"""
class PHMLinear_Implicit(nn.Module):
    """Implicit: einsum contraction, no materialized weight."""
    def __init__(self, n, in_features, out_features):
        super().__init__()
        self.n = n
        self.in_features = in_features
        self.out_features = out_features
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.a = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))
        self.s = nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.zeros((n, out_features // n, in_features // n))))
        fan_in = in_features
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        input = input.to(dtype=self.a.dtype)
        X = input.reshape(input.shape[0], self.n, -1)
        Y = torch.einsum('iab,ijk,Bbk->Baj', self.a, self.s, X)
        return Y.reshape(input.shape[0], -1) + self.bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)


