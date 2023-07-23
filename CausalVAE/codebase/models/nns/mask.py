#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import torch
import torch.nn.functional as F
from codebase import utils as ut
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")


class MaskLayer(nn.Module):
    def __init__(self, z_dim, concept=4, z2_dim=4):
        super().__init__()
        self.z_dim = z_dim
        self.z2_dim = z2_dim
        self.concept = concept

        self.elu = nn.ELU()
        self.net1 = nn.Sequential(
            nn.Linear(z2_dim , 32), # input : z1_dim=4  //  output = 32
            nn.ELU(),
            nn.Linear(32, z2_dim)
        )
        self.net2 = nn.Sequential(
            nn.Linear(z2_dim , 32),
            nn.ELU(),
            nn.Linear(32, z2_dim)
        )
        self.net3 = nn.Sequential(
            nn.Linear(z2_dim , 32),
            nn.ELU(),
            nn.Linear(32, z2_dim)
        )
        self.net4 = nn.Sequential(
            nn.Linear(z2_dim , 32),
            nn.ELU(),
            nn.Linear(32, z2_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(z_dim , 32),
            nn.ELU(),
            nn.Linear(32, z_dim)
        )

    def mix(self, z):
        zy = z.view(-1, self.concept * self.z2_dim)
        if self.z2_dim == 1:
            zy = zy.reshape(zy.size()[0],zy.size()[1],1)
            if self.concept ==4:
                zy1, zy2, zy3, zy4= zy[:,0],zy[:,1],zy[:,2],zy[:,3]
            elif self.concept ==3:
                zy1, zy2, zy3= zy[:,0],zy[:,1],zy[:,2]
        else:
            if self.concept ==4:
                zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim//self.concept, dim = 1) # floor division 15/2=7
            elif self.concept ==3:
                zy1, zy2, zy3= torch.split(zy, self.z_dim//self.concept, dim = 1)
        rx1 = self.net1(zy1)
        rx2 = self.net2(zy2)
        rx3 = self.net3(zy3)
        if self.concept ==4:
            rx4 = self.net4(zy4)
            h = torch.cat((rx1,rx2,rx3,rx4), dim=1)
        elif self.concept ==3:
            h = torch.cat((rx1,rx2,rx3), dim=1)
        #print(h.size())
        return h
  
   
class Attention(nn.Module):
    def __init__(self, in_features, bias=False):
        super().__init__()
        self.M =  nn.Parameter(torch.nn.init.normal_(torch.zeros(in_features,in_features), mean=0, std=1))
        self.sigmd = torch.nn.Sigmoid()
        #self.M =  nn.Parameter(torch.zeros(in_features,in_features))
        #self.A = torch.zeros(in_features,in_features).to(device)
    
    def attention(self, z, e):
        a = z.matmul(self.M).matmul(e.permute(0,2,1))
        a = self.sigmd(a)
        #print(self.M)
        A = torch.softmax(a, dim = 1)
        e = torch.matmul(A,e)
        return e, A
    
class DagLayer(nn.Linear):
    def __init__(self, in_features, out_features, i = False, bias=False):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.i = i
        self.a = torch.zeros(out_features, out_features)        
        #self.a[0][1], self.a[0][2], self.a[0][3] = 1,1,1
        #self.a[1][2], self.a[1][3] = 1,1
        self.A = nn.Parameter(self.a)
        
        self.b = torch.eye(out_features)
        self.B = nn.Parameter(self.b)
        
        self.I = nn.Parameter(torch.eye(out_features))
        self.I.requires_grad=False
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
    def mask_z(self, x):
        self.B = self.A
        #if self.i:
        #    x = x.view(-1, x.size()[1], 1)
        #    x = torch.matmul((self.B+0.5).t().int().float(), x)
        #    return x
        x = torch.matmul(self.B.t(), x)
        return x
        
    def mask_u(self, x):
        self.B = self.A
        #if self.i:
        #    x = x.view(-1, x.size()[1], 1)
        #    x = torch.matmul((self.B+0.5).t().int().float(), x)
        #    return x
        x = x.view(-1, x.size()[1], 1)
        x = torch.matmul(self.B.t(), x)
        return x


    def calculate_dag(self, x, v):
        #print(self.A)
        #x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)
        
        if x.dim()>2:
            x = x.permute(0,2,1)
        # x: B x 4 x 4 / A: 4 x 4
        # x[i] = x[i] A^T
        x = F.linear(x, torch.inverse(self.I - self.A.t()), self.bias)
        #print(x.size())

        if x.dim()>2:
            x = x.permute(0,2,1).contiguous()
        return x,v
        
        
class Encoder(nn.Module):
    def __init__(self, z_dim, channel=4, y_dim=4):
        super().__init__()
        self.channel = channel
        
        self.LReLU = nn.LeakyReLU(0.2, inplace=True)
        self.net = nn.Sequential(
            nn.Linear(self.channel*96*96, 900),
            nn.ELU(),
            nn.Linear(900, 300),
            nn.ELU(),
            nn.Linear(300, 2 * z_dim),
        )

    def encode(self, x, y=None):
        xy = x if y is None else torch.cat((x, y), dim=1)
        xy = xy.view(-1, self.channel*96*96)
        h = self.net(xy)
        m, v = ut.gaussian_parameters(h, dim=1)
        return m, v

class Conv_Encoder(nn.Module):
    def __init__(self, z_dim, channel=3, y_dim=None):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim if y_dim is not None else 0
        self.channel = channel

        self.net = nn.Sequential(
            # 3 x 128 x 128 -> 32 x 64 x 64
            nn.Conv2d(3, 32, 3, stride=2, padding=1), 
            nn.LeakyReLU(0.2, inplace=True),

            # 32 x 64 x 64 -> 64 x 32 x 32
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 x 32 x 32 -> 64 x 16 x 16
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 x 16 x 16 -> 64 x 8 x 8
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 x 8 x 8 -> 256 x 4 x 4
            nn.Conv2d(64, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 256 x 4 x 4 -> 256 x 1
            nn.Flatten(),
        )
        self.linear = nn.Linear(256*4*4 + self.y_dim, 2*self.z_dim)
    
    def encode(self, x, y=None):
        hh = self.net(x)
        if self.y_dim > 0 and y:
            h = self.linear(torch.cat((hh, y), dim=1))
        else:
            h = self.linear(hh)
        m, v = ut.gaussian_parameters(h, dim=1)
        return m, v


class Decoder_DAG(nn.Module):
    def __init__(self, z_dim, concept, z1_dim, channel = 4, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.concept = concept
        self.y_dim = y_dim
        self.channel = channel
        #print(self.channel)
        self.elu = nn.ELU()
        self.net1 = nn.Sequential(
            nn.Linear(z1_dim + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 1024),
            nn.ELU(),
            nn.Linear(1024, self.channel*96*96)
        )
        self.net2 = nn.Sequential(
            nn.Linear(z1_dim + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 1024),
            nn.ELU(),
            nn.Linear(1024, self.channel*96*96)
        )
        self.net3 = nn.Sequential(
            nn.Linear(z1_dim + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 1024),
            nn.ELU(),
            nn.Linear(1024, self.channel*96*96)
        )
        self.net4 = nn.Sequential(
            nn.Linear(z1_dim + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 1024),
            nn.ELU(),
            nn.Linear(1024, self.channel*96*96)
        )

    def decode_sep(self, z, u, y=None):
        z = z.view(-1, self.concept*self.z1_dim)
        zy = z if y is None else torch.cat((z, y), dim=1)
            
        if self.z1_dim == 1:
            zy = zy.reshape(zy.size()[0],zy.size()[1],1)
            if self.concept ==4:
                zy1, zy2, zy3, zy4= zy[:,0],zy[:,1],zy[:,2],zy[:,3]
            elif self.concept ==3:
                zy1, zy2, zy3= zy[:,0],zy[:,1],zy[:,2]
        else:
            if self.concept ==4:
                zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim//self.concept, dim = 1)
            elif self.concept ==3:
                zy1, zy2, zy3= torch.split(zy, self.z_dim//self.concept, dim = 1)
        rx1 = self.net1(zy1)
        rx2 = self.net2(zy2)
        rx3 = self.net3(zy3)
        if self.concept ==4:
            rx4 = self.net4(zy4)
            h = (rx1+rx2+rx3+rx4)/self.concept
        elif self.concept ==3:
            h = (rx1+rx2+rx3)/self.concept
        
        return h,h,h,h,h


class Conv_Decoder_DAG(nn.Module):
    def __init__(self, z_dim, concept, z2_dim, channel = 3, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.z2_dim = z2_dim
        self.concept = concept
        self.y_dim = y_dim
        self.channel = channel

        self.linear = nn.Sequential(
            nn.Linear(128, 128*4*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.net = nn.Sequential(
            # 64 x 8 x 8
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 x 16 x 16
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 x 32 x 32
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 x 64 x 64
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 3 x 128 x 128
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def decode(self, z, u, y=None):
        z = z.view(-1, self.concept * self.z2_dim)
        zy = z if y is None else torch.cat((z, y), dim=1)
        return self.net(self.linear(zy).view(-1, 128, 4, 4))
