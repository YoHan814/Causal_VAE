#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import torch
import numpy as np
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")



class CausalVAE(nn.Module):
    def __init__(self, nn='mask', name='vae', z_dim=16, z1_dim=4, z2_dim=4, inference = False, alpha=0.3, beta=1):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.channel = 4
        self.scale = np.array([[0,44], [100,40], [6.5, 3.5], [10,5]])
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, self.channel)
        self.dec = nn.Decoder_DAG(self.z_dim,self.z1_dim, self.z2_dim)
        self.dag = nn.DagLayer(self.z1_dim, self.z1_dim, i = inference)
        
        self.attn = nn.Attention(self.z1_dim)
        self.mask_z = nn.MaskLayer(self.z_dim)
        self.mask_u = nn.MaskLayer(self.z1_dim,z1_dim=1)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x, label, mask = None, sample = False, adj = None, alpha=0.3, beta=1, lambdav=0.001):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        assert label.size()[1] == self.z1_dim

        q_m, q_v = self.enc.encode(x.to(device)) # B * 16; e_mean, e_var
        # Reshape B x 16 -> B x 4 x 4
        q_m, q_v = q_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]),torch.ones(q_m.size()[0], self.z1_dim,self.z2_dim).to(device) # deterministic encoder
        # q_m, q_v = q_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]), q_v(q_v.size()[0], self.z1_dim, self.z2_dim) # random encoder 

        # decode_m: z = (I-A^t)^{-1} e / decode_v: 1 ; deterministic
        decode_m, decode_v = self.dag.calculate_dag(q_m.to(device), torch.ones(q_m.size()[0], self.z1_dim,self.z2_dim).to(device))
        decode_m = decode_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim])
        if sample == False:
            if mask != None and mask < 2: # mask: index of concept to intervene
                # intervene z before masking layer
                z_mask = torch.ones(q_m.size()[0], self.z1_dim, self.z2_dim).to(device) * adj
                decode_m[:, mask, :] = z_mask[:, mask, :]
                decode_v[:, mask, :] = z_mask[:, mask, :]
            m_zm = self.dag.mask_z(decode_m.to(device)).reshape([q_m.size()[0], self.z1_dim, self.z2_dim]) # A^T z
            m_zv = decode_v.reshape([q_m.size()[0], self.z1_dim, self.z2_dim]) # all is 1
            m_u = self.dag.mask_u(label.to(device)) # A^T u

            # Apply mask layer(MLP) for each concept(dim=1)
            f_z = self.mask_z.mix(m_zm).reshape([q_m.size()[0], self.z1_dim, self.z2_dim]).to(device) # g(A^T z)
            # e_tilde = Attention((I-A^t)^{-1}e, e)
#             e_tilde = self.attn.attention(decode_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]).to(device),
#                                           q_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]).to(device))[0]
            e_tilde = q_m
              
            f_z1 = f_z + e_tilde # z_i = g_i(A_i o z) + e_tilde_i
            if mask != None and mask < 2:
                z_mask = torch.ones(q_m.size()[0],self.z1_dim, self.z2_dim).to(device) * adj
                f_z1[:, mask, :] = z_mask[:, mask, :]
                m_zv[:, mask, :] = z_mask[:, mask, :]
            
            if mask!= None and (mask == 2 or mask == 3):
                # Only set z after masking layer to 0 for shadow position and length (no causal effect)
                z_mask = torch.ones(q_m.size()[0],self.z1_dim,self.z2_dim).to(device)*adj
                f_z1[:, mask, :] = z_mask[:, mask, :]
                m_zv[:, mask, :] = z_mask[:, mask, :]
            g_u = self.mask_u.mix(m_u).to(device) # u_i = g_i(A_i o z)
            z_given_dag = ut.conditional_sample_gaussian(f_z1, m_zv*lambdav)
        
        decoded_bernoulli_logits, x1, x2, x3, x4 = self.dec.decode_sep(z_given_dag.reshape([z_given_dag.size()[0], self.z_dim]), label.to(device))
        
        rec = ut.log_bernoulli_with_logits(x, decoded_bernoulli_logits.reshape(x.size()))
        rec = -torch.mean(rec)

        p_m, p_v = torch.zeros(q_m.size()), torch.ones(q_m.size())
        cp_m, cp_v = ut.condition_prior(self.scale, label, self.z2_dim) # cp_m: normalized u (0~1)
        cp_v = torch.ones([q_m.size()[0],self.z1_dim,self.z2_dim]).to(device)
        # cp_z = ut.conditional_sample_gaussian(cp_m.to(device), cp_v.to(device))
        kl = torch.zeros(1).to(device)
        # D_KL(q(e|x,u) || p(e)) = sum of 0.5 * ((e - 0)^2 - 1)
        kl = alpha * ut.kl_normal(q_m.view(-1,self.z_dim).to(device), q_v.view(-1,self.z_dim).to(device), p_m.view(-1,self.z_dim).to(device), p_v.view(-1,self.z_dim).to(device))
        
        # D_KL(q(z|x,u) || p(z|u)) = sum of 0.5 * ((z - u)^2 - 1)
        for i in range(self.z1_dim):
            kl = kl + beta * ut.kl_normal(decode_m[:,i,:].to(device), cp_v[:,i,:].to(device), cp_m[:,i,:].to(device), cp_v[:,i,:].to(device))
        kl = torch.mean(kl) # average on batch
        
        # mask_kl2 = torch.zeros(1).to(device)
        # KL = sum of 0.5 * ((f_z1 - cp_m)^2 - 1)
        mask_kl = torch.zeros(1).to(device)
        for i in range(4):
#             mask_kl = mask_kl + 1*ut.kl_normal(f_z1[:,i,:].to(device), cp_v[:,i,:].to(device), cp_m[:,i,:].to(device), cp_v[:,i,:].to(device))
            mask_kl = mask_kl + 1*ut.kl_normal(f_z[:,i,:].to(device), cp_v[:,i,:].to(device), decode_m[:,i,:].to(device), decode_v[:,i,:].to(device))
        
        u_loss = torch.nn.MSELoss()
        l_u = u_loss(g_u, label.float().to(device))
        mask_l = torch.mean(mask_kl) + l_u
        nelbo = rec + kl + mask_l

        return nelbo, kl, rec, decoded_bernoulli_logits.reshape(x.size()), torch.mean(mask_kl), l_u, z_given_dag

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries


    








