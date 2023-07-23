#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from codebase import utils as ut
import argparse
from pprint import pprint
cuda = torch.cuda.is_available()
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
import numpy as np
import math
import time
from torch.utils import data
from utils import  get_batch_unin_dataset_withlabel

import matplotlib.pyplot as plt
import random
import torch.utils.data as Data
from PIL import Image
import os
import numpy as np
from torchvision import transforms
from codebase import utils as ut
import codebase.models.mask_vae_pendulum as sup_dag
import argparse
from pprint import pprint
cuda = torch.cuda.is_available()
from torchvision.utils import save_image, make_grid

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epoch_max',   type=int, default=101,    help="Number of training epochs")
parser.add_argument('--iter_save',   type=int, default=5, help="Save model every n epochs")
parser.add_argument('--run',         type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',       type=int, default=1,     help="Flag for training")
parser.add_argument('--color',       type=int, default=False,     help="Flag for color")
parser.add_argument('--toy',       type=str, default="pendulum_mask",     help="Flag for toy")
parser.add_argument('--dag',       type=str, default="sup_dag",     help="Supervised")

args = parser.parse_args()
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

layout = [
    ('model={:s}',  'causalvae'),
    ('run={:04d}', args.run),
    ('color=True', args.color),
    ('toy={:s}', str(args.toy)),
]

model_name = '_'.join([t.format(v) for (t, v) in layout])
if args.dag == "sup_dag":
    lvae = sup_dag.CausalVAE(name=model_name, z_dim=16, inference = True).to(device)
    ut.load_model_by_name(lvae, 70)

if not os.path.exists(f'./figs_test_vae_pendulum/run_{args.run:04d}'): 
    os.makedirs(f'./figs_test_vae_pendulum/run_{args.run:04d}')
means = torch.zeros(2,3,4).to(device)
z_mask = torch.zeros(2,3,4).to(device)

dataset_dir = './causal_data/pendulum'
train_dataset = get_batch_unin_dataset_withlabel(dataset_dir, 100, dataset="test")

num_img = 10 # Only save 10 images
count = 0
sample = False
print('DAG:{}'.format(lvae.dag.A))
for u,l in train_dataset:
    for i in range(4):
        recon_images = torch.zeros((num_img*2, u.shape[1], u.shape[2], u.shape[3]))
        recon_images[:num_img] = u[:num_img]
        # for j in range(9):
        L, kl, rec, reconstructed_image, _, _, _ = lvae.negative_elbo_bound(u.to(device),l.to(device), i, sample = sample, adj=0)
        recon_images[num_img:(num_img*2)] = torch.sigmoid(reconstructed_image[:num_img])
        
        grid = make_grid(recon_images, nrow=5, normalize=True)
        save_image(grid, f"./figs_test_vae_pendulum/run_{args.run:04d}/concept_{i}.png")
    break
        # save_image(reconstructed_image[0], 'figs_test_vae_pendulum/reconstructed_image_{}_{}.png'.format(i, count),  range = (0,1)) 
    # save_image(u[0], './figs_test_vae_pendulum/true_{}.png'.format(count))
#     count += 1
#     if count == 10:
#         break

  
    
  
  
  
  
  
  
  
  
