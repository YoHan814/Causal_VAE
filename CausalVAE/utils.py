#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils import data
import torch.utils.data as Data
from torchvision import transforms

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")


def matrix_poly(matrix, d):
    x = torch.eye(d).to(device)+ torch.div(matrix.to(device), d).to(device)
    return torch.matrix_power(x, d)
    
def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

class dataload(data.Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        pil_img = Image.open(img_path)
        array = np.asarray(pil_img)
        data = torch.from_numpy(array)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img).reshape(96,96,4)
            data = torch.from_numpy(pil_img)
        return data

    def __len__(self):
        return len(self.imgs)

class CelebA(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, label_type=0):
        partition_file_path = "list_eval_partition.csv"
        label_path = "list_attr_celeba.csv"
        self.partition_file = pd.read_csv('%s/%s' % (data_dir, partition_file_path))
        self.label_file = pd.read_csv('%s/%s' % (data_dir, label_path))
        self.data_dir = data_dir
        self.split = split # 0: train / 1: validation / 2: test
        self.transform = transforms.Compose([
            transforms.CenterCrop((128, 128)),
            # transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # output: [-1, 1]
        ])
        self.partition_file_sub = self.partition_file[self.partition_file["partition"].isin(self.split)]
        
        if label_type == 0:
            self.label_cols = ["Male", "Smiling", "Narrow_Eyes", "Mouth_Slightly_Open"]
        else:
            self.label_cols = ["Young", "Male", "Bald", "No_Beard"]
    
    def __len__(self):
        return len(self.partition_file_sub)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, 'img_align_celeba', self.partition_file_sub.iloc[idx, 0])
        image = Image.open(img_name)
        label = torch.from_numpy((self.label_file[self.label_cols].iloc[idx,:].to_numpy() + 1.)/2.).float() # 0 or 1
        if self.transform:
            image = self.transform(image)
        return image, label

class dataload_withlabel(data.Dataset):
    def __init__(self, root, dataset="train"):
        root = root + "/" + dataset
       
        imgs = os.listdir(root)

        self.dataset = dataset
        
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.imglabel = [list(map(int,k[:-4].split("_")[1:]))  for k in imgs]
        #print(self.imglabel)
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        #print(idx)
        img_path = self.imgs[idx]
        
        label = torch.from_numpy(np.asarray(self.imglabel[idx]))
        #print(len(label))
        pil_img = Image.open(img_path)
        array = np.asarray(pil_img)
        array1 = np.asarray(label)
        label = torch.from_numpy(array1)
        data = torch.from_numpy(array)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img).reshape(96,96,4)
            data = torch.from_numpy(pil_img)
        
        return data, label.float()

    def __len__(self):
        return len(self.imgs)  

def get_batch_unin_dataset_withlabel(dataset_dir, batch_size, dataset="train"):
  
    dataset = dataload_withlabel(dataset_dir, dataset)
    dataset = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataset
