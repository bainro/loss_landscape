import os
import torch
import cifar10.model_loader
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, TensorDataset
import torch.nn.functional as F
from torch import nn, optim
import torchvision
import numpy as np
import sys
import torch


# pretty much the same as lenet5. +dropout & batchnorm
# class FashionCNN(nn.Module):
#     def __init__(self):
#         super(FashionCNN, self).__init__()
        
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
        
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
        
#         self.fc1 = nn.Linear(in_features=3136, out_features=600)
#         #self.drop = nn.Dropout2d(0.25)
#         self.fc2 = nn.Linear(in_features=600, out_features=120)
#         self.fc3 = nn.Linear(in_features=120, out_features=100)
        
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = out.view(out.size(0), -1)
#         #print(out.size())
#         out = self.fc1(out)
#         #out = self.drop(out)
#         out = self.fc2(out)
#         out = self.fc3(out)
        
#         return out

class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out

def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)
    elif dataset == 'mnist':
        net = models.mobilenet_v2(num_classes=10, width_mult=0.25)
        net.load_state_dict(torch.load(model_file))
        #net.load_state_dict(state['state_dict'])
    elif dataset == 'cifar100' or dataset == "fmnist":
        print("loading lenet5, you might not want this...")
        net = FashionCNN()
        try:
            net.load_state_dict(torch.load(model_file))
        except:
            # multi-gpu setup saves trained models differently: https://tinyurl.com/y35lj5bo
            # it might also complain about gpu buffers on the wrong device, just remove the
            # line of pytorch's set_device[] and it'll shutup

            # First load the model parameter dict file
            state_dict = torch.load(model_file)
            #print(state_dict);exit()
            from collections import OrderedDict
            # Initialize an empty dict
            new_state_dict = OrderedDict()
            # Modify key, no module field is required, if it is, you need to modify it to module.features
            for k, v in state_dict.items():
                if 'module' not in k:
                    k = 'module.'+k
                else:
                    k = k.replace('features.module.', 'module.features.')
                new_state_dict[k]=v
            # Load the modified new parameter dict file
            net = nn.DataParallel(net, device_ids=[0, 1])
            net.load_state_dict(new_state_dict)

    return net
