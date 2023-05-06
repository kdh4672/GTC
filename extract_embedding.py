import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter

# import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import clip
from Metrics import base_kmeans_model_evaluation, kmeans_with_init, base_agg_model_evaluation, extract_embedding
from networks import CustomCLIP, load_clip_to_cpu
from lr_scheduler import ConstantWarmupScheduler
from torch.autograd import Variable
import argparse
from sklearn.cluster import KMeans, AgglomerativeClustering
import pandas as pd
from dataset import custom_dataset
from torch.utils.data import Dataset, DataLoader
parser = argparse.ArgumentParser()
batch_size = 10

parser.add_argument("--clip_backbone", type=str,
                    default="ViT-B/16", choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'])
parser.add_argument("--dataset", type=str,
                    default="cifar10", choices=['cifar10', 'stl10', 'imagenet10','imagenet_dog','all_age_faces','stanford40','cropped_stanford40','bu101','cifar10_10000','cifar10_5000','cifar10_2500','stl10_10000','stl10_5000','stl10_2500','mit'])
parser.add_argument("--is_test_dataset", type=str,
                    default='False', choices=['True', 'False'])
parser.add_argument("--cluster", type=str,
                    default='kmean', choices=['kmean', 'agg'])
args = parser.parse_args()

backbone_name = args.clip_backbone
dataset_name = args.dataset
if args.is_test_dataset == 'True': is_test_dataset = True
else: is_test_dataset = False
device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
clip_model, preprocess = load_clip_to_cpu(backbone_name)
n_ctx=16

if dataset_name == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=preprocess) 

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=preprocess)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=False)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False)
    concatset = torch.utils.data.ConcatDataset([trainset, testset])
    concatloader = torch.utils.data.DataLoader(concatset, batch_size=batch_size,
                                            shuffle=False)


if dataset_name == 'stl10':
    trainset = torchvision.datasets.STL10(
        root='./data', split='train', download=True,
        transform=preprocess)
    testset = torchvision.datasets.STL10(
        root='./data', split='test', download=True,
        transform=preprocess)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False)
    concatset = torch.utils.data.ConcatDataset([trainset, testset])
    concatloader = torch.utils.data.DataLoader(concatset, batch_size=batch_size,
                                            shuffle=False)
    
if dataset_name == 'imagenet10':
    is_test_dataset = True
    testset = custom_dataset(root_dir='data/imagenet10', transform=preprocess)
    testloader = DataLoader(
        dataset=testset, 
        batch_size=batch_size,
        shuffle=False)
    
if dataset_name == 'imagenet_dog':
    is_test_dataset = True
    testset = custom_dataset(root_dir='data/imagenet_dog', transform=preprocess)
    testloader = DataLoader(
        dataset=testset, 
        batch_size=batch_size,
        shuffle=False)

if dataset_name == 'all_age_faces':
    is_test_dataset = True
    testset = custom_dataset(root_dir='/database/daehyeon/all_age_faces/aligned_faces/', transform=preprocess)
    testloader = DataLoader(
        dataset=testset, 
        batch_size=batch_size,
        shuffle=False)
    
if dataset_name == 'stanford40':
    is_test_dataset = True
    testset = custom_dataset(root_dir='data/stanford40', transform=preprocess)
    testloader = DataLoader(
        dataset=testset, 
        batch_size=batch_size,
        shuffle=False)

if dataset_name == 'cropped_stanford40':
    is_test_dataset = True
    testset = custom_dataset(root_dir='/database/daehyeon/Stanford40/cropped_stanford40', transform=preprocess)
    testloader = DataLoader(
        dataset=testset, 
        batch_size=batch_size,
        shuffle=False)
    
if dataset_name == 'bu101':
    is_test_dataset = True
    testset = custom_dataset(root_dir='/database/daehyeon/bu101', transform=preprocess)
    testloader = DataLoader(
        dataset=testset, 
        batch_size=batch_size,
        shuffle=False)
    
if dataset_name == 'cifar10_10000':
    is_test_dataset = True
    testset = custom_dataset(root_dir='data/cifar10_10000', transform=preprocess)
    testloader = DataLoader(
        dataset=testset, 
        batch_size=batch_size,
        shuffle=False)

if dataset_name == 'cifar10_5000':
    is_test_dataset = True
    testset = custom_dataset(root_dir='data/cifar10_5000', transform=preprocess)
    testloader = DataLoader(
        dataset=testset, 
        batch_size=batch_size,
        shuffle=False)

if dataset_name == 'cifar10_2500':
    is_test_dataset = True
    testset = custom_dataset(root_dir='data/cifar10_2500', transform=preprocess)
    testloader = DataLoader(
        dataset=testset, 
        batch_size=batch_size,
        shuffle=False)

if dataset_name == 'stl10_10000':
    is_test_dataset = True
    testset = custom_dataset(root_dir='data/stl10_10000', transform=preprocess)
    testloader = DataLoader(
        dataset=testset, 
        batch_size=batch_size,
        shuffle=False)

if dataset_name == 'stl10_5000':
    is_test_dataset = True
    testset = custom_dataset(root_dir='data/stl10_5000', transform=preprocess)
    testloader = DataLoader(
        dataset=testset, 
        batch_size=batch_size,
        shuffle=False)

if dataset_name == 'stl10_2500':
    is_test_dataset = True
    testset = custom_dataset(root_dir='data/stl10_2500', transform=preprocess)
    testloader = DataLoader(
        dataset=testset, 
        batch_size=batch_size,
        shuffle=False)

if dataset_name == 'mit':
    is_test_dataset = True
    testset = custom_dataset(root_dir='data/mit', transform=preprocess)
    testloader = DataLoader(
        dataset=testset, 
        batch_size=batch_size,
        shuffle=False)


num_classes = len(testset.classes)

if is_test_dataset:
    dataloader = testloader
    cluster_name = 'test_cluster'
else:
    dataloader = concatloader
    cluster_name = 'concat_cluster'
print('------------------')
print(backbone_name)
print("num_classes:",num_classes)
model = CustomCLIP(clip_model, num_classes, n_ctx=n_ctx)

for name, param in model.named_parameters():
    if "prompt_learner" not in name:
        param.requires_grad_(False)

model.to(device)
print('done')
    

with torch.no_grad():
    datas = extract_embedding(model, dataloader, num_classes)
    npy_file_path = '_'.join([f'npy_folder/{dataset_name}', backbone_name.replace('/', '_'), f'{cluster_name}_embedding.npy'])

np.save(npy_file_path , datas)