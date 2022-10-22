import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
from torchnet.meter import AUCMeter
import sys

def get_data(dataset_name,preprocess,batch_size,is_test_dataset):
    if dataset_name == 'cifar10':
        if is_test_dataset:
            fixed_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                        download=True, transform=preprocess)
            fixed_dataloader = torch.utils.data.DataLoader(fixed_dataset, batch_size=batch_size,
                                                        shuffle=False)
            dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=preprocess)
        else:
            fixed_train_c_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=preprocess)
            fixed_test_c_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True, transform=preprocess)
            fixed_dataset = torch.utils.data.ConcatDataset([fixed_train_c_set, fixed_test_c_set])

            fixed_dataloader = torch.utils.data.DataLoader(fixed_dataset, batch_size=batch_size,
                                                        shuffle=False)

            train_c_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=preprocess)
            test_c_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=preprocess)
            dataset = (train_c_set,test_c_set)
            
    elif dataset_name == 'stl10':
        if is_test_dataset:
            fixed_dataset = torchvision.datasets.STL10(root='./data', split='test',
                                                        download=True, transform=preprocess)
            fixed_dataloader = torch.utils.data.DataLoader(fixed_dataset, batch_size=batch_size,
                                                        shuffle=False)
            dataset = torchvision.datasets.STL10(root='./data', split='test',
                                                download=True, transform=preprocess)
        else:
            fixed_train_c_set = torchvision.datasets.STL10(root='./data', split='train',
                                                    download=True, transform=preprocess)
            fixed_test_c_set = torchvision.datasets.STL10(root='./data', split='test',
                                                    download=True, transform=preprocess)
            fixed_dataset = torch.utils.data.ConcatDataset([fixed_train_c_set, fixed_test_c_set])

            fixed_dataloader = torch.utils.data.DataLoader(fixed_dataset, batch_size=batch_size,
                                                        shuffle=False)

            train_c_set = torchvision.datasets.STL10(root='./data', split='train',
                                                download=True, transform=preprocess)
            test_c_set = torchvision.datasets.STL10(root='./data', split='test',
                                                download=True, transform=preprocess)
            dataset = (train_c_set,test_c_set)
    elif dataset_name == 'imagenet10':
        is_test_dataset = True
        fixed_dataset = custom_dataset(root_dir='data/imagenet10', transform=preprocess)
        fixed_dataloader = DataLoader(
            dataset=fixed_dataset, 
            batch_size=batch_size,
            shuffle=False)
        dataset = custom_dataset(root_dir='data/imagenet10', transform=preprocess)
        
    elif dataset_name == 'imagenet_dog':
        is_test_dataset = True
        fixed_dataset = custom_dataset(root_dir='data/imagenet_dog', transform=preprocess)
        fixed_dataloader = DataLoader(
            dataset=fixed_dataset, 
            batch_size=batch_size,
            shuffle=False)
        dataset = custom_dataset(root_dir='data/imagenet_dog', transform=preprocess)
        
    elif dataset_name == 'all_age_faces':
        is_test_dataset = True
        fixed_dataset = custom_dataset(root_dir='/database/daehyeon/all_age_faces/aligned_faces/', transform=preprocess)
        fixed_dataloader = DataLoader(
            dataset=fixed_dataset, 
            batch_size=batch_size,
            shuffle=False)
        dataset = custom_dataset(root_dir='/database/daehyeon/all_age_faces/aligned_faces/', transform=preprocess)

    elif dataset_name == 'stanford40':
        is_test_dataset = True
        fixed_dataset = custom_dataset(root_dir='/database/daehyeon/Stanford40/JPEGImages', transform=preprocess)
        fixed_dataloader = DataLoader(
            dataset=fixed_dataset, 
            batch_size=batch_size,
            shuffle=False)
        dataset = custom_dataset(root_dir='/database/daehyeon/Stanford40/JPEGImages', transform=preprocess)
        
    elif dataset_name == 'cropped_stanford40':
        is_test_dataset = True
        fixed_dataset = custom_dataset(root_dir='/database/daehyeon/Stanford40/cropped_stanford40', transform=preprocess)
        fixed_dataloader = DataLoader(
            dataset=fixed_dataset, 
            batch_size=batch_size,
            shuffle=False)
        dataset = custom_dataset(root_dir='/database/daehyeon/Stanford40/cropped_stanford40', transform=preprocess)
        
    elif dataset_name == 'bu101':
        is_test_dataset = True
        fixed_dataset = custom_dataset(root_dir='/database/daehyeon/bu101', transform=preprocess)
        fixed_dataloader = DataLoader(
            dataset=fixed_dataset, 
            batch_size=batch_size,
            shuffle=False)
        dataset = custom_dataset(root_dir='/database/daehyeon/bu101', transform=preprocess)
        
    elif dataset_name == 'cifar10_10000':
        is_test_dataset = True
        fixed_dataset = custom_dataset(root_dir='./data/cifar10_10000', transform=preprocess)
        fixed_dataloader = DataLoader(
            dataset=fixed_dataset, 
            batch_size=batch_size,
            shuffle=False)
        dataset = custom_dataset(root_dir='./data/cifar10_10000', transform=preprocess)
        
    elif dataset_name == 'cifar10_5000':
        is_test_dataset = True
        fixed_dataset = custom_dataset(root_dir='./data/cifar10_5000', transform=preprocess)
        fixed_dataloader = DataLoader(
            dataset=fixed_dataset, 
            batch_size=batch_size,
            shuffle=False)
        dataset = custom_dataset(root_dir='./data/cifar10_5000', transform=preprocess)


    elif dataset_name == 'cifar10_2500':
        is_test_dataset = True
        fixed_dataset = custom_dataset(root_dir='./data/cifar10_2500', transform=preprocess)
        fixed_dataloader = DataLoader(
            dataset=fixed_dataset, 
            batch_size=batch_size,
            shuffle=False)
        dataset = custom_dataset(root_dir='./data/cifar10_2500', transform=preprocess)

    elif dataset_name == 'stl10_10000':
        is_test_dataset = True
        fixed_dataset = custom_dataset(root_dir='./data/stl10_10000', transform=preprocess)
        fixed_dataloader = DataLoader(
            dataset=fixed_dataset, 
            batch_size=batch_size,
            shuffle=False)
        dataset = custom_dataset(root_dir='./data/stl10_10000', transform=preprocess)
        
    elif dataset_name == 'stl10_5000':
        is_test_dataset = True
        fixed_dataset = custom_dataset(root_dir='./data/stl10_5000', transform=preprocess)
        fixed_dataloader = DataLoader(
            dataset=fixed_dataset, 
            batch_size=batch_size,
            shuffle=False)
        dataset = custom_dataset(root_dir='./data/stl10_5000', transform=preprocess)


    elif dataset_name == 'stl10_2500':
        is_test_dataset = True
        fixed_dataset = custom_dataset(root_dir='./data/stl10_2500', transform=preprocess)
        fixed_dataloader = DataLoader(
            dataset=fixed_dataset, 
            batch_size=batch_size,
            shuffle=False)
        dataset = custom_dataset(root_dir='./data/stl10_2500', transform=preprocess)

    elif dataset_name == 'mit':
        is_test_dataset = True
        fixed_dataset = custom_dataset(root_dir='/database/daehyeon/mit', transform=preprocess)
        fixed_dataloader = DataLoader(
            dataset=fixed_dataset, 
            batch_size=batch_size,
            shuffle=False)
        dataset = custom_dataset(root_dir='/database/daehyeon/mit', transform=preprocess)


    if is_test_dataset:
        num_classes = len(dataset.classes)
    else:
        num_classes = len(dataset[0].classes)
    return num_classes,dataset,fixed_dataloader        





def get_cluster(clip_backbone,dataset_name,is_test_dataset,cluster_alg='kmean'):
        if is_test_dataset: 
            if cluster_alg == 'agg':
                cluster = np.load('_'.join(['npy_folder/',dataset_name,clip_backbone.replace('/','_'),'agg','test_cluster.npy']))
                print('cluster:',dataset_name,clip_backbone.replace('/','_'),'agg','test_cluster.npy')
            else:
                cluster = np.load('_'.join(['npy_folder/',dataset_name,clip_backbone.replace('/','_'),'test_cluster.npy']))
                print('cluster:',dataset_name,clip_backbone.replace('/','_'),'test_cluster.npy')
        else: 
            if cluster_alg == 'agg':
                cluster = np.load('_'.join(['npy_folder/',dataset_name,clip_backbone.replace('/','_'),'agg','concat_cluster.npy']))
                print('cluster:',dataset_name,clip_backbone.replace('/','_'),'agg','concat_cluster.npy')
            else:
                cluster = np.load('_'.join(['npy_folder/',dataset_name,clip_backbone.replace('/','_'),'concat_cluster.npy']))
                print('cluster:',dataset_name,clip_backbone.replace('/','_'),'concat_cluster.npy')
        return cluster
    
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class custom_dataset(Dataset): 
    def __init__(self,  root_dir='../data/imagenet10', transform=None): 
        self.transform = transform
        data = np.load(os.path.join(root_dir,'data.npy'))
        self.targets = np.load(os.path.join(root_dir,'label.npy'))
        self.classes = list(set(self.targets))
        # b,h,w,c = data.shape
        # data = data.reshape((b, c, h, w))
        # self.data = data.transpose((0, 2, 3, 1))
        self.data = data
                
    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img) 
        return img,label
    def __len__(self):
        return len(self.data)