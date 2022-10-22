from operator import is_
import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np

import clip

from Metrics import base_kmeans_model_evaluation, kmeans_with_init, cosine_kmeans_with_init
from networks import CustomCLIP, load_clip_to_cpu, save_model
from dataset import get_data, get_cluster
from lr_scheduler import ConstantWarmupScheduler

import argparse

# parser
parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str,
                    default='text_guided_record/real_fixed')
parser.add_argument("--clip_backbone", type=str,
                    default="ViT-L/14", choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'])
parser.add_argument("--batch_size", type=int,
                    default=256)
parser.add_argument("--total_epoch", type=int,
                    default=1000)
parser.add_argument("--lr", type=float,
                    default=3e-4)
parser.add_argument("--repeat", type=int,
                    default=4)
parser.add_argument("--is_test_dataset", type=str,
                    default='True', choices=['True', 'False'])
parser.add_argument("--cosine_sim", type=str,
                    default='False', choices=['True', 'False'])
parser.add_argument("--n_ctx", type=int,
                    default=16)
parser.add_argument("--dataset", type=str,
                    default='cifar10', choices=['cifar10', 'stl10', 'imagenet10', 'imagenet_dog', 'all_age_faces', 'stanford40', 'bu101','cifar10_10000','cifar10_5000','cifar10_2500','stl10_10000','stl10_5000','stl10_2500','mit'])
parser.add_argument("--loss", type=str,
                    default='CE', choices=['CE', 'MS'])
parser.add_argument("--num_centroids", type=int,
                    default=1)
parser.add_argument("--margin", type=float,
                    default=0)
parser.add_argument("--cluster_name", type=str,
                    default=None)
parser.add_argument("--cluster", type=str,
                    default='kmean')
parser.add_argument("--reload", type=str,
                    default=None)
args = parser.parse_args()

log_dir = args.log_dir
clip_backbone = args.clip_backbone
batch_size = args.batch_size
total_epoch = args.total_epoch
lr = args.lr
dataset_name = args.dataset
num_centroids = args.num_centroids
margin = args.margin
if args.reload == None:
    reload = False
else:
    reload = int(args.reload)
if args.cluster_name == None:
    cluster_name = False
else:
    cluster_name = args.cluster_name
if args.is_test_dataset == 'True':
    is_test_dataset = True
else:
    is_test_dataset = False
if args.cosine_sim == 'True':
    cosine_sim = True
else:
    cosine_sim = False

total_dpoch = args.repeat
n_ctx = args.n_ctx
# parser


writer = SummaryWriter(log_dir)

# parser
dict_ = vars(args)
print('--------------------------------')
for key, value in dict_.items():
    print(f"{key} : {value}")
print('--------------------------------')
print(clip.available_models())
device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

backbone_name = clip_backbone
clip_model, preprocess = load_clip_to_cpu(backbone_name)

num_classes, dataset, fixed_dataloader = get_data(
    dataset_name, preprocess, batch_size, is_test_dataset)
model = CustomCLIP(clip_model, num_classes, n_ctx=n_ctx,
                   num_centroids=num_centroids)
for name, param in model.named_parameters():
    if "prompt_learner" not in name:
        param.requires_grad_(False)
model.to(device)
print('done')

if args.loss == 'CE':
    criterion = F.cross_entropy
elif args.loss == 'MS':
    criterion = nn.MultiLabelSoftMarginLoss(reduction='sum')
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=5e-4,
    betas=(0.9, 0.999),
)

if reload:
    print('reload_model: from {} epoch'.format(reload))
    if cluster_name:
        model_fp = os.path.join('save', "{}_".format(cluster_name)+'{}cent'.format(str(num_centroids)),"checkpoint_{}.tar".format(reload))
    else:
        model_fp = os.path.join('save',"{}_".format(clip_backbone.replace('/', '_'))+"{}_".format(dataset_name)+'{}cent'.format(str(num_centroids)),"checkpoint_{}.tar".format(reload))
    checkpoint = torch.load(model_fp)
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = reload + total_dpoch
else:
    start_epoch = 0
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

# cluster = get_cluster(clip_backbone, dataset_name,
#                       is_test_dataset, cluster_alg=args.cluster)

if is_test_dataset:
    print('label_change_to_cluster_results')
    # dataset.targets = cluster.tolist()
else:
    train_c_set, test_c_set = dataset
    # train_c_set.targets = cluster[:len(train_c_set)].tolist()
    # test_c_set.targets = cluster[len(train_c_set):].tolist()
    dataset = torch.utils.data.ConcatDataset([train_c_set, test_c_set])

print("dataset_length:", len(dataset))
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True)

for epoch in range(start_epoch//total_dpoch, total_epoch):
    for dpoch in range(total_dpoch):
        score = 0
        total_loss = 0
        print('==================================================================')
        print('epoch:', dpoch + total_dpoch * epoch)
        for i, data in enumerate(dataloader, 0):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            output, centroids_similarity, centroids_margin_loss = model(inputs)
            if args.loss == 'CE':
                loss = criterion(output, targets)
            elif args.loss == 'MS':
                onehot_targets = torch.zeros(output.shape[0], num_classes).to(device).scatter_(
                    1, targets.view(-1, 1), 1)
                loss = criterion(output, onehot_targets)
            if margin:
                loss += centroids_margin_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = torch.argmax(output, dim=1)
            score += torch.sum(pred == targets)
            total_loss += loss.item()
            if i % 50 == 49:
                print('train_iter:', i, '/', len(dataloader))
        writer.add_scalar("Learning_rate", scheduler.get_lr()[
            0], dpoch + total_dpoch * epoch)
        writer.add_scalar("train_Loss", total_loss/len(dataloader.dataset),
                          dpoch + total_dpoch * epoch)
        writer.add_scalar("train_accuracy", score /
                          len(dataloader.dataset), dpoch + total_dpoch * epoch)
        writer.add_scalar("centroids_similarity", centroids_similarity.mean(
            dim=0).item(), dpoch + total_dpoch * epoch)

        print("train_accuracy: ", score/len(dataloader.dataset))
        print('centroids_similarity:', centroids_similarity)
        if (dpoch + total_dpoch * epoch) % 200 == 199:
            save_model(clip_backbone, dataset_name, num_centroids,
                       model, optimizer, dpoch + total_dpoch * epoch)
            print('save model at dpoch + total_dpoch * epoch')
            # save model

    with torch.no_grad():
        model.eval()

        prompts = model.prompt_learner()
        tokenized_prompts = model.tokenized_prompts
        # kdkd
        for i in range(len(prompts)):
            text_feature = model.text_encoder(prompts[i], tokenized_prompts)
            text_feature = text_feature / \
                text_feature.norm(dim=-1, keepdim=True)
            if i == 0:
                text_features = text_feature.expand(1, -1, -1)
            else:
                text_feature = text_feature.expand(1, -1, -1)
                text_features = torch.cat((text_features, text_feature), dim=0)
        text_centroids = text_features.mean(dim=0)
        # kdkd

        # no normalized
        if cosine_sim:
            knn = cosine_kmeans_with_init
        else:
            knn = kmeans_with_init
        new_label, acc, nmi, ari = knn(
            model, fixed_dataloader, num_classes, text_centroids)
        writer.add_scalar("val_acc", acc,
                          dpoch + total_dpoch * epoch)
        writer.add_scalar("val_nmi", nmi, dpoch + total_dpoch * epoch)
        writer.add_scalar("val_ari", ari, dpoch + total_dpoch * epoch)
        model.train()
