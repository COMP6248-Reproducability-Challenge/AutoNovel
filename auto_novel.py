import copy
from resnet import myResNet
from torchvision.models import resnet18
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils import extractRangeData, AverageMeter, sigmoid_rampup, PairEnum, TransformTwice, BCE, cluster_acc, RandomTranslateWithReflect
import numpy as np
import os 
from os.path import exists
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import json
from device import DEVICE
print(DEVICE)

#parameters
params = {
    'model': {
        'hyper_parameters': {
            'all_classes': 10,
            'unlabeled_classes': 5,
            'labeled_classes': 5,
        }
    },
    'dataset': {
        'root': './data/cifar10/',
        'batch_size': 128,
        'num_workers': 4
    },
    'optimizer': {
        'lr': 0.1,
        'momentum':0.9,
        'weight_decay': 5e-4,
    },
    'scheduler': {
        'milestones': [40, 80, 110, 140], 
        'gamma': 0.1,
        'step': 170
    },
    'coefficient': {
        'rampup_coefficient': 50.0,
        'rampup_length': 150
    },
    'epochs': 160,
    'dir': './auto_novel_2',
    'model_dir': './auto_novel_2/non_pretrained_auto_novel.pth',
    'selfsupervised_model_dir': './selfsupervised_learning/rotnet_cifar10.pth',
    'trial_dir': './auto_novel_2/non_pretrained_trial.txt',
    'supervised_learning': './supervised_learning/resnet_rotnet.pth',
    'topk': 5
}

def train(model, train_loader, labeled_eval_loader, unlabeled_eval_loader):
    optimizer = optim.SGD(model.parameters(), lr = params['optimizer']['lr'], momentum = params['optimizer']['momentum'], weight_decay = params['optimizer']['weight_decay'])
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, params['scheduler']['milestones'], params['scheduler']['gamma'])
    
    criterion1 = nn.CrossEntropyLoss() 
    criterion2 = BCE()
    history = {
        'train': [],
        'test': {
            'labeled': [],
            'unlabeled': []
        }
    }
    for epoch in range(params['epochs']):
        loss_record = AverageMeter()
        model.train()
        w = params['coefficient']['rampup_coefficient'] * sigmoid_rampup(epoch, params['coefficient']['rampup_length']) 
        for batch_idx, ((x, x_bar),  label) in enumerate(train_loader):
            x, x_bar, label = x.to(DEVICE), x_bar.to(DEVICE), label.to(DEVICE)
            output1, output2, feat = model(x)
            output1_bar, output2_bar, _ = model(x_bar)
            prob1, prob1_bar, prob2, prob2_bar=F.softmax(output1, dim=1),  F.softmax(output1_bar, dim=1), F.softmax(output2, dim=1), F.softmax(output2_bar, dim=1)

            mask_lb = label < params['model']['hyper_parameters']['labeled_classes']

            rank_feat = (feat[~mask_lb]).detach()
            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
            rank_idx1, rank_idx2= PairEnum(rank_idx)
            rank_idx1, rank_idx2=rank_idx1[:, :params['topk']], rank_idx2[:, :params['topk']]
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)

            rank_diff = rank_idx1 - rank_idx2
            rank_diff = torch.sum(torch.abs(rank_diff), dim = 1)
            target_ulb = torch.ones_like(rank_diff).float().to(DEVICE) 
            target_ulb[rank_diff > 0] = -1

            prob1_ulb, _ = PairEnum(prob2[~mask_lb]) 
            _, prob2_ulb = PairEnum(prob2_bar[~mask_lb]) 

            loss_ce = criterion1(output1[mask_lb], label[mask_lb])
            loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb)
            consistency_loss = F.mse_loss(prob1, prob1_bar) + F.mse_loss(prob2, prob2_bar)
            loss = loss_ce + loss_bce + w * consistency_loss
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        exp_lr_scheduler.step()
        print('Train Epoch: {} Avg Loss: {:.4f} lr: {}'.format(epoch, loss_record.avg, optimizer.param_groups[-1]['lr']))
        history['train'].append(loss_record.avg)
        print('test on labeled classes')
        acc1 = test(model, labeled_eval_loader, 'head1')
        history['test']['labeled'].append(acc1)
        print('test on unlabeled classes')
        acc2 = test(model, unlabeled_eval_loader, 'head2')
        history['test']['unlabeled'].append(acc2)
    
    with open(params['trial_dir'], 'w') as file:
        file.write(json.dumps(history))

def test(model, test_loader, head):
    model.eval()
    preds=np.array([])
    targets=np.array([])
    for batch_idx, (x, label) in enumerate(test_loader):
        x, label = x.to(DEVICE), label.to(DEVICE)
        output1, output2, _ = model(x)
        if head=='head1':
            output = output1
        else:
            output = output2
        _, pred = output.max(1)
        targets=np.append(targets, label.cpu().numpy())
        preds=np.append(preds, pred.cpu().numpy())
    acc = cluster_acc(targets.astype(int), preds.astype(int))
    print('Test acc {:.4f}'.format(acc))
    return acc

if __name__ == '__main__':
    if not exists(params['dir']):
        os.makedirs(params['dir'])
    # model
    model = myResNet(resnet18(), params['model']['hyper_parameters']['labeled_classes'], params['model']['hyper_parameters']['unlabeled_classes'], True).to(DEVICE)
    # state_dict = torch.load(params['supervised_learning'])
    # model.load_state_dict(state_dict, strict = False)
    # for name, param in model.named_parameters():
    #     if 'head' not in name and 'layer4' not in name:
    #         param.requires_grad = False

    print('Process data')
    # data set and dataloader
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_mix = transforms.Compose([
            RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR10(
        root = params['dataset']['root'],
        train = True,
        transform = transform_train,
        download = True
    )

    eval_dataset = CIFAR10(
        root = params['dataset']['root'],
        train = False,
        transform = transform_eval,
        download = True
    )

    labeled_train_dataset = extractRangeData(train_dataset, range(params['model']['hyper_parameters']['labeled_classes']))
    unlabeled_train_dataset = extractRangeData(train_dataset, range(params['model']['hyper_parameters']['labeled_classes'], params['model']['hyper_parameters']['all_classes']))

    labeled_eval_dataset = extractRangeData(eval_dataset, range(params['model']['hyper_parameters']['labeled_classes']))
    unlabeled_eval_dataset = extractRangeData(eval_dataset, range(params['model']['hyper_parameters']['labeled_classes'], params['model']['hyper_parameters']['all_classes']))

    # mix train loader
    train_dataset_2 = CIFAR10(
        root = params['dataset']['root'],
        train = True,
        transform = TransformTwice(transform_train),
        download = True
    )

    mix_labeled_train_eval_dataset = extractRangeData(train_dataset_2, range(params['model']['hyper_parameters']['labeled_classes']))
    unlabeled_train_dataset_2 = extractRangeData(train_dataset_2, range(params['model']['hyper_parameters']['labeled_classes'], params['model']['hyper_parameters']['all_classes']))
    mix_labeled_train_eval_dataset.targets = np.concatenate((mix_labeled_train_eval_dataset.targets,unlabeled_train_dataset_2.targets))
    mix_labeled_train_eval_dataset.data = np.concatenate((mix_labeled_train_eval_dataset.data,unlabeled_train_dataset_2.data),0)
    
    mix_label_train_eval_loader = DataLoader(
        dataset = mix_labeled_train_eval_dataset,
        batch_size = params['dataset']['batch_size'],
        shuffle = True,
        num_workers = params['dataset']['num_workers'],
    )

    # labeled train loader
    labeled_train_loader = DataLoader(
        dataset = labeled_train_dataset,
        batch_size = params['dataset']['batch_size'],
        shuffle = True,
        num_workers = params['dataset']['num_workers'],
    )

    # unlabeled train loader
    unlabeled_tarin_loader = DataLoader(
        dataset = unlabeled_train_dataset,
        batch_size = params['dataset']['batch_size'],
        shuffle = True,
        num_workers = params['dataset']['num_workers'],
    )

    # unlabeled eval loader
    unlabeled_eval_loader = DataLoader(
        dataset = unlabeled_eval_dataset,
        batch_size = params['dataset']['batch_size'],
        shuffle = False,
        num_workers = params['dataset']['num_workers'],
    )
    # labeled eval loader
    labeled_eval_loader = DataLoader(
        dataset = labeled_eval_dataset,
        batch_size = params['dataset']['batch_size'],
        shuffle = False,
        num_workers = params['dataset']['num_workers'],
    )

    # all eval dataset loader
    all_eval_loader_dataset = copy.deepcopy(labeled_eval_dataset)
    all_eval_loader_dataset.targets = np.concatenate((all_eval_loader_dataset.targets,unlabeled_eval_dataset.targets))
    all_eval_loader_dataset.data = np.concatenate((all_eval_loader_dataset.data,unlabeled_eval_dataset.data),0)
    all_eval_loader = DataLoader(
        dataset = all_eval_loader_dataset,
        batch_size = params['dataset']['batch_size'],
        shuffle = False,
        num_workers = params['dataset']['num_workers'],
    )

    
    if not exists(params['model_dir']):
        print('training')
        train(model, mix_label_train_eval_loader, labeled_eval_loader, unlabeled_tarin_loader)
        torch.save(model.state_dict(), params['model_dir'])
        print("model saved to {}.".format(params['model_dir']))