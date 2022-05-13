import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchbearer import Trial
import torchbearer
from device import DEVICE
from os.path import exists
import os
import json
from torchvision.models import resnet18
from resnet import myResNet
from utils import extractRangeData
print(DEVICE)

#parameters
params = {
    'model': {
        'hyper_parameters': {
            'unlabeled_classes': 5,
            'labeled_classes': 5
        }
    },
    'dataset': {
        'root': './data/cifar10/',
        'batch_size': 128,
        'num_workers': 2
    },
    'optimizer': {
        'lr': 0.01,
        'momentum':0.9,
        'weight_decay': 1e-4,
    },
    'scheduler': {
        'milestones': [50, 100, 150, 200],
        'gamma': 0.1,
    },
    'epochs': 200,
    'dir': './supervised_learning',
    'model_dir': './supervised_learning/resnet_rotnet.pth',
    'selfsupervised_model_dir': './selfsupervised_learning/rotnet_cifar10.pth',
    'trial_dir': './supervised_learning/trial.txt',
}

def main():
    # resnet model
    model = myResNet(resnet18()).to(DEVICE)
    # load pre-trained model
    state_dict = torch.load(params['selfsupervised_model_dir'])
    del state_dict['head1.weight']
    del state_dict['head1.bias']
    model.load_state_dict(state_dict, strict = False)

    # set the last layer and linear to be learn parameter
    for name, param in model.named_parameters(): 
        if 'head' not in name and 'layer4' not in name:
            param.requires_grad = False
    
    # dataset and dataloader
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

    labeled_train_dataset = CIFAR10(
        root = params['dataset']['root'],
        train = True,
        transform = transform_train,
        download=True
    )

    labeled_eval_dataset = CIFAR10(
        root = params['dataset']['root'],
        train = False,
        transform = transform_eval,
        download=True
    )
    
    labeled_train_dataset = extractRangeData(labeled_train_dataset, range(params['model']['hyper_parameters']['labeled_classes']))
    labeled_eval_dataset = extractRangeData(labeled_eval_dataset, range(params['model']['hyper_parameters']['labeled_classes']))

    labeled_train_loader = DataLoader(
        dataset = labeled_train_dataset,
        batch_size = params['dataset']['batch_size'],
        shuffle = True,
        num_workers = params['dataset']['num_workers']
    )

    labeled_eval_loader = DataLoader(
        dataset = labeled_eval_dataset,
        batch_size = params['dataset']['batch_size'],
        shuffle = False,
        num_workers = params['dataset']['num_workers']
    )
    
    # if do not have model file, train this model
    if not exists(params['model_dir']):
        
        optimizer = optim.SGD(model.parameters(), lr = params['optimizer']['lr'], momentum = params['optimizer']['momentum'], weight_decay = params['optimizer']['weight_decay'], nesterov=True)
        criterion = nn.CrossEntropyLoss()
        
        trial = Trial(model, optimizer, criterion, metrics=['loss', 'accuracy']).to(DEVICE)

        trial.with_generators(train_generator = labeled_train_loader, test_generator = labeled_eval_loader)
        trial.run(epochs = params['epochs'], verbose = 1)
        
        result = trial.state_dict()
    
        with open(params['trial_dir'], 'w') as file:
            _dict = {
                'history': result['history']
            }
            file.write(json.dumps(_dict))
        torch.save(trial.state_dict()['model'], params['model_dir'])

    else:
        model.load_state_dict(torch.load(params['model_dir']))
        model.eval()
        trial = Trial(model, metrics=['loss', 'accuracy']).to(DEVICE)
        trial.with_test_generator(generator = labeled_eval_loader)
   
    test_result = trial.evaluate(data_key = torchbearer.TEST_DATA)
    print(test_result)
    
    
if __name__ == '__main__':
    if not exists(params['dir']):
        os.makedirs(params['dir'])
    main()