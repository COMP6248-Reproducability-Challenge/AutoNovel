import json
import torch
from torchvision.models import resnet18
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torchbearer import Trial
from torch.optim import lr_scheduler
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, Dataset
from device import DEVICE
from os.path import exists
import os
from resnet import myResNet
# from tqdm import tqdm
from utils import AverageMeter, myAccuracy
print(DEVICE)

# parameters
params = {
    'dataset': {
        'name': 'cifar10',
        'root': './data/cifar10/',
        'batch_size': 128,
        'num_workers': 4
    },
    'optimizer': {
        'lr': 0.01,
        'momentum':0.9,
        'weight_decay': 5e-4,
    },
    'scheduler': {
        'milestones': [20, 30], 
        'gamma': 0.1,
    },
    'epochs': 40,
    'dir': './selfsupervised_learning',
    'model_dir': './selfsupervised_learning/rotnet_cifar10.pth',
    'trial_dir': './selfsupervised_learning/trial.txt'
}

# rotate images
def rot(dataset):
    data = []
    targets = []
    for idx in range(len(dataset)):
        img0, _ = dataset[idx]
        rotated_imgs = [
            img0,
            transforms.functional.rotate(img0, 90).clone(),
            transforms.functional.rotate(img0, 180).clone(),
            transforms.functional.rotate(img0, 270).clone(),
        ]
        rotation_labels = torch.LongTensor([0, 1, 2, 3])
        data.append(torch.stack(rotated_imgs, dim = 0))
        targets.append(rotation_labels)
    print('rotate image done')
    return data, targets

# my rotated cifar10 image dataset
class MyDataset(Dataset):
    def __init__(self, dataset):
        self.data, self.targets = rot(dataset)

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        return img, target

# merge rotation and batch dimension to one dimension
def _collate_fun(batch):
    batch = default_collate(batch)
    assert(len(batch) == 2)
    batch_size, rotations, channels, height, width = batch[0].size()
    batch[0] = batch[0].view([batch_size * rotations, channels, height, width])
    batch[1] = batch[1].view([batch_size * rotations])
    return batch

# def train(epoch, model, dataloader, optimizer, exp_lr_scheduler, criterion):
#     loss_record = AverageMeter()
#     acc_record = AverageMeter()
#     model.train()
#     for batch_idx, (data, label) in enumerate(dataloader):
#         data, label = data.to(DEVICE), label.to(DEVICE)
#         output = model(data)
#         loss = criterion(output, label)
     
#         # measure accuracy and record loss
#         acc = myAccuracy(output, label)
#         acc_record.update(acc[0].item(), data.size(0))
#         loss_record.update(loss.item(), data.size(0))

#         # compute gradient and do optimizer step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     exp_lr_scheduler.step()
#     print('Train Epoch: {} Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(epoch, loss_record.avg, acc_record.avg))
    
#     return acc_record, loss_record

# def test(model, dataloader):
#     acc_record = AverageMeter()
#     model.eval()
#     for batch_idx, (data, label) in enumerate(dataloader):
#         data, label = data.to(DEVICE), label.to(DEVICE)
#         output = model(data)
     
#         # measure accuracy and record loss
#         acc = myAccuracy(output, label)
#         acc_record.update(acc[0].item(), data.size(0))

#     print('Test Acc: {:.4f}'.format(acc_record.avg))
#     return acc_record 

def main():
    # dataset and dataloader
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset_train = CIFAR10(
        root =  params['dataset']['root'], 
        train = True,
        download = True,
        transform = transform_train,
    )
    dataset_train = MyDataset(dataset_train)
    
    dataset_test = CIFAR10(
        root =  params['dataset']['root'], 
        train = False,
        download = True,
        transform = transform_test
    )
    dataset_test = MyDataset(dataset_test)
    
    dloader_train = DataLoader(
        dataset = dataset_train,
        batch_size = params['dataset']['batch_size'],
        num_workers = params['dataset']['num_workers'],
        shuffle = True,
        collate_fn = _collate_fun
    )

    dloader_test = DataLoader(
        dataset = dataset_test,
        batch_size = params['dataset']['batch_size'],
        num_workers = params['dataset']['num_workers'],
        shuffle = False,
        collate_fn = _collate_fun
    )

    # model
    model = myResNet(resnet18(), 4).to(DEVICE)
    # model.fc = nn.Linear(512, 4) # four rotations
    
    print('go into training')
    # optimizer = optim.SGD(model.parameters(), lr = params['optimizer']['lr'], momentum = params['optimizer']['momentum'], weight_decay = params['optimizer']['weight_decay'])
    # exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, params['scheduler']['milestones'], params['scheduler']['gamma'])

    # criterion = nn.CrossEntropyLoss()
    # best_acc = 0
    # best_model = None
    # acc_history = {
    #     'train': [],
    #     'test': []
    # }
    # for epoch in range(params['epochs'] + 1):
    #     print(epoch, ":  ")
    #     train_acc_record, _ = train(epoch, model, dloader_train, optimizer, exp_lr_scheduler, criterion)
    #     acc_history['train'].append(train_acc_record.avg)
    #     test_acc_record = test(model, dloader_test)
    #     acc_history['test'].append(test_acc_record.avg)

    #     is_best = test_acc_record.avg > best_acc
    #     best_acc = max(test_acc_record.avg, best_acc)
    #     if is_best:
    #         best_model = model
    
    # with open(params['trial_dir'], 'w') as file:
    #     file.write(json.dumps(acc_history))

    # torch.save(best_model.state_dict(), params['model_dir'])
    
    optimizer = optim.SGD(model.parameters(), lr = params['optimizer']['lr'], momentum = params['optimizer']['momentum'], weight_decay = params['optimizer']['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    trial = Trial(model, optimizer, criterion, metrics = ['loss', 'accuracy']).to(DEVICE)

    trial.with_generators(dloader_train, val_generator = dloader_test)
    trial.run(epochs = params['epochs'], verbose = 1)
    result = trial.state_dict()
    
    with open(params['trial_dir'], 'w') as file:
        _dict = {
            'history': result['history']
        }
        file.write(json.dumps(_dict))
    
    torch.save(result['model'], params['model_dir'])

if __name__ == '__main__':
    if not exists(params['dir']):
        os.makedirs(params['dir'])
    
    if not exists(params['model_dir']):
        main()
    