from random import shuffle
from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import copy
import time

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])



data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

data_dir =  'data/hymenoptera_data'
image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])            for x in ['train', 'val']}
data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 4, shuffle=True, num_workers = 0)               for x in ['train', 'val'] }
dataset_sizes = {x:len(image_datasets[x])   for x in ['train', 'val']}
class_namess = image_datasets['train'].classes
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def model_train(model, optimizer, criterion, sceduler, num_epoch):
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    since = time.time()
    for epoch in range(num_epoch):
        print('Epoch: {}/{}'.format(epoch, num_epoch-1))
        print("--"*10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0
            running_corrects = 0
            for input, labels in data_loaders[phase]:
                input = input.to(device).float()
                labels = labels.to(device).type(torch.int64)
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(input)
                    loss = criterion(output, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    _, pred = torch.max(output, 1)
                    corcts = torch.sum(pred == labels)

                running_loss+= loss.item()
                running_corrects+=corcts.item()
            if phase == 'train':
                sceduler.step()    
            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = (running_corrects)/dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



            

##Non-freze
model = torchvision.models.resnet18(pretrained = True)
n_ftrs = model.fc.in_features
model.fc = nn.Linear(n_ftrs, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
skedulr = lr_scheduler.StepLR(optimizer, 7, 0.1)
Do_train = model_train(model, optimizer, criterion, skedulr,23)


##Freze

model = torchvision.models.resnet18(pretrained = True)
for param in model.parameters():
    param.requires_grad = False

n_ftrs = model.fc.in_features
model.fc = nn.Linear(n_ftrs, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
skedulr = lr_scheduler.StepLR(optimizer, 7, 0.1)
Do_train = model_train(model, optimizer, criterion, skedulr,23)



