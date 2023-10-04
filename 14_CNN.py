from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 4
num_epochs = 3
learning_rate = 0.01


transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True , download=True, transform = transform)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False , download=True, transform = transform)

train_dataloadeer = DataLoader(train_dataset, batch_size = batch_size, shuffle= True)
test_dataloadeer = DataLoader(test_dataset, batch_size = batch_size, shuffle= False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print(train_dataset)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.max1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.lin1 = nn.Linear(400, 120)
        self.lin2 = nn.Linear(120, 84)
        self.lin3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.max1(nn.functional.relu(self.conv1(x)))
        x = self.max1(nn.functional.relu(self.conv2(x)))
        x  = x.view(-1, 400)
        x = self.lin3(self.lin2(self.lin1(x)))
        return x

model = ConvNet().to(device)    
criterion = nn.CrossEntropyLoss()
optimizer  =  torch.optim.SGD(model.parameters(), lr = learning_rate)
total_num_steps = len(train_dataloadeer)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloadeer):
        images = images.to(device)  
        labels = labels.to(device)  

        output = model(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1)%1500 == 0:
            print(f'epoch: {epoch}/{num_epochs}, step: {i+1}/{total_num_steps}, loss: {loss.item():.4f}')

with torch.no_grad():
    total_class = [0 for i in range(10)]
    score_class = [0 for i in range(10)]
    class_wise_acc = [0 for i in range(10)]
    for i, (data, labels) in enumerate(test_dataloadeer):
        data = data.to(device)  
        labels = labels.to(device)  
        y_pred = model(data)
        _, prediction = torch.max(y_pred, 1)


        for j in range(batch_size):
            label = labels[j]

            if label == prediction[j]:
                score_class[label] +=1
            total_class[label] +=1



    net_score_class = sum(score_class)
    net_total_class = sum(total_class)
    final_acc = ((net_score_class/net_total_class)*100)

    for i in range(10):
        class_wise_acc[i] = (score_class[i]/total_class[i])*100
        print(f'The acc of Class {i+1} = {class_wise_acc[i]:.4f}')
    
    
    print(f'The final acc is {final_acc:.4f}')    