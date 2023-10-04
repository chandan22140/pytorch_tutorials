import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class Winedataset(Dataset):
    def __init__(self, transform = None):
        super().__init__()
        xy = np.loadtxt('./data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]
        self.n_samples = xy.shape[0]
        self.transform = transform
    def __getitem__(self, index):
        samples =  self.x[index], self.y[index]
        if self.transform:
            samples = self.transform(samples)
        return samples
    def __len__(self):
        return self.n_samples
class ToTensor():
    def __call__(self, sample):
        input, label = sample
        return torch.from_numpy(input), torch.from_numpy(label)

class Multensor():
    def __init__(self, factor):
        self.factor = factor 

    def __call__(self, tensorrr):
        input, label = tensorrr
        input *= self.factor
        return input, label
        
composed = torchvision.transforms.Compose([ToTensor(), Multensor(2)])
# datast = Winedataset()
# dataloader = DataLoader(dataset=datast, batch_size=4, shuffle=True)

# datainr = iter(dataloader)
# data = datainr.next()
# features, labels = data
# print(features, labels)
# num_epochs = 2
# total_samples = len(datast)
# n_iterations = math.ceil(total_samples / 4)
# # model = nn.Linear(datast.x.shape[1], 1)

# print(total_samples, n_iterations)
# for epoch in range(num_epochs):
#     for i, (inputs, labels) in enumerate(dataloader):
#         # print(inputs, labels)
#         # y_hat = model()
#         if (i+1) % 5 == 0:
#             print(f'epoch = {epoch+1}/{num_epochs}, step = {i+1}/{n_iterations}')
        
