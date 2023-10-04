import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class Winedataset(Dataset):
    def __init__(self):
        super().__init__()
        xy = np.loadtxt('./data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

datast = Winedataset()
dataloader = DataLoader(dataset=datast, batch_size=4, shuffle=True)

datainr = iter(dataloader)
data = datainr.next()
features, labels = data
print(features, labels)
num_epochs = 2
total_samples = len(datast)
n_iterations = math.ceil(total_samples / 4)
# model = nn.Linear(datast.x.shape[1], 1)

print(total_samples, n_iterations)
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # print(inputs, labels)
        # y_hat = model()
        if (i+1) % 5 == 0:
            print(f'epoch = {epoch+1}/{num_epochs}, step = {i+1}/{n_iterations}')
        
