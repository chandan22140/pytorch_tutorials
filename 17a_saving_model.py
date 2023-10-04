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

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model(n_input_features=6)

# torch.save(model.state_dict(), 'folder.pth')

model.load_state_dict(torch.load('folder.pth'))
model.eval()
optimizer = torch.optim.SGD(model.parameters(), lr  = 0.01)
checkpoint = {"epoch":90, 
"model_dict":model.state_dict(),
"optim_dict":optimizer.state_dict()
}

torch.save(checkpoint, "chkpnt.pth")

