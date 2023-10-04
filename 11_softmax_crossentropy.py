from turtle import forward
import torch
import torch.nn as nn
import numpy as np

loss = nn.CrossEntropyLoss()

Y = torch.tensor([0])
Y_Good = torch.tensor([[2, 1, 0.1]])
print(loss(Y_Good, Y).item())

class NeuralNet2():
    def __init__(self, input_siz,hidden_size, output_size):
        self.lin1= nn.Linear(input_siz, hidden_size)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.lin1(x)
        out = self.relu(out)
        out = self.lin2(out)

