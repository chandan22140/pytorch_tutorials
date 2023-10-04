# from random import shuffle
# from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim


class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr  = 0)



loaded_checkpoint = torch.load("chkpnt.pth")

epoch = loaded_checkpoint["epoch"]
model.load_state_dict(loaded_checkpoint["model_dict"])
optimizer.load_state_dict(loaded_checkpoint["optim_dict"])
print(epoch)
print(model.state_dict)
print(optimizer.state_dict)