import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda")

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
# x = torch.tensor([[1, 2], [3, 4]])
# model = Model(n_input_features=6)


# x.to(device)
# model.to(device)

# torch.save(model.state_dict(), "path.pth")
# torch.save(x, "path2.pth" )

model = Model(6)
x = torch.ones(2,2)

model.load_state_dict(torch.load("path.pth"))
x = torch.load("path2.pth")

model.to(device)
x.to(device)

print(model.state_dict)
print(x)


