import torch
import torch.nn as nn


X = torch.tensor([[1],[2],[3],[4]], dtype = torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype = torch.float32)
X_test = torch.tensor([5], dtype = torch.float32)
n_samples, n_features = X.shape
input_size = n_features
print(n_features)
output_size = n_features
# model = nn.Linear(input_size, output_size)
class Linearregression(nn.Module):

    def __init__(self, inp_dim, out_dim):
        super(Linearregression, self).__init__()
        self.lin = nn.Linear(inp_dim, out_dim)

    def forward(self, x):
        return self.lin(x)

model = Linearregression(input_size, output_size)



print(f'prediction before training: f(5)= {model(X_test).item():.3f}')
n_itrs = 100
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters() , lr=0.1)


for epoch in range(n_itrs):
    #prediction
    Y_HAT = model(X)
    #loss compute
    l = loss(Y, Y_HAT)

    #backprop
    l.backward()

    #update weight
    optimizer.step()

    #grad value clean
    optimizer.zero_grad()  

    [w, b] = model.parameters()
    if epoch%10 == 0:

        print(f'epoch: {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')
    
print(f'prediction after training: f(5)= {model(X_test).item():.3f}')


