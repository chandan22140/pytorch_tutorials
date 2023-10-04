import torch

X = torch.tensor([1,2,3,4], dtype = torch.float32)
Y = torch.tensor([2,4,6,8], dtype = torch.float32)
w = torch.tensor(0, dtype = torch.float32, requires_grad=True)

def forward(x):
    return w*x

def loss(y,y_hat):
    return ((y-y_hat)**2).mean()

    
def grad(x,y,y_hat):
    return torch.dot(2*x, y_hat-y).mean()


print(f'prediction before training: f(5)= {forward(5):.3f}')
lr = 0.01
n_itrs = 20
for epoch in range(n_itrs):
    Y_HAT = forward(X)  
     
    loss(Y, Y_HAT).backward()
    with torch.no_grad():
        w = w - (w.grad)*0.1
    w.grad.zero_()
print(f'prediction after training: f(5)= {forward(5):.3f}')
    
    

    




    