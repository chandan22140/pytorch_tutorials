import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy

# 0) Prepate data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise = 20, random_state = 1)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0],1)
print(X.shape)
# print(X)
# print(y)
n_samples, n_features = X.shape
in_size = 1
out_size = 1

# model reation
model = nn.Linear(in_size, out_size)

#optimizer
learning_rate = 0.1
critrn  = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
#traning

num_epochs = 100
for epoch in range(num_epochs):
    y_hat = model(X)


    loss = critrn(y_hat, y)
    loss.backward()
    

    optimizer.step()
    optimizer.zero_grad()
    if epoch%10 == 0:

        print(f'epoch: {epoch}, loss = {loss:.8f}')
     

predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, "ro")
plt.plot(X_numpy, predicted, "b")

plt.show()

##forward
##backward
##updates

