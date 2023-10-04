from cmath import exp
from pickletools import optimize
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1234)

sc = StandardScaler()
X_test = sc.fit_transform(X_test)
X_train = sc.fit_transform(X_train)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))


y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

##model
class Logisticregression(nn.Module):
    def __init__(self, n_inp_features):
        super(Logisticregression, self).__init__()
        self.linear = nn.Linear(n_inp_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))
##loss and optimizer
learn_rate = 0.9
model = Logisticregression(n_features)
optimizer = torch.optim.SGD(model.parameters(), lr = learn_rate)
criterion = nn.BCELoss()
##training loop

num_epochs = 100
for epoch in range(num_epochs):
    #forward
    y_hat = model(X_train)

    #loss compute
    loss = criterion(y_hat, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch%10 == 9:
        print(f'epoch = {epoch} and loss = {loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(X_test)
    y_pred_cls = y_predicted.round()
    acc = y_pred_cls.eq(y_test).sum()/y_test.shape[0]
    print(f'The accuracy of model is: {acc:.4f}')


    



















