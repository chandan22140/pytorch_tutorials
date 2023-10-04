from pickletools import optimize
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tt
from itertools import chain


input1_size = 28
input2_siz = 20
hidden_size = 100
recurrence_size = 28
output_size = 10
num_epochs = 1
learning_rate = 0.01
b_size = 1

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
transform = tt.Compose([tt.ToTensor(), tt.Normalize((0.1037),(0.3081))])

train_dataset = torchvision.datasets.MNIST(root='data' , train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='data' , train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset , batch_size =b_size, shuffle = True )
test_loader = torch.utils.data.DataLoader(dataset = test_dataset , batch_size =b_size, shuffle = True )


class RNN1(nn.Module):
    def __init__(self, input1_size,hidden1_size,input2_size):
        super().__init__()
        self.hidden1_size = hidden1_size        
        self.i1toh1= nn.Linear(input1_size+hidden1_size, hidden1_size)
        self.i1toi2 = nn.Linear(input1_size + hidden1_size, input2_size)

    def init_hidden(self):
        return torch.zeros(1, self.hidden1_size)
    
    def forward(self, input1, hidden1):
        input1 = input1.reshape(1, 28)
        combined1 = torch.cat((input1, hidden1), -1)
        hidden1 = self.i1toh1(combined1)
        input2 = self.i1toi2(combined1)
        return input2, hidden1


class RNN2(nn.Module):
   
    def __init__(self,input2_size, hidden2_size,  output_size):
        super().__init__()
        self.hidden2_size = hidden2_size  
        self.i2toh2= nn.Linear(input2_size+hidden2_size, hidden2_size)
        self.i2toO = nn.Linear(input2_size + hidden2_size, output_size)        
        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self):
        return torch.zeros(1, self.hidden2_size)
  
    def forward(self, input2, hidden2):
        input2 = input2.reshape(1, 20)
        combined2 = torch.cat((input2, hidden2), -1)
        hidden2 = self.i2toh2(combined2)
        Output = self.i2toO(combined2)
        return Output, hidden2


model1 = RNN1(input1_size, hidden_size, input2_siz).to(device)
model2 = RNN2(input2_siz, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
params = chain(model1.parameters(), model2.parameters())
optimizer = torch.optim.SGD(params, lr = learning_rate)

for i, (data, label) in enumerate(train_loader):
    if True:
        hidden1 = model1.init_hidden().to(device)
        hidden2 = model2.init_hidden().to(device)
        data = data.reshape(-1, 28).to(device)
        label = label.to(device)
        input2, hidden1 = model1(data[0].reshape(1, 28),hidden1)
        for j in range(recurrence_size):
            Output, hidden2 = model2(input2, hidden2)
            if not j == (recurrence_size-1):
                input2, hidden1 = model1(data[j+1], hidden1)
        loss = criterion(Output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# test_loader = torch.utils.data.DataLoader(dataset = test_dataset , batch_size =b_size, shuffle = True )



for i, (data, label) in enumerate(test_loader):
    with torch.no_grad():
        hidden1 = model1.init_hidden().to(device)
        hidden2 = model2.init_hidden().to(device)
        data = data.reshape(-1, 28).to(device)
        label = label.to(device)
        input2, hidden1 = model1(data[0].reshape(1, 28),hidden1)
        for j in range(recurrence_size):
            Output, hidden2 = model2(input2, hidden2)
            if not j == (recurrence_size-1):
                input2, hidden1 = model1(data[j+1], hidden1)
        _, pred = torch.max(Output, 1)
        if pred == label:
            check = "Correct" 
        else:
            check = "Incorrect"  
        print(f'output: {pred}, label:{label}. hence, {check}')
