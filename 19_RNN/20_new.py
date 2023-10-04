from turtle import forward
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 28
sequence_length = 28
hidden_size = 128


num_layers = 2
num_epochs = 2
learning_rate = 0.001
b_size = 100
num_classes = 10

# transform = tt.Compose([tt.ToTensor(), tt.Normalize((0.1037),(0.3081))])

train_dataset = torchvision.datasets.MNIST(root='data' , train=True, transform=tt.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data' , train=False, transform=tt.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset , batch_size =b_size, shuffle = True )
test_loader = torch.utils.data.DataLoader(dataset = test_dataset , batch_size =b_size, shuffle = False )

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size,num_layers, batch_first = True )  
        # x-> batc_size, seq_lenth, input_size
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.rnn(x, h0)
        # out-> batc_size, seq_lenth, hidden_size
        out = out[:, -1, :]
        out = self.fc(out)
        return out



model = RNN(input_size, hidden_size,num_layers,  num_classes).to(device)
criterion  = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
n_total_steps = len(train_loader)

running_correct = 0
running_loss = 0


for epoch in range(num_epochs):
    for i, (images, label) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        label = label.to(device)
        output = model(images) 
        loss = criterion(output, label)
        _, pred = torch.max(output, 1)
        
        #back
        optimizer.zero_grad()

        loss.backward()        
        optimizer.step()
        running_loss += loss.item()
        running_correct+= (pred == label).sum().item()
 
        if (i+1)%100 == 0:
            print(f'epoch:{epoch+1} , step:{i+1}, loss = {running_loss:.4f}')
            running_correct = 0
            running_loss = 0

            
class_labels = []
class_preds = []


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        output = model(images)

        _, pred = torch.max(output, 1)
        n_samples += labels.shape[0]
        n_correct += (pred ==labels).sum().item()
  
        # class_probs_batch = [F.softmax(i, dim=0) for i in output]
    #     class_preds.append(class_probs_batch)
    #     class_labels.append(pred)

    # # stack concatenates 10000*10 tensors along a new dimension to return a list of tensors 
    # # cat concatenates that list of tensors(of dim - 64 ) in the given dimension
 
    # class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
    # class_labels = torch.cat(class_labels)
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
