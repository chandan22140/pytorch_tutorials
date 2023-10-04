import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tt
import datetime 
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

import sys
writer = SummaryWriter("runs/mnist" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 784
hidden_size = 500
output_size = 10
num_epochs = 1
learning_rate = 0.01
b_size = 64 


transform = tt.Compose([tt.ToTensor(), tt.Normalize((0.1037),(0.3081))])



train_dataset = torchvision.datasets.MNIST(root='data' , train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data' , train=False, transform=transform)
# print(type(train_dataset))
train_loader = torch.utils.data.DataLoader(dataset = train_dataset , batch_size =b_size, shuffle = True )
test_loader = torch.utils.data.DataLoader(dataset = test_dataset , batch_size =b_size, shuffle = False )
# example_iter = iter(test_loader)
# example_data, example_target = example_iter.next()
# print(type(train_loader))
# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.imshow(example_data[i][0], cmap = 'grey')

print(type(train_dataset))
print(type(train_loader))

#tboard
# img_grid = torchvision.utils.make_grid(example_data)
# writer.add_image('mnist_image', img_grid)
# writer.close()


class NeuralNet2(nn.Module):
    def __init__(self, input_siz,hidden_size, output_size):
        super().__init__()
        self.lin1= nn.Linear(input_siz, hidden_size)
        # self.lin15 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.lin1(x)
        out = self.relu(out)
        # out = self.lin15(out)
        # out = self.relu(out)
        out = self.lin2(out)
        return out
    

# model = NeuralNet2(input_size, hidden_size, output_size).to(device)
# criterion  = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
# n_total_steps = len(train_loader)

# writer.add_graph(model, example_data.reshape(-1, 28*28))
# writer.close()

# running_correct = 0
# running_loss = 0
# for epoch in range(num_epochs):
#     for i, (images, label) in enumerate(train_loader):
#         images = images.reshape(-1, 28*28).to(device)
#         label = label.to(device)
#         output = model(images)
#         if (i)%100000==0:
#             print(label.size())
#         loss = criterion(output, label)
#         _, pred = torch.max(output, 1)
        
#         #back
#         optimizer.zero_grad()

#         loss.backward()        
#         optimizer.step()
#         running_loss += loss.item()
#         running_correct+= (pred == label).sum().item()
#         if (i+1)%100 == 0:
#             print(f'epoch:{epoch+1} , step:{i+1}, loss = {loss.item():}')
#             writer.add_scalar('training_loss', running_loss/100, i+ epoch*n_total_steps)
#             writer.add_scalar('training_acc', running_correct/100, i+ epoch*n_total_steps)

        
#         running_correct = 0
#         running_loss = 0
            
# class_labels = []
# class_preds = []

# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     for images, labels in test_loader:
#         images = images.reshape(-1, 28*28).to(device)
#         labels = labels.to(device)
#         output = model(images)

#         _, pred = torch.max(output, 1)
#         n_samples += labels.shape[0]
#         n_correct += (pred ==labels).sum().item()

#         class_probs_batch = [F.softmax(i, dim=0) for i in output]
#         class_preds.append(class_probs_batch)
#         class_labels.append(pred)

#     # stack concatenates 10000*10 tensors along a new dimension to return a list of tensors 
#     # cat concatenates that list of tensors(of dim - 64 ) in the given dimension
 
#     class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
#     class_labels = torch.cat(class_labels)
#     acc = 100.0 * n_correct / n_samples
#     print(f'Accuracy of the network on the 10000 test images: {acc} %')


#     ############## TENSORBOARD ########################
#     classes = range(10)
#     for i in classes:
#         labels_i = class_labels == i
#         preds_i = class_preds[:, i]
#         writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
#         writer.close()


# torch.save(model.state_dict(), "ffn_001.pth")




