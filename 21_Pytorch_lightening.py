
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tt
import datetime 
import torch.nn.functional as F
import pytorch_lightning  as pl
from pytorch_lightning import Trainer

input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 1
b_size = 64 
learning_rate = 0.01 
transform = tt.Compose([tt.ToTensor(), tt.Normalize((0.1037),(0.3081))])




class LitNeuralNet2(pl.LightningModule ):
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

    def train_dataloader(self):
        # MNIST dataset
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, transform=transform, download=True
        )
        # Data loader
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=b_size, num_workers = 4, shuffle=True
        )
        return train_loader


    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28 * 28)

        # Forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        
        # use key 'log'
        return {"loss": loss}

    # define what happens for testing here


    def val_dataloader(self):
        # MNIST dataset
        val_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, transform=transform
        )
        # Data loader
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=b_size, num_workers = 4, shuffle=False
        )
        return val_loader

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28 * 28)

        # Forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        
        # use key 'log'
        return {"val_loss": loss}

    # define what happens for testing here
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)
 
if __name__ == '__main__':
    trainer = Trainer(max_epochs =num_epochs  ,  fast_dev_run=False)
    model = LitNeuralNet2(input_size, hidden_size, num_classes)
    trainer.fit(model)