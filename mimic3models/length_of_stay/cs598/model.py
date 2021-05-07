import torch
import torch.nn as nn
import torch.nn.functional as F 


class EpisodeCNN(nn.Module):
    def __init__(self):
        super(EpisodeCNN, self).__init__()
        #input shape 34 * 8
        #output shape 16 * 8
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride = 1)
        #input shape 16 * 8
        #output shape 32 * 8
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride = 1)
        #input shape 32 * 8
        #output shape 16 * 4
        self.pool1 = nn.MaxPool2d(2,2)
        #input shape 16 * 4
        #output shape 32 * 4
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride = 1)
        self.fc1 = nn.Linear(64*17*4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        #input is of shape (batch_size=32, 3, 224, 224) if you did the dataloader right
        #input is of shape (batch_size=32, 3, 224, 224) if you did the dataloader right
        x = x.unsqueeze(1)
        #print(x.shape)
        x = F.leaky_relu(self.conv1(x))
        #print(x.shape)
        x = F.leaky_relu(self.conv2(x))
        #print(x.shape)
        x = self.pool1(x)
        #print(x.shape)
        x = F.leaky_relu(self.conv3(x))
        #print(x.shape)
        x = x.view(-1, 64*17*4)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        #print(x.shape)
        x = F.leaky_relu(self.fc2(x))
        #print(x.shape)
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x