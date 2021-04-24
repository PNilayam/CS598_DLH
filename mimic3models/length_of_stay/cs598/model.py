import torch
import torch.nn as nn
import torch.nn.functional as F 


class EpisiodeCNN(nn.Module):
    def __init__(self):
        super(EpisiodeCNN, self).__init__()
        #input shape 1 * 17 * 4
        #output shape 17 * 17 * 4
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=17, kernel_size=3, padding=1, stride = 1)
        #input shape 17 * 17 * 4
        #output shape 68 * 17 * 4
        self.conv2 = nn.Conv2d(in_channels=17, out_channels=34, kernel_size=3, padding=1, stride = 1)
        #input shape 34 * 17 * 4
        #output shape 34 * 8 * 2
        self.pool1 = nn.MaxPool2d(2,2)
        #input shape 34 * 8 * 2
        #output shape 68 * 8 * 2
        self.conv3 = nn.Conv2d(in_channels=34, out_channels=68, kernel_size=3, padding=1, stride = 1)
        self.fc1 = nn.Linear(68*8*2, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        #input is of shape (batch_size=32, 3, 224, 224) if you did the dataloader right
        x = x.unsqueeze(1)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.conv3(x))
        x = x.view(-1, 68 * 8 * 2)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x