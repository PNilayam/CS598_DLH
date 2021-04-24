import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import pandas as pd
from preprocess import preprocess
from model import EpisiodeCNN
from sklearn.metrics import mean_squared_error

seed = 29
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

def eval_model(model, val_loader):
    model.eval()
    all_y_true = torch.DoubleTensor()
    all_y_pred = torch.DoubleTensor()
    for x, y in val_loader:
        y_hat = model(x)
        all_y_true = torch.cat((all_y_true, y.to('cpu')), dim=0)
        all_y_pred = torch.cat((all_y_pred,  y_hat.to('cpu')), dim=0)
    mse= mean_squared_error(all_y_true.detach().numpy(), all_y_pred.detach().numpy())
    print(f"mse: {mse:.3f}")
    return mse

def train(model, train_loader, val_loader, n_epochs, optimizer, criterion):
    model.train()
    for epoch in range(n_epochs):
        train_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            y_hat = model(x)
            y_hat = y_hat.view(y_hat.shape[0]).double()
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loss = train_loss / len(train_loader)
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))
        eval_model(model, val_loader)

from torch.utils.data import Dataset

class EpisodeDataset(Dataset):
    
    def __init__(self, obs, los):
        self.x = obs
        self.y = los
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return (self.x[index], self.y[index])
        


if __name__ == "__main__":
    print("main")
    DATA_PATH = "/Users/prashanti.nilayam/Desktop/temp/"
    X_train, Y_train = preprocess('train')
    train_dataset = EpisodeDataset(X_train, Y_train)
    X_val, Y_val = preprocess('val')
    val_dataset = EpisodeDataset(X_val, Y_val)
    learning_rate = 0.00001
    criterion = nn.MSELoss()
    model = EpisiodeCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr =learning_rate )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,shuffle=True)                              
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=32, shuffle=False)    
    train(model= model, train_loader = train_loader, val_loader= val_loader, n_epochs = 250, optimizer= optimizer, criterion = criterion)
    torch.save(model.state_dict(), DATA_PATH+"/model.pt")