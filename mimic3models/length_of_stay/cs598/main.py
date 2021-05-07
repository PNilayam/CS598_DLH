import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import pandas as pd
from preprocess import preprocess
from model import EpisodeCNN
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from tqdm import trange
import time
import argparse
import statistics
import matplotlib
import matplotlib.pyplot as plt
from math import log

seed = 29
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default = 0.001, help= "Learning rate for the model")
parser.add_argument('--epochs', type=int, default = 20, help= "number of epochs")
parser.add_argument('--batch_size', type=int, default = 128, help= "batch size")
parser.add_argument('--window', type=int, default=5, help="minimum 5")
parser.add_argument('--use_saved', type=bool, default=True, help="use saved embeddings")
parser.add_argument('--data', type=str, help='sample / all',default="all")
parser.add_argument('--train_sample_size', type=int, help='if data == sample, provide training sample size',default=5000)
parser.add_argument('--val_sample_size', type=int, help='if data == sample, provide training sample size',default=500)
parser.add_argument('--test_sample_size', type=int, help='if data == sample, provide training sample size',default=5000)
parser.add_argument('--model_output_dir', type=str, help='Model output dir',default='/mnt/data01/models/cnn/model_v2.pt')

args = parser.parse_args()
print(args)

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
    train_losses = []
    for epoch in trange(n_epochs):
        loss_per_epoch = []
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            y_hat = model(x)
            y_hat = y_hat.view(y_hat.shape[0]).double()
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            loss_per_epoch.append(train_loss)
            #print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))
        epoch_loss = statistics.mean(loss_per_epoch)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, epoch_loss))
        eval_model(model, val_loader)
        train_losses.append(epoch_loss)
    return train_losses

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
    start_time = time.time()
    DATA_PATH =  "/mnt/data01/nilayam2/length-of-stay/"
    sample = False
    if args.data == 'sample':
        sample = True
    X_train, Y_train = preprocess('train', use_saved=args.use_saved, window_len=args.window, sample=sample, sample_size=args.train_sample_size)
    train_dataset = EpisodeDataset(X_train, Y_train)
    X_val, Y_val = preprocess('val', use_saved=args.use_saved, window_len = args.window, sample=sample, sample_size=args.val_sample_size)
    val_dataset = EpisodeDataset(X_val, Y_val)
    #criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    model = EpisodeCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr =args.lr)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)                              
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size, shuffle=False)    
    train_losses = train(model= model, train_loader = train_loader, val_loader= val_loader, n_epochs = args.epochs, optimizer= optimizer, criterion = criterion)
    torch.save(model.state_dict(), args.model_output_dir)

    #save train losses as png
    matplotlib.use('Agg')
    train_losses = [log(y) for y in train_losses]
    plt.plot(train_losses)
    plt.savefig('train_{}_{}_{}_{}.png'.format(args.window,args.lr,args.batch_size, args.epochs))

    end_time = time.time()
    total_time = end_time - start_time
    print("Total time taken : {} secs".format(total_time))