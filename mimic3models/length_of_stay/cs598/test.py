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
from torch.utils.data import Dataset
from main import EpisodeDataset
from main import eval_model


if __name__ == "__main__":
    print("Testing")
    start_time = time.time()
    DATA_PATH =  "/mnt/data01/nilayam2/length-of-stay/"
    X_test, Y_test = preprocess('test', True)
    test_dataset = EpisodeDataset(X_test, Y_test)
    model = EpisodeCNN()
    model.load_state_dict(torch.load(DATA_PATH+"/model.pt"))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128,shuffle=True)                              
    eval_model(model, test_loader)
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time taken : {} secs".format(total_time))