import json
import numpy as np
import torch
import os
import pandas as pd
from tqdm import tqdm

replacement_map = None
default_value_map = None
prev_value_map = {}
DATA_PATH = "/mnt/data01/nilayam2/length-of-stay/"
SAVED_TENSOR_PATH = DATA_PATH + "saved/"
TRAIN_PATH = DATA_PATH + "train/"
train_y_df = pd.read_csv(DATA_PATH + 'train_listfile.csv') 
train_files = train_y_df["stay"].unique().tolist()
val_y_df = pd.read_csv(DATA_PATH + 'val_listfile.csv') 
val_files = val_y_df["stay"].unique().tolist()

with open('config.json') as f:
    config = json.load(f)
    replacement_map = config['replacement_map']
    default_value_map = config['default_value_map']


def to_np(elem):
    return np.concatenate([np.array(i) for i in elem])

def cleanup(episode_df):
    episode_df["Glascow coma scale eye opening"] = episode_df["Glascow coma scale eye opening"].apply(lambda x: replacement_map["Glascow coma scale eye opening"][x] if x in replacement_map["Glascow coma scale eye opening"] else x)
    episode_df["Glascow coma scale motor response"] = episode_df["Glascow coma scale motor response"].apply(lambda x: replacement_map["Glascow coma scale motor response"][x] if x in replacement_map["Glascow coma scale motor response"] else x)
    episode_df["Glascow coma scale verbal response"] = episode_df["Glascow coma scale verbal response"].apply(lambda x: replacement_map["Glascow coma scale verbal response"][x] if x in replacement_map["Glascow coma scale verbal response"] else x)


def process_column(person_id, value, colname):
    if value is not None and not np.isnan(value):
        prev_value_map[person_id][colname] = value
        return value
    if person_id in prev_value_map and colname in prev_value_map[person_id] and prev_value_map[person_id][colname] is not None:
        prev = prev_value_map[person_id][colname]
    else:
        prev = default_value_map[colname]
    return prev

def fill_missing_values(pateint_id, episode_df):
    prev_value_map[pateint_id] = {}
    episode_df["Capillary refill rate"] = episode_df["Capillary refill rate"].apply(lambda x: process_column(pateint_id, x, "Capillary refill rate"))
    episode_df["Diastolic blood pressure"] = episode_df["Diastolic blood pressure"].apply(lambda x: process_column(pateint_id, x, "Diastolic blood pressure"))
    episode_df["Fraction inspired oxygen"] = episode_df["Fraction inspired oxygen"].apply(lambda x: process_column(pateint_id, x, "Fraction inspired oxygen"))
    episode_df["Glascow coma scale eye opening"] = episode_df["Glascow coma scale eye opening"].apply(lambda x: process_column(pateint_id, x, "Glascow coma scale eye opening"))
    episode_df["Glascow coma scale motor response"] = episode_df["Glascow coma scale motor response"].apply(lambda x: process_column(pateint_id, x, "Glascow coma scale motor response"))
    episode_df["Glascow coma scale total"] = episode_df["Glascow coma scale total"].apply(lambda x: process_column(pateint_id, x, "Glascow coma scale total"))
    episode_df["Glascow coma scale verbal response"] = episode_df["Glascow coma scale verbal response"].apply(lambda x: process_column(pateint_id,  x, "Glascow coma scale verbal response"))
    episode_df["Glucose"] = episode_df["Glucose"].apply(lambda x: process_column(pateint_id, x, "Glucose"))
    episode_df["Heart Rate"] = episode_df["Heart Rate"].apply(lambda x: process_column(pateint_id, x, "Heart Rate"))
    episode_df["Mean blood pressure"] = episode_df["Mean blood pressure"].apply(lambda x: process_column(pateint_id, x, "Mean blood pressure"))
    episode_df["Height"] = episode_df["Height"].apply(lambda x: process_column(pateint_id,x, "Height"))
    episode_df["Oxygen saturation"] = episode_df["Oxygen saturation"].apply(lambda x: process_column(pateint_id,  x, "Oxygen saturation"))
    episode_df["Respiratory rate"] = episode_df["Respiratory rate"].apply(lambda x: process_column(pateint_id, x, "Respiratory rate"))
    episode_df["Systolic blood pressure"] = episode_df["Systolic blood pressure"].apply(lambda x: process_column(pateint_id, x, "Systolic blood pressure"))
    episode_df["Temperature"] = episode_df["Temperature"].apply(lambda x: process_column(pateint_id, x, "Temperature"))
    episode_df["Weight"] = episode_df["Weight"].apply(lambda x: process_column(pateint_id, x, "Weight"))
    episode_df["pH"] = episode_df["pH"].apply(lambda x: process_column(pateint_id, x, "pH"))
    del prev_value_map[pateint_id]

def get_window_indices(data_len):
    i = 0
    indices = []
    while i <= data_len-4:
        indices.append([i, i+1, i+2, i+3])
        i +=1
    return indices


def preprocess(path, use_saved):
    x_file_name = path+"_X.pt"
    y_file_name = path+"_Y.pt"
    if use_saved:
        print("Attempting to use saved files")
        saved_files = os.listdir(SAVED_TENSOR_PATH)
        print("Saved files : ", saved_files)
        if x_file_name in saved_files and y_file_name in saved_files:
            print("Loading X and Y from saved files. Will Skip processing")
            X = torch.load(SAVED_TENSOR_PATH+x_file_name)
            Y = torch.load(SAVED_TENSOR_PATH+y_file_name) 
            return (X, Y)
    print("Processing "+ path +" files.")
    if path == 'train':
        y_df = train_y_df
        #data_files = np.random.choice(train_files, 5000, replace=False)
        data_files = train_files
    elif path == 'val':
        y_df = val_y_df
        #data_files = np.random.choice(val_files, 500, replace=False)
        data_files = val_files

    #x_path = DATA_PATH +'/'+path+'/'
    X = torch.empty(0,17,4)
    Y = torch.empty(0,)
    #y_df = pd.read_csv(DATA_PATH + path +'_listfile.csv') 
    #data_files = os.listdir(x_path)
    print("Number of "+ path+ " files = ",len(data_files))
    for data_file in tqdm(data_files):
        if data_file.endswith(".csv"):
            episode_df = pd.read_csv(TRAIN_PATH + data_file)
            cleanup(episode_df)
            fill_missing_values(data_file, episode_df)
            episode_df["H_IDX"] = episode_df.Hours.apply(np.floor).astype('int32')
            episode_df = episode_df.groupby(by = "H_IDX").mean()
            episode_df = episode_df[episode_df.Hours>=5].reset_index(drop = True)
            temp_y = y_df[y_df.stay == data_file].sort_values(by = "period_length").reset_index(drop = True)
            temp_y = temp_y[["period_length", "y_true"]].set_index("period_length")
            episode_df = episode_df.join(temp_y, how = "inner").drop('Hours', axis=1).reset_index(drop = True)
            if(len(episode_df) >0):
                indices = get_window_indices(len(episode_df))
                windows = []
                y_values = []
                for idx in indices:
                    window = episode_df.loc[idx]
                    y_values.append(window.loc[idx[-1]].y_true)
                    windows.append(window.drop("y_true", axis=1).transpose().values.astype(np.float32))
                t_windows = torch.tensor(windows)
                t_y_values = torch.tensor(y_values)
                X = torch.cat((X, t_windows), 0)
                Y = torch.cat((Y, t_y_values), 0)
    torch.save(X, SAVED_TENSOR_PATH+x_file_name)
    torch.save(Y, SAVED_TENSOR_PATH+y_file_name)
    return (X, Y)