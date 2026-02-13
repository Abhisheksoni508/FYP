import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import torch
import os

# Define column names for CMAPSS
COLS = ['unit', 'time', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]

def load_data(data_path):
    # Read the raw text file
    df = pd.read_csv(data_path, sep=r'\s+', header=None, names=COLS)
    return df

def load_combined_data(data_dir='data'):
    """
    Loads FD001 and FD002, fixes unit IDs, and merges them.
    """
    # 1. Load FD001
    path1 = os.path.join(data_dir, 'train_FD001.txt')
    if not os.path.exists(path1):
        raise FileNotFoundError(f"Missing {path1}")
    df1 = load_data(path1)
    print(f"FD001 loaded: {len(df1)} rows, {df1['unit'].nunique()} units")
    
    # 2. Load FD002
    path2 = os.path.join(data_dir, 'train_FD002.txt')
    if not os.path.exists(path2):
        print(f"Warning: {path2} not found. Using FD001 only.")
        return df1
    df2 = load_data(path2)
    print(f"FD002 loaded: {len(df2)} rows, {df2['unit'].nunique()} units")
    
    # 3. FIX UNIT IDs so they don't overlap
    # FD001 ends at Unit 100. We shift FD002 units by 100.
    max_id_1 = df1['unit'].max()
    df2['unit'] = df2['unit'] + max_id_1
    
    # 4. Concatenate
    df_combined = pd.concat([df1, df2], ignore_index=True)
    print(f"Combined Dataset: {len(df_combined)} rows, {df_combined['unit'].nunique()} units")
    
    return df_combined

def calculate_rul(df):
    # Get the last time cycle for each unit
    max_cycles = df.groupby('unit')['time'].max().reset_index()
    max_cycles.columns = ['unit', 'max_time']
    
    # Merge and subtract to get RUL
    df = df.merge(max_cycles, on='unit', how='left')
    df['RUL'] = df['max_time'] - df['time']
    
    # Cap RUL at 125 (Standard practice)
    df['RUL'] = df['RUL'].clip(upper=125)
    
    return df.drop(columns=['max_time'])

def process_data(df, drop_sensors, drop_settings, scaler=None):
    # Drop useless columns
    cols_to_drop = drop_sensors + drop_settings
    df_clean = df.drop(columns=cols_to_drop)
    
    # Separate features and target
    features = [c for c in df_clean.columns if c not in ['unit', 'time', 'RUL']]
    
    # Normalize
    if scaler is None:
        scaler = MinMaxScaler()
        df_clean[features] = scaler.fit_transform(df_clean[features])
    else:
        df_clean[features] = scaler.transform(df_clean[features])
        
    return df_clean, scaler

def create_sequences(df, window_size, features):
    sequences = []
    labels = []
    units = df['unit'].unique()
    
    for unit in units:
        unit_data = df[df['unit'] == unit]
        data_array = unit_data[features].values
        rul_array = unit_data['RUL'].values
        
        if len(unit_data) < window_size:
            continue
            
        for i in range(len(unit_data) - window_size):
            seq = data_array[i : i + window_size]
            label = rul_array[i + window_size]
            sequences.append(seq)
            labels.append(label)
            
    return np.array(sequences), np.array(labels)

class CMAPSSDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        # Normalize Target RUL (0-125 -> 0.0-1.0)
        self.labels = torch.FloatTensor(labels).unsqueeze(1) / 125.0 
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]