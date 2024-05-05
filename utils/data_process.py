import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd


class StandardScaler():
    """Description: Normalize the data"""
    def __init__(self):
        self.mean = 0.
        self.std = 1.
 
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)
 
    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std
 
    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
 
    def __len__(self):
        return len(self.sequences)
 
    def __getitem__(self, index):
        sequence, sequence_filled, label = self.sequences[index]
        return torch.Tensor(sequence), torch.Tensor(sequence_filled), torch.Tensor(label)
    
    def update_item(self, index, new_item):
        self.sequences[index] = new_item


def create_inout_sequences(input_data, input_data_filled, tw, pre_len, config):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        seq = input_data[i:i + tw]
        seq_filled = input_data_filled[i:i + tw]
        if config.fill == True:
            seq_filled = fill(seq_filled)
        if (i + tw + pre_len) > len(input_data):
            break
        if config.feature == 'MS':
            label_filled = input_data_filled[:, -1:][i + tw:i + tw + pre_len]
        else:
            label_filled = input_data_filled[i + tw:i + tw + pre_len]
        inout_seq.append((seq, seq_filled, label_filled))
    return inout_seq

def create_dataloader(config, device):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>Creating DataLoader<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    df = pd.read_csv(config.data_path)  
    pre_len = config.pre_len  
    train_window = config.window_size  

    # Move the feature column to the end
    target_data = df[[config.target]]
    df = df.drop(config.target, axis=1)
    df = pd.concat((df, target_data), axis=1)

    cols_data = df.columns[1:]
    df_data = df[cols_data]

    true_data = df_data.values

    train_data = true_data[:int(0.75 * len(true_data))]
    valid_data = true_data[int(0.75 * len(true_data)):int(0.8 * len(true_data))]
    test_data = true_data[int(0.8 * len(true_data)):]

    print("Training set size:", len(train_data), "Test set size:", len(test_data), "Validation set size:", len(valid_data))
    
    # Define standardization optimizer
    scaler = StandardScaler()
    scaler.fit(train_data)

    # Perform standardization
    train_data_normalized = scaler.transform(train_data)
    test_data_normalized = scaler.transform(test_data)
    valid_data_normalized = scaler.transform(valid_data)

    # Convert to Tensor
    train_data_normalized = torch.FloatTensor(train_data_normalized).to(device)
    test_data_normalized = torch.FloatTensor(test_data_normalized).to(device)
    valid_data_normalized = torch.FloatTensor(valid_data_normalized).to(device)
    
    # Replace 0 values with 10 in normalized data
    train_data_filled = train_data_normalized.clone()
    train_data_filled[train_data == 0] = 100
    
    test_data_filled = test_data_normalized.clone()
    test_data_filled[test_data == 0] = 100
    
    valid_data_filled = valid_data_normalized.clone()
    valid_data_filled[valid_data == 0] = 100

    # Define the input of the trainer
    train_inout_seq = create_inout_sequences(train_data_normalized, train_data_filled, train_window, pre_len, config)
    test_inout_seq = create_inout_sequences(test_data_normalized, train_data_filled, train_window, pre_len, config)
    valid_inout_seq = create_inout_sequences(valid_data_normalized, train_data_filled, train_window, pre_len, config)

    # Create dataset
    train_dataset = TimeSeriesDataset(train_inout_seq)
    test_dataset = TimeSeriesDataset(test_inout_seq)
    valid_dataset = TimeSeriesDataset(valid_inout_seq)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

    print("Training set data:   ", len(train_inout_seq), "Converted to batch data:", len(train_loader))
    print("Test set data:       ", len(test_inout_seq), "Converted to batch data:", len(test_loader))
    print("Validation set data: ", len(valid_inout_seq), "Converted to batch data:", len(valid_loader))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>DataLoader Created<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    return train_loader, test_loader, valid_loader, scaler


def fill(tensor):
    # 交换维度
    tensor_swapped = tensor.t()

    # 找出所有为100的元素的索引
    idx_100 = (tensor_swapped == 100)

    # 计算每列除了100以外的元素的平均值
    mean_except_100 = (tensor_swapped * (~idx_100)).sum(dim=1) / (~idx_100).sum(dim=1)

    # 将100替换为平均值
    tensor_swapped[idx_100] = mean_except_100.unsqueeze(1).repeat(1, tensor_swapped.size(1))[idx_100]

    # 交换维度
    tensor_swapped = tensor_swapped.t()

    return tensor_swapped

import torch
import torch.nn.functional as F

def fill_missing_tensor(X, missing_value=100.0, rank=10, n_iter=10):
    """
    Fill missing values in a 2D tensor using tensor factorization.
    
    Args:
        X (torch.Tensor): 2D tensor with missing values represented as missing_value
        missing_value (float, optional): Value used to represent missing values. Defaults to 100.0.
        rank (int, optional): Rank for tensor factorization. Defaults to 10.
        n_iter (int, optional): Number of iterations for ALS. Defaults to 10.
        
    Returns:
        torch.Tensor: Tensor with missing values filled.
    """
    
    # Get indices of missing values
    is_missing = (X == missing_value)
    
    # Initialize factor matrices randomly
    m, n = X.shape
    U = torch.randn(m, rank)
    V = torch.randn(rank, n)
    
    # ALS algorithm
    for _ in range(n_iter):
        # Update U
        V_t = V.T
        for i in range(m):
            row = X[i]
            missing_idx = is_missing[i]
            row_missing = row[~missing_idx]
            V_missing_idx = ~missing_idx
            if V_missing_idx.numel() == 0:  # Handle empty index case
                U[i, :] = torch.zeros(rank)
            else:
                V_missing = V_t[:, V_missing_idx]
                U[i, V_missing_idx] = torch.linalg.lstsq(row_missing, V_missing).solution.squeeze()
        
        # Update V
        for j in range(n):
            col = X[:, j]
            missing_idx = is_missing[:, j]
            col_missing = col[~missing_idx]
            U_missing_idx = ~missing_idx
            if U_missing_idx.numel() == 0:  # Handle empty index case
                V[:, j] = torch.zeros(rank)
            else:
                U_missing = U[U_missing_idx]
                V[:, j][U_missing_idx] = torch.linalg.lstsq(col_missing, U_missing).solution.squeeze()
    
    # Reconstruct tensor
    X_reconstructed = torch.matmul(U, V)
    
    # Fill missing values
    X_filled = X.clone()
    X_filled[is_missing] = X_reconstructed[is_missing]
    
    return X_filled