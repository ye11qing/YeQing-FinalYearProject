import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tensorly.decomposition import CP
import tensorly as tl
from tensorly.cp_tensor import cp_to_tensor
import numpy as np

# Author: Ye Qing
# Affiliation: National University of Singapore, Suzhou Research Institute

class StandardScaler:
    """StandardScaler normalizes data by removing the mean and scaling to unit variance"""
    
    def __init__(self):
        self.mean = 0.
        self.std = 1.
 
    def fit(self, data):
        """Calculate mean and std deviation of the data"""
        self.mean = data.mean(0)
        self.std = data.std(0)
 
    def transform(self, data):
        """Transform data by normalizing it"""
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std
 
    def inverse_transform(self, data):
        """Revert the data back to its original form before scaling"""
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean

class TimeSeriesDataset(Dataset):
    """A dataset class for time series data"""
    
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
    """Create input-output sequences from time series data"""
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        seq = input_data[i:i + tw]
        seq_filled = input_data_filled[i:i + tw]
        if config.fill:
            seq_filled = fill_missing_values_with_svd(seq_filled)
        if (i + tw + pre_len) > len(input_data):
            break
        label_filled = input_data_filled[:, -1:][i + tw:i + tw + pre_len] if config.feature == 'MS' else input_data_filled[i + tw:i + tw + pre_len]
        inout_seq.append((seq, seq_filled, label_filled))
    return inout_seq

def create_dataloader(config, device):
    """Create DataLoader for training, testing, and validation datasets"""
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
    
    # Define and apply standardization
    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data_normalized = scaler.transform(train_data)
    test_data_normalized = scaler.transform(test_data)
    valid_data_normalized = scaler.transform(valid_data)

    # Convert to Tensor and replace 0 values with 100 in normalized data
    train_data_normalized, test_data_normalized, valid_data_normalized = [torch.FloatTensor(data).to(device) for data in [train_data_normalized, test_data_normalized, valid_data_normalized]]
    train_data_filled = train_data_normalized.clone()
    train_data_filled[train_data == 0] = 100
    test_data_filled = test_data_normalized.clone()
    test_data_filled[test_data == 0] = 100
    valid_data_filled = valid_data_normalized.clone()
    valid_data_filled[valid_data == 0] = 100

    # Define the input of the trainer
    train_inout_seq = create_inout_sequences(train_data_normalized, train_data_filled, train_window, pre_len, config)
    test_inout_seq = create_inout_sequences(test_data_normalized, test_data_filled, train_window, pre_len, config)
    valid_inout_seq = create_inout_sequences(valid_data_normalized, valid_data_filled, train_window, pre_len, config)

    # Create dataset and DataLoader
    train_dataset = TimeSeriesDataset(train_inout_seq)
    test_dataset = TimeSeriesDataset(test_inout_seq)
    valid_dataset = TimeSeriesDataset(valid_inout_seq)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

    print("Training set data:   ", len(train_inout_seq), "Converted to batch data:", len(train_loader))
    print("Test set data:       ", len(test_inout_seq), "Converted to batch data:", len(test_loader))
    print("Validation set data: ", len(valid_inout_seq), "Converted to batch data:", len(valid_loader))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>DataLoader Created<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    return train_loader, test_loader, valid_loader, scaler



def fill_missing_values_with_mean(tensor):
    # 找出所有为100的元素的索引
    idx_100 = (tensor == 100)

    # 计算每个节点（沿seq维度）的平均值，除去值为100的
    mean_except_100 = (tensor * (~idx_100)).sum(dim=1) / (~idx_100).sum(dim=1)

    # 将100替换为各节点的平均值
    # 为此我们需要扩展mean_except_100使其维度与tensor匹配
    mean_except_100 = mean_except_100.unsqueeze(1).expand_as(tensor)
    tensor[idx_100] = mean_except_100[idx_100]

    return tensor


def rank_k_approx(x, k):
    # Compute SVD using PyTorch
    u, s, vh = torch.svd(x)
    # Keep only the first k singular values
    s = torch.diag(s[:k])
    return u[:, :k] @ s @ vh[:k, :]

def fill_missing_values_with_svd(x, k=1, num_iters=10):
    # Identify missing values (marked as 100)
    missing_mask = x == 100
    # Replace missing values with NaN for calculation
    x = x.masked_fill(missing_mask, float('nan'))

    # Fill missing values with the mean of each column
    means = torch.nanmean(x, dim=0, keepdim=True)
    x_filled = x.clone()
    
    # Ensure means are broadcast correctly over the missing values
    for col in range(x.shape[1]):
        col_missing_mask = missing_mask[:, col]
        x_filled[col_missing_mask, col] = means[:, col]

    for i in range(num_iters):
        # Compute the rank-k approximation
        x_approx = rank_k_approx(x_filled, k)
        # Only update the missing values
        x_filled = torch.where(missing_mask, x_approx, x_filled)

    return x_filled