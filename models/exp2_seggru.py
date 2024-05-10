import torch
import torch.nn as nn

class SEGGRU(nn.Module):
    def __init__(self, args, device):
        super(SEGGRU, self).__init__()
        self.hidden_size = args.hidden_size
        self.gru_cell = nn.GRUCell(args.input_size, args.hidden_size)
        self.args = args
        self.device = device
        self.pre_len = args.pre_len
        self.fc1 = nn.Linear(args.window_size, int(args.window_size * args.m))
        self.fc2 = nn.Linear(args.hidden_size, args.input_size)

    def forward(self, x, h=None):
        # x: (batch_size, seq_len, input_size)
        batch_size, seq_len, input_size = x.shape
        x = x.transpose(1, 2)  # x: (batch_size, input_size, seq_len)
        x = self.fc1(x)  # x: (batch_size, input_size, new_feature_size)
        x = x.transpose(1, 2)  # x: (batch_size, new_feature_size, input_size)

        # New shape after fc1 adjustment
        batch_size, new_seq_len, new_input_size = x.shape
        
        if h is None:
            h = torch.zeros((batch_size, self.hidden_size), device=self.device)  # h: (batch_size, hidden_size)
        
        h_seq = []  # 创建一个空的列表来保存每一步的h
        for i in range(new_seq_len):
            h = self.gru_cell(x[:, i, :], h)  # h: (batch_size, hidden_size)
            h_seq.append(h.unsqueeze(1))  # (batch_size, 1, hidden_size)

        h_seq = torch.cat(h_seq, dim=1)  # h_seq: (batch_size, new_seq_len, hidden_size)
        h_seq = h_seq[:, -self.pre_len:, :]  # h_seq: (batch_size, pre_len, hidden_size)

        new_h_seq = []
        for i in range(self.pre_len):
            h = self.gru_cell(self.fc2(h_seq[:, i, :]), h_seq[:, i, :])  # h: (batch_size, hidden_size)
            new_h_seq.append(h.unsqueeze(1))  # (batch_size, 1, hidden_size)

        final_h_seq = torch.cat(new_h_seq, dim=1)  # final_h_seq: (batch_size, pre_len, hidden_size)
        final_h_seq = self.fc2(final_h_seq)

        return final_h_seq
