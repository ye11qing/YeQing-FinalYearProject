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
        # h: (1, batch_size, hidden_size)
        batch_size, seq_len, input_size = x.shape
        x = x.transpose(1, 2)
        x = self.fc1(x)
        x = x.transpose(1, 2)
        batch_size, new_seq_len, input_size = x.shape
        
        if h is None:
            h = torch.zeros((batch_size, self.hidden_size), device=self.device)  # h: (batch_size, hidden_size)
        
        h_seq = []  # 创建一个空的列表来保存每一步的h
        for i in range(new_seq_len):
            h = self.gru_cell(x[:, i, :], h)  # h: (batch_size, hidden_size)
            h_seq.append(h.unsqueeze(1))  # (seq_len, 1, hidden_size)

        h_seq = torch.cat(h_seq, dim=1)  # 沿着seq_len维度拼接所有的h
        h_seq = self.fc2(h_seq)
        h_seq = h_seq[:, - self.pre_len:, :]
        return h_seq