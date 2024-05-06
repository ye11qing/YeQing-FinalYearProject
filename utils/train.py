import time
import torch
from tqdm import tqdm
import torch.nn as nn
import plotly.graph_objects as go
import numpy as np
import time
import torch.nn as nn
from tqdm import tqdm


def train(model, args, train_loader, scaler):
    start_time = time.time()  # 计算开始时间
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    epochs = args.epochs
    model.train()  # 训练模式
    results_loss = []
    
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        for seq, seq_filled, labels_filled in train_loader:
            optimizer.zero_grad()
            input = seq_filled if args.fill else seq
            y_pred = model(input)
            if args.feature == 'MS':
                y_pred = y_pred[:, :, -1].unsqueeze(2)
            mask = labels_filled != 100
            y_pred_masked = y_pred[mask]
            labels_masked = labels_filled[mask]
            loss = loss_function(y_pred_masked, labels_masked)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        results_loss.append(epoch_loss / len(train_loader))
    
    # 训练完成后保存模型
    torch.save(model.state_dict(), f"trained_models/{args.model}_{args.window_size}_{args.pre_len}.pth")
    
    print(f">>>>>>>>>>>>>>>>>>>>>>模型已保存,用时:{(time.time() - start_time) / 60:.4f} min<<<<<<<<<<<<<<<<<<")

    plot_loss_data(results_loss)

def plot_loss_data(data):
    # Draw the curve by plotly
    fig = go.Figure()

    # Change the color of the line to gold
    fig.add_trace(go.Scatter(y=data, mode='lines+markers', name='Loss', line=dict(color='gold',width = 5)))

    # Find the minimum point
    min_val = np.min(data)
    min_idx = np.argmin(data)

    # Add a marker for the minimum point
    fig.add_trace(go.Scatter(x=[min_idx], y=[min_val], mode='markers', 
                             marker=dict(color='purple', size=10), 
                             text=f"Min Loss: {min_val} at {min_idx}", 
                             name='Min Loss'))

    # Add title
    fig.update_layout(title='Loss Results Plot',
                      xaxis=dict(showgrid=True, gridwidth=1, gridcolor='Purple'),
                      yaxis=dict(showgrid=True, gridwidth=1, gridcolor='Purple'))

    fig.show()