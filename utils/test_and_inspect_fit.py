import torch
from sklearn.metrics import r2_score
import numpy as np
import plotly.graph_objects as go

# Author: Ye Qing
# Affiliation: National University of Singapore, Suzhou Research Institute

def test(model, args, test_loader, scaler):
    """Evaluate the model on the test set and plot results."""
    mae_list = []
    rmse_list = []
    model = model
    model.load_state_dict(torch.load( f"trained_models/{args.model}_{args.window_size}_{args.pre_len}.pth"))
    model.eval()  # 评估模式
    results = []
    labels = []
    for seq, seq_filled, labels_filled in test_loader:
        if args.fill == True:
            input = seq_filled
        else:
            input = seq
        pred = model(input)
        if args.feature == 'MS':
            pred = pred[:, :, -1]
            labels_filled = labels_filled[:, :, -1]
        mask = labels_filled != 100
        pred_masked = pred[mask]
        labels_masked = labels_filled[mask]
        mae = calculate_mae(pred_masked.detach().cpu().numpy(),
                            np.array(labels_masked.detach().cpu()))
        rmse = calculate_rmse(pred_masked.detach().cpu().numpy(),
                            np.array(labels_masked.detach().cpu()))
        if args.feature == 'M':
            pred = pred[:, :, -1]
            labels_filled = labels_filled[:, :, -1]
        pred = pred[:, 0]
        labels_filled = labels_filled[:, 0]
        mask = labels_filled != 100
        pred = pred[mask]
        labels_filled = labels_filled[mask]
        pred = scaler.inverse_transform(pred.detach().cpu().numpy())
        labels_filled = scaler.inverse_transform(labels_filled.detach().cpu().numpy())
        mae_list.append(mae)
        rmse_list.append(rmse)
        for i in range(len(pred)):
            results.append(pred[i])
            labels.append(labels_filled[i])

    print("Testset Mean Absolute Error(测试集平均绝对误差):", np.mean(mae_list))
    print("Testset Root Mean Squared Error(测试集均方根误差):",np.sqrt(np.mean(np.array(rmse_list)**2)) )
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(y=labels, mode='lines', name='TrueValue', line=dict(color='purple', width=5)))
    fig.add_trace(go.Scatter(y=results, mode='lines', name='Prediction', line=dict(color='gold', width=5)))

    fig.update_layout(title='Test State')

    fig.show()
    print("Target Feature Testset R2(目标特征测试集拟合曲线决定系数):",r2_score(labels, results))

    return np.mean(mae_list)


# 检验模型拟合情况
def inspect_model_fit(model, args, train_loader, scaler):
    model = model
    model.load_state_dict(torch.load( f"trained_models/{args.model}_{args.window_size}_{args.pre_len}.pth"))
    model.eval()  # 评估模式
    results = []
    labels = []
    for seq, seq_filled, labels_filled in train_loader:
        if args.fill is True:
            input = seq_filled
        else:
            input = seq
        pred = model(input)
        pred = pred[:, 0, -1]
        labels_filled = labels_filled[:, 0, -1]
        mask = labels_filled != 100
        pred = pred[mask]
        label = labels_filled[mask]
        pred = scaler.inverse_transform(pred.detach().cpu().numpy())
        label = scaler.inverse_transform(label.detach().cpu().numpy())
        for i in range(len(pred)):
            results.append(pred[i])
            labels.append(label[i])

    fig = go.Figure()

    fig.add_trace(go.Scatter(y=labels, mode='lines', name='History', line=dict(color='purple',width=5)))
    fig.add_trace(go.Scatter(y=results, mode='lines', name='Prediction', line=dict(color='gold', width=5)))

    fig.update_layout(title='Inspect model fit state')

    fig.show()
    print("Target Feature Trainingset R2(目标特征训练集拟合曲线决定系数):", r2_score(labels, results))

def calculate_mae(y_true, y_pred):
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(y_pred - y_true))
    return mae

def calculate_rmse(y_true, y_pred):
    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    return rmse