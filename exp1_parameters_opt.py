from utils.data_process import create_dataloader
from instance_parameters import Predict
from models.exp1_gru import GRU
from models.exp1_tgcn import TGCN, load_adjacency_matrix
from utils.predict import predict
from utils.test_and_inspect_fit import test, inspect_model_fit
from utils.train import train
import torch
import optuna
import torch.nn as nn
import os

def objective(trial):
    # 提议超参数
    h_epochs = trial.suggest_int('epochs', 4, 128)
    h_hidden_size = trial.suggest_int('hidden_size', 4, 32)
    h_window_size = trial.suggest_int('window_size', 64, 128)
    h_batch_size = trial.suggest_int('batch_size', 48, 72)
    h_pre_len = trial.suggest_int('pre_len', 1, 16)
    model_type = trial.suggest_categorical('model_type', ['GRU', 'TGCN'])
    fill_type = trial.suggest_categorical('fill_type', [True, False])
    
    # 创建和训练模型
    instance = Predict()
    instance.model = model_type
    instance.input_size = 50
    instance.hidden_size = h_hidden_size
    instance.data_path = 'dataset/sz_speed.csv'
    instance.target = '90224'
    instance.feature = 'M'
    instance.window_size = h_window_size
    instance.pre_len = h_pre_len
    instance.batch_size = h_batch_size
    instance.epochs = h_epochs
    instance.fill = fill_type
    instance.adj = 'dataset/sz_adj.csv'

    device = torch.device("cpu")
    train_loader, test_loader, _ , scaler = create_dataloader(instance, device)
    if instance.model == 'GRU':
        model = GRU(input_dim = instance.input_size, hidden_size = instance.hidden_size, args = instance)
    elif instance.model == 'TGCN':
        adj = load_adjacency_matrix(instance.adj)
        model = TGCN(adj = adj, hidden_size = instance.hidden_size ,args = instance)
    if instance.train:
        train(model, instance, train_loader, scaler)
    
    # 计算并返回验证集上的损失
    val_loss = test(model, instance, test_loader, scaler)
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials = 48)

# 可视化
fig_importances = optuna.visualization.plot_param_importances(study)
fig_importances.show()
fig_paralled = optuna.visualization.plot_parallel_coordinate(study)
fig_paralled.show()
fig_slice = optuna.visualization.plot_slice(study)
fig_slice.show()
fig_history = optuna.visualization.plot_optimization_history(study)
fig_history.show()


# save plots
plot_folder = 'hyperparameters_plot'
os.makedirs(plot_folder, exist_ok=True)
fig_importances.savefig(os.path.join(plot_folder, 'param_importances.png'))
fig_slice.savefig(os.path.join(plot_folder, 'param_slice.png'))
fig_history.savefig(os.path.join(plot_folder, 'optimization_history.png'))
fig_paralled.savefig(os.path.join(plot_folder, 'parallel_coordinate.png'))

fig_contour = optuna.visualization.plot_contour(study, params=['pre_len', "window_size"])
fig_contour.show()

fig_contour2 = optuna.visualization.plot_contour(study, params=['pre_len', "model_type"])
fig_contour2.show()

fig_contour3 = optuna.visualization.plot_contour(study, params=['fill_type', "model_type"])
fig_contour3.show()