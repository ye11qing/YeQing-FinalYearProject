import argparse
import torch
import random

from utils.data_process import create_dataloader
from models.tcn import TemporalConvNet
from models.gru import GRU
from utils.train import train
from utils.test_and_inspect_fit import test, inspect_model_fit
from utils.predict import predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series forecast')
    parser.add_argument('-model', type=str, default='GRU', help="模型持续更新")
    parser.add_argument('-window_size', type=int, default=128, help="时间窗口大小, window_size > pre_len")
    parser.add_argument('-pre_len', type=int, default=4, help="预测未来数据长度")
    # data
    parser.add_argument('-shuffle', action='store_true', default=True, help="是否打乱数据加载器中的数据顺序")
    parser.add_argument('-data_path', type=str, default='dataset/filled_sz_speed.csv', help="你的数据数据地址")
    parser.add_argument('-target', type=str, default='92693', help='你需要预测的特征列,这个值会最后保存在csv文件里')
    parser.add_argument('-input_size', type=int, default=40, help='你的特征个数不算时间那一列')
    parser.add_argument('-feature', type=str, default='MS', help='[M, S, MS],多元预测多元,单元预测单元,多元预测单元')
    parser.add_argument('-model_dim', type=list, default=[64, 128, 256], help='这个地方是这个TCN卷积的关键部分,它代表了TCN的层数我这里输'
                                                                              '入list中包含三个元素那么我的TCN就是三层,这个根据你的数据复杂度来设置'
                                                                              '层数越多对应数据越复杂但是不要超过5层')

    # learning
    parser.add_argument('-lr', type=float, default=0.001, help="学习率")
    parser.add_argument('-drop_out', type=float, default=0.2, help="随机丢弃概率,防止过拟合")
    parser.add_argument('-epochs', type=int, default=20, help="训练轮次")
    parser.add_argument('-batch_size', type=int, default=8, help="批次大小")
    parser.add_argument('-save_path', type=str, default='models')
 
    # model
    parser.add_argument('-hidden_size', type=int, default=128, help="隐藏层单元数")
    parser.add_argument('-kernel_sizes', type=int, default=3)
    parser.add_argument('-laryer_num', type=int, default=1)
    # device
    parser.add_argument('-use_gpu', type=bool, default=False)
    parser.add_argument('-device', type=int, default=0, help="只设置最多支持单个gpu训练")
 
    # option
    parser.add_argument('-train', type=bool, default=True)
    parser.add_argument('-test', type=bool, default=True)
    parser.add_argument('-predict', type=bool, default=True)
    parser.add_argument('-inspect_fit', type=bool, default=True)
    parser.add_argument('-lr-scheduler', type=bool, default=True)

    args = parser.parse_args()
 
    if isinstance(args.device, int) and args.use_gpu:
        device = torch.device("cuda:" + f'{args.device}')
    else:
        device = torch.device("cpu")
    print("使用设备:", device)
    train_loader, test_loader, valid_loader, scaler = create_dataloader(args, device)
 
    if args.feature == 'MS' or args.feature == 'S':
        args.output_size = args.input_size
    else:
        args.output_size = args.input_size
 
    # 实例化模型
    try:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        if args.model == 'TCN':
            model = TemporalConvNet(args.input_size,args.output_size, args.pre_len,args.model_dim, args.kernel_sizes).to(device)
        elif args.model == 'GRU':
            model = GRU(args, device).to(device)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型成功<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    except Exception as e:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型失败<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print(f"错误详情：{e}")

    # 训练模型
    if args.train:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型训练<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        train(model, args, train_loader, scaler, device)
    if args.test:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型测试<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        test(model, args, test_loader, scaler)
    if args.inspect_fit:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始检验{args.model}模型拟合情况<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        inspect_model_fit(model, args, train_loader, scaler)
    if args.predict:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>预测未来{args.pre_len}条数据<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        predict(model, args, device, scaler)