import torch
import pandas as pd
import plotly.graph_objects as go
def predict(model, args, device, scaler):
    # 预测未知数据的功能
    df = pd.read_csv(args.data_path)
    df = df.iloc[:, 1:][-args.window_size:].values  # 转换为nadarry
    pre_data = scaler.transform(df)
    tensor_pred = torch.FloatTensor(pre_data).to(device)
    tensor_pred = tensor_pred.unsqueeze(0)  # 单次预测 , 滚动预测功能暂未开发后期补上
    model = model
    model.load_state_dict(torch.load(f"trained_models/{args.model}_{args.window_size}_{args.pre_len}.pth"))
    model.eval()  # 评估模式

    x = tensor_pred

    pred = model(x)[:, :, -1]
    pred = torch.squeeze(pred)
    
 
    pred = scaler.inverse_transform(pred.detach().cpu().numpy())
 
    # 假设 df 和 pred 是你的历史和预测数据
 
    # 计算历史数据的长度
    history_length = len(df[:, -1])
 
    # 为历史数据生成x轴坐标
    history_x = range(history_length)
 
    # 为预测数据生成x轴坐标
    # 开始于历史数据的最后一个点的x坐标
    prediction_x = range(history_length, history_length + len(pred))
 
    history_length = len(df[:, -1])
    history_x = list(range(history_length))
    prediction_x = list(range(history_length, history_length + len(pred)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_x, y=df[:, -1], mode='lines', name='History'))
    fig.add_trace(go.Scatter(x=prediction_x, y=pred, mode='lines+markers', name='Prediction'))

    fig.add_shape(type="line",
                  x0=history_length - 1, y0=df[-1, -1],
                  x1=history_length, y1=pred[0],
                  yref='y', xref='x',
                  line=dict(color="Red"))
    fig.add_shape(type="line",
                  x0=history_length - 1, y0=0,
                  x1=history_length - 1, y1=1,
                  yref='paper', xref='x',
                  line=dict(color="Red", width=3))

    fig.update_layout(title='History and Prediction')
    fig.show()