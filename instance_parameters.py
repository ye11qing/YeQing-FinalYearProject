
class Predict:
    def __init__(self):
        # model parameters
        self.model = 'GRU'
        self.input_size = 7
        self.feature = 'M'
        self.output_size = self.input_size
        self.use_gpu = False
        self.device = 0

        # train parameters
        self.m = 0.5
        self.lr = 0.001
        self.epochs = 100
        self.batch_size = 64
        self.hidden_size = 96
        self.train = True
        self.test = True
        self.predict = True
        self.fill = True

        # data process parameters
        self.data_path = 'dataset/ETTh1'
        self.adj = 'dataset/sz_adj.csv'
        self.target = 'OT'
        self.window_size = 64
        self.pre_len = 4
        self.shuffle = False

        # others
        self.inspect_fit = True