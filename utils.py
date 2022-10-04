import datetime
import os, logging
import numpy as np
import pandas as pd

import torch
from pytorch_tabnet.tab_model import TabNetRegressor

from torch.utils.data import TensorDataset, DataLoader


def get_filename(PATH, head, tail):
    i = 0
    today = datetime.datetime.now()
    today = today.strftime('%Y%m%d')
    if not os.path.exists(os.path.join(PATH, today)):
        os.mkdir(os.path.join(PATH, today))
    name = today+'/'+head+'-'+today+'-'+'%02d'%i+tail
    while os.path.exists(os.path.join(PATH, name)):
        i += 1
        name = today+'/'+head+'-'+today+'-'+'%02d'%i+tail
    return name

def get_logger(log_path='./logs'):

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    logger = logging.getLogger()
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter('%(asctime)s[%(levelname)s] %(filename)s:%(lineno)s  %(message)s', date_format)
    name = get_filename(log_path, 'log', '.log')
    
    fileHandler = logging.FileHandler(os.path.join(log_path, name))
    streamHandler = logging.StreamHandler()
    
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)
    
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    
    logger.setLevel(logging.INFO)
    logger.info('Writing logs at {}'.format(os.path.join(log_path, name)))
    return logger, os.path.join(log_path, name)

def rocksdb_knobs_make_dict(knobs_path):
    '''
        input: DataFrame form (samples_num, knobs_num)
        output: Dictionary form 
            ex. dict_knobs = {'columnlabels'=array([['knobs_1', 'knobs_2', ...],['knobs_1', 'knobs_2', ...], ...]),
                                'rowlabels'=array([1, 2, ...]),
                                'data'=array([[1,2,3,...], [2,3,4,...], ...[]])}

        For mode selection knob, "yes" -> 1 , "no" -> 0
    '''
    config_files = os.listdir(knobs_path)

    dict_data = {}
    datas = []
    columns = []
    rowlabels = []

    compression_type = ["snappy", "none", "lz4", "zlib"]
    cache_index_and_filter_blocks = ["false", "true"]

    for m in range(len(config_files)):
        flag = 0
        config_datas = []
        config_columns = []
        knob_path = os.path.join(knobs_path, 'config'+str(m+1)+'.cnf')
        f = open(knob_path, 'r')
        config_file = f.readlines()
        knobs_list = config_file[1:-1]
        cmp_type = 0
        
        for l in knobs_list:
            col, _, d = l.split()
            if d in compression_type:
                if d == "none":
                    cmp_type = 1
                d = compression_type.index(d)
            elif d in cache_index_and_filter_blocks:
                d = cache_index_and_filter_blocks.index(d)
            if col == "compression_ratio" and cmp_type:
                d = 1
            config_datas.append(d)
            config_columns.append(col)

        datas.append(config_datas)
        columns.append(config_columns)
        rowlabels.append(m+1)

    dict_data['data'] = np.array(datas)
    dict_data['rowlabels'] = np.array(rowlabels)
    dict_data['columnlabels'] = np.array(columns[0])
    return dict_data


## For maml steps
class Sampler():  
    def __init__(self, dataloaders):
        self.wk_num = len(dataloaders)
        self.dataloaders = dataloaders
        self.iterators = self.get_iterators()

    def get_iterators(self):
        iterators = []
        for i in range(self.wk_num):
            # wk = self.wk_num[i]
            iterators.append(iter(self.dataloaders[i]))
        return iterators

    def get_sample(self):
        samples = {}
        for i in range(self.wk_num):
            # wk = self.wk_num[i]            
            samples[i] = next(self.iterators[i])
        return samples

def MAML_dataset(entire_X_tr, entire_y_tr, entire_X_te, entire_y_te, scaler_X, scaler_y, wk, batch_size=1):
    DL_tr = []
    DL_te = []  
    test_X_te = pd.DataFrame()
    test_y_te = pd.DataFrame()  
    for i in range(len(wk)):
        ############## need to edit number of sample for data
        X_tr_ = entire_X_tr.iloc[(16000*wk[i]):16000*(wk[i]+1),:] #train data set for each workload is 16000
        y_tr_ = entire_y_tr.iloc[(16000*wk[i]):16000*(wk[i]+1),:]
        X_te_ = entire_X_te.iloc[(4000*wk[i]):4000*(wk[i]+1),:]   #test data set for each workload is 4000
        y_te_ = entire_y_te.iloc[(4000*wk[i]):4000*(wk[i]+1),:] 
        ############## need to edit number of sample for data

        s_X_tr = torch.Tensor(scaler_X.transform(X_tr_)).cuda()
        s_X_te = torch.Tensor(scaler_X.transform(X_te_)).cuda()
        s_y_tr = torch.Tensor(scaler_y.transform(y_tr_)).cuda()
        s_y_te = torch.Tensor(scaler_y.transform(y_te_)).cuda()

        # s_X_tr = torch.Tensor(X_tr_.values).cuda()
        # s_X_te = torch.Tensor(X_te_.values).cuda()
        # s_y_tr = torch.Tensor(y_tr_.values).cuda()
        # s_y_te = torch.Tensor(y_te_.values).cuda()
        test_X_te = pd.concat((test_X_te, X_te_))
        test_y_te = pd.concat((test_y_te, y_te_))
        

        dataset_tr = TensorDataset(s_X_tr, s_y_tr)
        dataset_te = TensorDataset(s_X_te, s_y_te)
        dataloader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
        dataloader_te = DataLoader(dataset_te, batch_size=batch_size, shuffle=True)
        DL_tr.append(dataloader_tr)
        DL_te.append(dataloader_te)

    s_test_X_te = torch.Tensor(scaler_X.transform(test_X_te)).cuda()
    s_test_y_te = torch.Tensor(scaler_y.transform(test_y_te)).cuda()


def pretrain_dataset(entire_X_tr, entire_y_tr, entire_X_te, entire_y_te, scaler_X, scaler_y, wk, batch_size=1):   # wk : using workload ex): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  
    selected_X_tr = pd.DataFrame()
    selected_y_tr = pd.DataFrame()
    selected_X_te = pd.DataFrame()
    selected_y_te = pd.DataFrame()

    test_X_te = pd.DataFrame()
    test_y_te = pd.DataFrame()  
    for i in range(len(wk)):
        X_tr_ = entire_X_tr.iloc[(16000*wk[i]):16000*(wk[i]+1),:] #train data set for each workload is 16000
        y_tr_ = entire_y_tr.iloc[(16000*wk[i]):16000*(wk[i]+1),:]
        X_te_ = entire_X_te.iloc[(4000*wk[i]):4000*(wk[i]+1),:]   #test data set for each workload is 4000
        y_te_ = entire_y_te.iloc[(4000*wk[i]):4000*(wk[i]+1),:] 
        
        selected_X_tr = pd.concat((selected_X_tr, X_tr_))
        selected_y_tr = pd.concat((selected_y_tr, y_tr_))
        selected_X_te = pd.concat((selected_X_te, X_te_))
        selected_y_te = pd.concat((selected_y_te, y_te_))

        test_X_te = pd.concat((test_X_te, X_te_))
        test_y_te = pd.concat((test_y_te, y_te_))
        
    s_X_tr = torch.Tensor(scaler_X.transform(selected_X_tr)).cuda()
    s_X_te = torch.Tensor(scaler_X.transform(selected_X_te)).cuda()
    s_y_tr = torch.Tensor(scaler_y.transform(selected_y_tr)).cuda()
    s_y_te = torch.Tensor(scaler_y.transform(selected_y_te)).cuda()      

        # dataset_tr = TensorDataset(s_X_tr, s_y_tr)
        # dataset_te = TensorDataset(s_X_te, s_y_te)
        # DL_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
        # DL_te = DataLoader(dataset_te, batch_size=batch_size, shuffle=True)
    
    dataset_tr = TensorDataset(s_X_tr, s_y_tr)
    dataset_te = TensorDataset(s_X_te, s_y_te)
    DL_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
    DL_te = DataLoader(dataset_te, batch_size=batch_size, shuffle=True)

    s_test_X_te = torch.Tensor(scaler_X.transform(test_X_te)).cuda()
    s_test_y_te = torch.Tensor(scaler_y.transform(test_y_te)).cuda()

    return DL_tr, DL_te, s_test_X_te, s_test_y_te

class Tabnet_architecture(TabNetRegressor):
    def __init__(self):
        super(Tabnet_architecture, self).__init__()

    def fit(
        self,
        X_train,
        y_train,
        input_dim=10,
        eval_set=None,
        weights=0,
        batch_size=1024,
        virtual_batch_size=128,
        drop_last=False
    ):
        self.virtual_batch_size = virtual_batch_size
        self.drop_last = drop_last
        self.input_dim = input_dim
        self._stop_training = False

        self.update_fit_params(
            X_train,
            y_train,
            eval_set,
            weights,
        )

        if not hasattr(self, "network"):
            self._set_network()
        self._update_network_params()

class Set_tabnet_network(nn.Module):
    def __init__(self, m, x_train, y_train, x_eval, y_eval):
        super(Set_tabnet_network, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_eval = x_eval
        self.y_eval = y_eval
        self.set_network(m)

    def set_network(self, m):
        m.fit(self.x_train, self.y_train, eval_set=[(self.x_eval, self.y_eval)])
        # m.fit(knobs.norm_X_dict['tr'][0].detach().cpu().numpy(), knobs.norm_em_dict['tr'][0].detach().cpu().numpy(), 
        #                eval_set=[(knobs.norm_X_dict['val'][0].detach().cpu().numpy(), knobs.norm_em_dict['val'][0].detach().cpu().numpy())])
        self.model = m.network

    def forward(self, x):
        return self.model(x)