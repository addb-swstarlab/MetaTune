import datetime
import os, logging
import numpy as np
import pandas as pd
import configparser

import torch
import torch.nn as nn
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

def mysql_knob_dataframe(wk, knobs_path):
    knobs_path = os.path.join(knobs_path, str(wk))
    config_len = len(os.listdir(knobs_path))
    cnf_parser = configparser.ConfigParser()
    pd_mysql = pd.DataFrame()
    for idx in range(config_len):
        cnf_parser.read(os.path.join(knobs_path, f'my_{idx}.cnf'))
        conf_dict = cnf_parser._sections['mysqld']
        tmp = pd.DataFrame(data=[conf_dict.values()], columns=conf_dict.keys())
        pd_mysql = pd.concat([pd_mysql, tmp])
        
    pd_mysql = pd_mysql.reset_index(drop=True)
    pd_mysql = pd_mysql.drop(columns=['log-error', 'bind-address'])
    return pd_mysql

def mysql_metrics_dataframe(wk, internal_path, external_path):
    internal = pd.read_csv(os.path.join(internal_path, f'internal_results_{wk}.csv'), index_col=0)
    ## Drop oolumns contained unique data
    unique_data_column = []
    for col in internal.columns:
        if len(pd.value_counts(internal[col])) == 1:
            unique_data_column.append(col)
    internal = internal.drop(columns=unique_data_column)
    
    external = pd.read_csv(os.path.join(external_path, f'external_results_{wk}.csv'), index_col=0)
    latency_columns = []
    for col in external.columns:
        if col.find("latency") == 0 and col != 'latency_max' and col != 'latency_CLEANUP':
            latency_columns.append(col)
    external_ = external[['tps']].copy()
    external_['latency'] = external[latency_columns].max(axis=1)
    return internal, external_


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
            # iterators.append(iter(self.dataloaders[i]))
            d_iter = self.dataloaders[i].__iter__()
            iterators.append(d_iter)
        return iterators

    def get_sample(self):
        samples = {}
        for i in range(self.wk_num):
            # wk = self.wk_num[i]            
            samples[i] = next(self.iterators[i])
        return samples

# for train tabnet with wmaml
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

# for train tabnet with wmaml
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

def make_wmaml_dataloader(knobs, opt):
    """ make dataset"""
    ## wmaml dataset
    X_target_tr = knobs.norm_X_dict['tr'][opt.target]
    y_target_tr = knobs.norm_em_dict['tr'][opt.target]
    X_target_val = knobs.norm_X_dict['val'][opt.target]
    y_target_val = knobs.norm_em_dict['val'][opt.target]
    eval_set = [(X_target_val, y_target_val)]

    X_maml_tr = knobs.norm_X_dict['tr'].copy()
    del(X_maml_tr[opt.target])
    y_maml_tr = knobs.norm_em_dict['tr'].copy()
    del(y_maml_tr[opt.target])

    X_maml_val = knobs.norm_X_dict['val'].copy()
    del(X_maml_val[opt.target])
    y_maml_val = knobs.norm_em_dict['val'].copy()
    del(y_maml_val[opt.target])

    X_maml_te = knobs.norm_X_dict['te'].copy()
    del(X_maml_te[opt.target])
    y_maml_te = knobs.norm_em_dict['te'].copy()
    del(y_maml_te[opt.target])

    eval_set_maml = []
    for i in range(len(X_maml_val)):
        X_val_ = X_maml_val[i]
        y_val_ = y_maml_val[i]
        eval_set_maml.append([(X_val_, y_val_)])

    ## adaptation dataset
    X_target_tr = knobs.norm_X_dict['tr'][opt.target]
    y_target_tr = knobs.norm_em_dict['tr'][opt.target]
    X_target_val = knobs.norm_X_dict['val'][opt.target]
    y_target_val = knobs.norm_em_dict['val'][opt.target]
    X_target_te = knobs.norm_X_dict['te'][opt.target]
    y_target_te = knobs.norm_em_dict['te'][opt.target]

    """ make dataloader """
    ## wmaml dataloader
    maml_dl_tr = []
    maml_dl_val = []
    maml_dl_te = []

    for i in range(len(X_maml_tr)):
        dataset_tr = TensorDataset(X_maml_tr[i], y_maml_tr[i])
        dataset_val = TensorDataset(X_maml_val[i], y_maml_val[i])
        dataset_te = TensorDataset(X_maml_te[i], y_maml_te[i])

        dataloader_tr = DataLoader(dataset_tr, batch_size=opt.batch_size, shuffle=True)
        dataloader_val = DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=True)
        dataloader_te = DataLoader(dataset_te, batch_size=opt.batch_size, shuffle=True)

        maml_dl_tr.append(dataloader_tr)
        maml_dl_val.append(dataloader_val)
        maml_dl_te.append(dataloader_te)

    ## adaptation dataloader
    dataset_tr = TensorDataset(X_maml_tr[i], y_maml_tr[i])
    dataset_val = TensorDataset(X_maml_val[i], y_maml_val[i])
    dataset_te = TensorDataset(X_maml_te[i], y_maml_te[i])

    dataloader_tr = DataLoader(dataset_tr, batch_size=opt.batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=True)
    dataloader_te = DataLoader(dataset_te, batch_size=opt.batch_size, shuffle=True)

    maml_dl_tr.append(dataloader_tr)
    maml_dl_val.append(dataloader_val)
    maml_dl_te.append(dataloader_te)

    

    return maml_dl_tr, maml_dl_val, maml_dl_te