import torch
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from main import WK_NUM

# WK_NUM = 16

class Knob:
    def __init__(self, knobs, internal_metrics, external_metrics, opt):
        """
            This class includes knobs, internal metrics, external metrics, target workload info and scaler of data
        """
        self.knobs = knobs
        self.internal_metrics = internal_metrics
        self.external_metrics = external_metrics
        self.target_wk = opt.target
        self.dbms = opt.dbms
        self.external_metrics_size = self.external_metrics[self.target_wk].shape[-1]
        self.columns = self.knobs.columns
        self.DEFAULT_EM_PATH = '/home/jieun/DBMOO/data/rocksdb/external/default_external.csv'
        self.default_trg_em = self.get_trg_default()
        self.get_range()

    def split_data(self): # s_wk is similar workload with target workload
        self.s_internal_metrics = self.internal_metrics[self.target_wk]
        self.s_external_metrics = self.external_metrics[self.target_wk]
        self.X_tr, self.X_te, self.im_tr, self.im_te, self.em_tr, self.em_te = \
            train_test_split(self.knobs, self.s_internal_metrics, self.s_external_metrics, test_size=0.2, random_state=22)

    def scale_data(self):
        self.scaler_X = MinMaxScaler().fit(self.X_tr)
        self.scaler_im = MinMaxScaler().fit(self.im_tr) # [0, 1]
        self.scaler_em = MinMaxScaler().fit(self.em_tr)
        
        self.norm_X_tr = torch.Tensor(self.scaler_X.transform(self.X_tr)).cuda()
        self.norm_X_te = torch.Tensor(self.scaler_X.transform(self.X_te)).cuda()
        self.norm_im_tr = torch.Tensor(self.scaler_im.transform(self.im_tr)).cuda()
        self.norm_im_te = torch.Tensor(self.scaler_im.transform(self.im_te)).cuda()
        self.norm_em_tr = torch.Tensor(self.scaler_em.transform(self.em_tr)).cuda()
        self.norm_em_te = torch.Tensor(self.scaler_em.transform(self.em_te)).cuda()
        
        # self.default_trg_em = self.scaler_em.transform([self.default_trg_em])[0]
        if self.dbms == 'rocksdb':
            self.default_trg_em = self.scaler_em.transform(self.default_trg_em)

    def get_trg_default(self):
        '''
            To get default results on target workload, self.target_wk
        '''
        default_em = pd.read_csv(self.DEFAULT_EM_PATH,index_col=0)
        # default_em = default_em.to_numpy()
        return default_em.iloc[self.target_wk:self.target_wk+1, :] # [time, rate, waf, sa]

    def get_range(self):
        self.lower_boundary = np.array(self.knobs.astype(float).min())
        self.upper_boundary = np.array(self.knobs.astype(float).max())