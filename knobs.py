import torch
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial import distance
import logging

class Knob:
    def __init__(self, knobs, internal_metrics, external_metrics, target, opt, similar_wk=None):
        """
            This class includes knobs, internal metrics, external metrics, target workload info and scaler of data
        """
        self.opt = opt
        self.knobs = knobs
        self.internal_metrics = internal_metrics
        self.external_metrics = external_metrics
        self.target = target
        self.similar_wk = similar_wk
        self.columns = self.knobs[0].columns
        self.train_type = opt.train_type
        self.d_threshold = opt.d_threshold
        self.DEFAULT_EM_PATH = 'data/rocksdb/external/default_external.csv'
        self.default_trg_em = self.get_trg_default()
        self.get_range()
        
            
    def _calculate_mahalanobis_distance(self):
        external_metrics_ = self.external_metrics.copy()
        if self.target+1 in self.external_metrics.keys():
            del external_metrics_[self.target+1]    # delete test dataset of adaptation step
            
        data_u = external_metrics_[self.target]
        self.ds = []
        for wk in external_metrics_.keys():
            data_v = external_metrics_[wk]
            sum_d = 0
            for conf_idx in range(len(data_u)):
                d = distance.mahalanobis(u=data_u.iloc[conf_idx],
                                        v=data_v.mean(),
                                        VI=np.linalg.pinv(data_v.cov()))
                sum_d += d
            self.ds.append(sum_d)
        self.div_ds = self.ds / np.min(self.ds) # divide distances by a distance of target workload
        
        logging.info("====calculate mahalanobis distance====")
        for i, wk in enumerate(external_metrics_.keys()):
            logging.info(f'{wk:2}th divided distance : {self.ds[i]}')
        
        logging.info("====calculate div-mahalanobis distance====")
        for i, wk in enumerate(external_metrics_.keys()):
            logging.info(f'{wk:2}th divided distance : {self.div_ds[i]}')

    def _select_training_workload(self):
        logging.info("====select workloads to replace target samples===")
        trg_except_div_ds = self.div_ds[:-1]
        self.avail_wk_idx = np.where(trg_except_div_ds<=self.d_threshold)[0] # except target workload data
        
        if self.avail_wk_idx.size == 0:
            self.avail_wk_idx = [np.argmin(trg_except_div_ds)]
            logging.info(f"there is no similar workloads whose distance are lower than {self.d_threshold}")
        logging.info(f"use {self.avail_wk_idx} workloads")
        
        if self.opt.train_type == 'general':    # concat available workload data
            knob_data = []
            ex_data = []
            for i in range(len(self.avail_wk_idx)):
                knob_data.append(self.knobs[self.avail_wk_idx[i]])
                ex_data.append(self.external_metrics[self.avail_wk_idx[i]])
                
            self.concated_X = pd.concat(knob_data)
            self.concated_y = pd.concat(ex_data)
            self.concated_X = self.concated_X.reset_index(drop=True)  
            self.concated_y = self.concated_y.reset_index(drop=True)  
            
        self.selected_X = dict((_, self.knobs[_]) for _ in self.avail_wk_idx)
        self.selected_y = dict((_, self.external_metrics[_]) for _ in self.avail_wk_idx)

    def split_data(self):
        if self.train_type == 'general' or self.train_type == 'replace':
            if self.train_type == 'general': 
                self._calculate_mahalanobis_distance()
                self._select_training_workload()
            self._normal_split_data()
            
        else:
            self._calculate_mahalanobis_distance()
            self._select_training_workload()
            self._maml_split_data()
    
    def scale_data(self):
        if self.train_type == 'general' or self.train_type == 'replace':
            self._normal_scale_data()
        else:
            self._maml_scale_data()
    
    def _normal_split_data(self):
        pass

    def _maml_split_data(self):
        self.X_tr_dict, self.X_val_dict, self.X_te_dict, self.X_tmp_dict = {}, {}, {}, {}
        self.y_tr_dict, self.y_val_dict, self.y_te_dict, self.y_tmp_dict = {}, {}, {}, {}
        
        # train, valid, test data for repository workload data
        for i in self.selected_X.keys():
            self.X_tmp_dict[i], self.X_te_dict[i], self.y_tmp_dict[i], self.y_te_dict[i] = \
                train_test_split(self.selected_X[i], self.selected_y[i], test_size=0.1)
            self.X_tr_dict[i], self.X_val_dict[i], self.y_tr_dict[i], self.y_val_dict[i] = \
                train_test_split(self.X_tmp_dict[i], self.y_tmp_dict[i], test_size=0.5)   
 
    def _normal_scale_data(self):
        self.scaler_X = MinMaxScaler().fit(self.X_tr)
        self.scaler_y = MinMaxScaler().fit(self.y_tr)
        
        self.norm_X_tr = torch.Tensor(self.scaler_X.transform(self.X_tr)).cuda()
        self.norm_X_te = torch.Tensor(self.scaler_X.transform(self.X_te)).cuda()
        self.norm_y_tr = torch.Tensor(self.scaler_y.transform(self.y_tr)).cuda()
        self.norm_y_te = torch.Tensor(self.scaler_y.transform(self.y_te)).cuda()
        
        self.target_norm_X_te = torch.Tensor(self.scaler_X.transform(self.knobs[self.target+1])).cuda()
        self.target_norm_y_te = torch.Tensor(self.scaler_y.transform(self.groud_truth)).cuda()

    def _maml_scale_data(self):
        self.scaler_X = MinMaxScaler().fit(self.X_tr)
        self.scaler_y = MinMaxScaler().fit(self.y_tr)
        
        self.norm_X_tr = torch.Tensor(self.scaler_X.transform(self.X_tr)).cuda()
        self.norm_X_te = torch.Tensor(self.scaler_X.transform(self.X_te)).cuda()
        self.norm_y_tr = torch.Tensor(self.scaler_y.transform(self.y_tr)).cuda()
        self.norm_y_te = torch.Tensor(self.scaler_y.transform(self.y_te)).cuda()
        
        self.target_norm_X_te = torch.Tensor(self.scaler_X.transform(self.knobs[self.target+1])).cuda()
        self.target_norm_y_te = torch.Tensor(self.scaler_y.transform(self.groud_truth)).cuda()  

    def get_trg_default(self):
        '''
            To get default results on target workload, self.target
        '''
        default_em = pd.read_csv(self.DEFAULT_EM_PATH,index_col=0)
        return default_em.iloc[self.opt.target:self.opt.target+1, :] # [time, rate, waf, sa]

    def get_range(self):
        entire_knobs = pd.concat(self.knobs)
        self.lower_boundary = np.array(entire_knobs.astype(float).min())
        self.upper_boundary = np.array(entire_knobs.astype(float).max())
