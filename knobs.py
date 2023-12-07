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
        pass

    def split_data(self):
        pass
    
    def scale_data(self):
        pass
    
    def _normal_split_data(self):
        pass

    def _maml_split_data(self):
        pass
 
    def _normal_scale_data(self):
        pass

    def _maml_scale_data(self):
        pass

    def get_trg_default(self):
        '''
            To get default results on target workload, self.target
        '''
        default_em = pd.read_csv(self.DEFAULT_EM_PATH,index_col=0)
        return default_em.iloc[self.opt.target:self.opt.target+1, :] # [time, rate, waf, sa]

    def get_range(self):
        pass
