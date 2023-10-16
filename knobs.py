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
        pass

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
        pass

    def get_range(self):
        pass
