import datetime
import os, logging
import numpy as np
import pandas as pd
import configparser
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

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

def rocksdb_knob_dataframe(_, knobs_path, target=False):
    if target:
        return pd.read_csv(f'{knobs_path}/configs_{_}.csv', index_col=0)
    
    config_files = os.listdir(knobs_path)

    datas = []
    columns = []
    rowlabels = []

    compression_type = ["snappy", "none", "lz4", "zlib"]
    cache_index_and_filter_blocks = ["false", "true"]
    open_files = ['-1', '10000', '100000', '1000000']
    
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
            if col == 'open_files':
                d = open_files.index(d)
            config_datas.append(d)
            config_columns.append(col)

        datas.append(config_datas)
        columns.append(config_columns)
        rowlabels.append(m+1)

    pd_knobs = pd.DataFrame(data=np.array(datas).astype(np.float32), columns=np.array(columns[0]))
    return pd_knobs

def rocksdb_metrics_dataframe(wk, pruned_im, internal_path, external_path):
    internal = pd.read_csv(os.path.join(internal_path, f'internal_results_{wk}.csv'), index_col=0)
    internal = internal[pruned_im.columns]
    external = pd.read_csv(os.path.join(external_path, f'external_results_{wk}.csv'), index_col=0)
    return internal, external

def PCC(true, pred):
    pcc_res = np.zeros(4)
    for idx in range(true.shape[-1]):
        res, _ = pearsonr(true[:,idx], pred[:,idx])  
        pcc_res[idx] = res
    return np.average(pcc_res)

def RMSE(true, pred):
    return mean_squared_error(true, pred)**0.5

def MSE(true, pred):
    return mean_squared_error(true, pred)

class Sampler():  
    def __init__(self, dataloaders):
        self.wk_num = len(dataloaders)
        self.dataloaders = dataloaders
        self.iterators = self.get_iterators()

    def get_iterators(self):
        return

    def get_sample(self):
        return