import datetime
import os, logging
import numpy as np
import pandas as pd
import configparser

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