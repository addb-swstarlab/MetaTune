import os
import numpy as np
import pandas as pd
from datetime import datetime
# from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader
from main import WK_NUM
from network import RocksDBDataset, SingleNet
from train import train, valid
from utils import get_filename
import rocksdb_option as option
from scipy.stats import gmean
from sklearn.ensemble import RandomForestRegressor
from ga import RocksDBSingleProblem, RocksDBMultiProblem, genetic_algorithm

from models.wmaml import *

# def euclidean_distance(a, b):
#     res = a - b
#     res = res ** 2
#     res = np.sqrt(res)
#     return np.average(res)

# def get_euclidean_distance(internal_dict, logger, opt):
#     scaler = MinMaxScaler().fit(pd.concat(internal_dict))
#     target_config_idx = [1004,1214,1312,2465,3306,3333,4569,4801,5124,5389,8490,9131,9143,11896,12065,12293,12491,13098,18088,19052]
    
#     trg = opt.target
#     if trg > 15:
#         trg = 16
    
#     wk = []
#     # for im_d in internal_dict:
#     #     wk.append(scaler.transform(internal_dict[im_d].iloc[:opt.target_size, :]))
#     for im_d in internal_dict:
#         if im_d == 16:
#             wk.append(scaler.transform(internal_dict[im_d]))
#         else:
#             wk.append(scaler.transform(internal_dict[im_d].iloc[target_config_idx, :]))

#     big = 100
#     for i in range(len(wk)):
#         ed = euclidean_distance(wk[trg], wk[i])
#         if ed<big and trg != i: 
#             big=ed
#             idx = i
#         logger.info(f'{i:4}th   {ed:.5f}')
#     logger.info(f'best similar workload is {idx}th')

#     return idx

def train_fitness_function(knobs, logger, opt):
    if opt.mode == 'RF':
        rf = RandomForestRegressor(max_depth=2, random_state=0)
        rf.fit(knobs.norm_X_tr.cpu().detach().numpy(), knobs.norm_em_tr.cpu().detach().numpy())
        return rf, rf.predict(knobs.norm_X_te.cpu().detach().numpy())

    # elif opt.mode == 'dnn': tmp_model = SingleNet(input_dim=knobs.norm_X_tr.shape[1], hidden_dim=16, output_dim=knobs.norm_em_tr.shape[-1]).cuda()
    # elif opt.mode == 'tabnet': tmp_model = TabNetRegressor()

    if opt.train_type == 'wmaml': 

        logger.info(f"[Train MODE] 1st step of train model (wmaml)")
        data_mapping = []
        origin_model = TabNetRegressor()    # if opt.mode == 'dnn': origin_model = SingleNet(@@@) ?
        wmaml = MAML_one_batch(origin_model, 
                               knobs.norm_X_dict, knobs.norm_im_dict, knobs.norm_em_dict, 
                               num_epochs=opt.epochs, inner_lr=opt.inner_lr, meta_lr=opt._lr)
        wmaml.main_loop()
             
        logger.info(f"[Train MODE] 2nd step of train model (adaptation)")
        model = wmaml.model

    ######################### 나중에 wmaml, ranking_step_pretrain, normal if문 정리############################
    # elif opt.train_type == 'ranking_step_pretrain' or opt.train_type == 'rsp':
    #     best_loss = 100
    #     for i in range(WK_NUM): # WK_NUM 개수만큼 반복 (similarity가 낮은 워크로드부터 훈련)           
    #         name = get_filename('model_save', 'model', '.pt')
    #         for epoch in range(opt.epochs):
    #             loss_tr = train(model, loader_tr, opt.lr)
    #             loss_te, outputs = valid(model, loader_te)
            
    #             logger.info(f"[{epoch:02d}/{opt.epochs}] loss_tr: {loss_tr:.8f}\tloss_te:{loss_te:.8f}")

    #             if best_loss > loss_te and epoch>15:
    #                 best_loss = loss_te
    #                 best_model = model
    #                 best_outputs = outputs
    #                 torch.save(best_model, os.path.join('model_save', name))
    #         logger.info(f"loss is {best_loss:.4f}, save model to {os.path.join('model_save', name)}")
                       
    else:
        dataset_tr = RocksDBDataset(knobs.norm_X_tr, knobs.norm_em_tr)
        dataset_te = RocksDBDataset(knobs.norm_X_te, knobs.norm_em_te)


        loader_tr = DataLoader(dataset = dataset_tr, batch_size = opt.batch_size, shuffle=True)
        loader_te = DataLoader(dataset = dataset_te, batch_size = opt.batch_size, shuffle=False)

        model = SingleNet(input_dim=knobs.norm_X_tr.shape[1], hidden_dim=16, output_dim=knobs.norm_em_tr.shape[-1]).cuda()        

        # if opt.train:       
        logger.info(f"[Train MODE] Training Model") 

    ###################################################################################################     
    best_loss = 100
    # name = get_filename('model_save', 'model', '.pt')
    name = get_filename('model_save', opt.train_type, '.pt')
    for epoch in range(opt.epochs):
        loss_tr = train(model, loader_tr, opt.lr)
        loss_te, outputs = valid(model, loader_te)
    
        logger.info(f"[{epoch:02d}/{opt.epochs}] loss_tr: {loss_tr:.8f}\tloss_te:{loss_te:.8f}")

        if best_loss > loss_te and epoch>15:
            best_loss = loss_te
            best_model = model
            best_outputs = outputs
            torch.save(best_model, os.path.join('model_save', name))
    logger.info(f"loss is {best_loss:.4f}, save model to {os.path.join('model_save', name)}")
    
    return best_model, best_outputs
    # elif opt.eval:
    #     logger.info(f"[Eval MODE] Trained Model Loading with path: {opt.model_path}")
    #     model = torch.load(os.path.join('model_save',opt.model_path))
    #     _, outputs = valid(model, loader_te)
    #     return model, outputs

def GA_optimization(knobs, fitness_function, logger, opt):
    if opt.ga == 'GA':
        problem = RocksDBSingleProblem(knobs=knobs, model=fitness_function, model_mode=opt.mode)
    elif opt.ga == 'NSGA2' or opt.ga == 'NSGA3':
        problem = RocksDBMultiProblem(knobs=knobs, model=fitness_function, model_mode=opt.mode)

    res = genetic_algorithm(mode=opt.ga, problem=problem, pop_size=opt.population)
    
    if len(res.X.shape) == 2:
        results = res.X[0] # NSGA2, MOO
        res_F = res.F[0]
    else:
        results = res.X # GA, SOO
        res_F = res.F
    
    recommend_command = ''
    
    for idx, col in enumerate(knobs.columns):                 
        recommend_command = convert_int_to_category(col, recommend_command, results[idx])

    recommend_command = make_dbbench_command(opt.target, recommend_command)
    logger.info(f"db_bench command is  {recommend_command}")
    
    return res_F, recommend_command

def convert_int_to_category(col, cmd, s):
    if col=='compression_type':
            ct = ["snappy", "zlib", "lz4", "none"]
            cmd += f'-{col}={ct[int(s)]} '
    else:
        cmd += f'-{col}={int(s)} '
    return cmd

def make_dbbench_command(trg_wk, rc_cmd):
    wk_info = pd.read_csv('data/rocksdb_workload_info.csv', index_col=0)
    f_wk_info = wk_info.loc[:,:'num']
    b_wk_info = wk_info.loc[:, 'benchmarks':]
    cmd = 'rocksdb/db_bench '   
    f_cmd = " ".join([f'-{_}={int(f_wk_info.loc[trg_wk][_])}' for _ in f_wk_info.columns])
    b_cmd = f"--{b_wk_info.columns[0]}={b_wk_info.loc[trg_wk][0]} "
    b_cmd += f"--statistics "
    if not np.isnan(b_wk_info.loc[trg_wk][1]):
        b_cmd += f"--{b_wk_info.columns[1]}={int(b_wk_info.loc[trg_wk][1])}"

    cmd += f_cmd + " " + rc_cmd + b_cmd + ' > /jieun/result.txt'

    return cmd
