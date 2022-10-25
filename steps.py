import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime
# from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader
from network import RocksDBDataset, SingleNet
from train import train, valid, wmaml_train, wmaml_valid
from utils import (
    get_filename, Tabnet_architecture, 
    Set_tabnet_network, WmamlData,
    )

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
import rocksdb_option as option
from scipy.stats import gmean
from sklearn.ensemble import RandomForestRegressor
# from ga import RocksDBSingleProblem, RocksDBMultiProblem, genetic_algorithm
from ga import DBMSProblem, genetic_algorithm

import logging
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
def get_distance_list(knobs, opt): # uclidean or matalobis distance
    im_statics = {}
    d_list = {}
    drop_columns = ['count', 'min', 'max']
    for i in range(len(knobs.internal_metrics)):
        im_statics[i] = knobs.internal_metrics[0].describe().T.drop(columns=drop_columns).T # (5, 148)
        
    target_im_statics = im_statics[opt.target]
    
    d = distance_function() # calculate distance 

    # Get uclidean distance

    ## Get Convariance from internal metric data
    # cov_internal_metrics = pd.concat(wk_stats_list, axis=1).T.cov() # (5, 148*16) => (148*16, 5) => (5, 5)

# def train_fitness_function(knobs, logger, opt):
def train_fitness_function(knobs, opt):

    if opt.train_type == 'wmaml':
        from learn2learn.algorithms import MAML
        from utils import Tabnet_architecture
        """ calculate weighted workload similarity step"""        
        # w = get_mahalobis_distance(knobs, opt)  # weight of workload similarity len(w) is number of workload

        if opt.mode == 'tabnet':
            logging.info(f"[Make dataset for wmaml]")      
            # maml_dl_tr, maml_dl_val, maml_dl_te = make_wmaml_dataloader(knobs, opt)
            data = WmamlData(knobs, opt)
            data.make_data()
            data.make_dataloader()

            logging.info(f"[Train MODE] Training Model WMAML step start")         
            arch = Tabnet_architecture(knobs.norm_X_dict['tr'][0], knobs.norm_em_dict['tr'][0])
            tabnet_model = arch.set_network()
            # tabnet_model.network = tabnet_model.network.cuda() 
            maml = MAML(tabnet_model, lr=opt.wmaml_in_lr).cuda()
            optimizer = torch.optim.Adam(maml.parameters(), lr=opt.wmaml_lr)
            best_loss = np.inf
            patience = 0           
            name = get_filename('model_save/wmaml', 'model', '.pt')

            for epoch in range(opt.wmaml_epochs):
                loss_tr = wmaml_train(maml, data.maml_dl_tr, data.maml_dl_val, optimizer)
                loss_te, r2_res = wmaml_valid(maml, data.maml_dl_te)
                logging.info(f"[{(epoch+1):02d}/{opt.wmaml_epochs}] loss_tr: {loss_tr:.8f} | loss_te:{loss_te:.8f} | R2 : {r2_res:.8f} | patience : {patience}")

                # update best model
                if best_loss > loss_te:
                    best_loss = loss_te
                    best_model = maml
                    best_epoch = epoch + 1
                    best_model = maml
                    best_r2 = r2_res
                    patience = 0
                    torch.save(best_model.module.state_dict(), os.path.join('model_save/wmaml', name))
                else:
                    patience += 1
                
                if patience >= opt.patience:
                    logging.info(f"Early stopping") 
                    # logging.info(f"Epoch [{best_epoch}/{opt.wmaml_epochs}] : Total_loss : {best_loss:.6f} | R2 : {best_r2:.6f} | patience : {patience}, save model to {os.path.join('model_save/wmaml', name)}")                   
                    break
            logging.info(f"---------------wmaml train step finish---------------") 
            logging.info(f"Epoch [{best_epoch}/{opt.wmaml_epochs}] : Total_loss : {best_loss:.6f} | R2 : {best_r2:.6f} | patience : {patience}, save model to {os.path.join('model_save/wmaml', name)}")
                
            logging.info(f"[Train MODE] Training Model Adaptation step start") 

            model = Tabnet_architecture(knobs.norm_X_dict['tr'][opt.target], knobs.norm_em_dict['tr'][opt.target])
            model.set_network()
            model.network = best_model.module.state_dict()
            model.fit(data.X_target_tr.cpu().detach().numpy(), 
                      data.y_target_tr.cpu().detach().numpy(),
                      eval_set=[(
                                data.X_target_val.cpu().detach().numpy(), 
                                data.y_target_val.cpu().detach().numpy()
                                )],
                      max_epochs=opt.epochs, 
                      patience=opt.patience)
            preds = model.predict(data.X_target_te.cpu().detach().numpy())
            # preds = outputs.cpu().detach().numpy()
            trues = data.y_target_te.cpu().detach().numpy()
            mse_res = mse(trues, preds)
            r2_res = r2_score(trues, preds)
            # mse_res = mse(data.y_target_te, preds)
            # r2_res = r2_score(data.y_target_te, preds)
            logging.info(f"[Train MODE] Training Model Adaptation step finish | MSE : {mse_res:.6f} | R2 : {r2_res:.6f}")  
            
            # global
            # features = [ col for col in data.X_target_te.cpu().detach().numpy().columns]
            features = [ col for col in data.knobs.columns]

            # feat_importances = pd.Series(model.feature_importances_, index=features)    
            # 원핫벡터가 포함된 importance라서 index랑 데이터의 길이가 다르다는 오류
            # ValueError: Length of values (38) does not match length of index (22)
            # model.feature_importances_ # list임
            """ feature_importance값이 0인 index를 제외 --> 선별된 knob들로 ga """

            #local
            explain_matrix, masks = model.explain(data.X_target_te.cpu().detach().numpy())
            masks[3] = masks[0] + masks[1] + masks[2]
            # model.save_model('model_save')
            
            return model, preds           
        
        elif opt.mode == 'dnn':
            """ make dataset ( train, valid, test) """

            """ wmaml-step """

            """ adaptation-step """


    elif opt.train_type == 'rsp':
        pass

    elif opt.train_type == 'normal':

        if opt.mode == 'RF':
            rf = RandomForestRegressor(max_depth=2, random_state=0)
            rf.fit(knobs.norm_X_tr.cpu().detach().numpy(), knobs.norm_em_tr.cpu().detach().numpy())
            return rf, rf.predict(knobs.norm_X_te.cpu().detach().numpy())

        dataset_tr = RocksDBDataset(knobs.norm_X_tr, knobs.norm_em_tr)
        dataset_te = RocksDBDataset(knobs.norm_X_te, knobs.norm_em_te)


        loader_tr = DataLoader(dataset = dataset_tr, batch_size = opt.batch_size, shuffle=True)
        loader_te = DataLoader(dataset = dataset_te, batch_size = opt.batch_size, shuffle=False)

        model = SingleNet(input_dim=knobs.norm_X_tr.shape[1], hidden_dim=16, output_dim=knobs.norm_em_tr.shape[-1]).cuda()

        # if opt.train:       
        logging.info(f"[Train MODE] Training Model") 
        best_loss = 100
        name = get_filename('model_save', 'model', '.pt')
        for epoch in range(opt.epochs):
            loss_tr = train(model, loader_tr, opt.lr)
            loss_te, outputs = valid(model, loader_te)
        
            logging.info(f"[{epoch:02d}/{opt.epochs}] loss_tr: {loss_tr:.8f}\tloss_te:{loss_te:.8f}")

            if best_loss > loss_te and epoch>15:
                best_loss = loss_te
                best_model = model
                best_outputs = outputs
                torch.save(best_model, os.path.join('model_save', name))
        logging.info(f"loss is {best_loss:.4f}, save model to {os.path.join('model_save', name)}")
        
        return best_model, best_outputs
        # elif opt.eval:
        #     logger.info(f"[Eval MODE] Trained Model Loading with path: {opt.model_path}")
        #     model = torch.load(os.path.join('model_save',opt.model_path))
        #     _, outputs = valid(model, loader_te)
        #     return model, outputs

def GA_optimization(knobs, fitness_function, logger, opt):
    problem = DBMSProblem(knobs=knobs, model=fitness_function, opt=opt).get_problem()
    # if opt.ga == 'GA':
    #     problem = RocksDBSingleProblem(knobs=knobs, model=fitness_function, model_mode=opt.mode)
    # elif opt.ga == 'NSGA2' or opt.ga == 'NSGA3':
    #     problem = RocksDBMultiProblem(knobs=knobs, model=fitness_function, model_mode=opt.mode)

    res = genetic_algorithm(mode=opt.ga, problem=problem, pop_size=opt.population)
    
    if len(res.X.shape) == 2:
        results = res.X[0] # NSGA2, MOO
        res_F = res.F[0]
    else:
        results = res.X # GA, SOO
        res_F = res.F
    
    return res_F, results
    recommend_command = make_dbbench_command(knobs, results, opt.target)
    # recommend_command = ''
    
    # for idx, col in enumerate(knobs.columns):                 
    #     recommend_command = convert_int_to_category(col, recommend_command, results[idx])

    # recommend_command = make_dbbench_command(opt.target, recommend_command)
    logger.info(f"db_bench command is  {recommend_command}")
    
    return res_F, recommend_command

def convert_int_to_category(col, cmd, s):
    if col=='compression_type':
            ct = ["snappy", "zlib", "lz4", "none"]
            cmd += f'-{col}={ct[int(s)]} '
    else:
        cmd += f'-{col}={int(s)} '
    return cmd

# def make_dbbench_command(trg_wk, rc_cmd):
#     wk_info = pd.read_csv('data/rocksdb_workload_info.csv', index_col=0)
#     f_wk_info = wk_info.loc[:,:'num']
#     b_wk_info = wk_info.loc[:, 'benchmarks':]
#     cmd = 'rocksdb/db_bench '   
#     f_cmd = " ".join([f'-{_}={int(f_wk_info.loc[trg_wk][_])}' for _ in f_wk_info.columns])
#     b_cmd = f"--{b_wk_info.columns[0]}={b_wk_info.loc[trg_wk][0]} "
#     b_cmd += f"--statistics "
#     if not np.isnan(b_wk_info.loc[trg_wk][1]):
#         b_cmd += f"--{b_wk_info.columns[1]}={int(b_wk_info.loc[trg_wk][1])}"

#     cmd += f_cmd + " " + rc_cmd + b_cmd + ' > /jieun/result.txt'

#     return cmd

def make_dbbench_command(knobs, results, trg_wk):
    recommend_command = ''
    
    for idx, col in enumerate(knobs.columns):                 
        recommend_command = convert_int_to_category(col, recommend_command, results[idx])
    
    wk_info = pd.read_csv('data/rocksdb/rocksdb_workload_info.csv', index_col=0)
    f_wk_info = wk_info.loc[:,:'num']
    b_wk_info = wk_info.loc[:, 'benchmarks':]
    cmd = 'rocksdb/db_bench '   
    f_cmd = " ".join([f'-{_}={int(f_wk_info.loc[trg_wk][_])}' for _ in f_wk_info.columns])
    b_cmd = f"--{b_wk_info.columns[0]}={b_wk_info.loc[trg_wk][0]} "
    b_cmd += f"--statistics "
    if not np.isnan(b_wk_info.loc[trg_wk][1]):
        b_cmd += f"--{b_wk_info.columns[1]}={int(b_wk_info.loc[trg_wk][1])}"

    cmd += f_cmd + " " + recommend_command + b_cmd + ' > /jieun/result.txt'

    return cmd

def make_mysql_configuration(knobs, results):
    R_MYSQL_PATH = 'rcmd_mysql_config'
    if os.path.exists(R_MYSQL_PATH) is False:
        os.makedirs(R_MYSQL_PATH)
    
    CONFIG_NAME = 'rcmd_my.cnf'
    f = open(os.path.join(R_MYSQL_PATH, CONFIG_NAME), 'w')
    f.write('[mysqld]\n')
    f.write('log-error = /var/log/mysqld.log\n')
    f.write('bind-address = 127.0.0.1\n')
    
    for idx, col in enumerate(knobs.columns):
        f.write(f'{col} = {np.round(results[idx]).astype(int)}\n')
        # logging.info(f'{col} : {np.round(results[idx]).astype(int)}')
        
    f.close()