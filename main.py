import os
import argparse
import pandas as pd
import numpy as np
from benchmark import exec_benchmark
from utils import get_logger, rocksdb_knobs_make_dict, mysql_knob_dataframe, mysql_metrics_dataframe
from knobs import Knob
from steps import train_fitness_function, GA_optimization
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from lifelines.utils import concordance_index
# from benchmark import exec_benchmark
# from datetime import datetime

os.system('clear')

parser = argparse.ArgumentParser()
parser.add_argument('--dbms', type=str, choices=['rocksdb', 'mysql'], help='choose dbms, rocksdb or mysql')
parser.add_argument('--target', type=int, default=0, help='Choose target workload')
# parser.add_argument('--target_size', type=int, default=10, help='Define target workload size')
parser.add_argument('--lr', type=float, default=0.001, help='Define learning rate')
parser.add_argument('--epochs', type=int, default=50, help='Define train epochs')
parser.add_argument('--hidden_size', type=int, default=64, help='Define model hidden size')
parser.add_argument('--batch_size', type=int, default=32, help='Define model batch size')
parser.add_argument('--ga', type=str, default='GA', choices=['GA', 'NSGA2', 'NSGA3'], help='choose genetic algorithm')
parser.add_argument('--mode', type=str, default='dnn', help='choose which model be used on fitness function')
# parser.add_argument('--eval', action='store_true', help='if trigger, model goes eval mode')
# parser.add_argument('--train', action='store_true', help='if trigger, model goes triain mode')
# parser.add_argument('--model_path', type=str, help='Define which .pt will be loaded on model')
parser.add_argument('--population', type=int, default=100, help='Define the number of generation to GA algorithm')
parser.add_argument('--GA_batch_size', type=int, default=32, help='Define GA batch size')
# parser.add_argument('--ex_weight', type=float, action='append', help='Define external metrics weight to calculate score')
# parser.add_argument('--save', action='store_true', help='Choose save the score on csv file or just show')
# parser.add_argument('--sample_size', type=int, default=20000, help='Define train sample size, max is 20000')


opt = parser.parse_args()

if not os.path.exists('logs'):
    os.mkdir('logs')

if not os.path.exists('model_save'):
    os.mkdir('model_save')


logger, log_dir = get_logger(os.path.join('./logs'))

## print parser info
logger.info("## model hyperparameter information ##")
for i in vars(opt):
    logger.info(f'{i}: {vars(opt)[i]}')

DBMS_PATH = f'{opt.dbms}'
KNOB_PATH = os.path.join('data', DBMS_PATH, 'configs')
EXTERNAL_PATH = os.path.join('data', DBMS_PATH, 'external')
INTERNAL_PATH = os.path.join('data', DBMS_PATH, 'internal')
WK_NUM = 3 # MySQL
# WK_NUM = 16 # RocksDB

def main():
    logger.info("## get raw datas ##")
    
    internal_dict = {}
    external_dict = {}
    
    if opt.dbms == "rocksdb":
        '''
            When using rocksdb dbms, raw_knobs is a dataframe type
        '''
        raw_knobs = rocksdb_knobs_make_dict(KNOB_PATH)
        raw_knobs = pd.DataFrame(data=raw_knobs['data'].astype(np.float32), columns=raw_knobs['columnlabels'])  
        
        pruned_im = pd.read_csv(os.path.join(INTERNAL_PATH, 'internal_ensemble_pruned_tmp.csv'), index_col=0)
        for wk in range(WK_NUM):
            im = pd.read_csv(os.path.join(INTERNAL_PATH, f'internal_results_{wk}.csv'), index_col=0)
            internal_dict[wk] = im[pruned_im.columns]
        if opt.target > 15:
            im = pd.read_csv(f'data/target_workload/{opt.target}/internal_results_11.csv', index_col=0)
            internal_dict[wk+1] = im[pruned_im.columns]

        for wk in range(WK_NUM):
            ex = pd.read_csv(os.path.join(EXTERNAL_PATH, f'external_results_{wk}.csv'), index_col=0)
            external_dict[wk] = ex
        if opt.target > 15:
            ex = pd.read_csv(f'data/target_workload/{opt.target}/external_results_11.csv', index_col=0)
            external_dict[wk+1] = ex
        
    elif opt.dbms == "mysql":
        '''
            When using mysql dbms, raw_knobs is a dictionary type; keys represent workload number
        '''
        raw_knobs = {}
        for wk in range(WK_NUM):
            raw_knobs[wk] = mysql_knob_dataframe(wk, KNOB_PATH)
            
            internal_dict[wk], external_dict[wk] = mysql_metrics_dataframe(wk, INTERNAL_PATH, EXTERNAL_PATH)
        raw_knobs = raw_knobs[opt.target]
        
    
    logger.info('## get raw datas DONE ##')


    knobs = Knob(raw_knobs, internal_dict, external_dict, opt)


    knobs.split_data()
    knobs.scale_data()
  
    # if opt.train:
    logger.info("## Train Fitness Function ##")
    fitness_function, outputs = train_fitness_function(knobs=knobs, logger=logger, opt=opt)   
    
    if opt.mode == "RF":
        # if outputs' type are numpy array
        pred = np.round(knobs.scaler_em.inverse_transform(outputs), 2)
    else:
        # if outputs' type are torch.tensor
        pred = np.round(knobs.scaler_em.inverse_transform(outputs.cpu().detach().numpy()), 2)    
        
    true = knobs.em_te.to_numpy()

    for i in range(10):
        logger.info(f'predict rslt: {pred[i]}')
        logger.info(f'ground truth: {true[i]}\n')
    
    # elif opt.eval:
    #     logger.info("## Load Trained Fitness Function ##")
    #     fitness_function, outputs = train_fitness_function(knobs=knobs, logger=logger, opt=opt)
    #     pred = np.round(knobs.scaler_em.inverse_transform(outputs.cpu().detach().numpy()), 2)
    #     true = knobs.em_te.to_numpy()
        
    # else:
    #     logger.exception("Choose Model mode, '--train' or '--eval'")      

    r2_res = r2_score(true, pred, multioutput='raw_values')
    logger.info('[R2 SCORE]')
    if opt.dbms == 'rocksdb':
        logger.info(f'TIME:{r2_res[0]:.4f}, RATE:{r2_res[1]:.4f}, WAF:{r2_res[2]:.4f}, SA:{r2_res[3]:.4f}')
    elif opt.dbms == 'mysql':
        logger.info(f'TPS:{r2_res[0]:.4f}, LATENCY:{r2_res[1]:.4f}')
    r2_res = np.average(r2_res)
    logger.info(f'average r2 score = {r2_res:.4f}')
    
    pcc_res = 0
    ci_res = 0
    for idx in range(knobs.external_metrics_size):
        res_, _ =  pearsonr(true[:,idx], pred[:,idx])
        pcc_res += res_
        res_ = concordance_index(true[:,idx], pred[:,idx])
        ci_res += res_
    pcc_res = pcc_res/knobs.external_metrics_size
    ci_res = ci_res/knobs.external_metrics_size
    
    logger.info('[PCC SCORE]')
    logger.info(f'average pcc score = {pcc_res:.4f}')
    logger.info('[CI SCORE]')
    logger.info(f'average ci score = {ci_res:.4f}')
    
    res_F, recommend_command = GA_optimization(knobs=knobs, fitness_function=fitness_function, logger=logger, opt=opt)
    
    if opt.ga == "NSGA2":
        logger.info(f'## Predicted External metrics from genetic algorithm ##')
        if opt.dbms == 'rocksdb':
            logger.info(f'TIME: {res_F[0]}')
            logger.info(f'RATE: {res_F[1]}')
            logger.info(f'WAF: {res_F[2]}')
            logger.info(f'SA: {res_F[3]}')
        elif opt.dbms == 'mysql':
            logger.info(f'TPS: {res_F[0]}')
            logger.info(f'LATENCY: {res_F[1]}')
    
    logger.info("## Train/Load Fitness Function DONE ##")
    logger.info("## Configuration Recommendation DONE ##")
    
    exec_benchmark(recommend_command, opt)
    
    if os.path.exists('res.txt'):
        logger.info('## Results of External Metrics ##')
        f = open('res.txt')
        bench_res = f.readlines()
        for _ in bench_res:
            logger.info(f'{_[:-1]}')
    # if opt.step:
    #     for s_cmd in step_recommend_command:
    #         exec_benchmark(s_cmd, opt)
    #     file_name = f"{datetime.today().strftime('%Y%m%d')}_{opt.sample_size}_steps_fitness.csv"
    #     if os.path.isfile(file_name) is False:
    #         pd.DataFrame(data=range(1,101), columns=['idx']).to_csv(file_name, index=False)
    #     pd_steps = pd.read_csv(file_name, index_col=0)
    #     if opt.bidirect:
    #         pd_steps[f'{opt.target}_bi{opt.mode}'] = step_best_fitness
    #     else:
    #         pd_steps[f'{opt.target}_{opt.mode}'] = step_best_fitness
    #     pd_steps.to_csv(file_name)
    # else:
    #     ## Execute db_benchmark with recommended commands by transporting to other server
    #     exec_benchmark(recommend_command, opt)



if __name__ == '__main__':
    try:
        main()
    except:
        logger.exception("ERROR!!")
    finally:
        logger.handlers.clear()
