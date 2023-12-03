import os
import argparse
import pandas as pd
import numpy as np
from benchmark import exec_benchmark
import utils
from knobs import Knob
from steps import train_fitness_function, GA_optimization
from configs import *
import random
import numpy as np
import torch


os.system('clear')
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=str, default='random', help='random seed')
parser.add_argument('--dbms', type=str, default='rocksdb', help='select dbms')
parser.add_argument('--target', type=str, help='Choose target workload')
parser.add_argument('--lr', type=float, default=0.0001, help='Define learning rate')
parser.add_argument('--epochs', type=int, default=50, help='Define train epochs')
parser.add_argument('--patience', type=int, default=15, help='Define earlystop count')
parser.add_argument('--hidden_size', type=int, default=128, help='Define model hidden size')
parser.add_argument('--n_layers', type=int, default=8, help='Define # of conv layer')
parser.add_argument('--kernel_size', type=int, default=3, help='Define # of kernel on conv')
parser.add_argument('--batch_size', type=int, default=256, help='Define model batch size')
parser.add_argument('--train_type', type=str, default='maml',choices=['replace','maml'])
parser.add_argument('--d_threshold', type=float, default=1.5, help='select value of threshold')
parser.add_argument('--maml_in_lr', type=float, default=0.01, help='Define inner_learning rate for inner loop of maml train')
parser.add_argument('--maml_lr', type=float, default=0.001, help='Define maml learning rate')
parser.add_argument('--maml_epochs', type=int, default=50, help='Define maml train epochs')
parser.add_argument('--ad_batch_size', type=int, help='Define model batch size of adaptation step')
parser.add_argument('--population', type=int, default=100, help='Define the number of generation to GA algorithm')
parser.add_argument('--GA_batch_size', type=int, default=32, help='Define GA batch size')

opt = parser.parse_args()

### set random seed
if opt.seed == 'random':
    logger.info(f"## Random seed [random] ##")    # random
    num = random.randrange(1, 1004)
else:
    num = int(opt.seed)

seed = num
opt.seed = seed
deterministic = True

random.seed(seed)       # fix random seed of python
np.random.seed(seed)    # fix random seed of numpy

# fix random seed of pytorch
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
logger.info(f"## Random seed={seed} ##")

# fix random seed of CuDNN
if deterministic:
    torch.backends.cudnn.deterministic = True    # If set True, slow down computational speed. Set True when reproductable experiment model is required
    torch.backends.cudnn.benchmark = False



if not os.path.exists('logs'):
    os.mkdir('logs')

if not os.path.exists('model_save'):
    os.mkdir('model_save')


logger, log_dir = utils.get_logger(os.path.join('./logs'))
opt.log_dir = log_dir


## print parser info
logger.info("## model hyperparameter information ##")
for i in vars(opt):
    logger.info(f'{i}: {vars(opt)[i]}')

DBMS_PATH = f'{opt.dbms}'
KNOB_PATH = os.path.join('data', DBMS_PATH, 'configs')
EXTERNAL_PATH = os.path.join('data', DBMS_PATH, 'external')
INTERNAL_PATH = os.path.join('data', DBMS_PATH, 'internal')

if opt.dbms == 'rocksdb':
    WK_NUM = 16    
else : 
    assert False, "Not support the dbms"
    
def main():
    
    logger.info("## get raw datas ##")
    
    # Prepare the data
    raw_knobs = {}
    internal_dict = {}
    external_dict = {}
    
    pruned_im = pd.read_csv(os.path.join(INTERNAL_PATH, 'internal_ensemble_pruned_tmp.csv'), index_col=0)
    
    for wk in range(WK_NUM):
        raw_knobs[wk] = getattr(utils, f'{opt.dbms}_knob_dataframe')(wk, KNOB_PATH)
        internal_dict[wk], external_dict[wk] = getattr(utils, f'{opt.dbms}_metrics_dataframe')(wk, pruned_im, INTERNAL_PATH, EXTERNAL_PATH)
    
    
    raw_knobs[wk+1] = getattr(utils, f'{opt.dbms}_knob_dataframe')(opt.target, os.path.join('data', DBMS_PATH, 'target_workload', 'adaptation', 'configs'), target=True)
    internal_dict[wk+1], external_dict[wk+1] = getattr(utils, f'{opt.dbms}_metrics_dataframe')(opt.target, pruned_im, 
                                                                                               os.path.join('data', DBMS_PATH, 'target_workload', 'adaptation', 'results'), 
                                                                                               os.path.join('data', DBMS_PATH, 'target_workload', 'adaptation', 'results')) 
    raw_knobs[wk].reset_index(drop=True)
    internal_dict[wk+1].reset_index(drop=True)
    external_dict[wk+1].reset_index(drop=True)

    
    raw_knobs[wk+2] = getattr(utils, f'{opt.dbms}_knob_dataframe')(opt.target, os.path.join('data', DBMS_PATH, 'target_workload', 'test', 'configs'), target=True)
    internal_dict[wk+2], external_dict[wk+2] = getattr(utils, f'{opt.dbms}_metrics_dataframe')(opt.target, pruned_im,
                                                                                            os.path.join('data', DBMS_PATH, 'target_workload', 'test', 'results'), 
                                                                                            os.path.join('data', DBMS_PATH, 'target_workload', 'test', 'results'))
    raw_knobs[wk+2].reset_index(drop=True)
    internal_dict[wk+2].reset_index(drop=True)
    external_dict[wk+2].reset_index(drop=True)
    
    knobs = Knob(raw_knobs, internal_dict, external_dict, WK_NUM, opt)
    knobs.split_data()
    knobs.scale_data()
    
    # Model training
    logger.info("## Train/Load Fitness Function start ##")
    fitness_function, scores = train_fitness_function(knobs=knobs, opt=opt) 
    logger.info("## Train/Load Fitness Function finish ##")
    
    # Configuration recommendation    
    logger.info("## configuration recommendation start ##")
    res, recommendation_command = GA_optimization(knobs=knobs, fitness_function=fitness_function, logger=logger, opt=opt)
    exec_benchmark(recommendation_command, opt)
    logger.info("## configuration recommendation finish ##")

if __name__ == '__main__':
    try:
        main()
        torch.cuda.empty_cache()
    except:
        logger.exception("ERROR!!")
    finally:
        logger.handlers.clear()
