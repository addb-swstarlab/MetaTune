import os
import argparse
import pandas as pd
import numpy as np
from benchmark import exec_benchmark
import utils
from knobs import Knob
from steps import train_fitness_function, GA_optimization
from configs import *
import torch


os.system('clear')
parser = argparse.ArgumentParser()

parser.add_argument('--dbms', type=str, default='rocksdb', help='select dbms')
parser.add_argument('--target', type=str, help='Choose target workload')
parser.add_argument('--lr', type=float, default=0.0001, help='Define learning rate')
parser.add_argument('--epochs', type=int, default=200, help='Define train epochs')
parser.add_argument('--hidden_size', type=int, default=128, help='Define model hidden size')
parser.add_argument('--n_layers', type=int, default=8, help='Define # of conv layer')
parser.add_argument('--kernel_size', type=int, default=3, help='Define # of kernel on conv')
parser.add_argument('--batch_size', type=int, default=256, help='Define model batch size')
parser.add_argument('--train_type', type=str, default='maml',choices=['replace','maml'])
parser.add_argument('--d_threshold', type=float, default=1.5, help='select value of threshold')
parser.add_argument('--maml_in_lr', type=float, default=0.01, help='Define inner_learning rate for inner loop of maml train')
parser.add_argument('--maml_lr', type=float, default=0.001, help='Define maml learning rate')
parser.add_argument('--maml_epochs', type=int, default=200, help='Define maml train epochs')

opt = parser.parse_args()

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
    
    knobs = Knob(raw_knobs, internal_dict, external_dict, WK_NUM, opt)
    knobs.split_data()
    knobs.scale_data()
    
    # Model training
    fitness_function, scores = train_fitness_function(knobs=knobs, opt=opt) 
    logger.info("## Train/Load Fitness Function DONE ##")
    
    # Configuration recommendation    
    res, recommend_command = GA_optimization(knobs=knobs, fitness_function=fitness_function, logger=logger, opt=opt)

    exec_benchmark(recommend_command, opt)

if __name__ == '__main__':
    try:
        main()
        torch.cuda.empty_cache()
    except:
        logger.exception("ERROR!!")
    finally:
        logger.handlers.clear()