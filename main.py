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
    
def main():
    pass


if __name__ == '__main__':
    try:
        main()
        torch.cuda.empty_cache()
    except:
        logger.exception("ERROR!!")
    finally:
        logger.handlers.clear()