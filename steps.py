import os
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from network import * 
from train import *
from utils import get_filename, PCC, MSE, RMSE
from scipy.stats import gmean
from ga import RocksDBSingleProblem, RocksDBMultiProblem, genetic_algorithm
import logging
from learn2learn.algorithms import MAML
from dataset import *
from configs import *

def train_fitness_function(knobs, opt):

    encoder = EncoderConv(embed_dim=opt.hidden_size//2, hidden_dim=opt.hidden_size, 
                            n_layers=opt.n_layers, kernel_size=opt.kernel_size, 
                            dropout_p=0.25, max_length=knobs.knobs[0].shape[-1])
    decoder = DecoderConv(embed_dim=opt.hidden_size//2, hidden_dim=opt.hidden_size, 
                            n_layers=opt.n_layers, kernel_size=opt.kernel_size,
                            dropout_p=0.25, max_length=knobs.external_metrics[0].shape[-1])
    model = ConvNet(encoder=encoder, decoder=decoder).cuda()
    
    if opt.train_type == 'maml':
        maml_data = MAMLdata(knobs, opt)
        
        trainer = MAMLTrainer(model, opt)
        
        best_loss = np.inf
        patience = 0
        
        if not os.path.exists('model_save/maml'):
            os.mkdir('model_save/maml')
        name = get_filename('model_save/maml', 'model', '.pt')
        
        for epoch in range(opt.wmaml_epochs):
            loss_tr = trainer.train(maml_data.maml_dl_tr, maml_data.maml_dl_val)
            
            loss_te, r2_res = trainer.valid(maml_data.maml_dl_te)

            # update best model
            if best_loss > loss_te:
                best_loss = loss_te
                best_model = trainer.model
                best_epoch = epoch + 1
                best_r2 = r2_res
                patience = 0
                torch.save(best_model, os.path.join('model_save/maml', name))
            else:
                patience += 1
            
            if patience >= opt.patience:
                logging.info(f"Early stopping") 
                break
        logging.info(f"---------------maml train step finish---------------") 