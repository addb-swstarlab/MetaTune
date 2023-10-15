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
        logging.info(f"---------------MAML train step finish---------------") 
        

        best_loss = np.inf
        adapt_model = torch.load(os.path.join('model_save/maml', name))
        name = get_filename('model_save', 'model', '.pt')
        for epoch in range(opt.epochs):
            loss_tr = train(adapt_model, maml_data.adapt_dl_tr, opt.lr, loss_f=opt.loss)
            loss_te, outputs = valid(adapt_model, maml_data.adapt_dl_te, loss_f=opt.loss)
        
            if best_loss > loss_te:
                best_loss = loss_te
                best_model = adapt_model
                best_outputs = outputs
                best_epoch = epoch + 1
                patience = 0                
                torch.save(best_model, os.path.join('model_save', name))
            else:
                patience += 1
            
            if patience >= opt.patience:
                logging.info(f"Early stopping") 
                break
        
        model_path = os.path.join('model_save', name)
        logging.info(f"---------------Adaptation train step finish---------------") 



    ## Evaluate model
    pred = np.round(knobs.scaler_y.inverse_transform(best_outputs.cpu().detach().numpy().squeeze()), 2)
        
    if opt.train_type == 'maml' :
        true = np.round(knobs.scaler_y.inverse_transform(maml_data.knobs.norm_target_y_te.cpu().detach().numpy()), 2)

    logging.info('[PCC SCORE]')
    pcc = PCC(true, pred)
    logging.info(f'average PCC score = {pcc:.4f}')
    
    logging.info('[MSE SCORE]')
    mse = MSE(true, pred)
    logging.info(f'average MSE score = {mse:.4f}')
    
    logging.info('[RMSE SCORE]')
    rmse = RMSE(true, pred)
    logging.info(f'average RMSE score = {rmse:.4f}')
      
    os.system(f'echo {opt.log_dir} >> wk_{opt.target}_scores.txt')
    os.system(f'echo PCC={PCC(true, pred):.4f} MSE={MSE(true, pred):.4f} RMSE={RMSE(true, pred):.4f} >> wk_{opt.target}_scores.txt')
    
    return best_model, pd.DataFrame(data=[[pcc, mse, rmse, model_path]], columns=['PCC', 'MSE', 'RMSE', 'model_path'])
