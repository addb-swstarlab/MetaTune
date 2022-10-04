import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
# from src.tasks import Sine_Task, Sine_Task_Distribution
import matplotlib.pyplot as plt
# from models.utils import *
import os
import datetime
import copy
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from lifelines.utils import concordance_index
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.nn.parameter import Parameter
from torchsummary import summary as summary
# from models.tasks import Workload_Task
# from pytorchtools import EarlyStopping
from models.utils import *
from pytorch_tabnet.tab_model import TabNetRegressor

from utils import Set_tabnet_network, Tabnet_architecture
# interpreter : py3.8         

class MAML():
    # def __init__(self, model, train_dataloaders, val_dataloaders, test_dataloaders, num_epochs, inner_lr, meta_lr, inner_steps=1, dot=True, lamb=0.6):
    def __init__(self, model, train_dataloaders, val_dataloaders, test_dataloaders, num_epochs, inner_lr, meta_lr, inner_steps=1):
    # def __init__(self, model, datas, num_epochs, inner_lr, meta_lr, inner_steps=1):

    
    # def __init__(self, model, train_dataloaders, test_dataloaders, meta_tasks, inner_lr, meta_lr, K=K, inner_steps=1):   # K : number of sample data of task
        
        #############################################
        self.patience = 10  # for early stopping 
        #############################################

        # important objects
        self.model = model        
        self.train_dataloaders = train_dataloaders
        self.val_dataloaders = val_dataloaders
        self.test_dataloaders = test_dataloaders
        # self.datas = datas
        # self.train_dataloaders = self.datas
        # self.val_dataloaders = self.datas
        # self.test_dataloaders = self.datas


        # hyperparameters
        self.num_epochs = num_epochs
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        # self.dot = dot
        # self.lamb = lamb

        # self.meta_tasks = meta_tasks    # list of using workload number for MAML
        self.num_meta_tasks = len(self.train_dataloaders)    # len(train_dataloaders) = len(test_dataloaders) 
        self.criterion = nn.MSELoss()
        # self.weights = list(self.model.parameters()) # the maml weights we will be meta-optimising

        # ######################################################
        # # self.weights = list(torch.nn.init.xavier_uniform_(self.model.parameters())) # the maml weights we will be meta-optimising
        # self.weights = list(self.model.parameters()) # the maml weights we will be meta-optimising
        # for i in range(len(self.weights)):
        #     self.weights[i]=torch.nn.init.kaiming_normal_(self.weights[i])
        # ######################################################
        # self.meta_optimizer = torch.optim.Adam(self.weights, self.meta_lr)
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), self.meta_lr)
        
        # self.K = K
        self.inner_steps = inner_steps # with the current design of MAML, >1 is unlikely to work well         
        
        # metrics
        self.meta_losses_tr = []
        self.meta_losses_te = []
        self.r2_score = []
        self.best_loss = np.inf
        # self.best_loss = 

    
    # def inner_loop(self, iter):     # i: task , iteration : iteration
    #     # reset inner model to current maml weights
    #     temp_weights = [w.clone() for w in self.weights]         
    #     # perform training on data sampled from task

    #     X, y = self.sample_tr[0], self.sample_tr[1]
    #     inner_loss = self.criterion(self.model.parameterised(X, temp_weights), y)
    #     grad = torch.autograd.grad(inner_loss, temp_weights)
    #     temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]

    #     temp_pred = self.model.parameterised(X, temp_weights)
    #     # calculate loss for update maml weight (with update inner loop weight)
    #     if self.dot:
    #         d = torch.bmm(self.model.p_res_x, self.model.p_res_x.transpose(1, 2))
    #         dot_loss = F.mse_loss(d, torch.eye(d.size(1)).repeat(X.shape[0], 1, 1).cuda())
    #         meta_loss = (1-self.lamb)*self.criterion(temp_pred, y) + self.lamb*dot_loss
    #         # meta_loss = (1-self.lamb)*F.mse_loss(self.model.parameterised(X, temp_weights), y) + self.lamb*dot_loss
    #     else:
    #         meta_loss = self.criterion(self.model.parameterised(X, temp_weights), y)
        
    #     return inner_loss, meta_loss

    # def inner_loop(self, model, iter):     # i: task , iteration : iteration
    def inner_loop(self, tmp_model, data_tr, data_val):     # i: task , iteration : iteration
        # reset inner model to current maml weights

        # define tmp_model for wmaml (copy original model)
        # tmp_model = TabNetRegressor()
        tmp_model.load_state_dict(self.model.state_dict())   # copy weight of origin model

        # perform training on data sampled from task
        # X, y = self.sample_tr[0], self.sample_tr[1]
        X_tr, y_tr = data_tr[0], data_tr[1]                
        tmp_model.fit(X_tr, y_tr)

        X_val, y_val = data_val[0], data_val[1]
        y_val_pred = tmp_model.predict(X_val)

        meta_loss = tmp_model.loss_fn(y_val, y_val_pred)    # 수정 필요
        
        return meta_loss
    

    def main_loop(self):
        # epoch_loss = 0 ####
        trigger_times = 0
        breaker = False ####
        for e in range(1, self.num_epochs+1):    # epoch
            sampler_tr = Sampler(dataloaders=self.train_dataloaders)
            sampler_val = Sampler(dataloaders=self.train_dataloaders)        
            for iter in range(1, len(self.train_dataloaders[0])+1):    # iteration    
                self.meta_optimizer.zero_grad()   
                total_meta_loss_tr = 0
                meta_loss_sum = 0
                wk_loss =[]    

                sample_tr = sampler_tr.get_sample() # get sample for each metatask data [[], [], [], ...]
                sample_val = sampler_val.get_sample()   #############
                 
                for num_wk in range(self.num_meta_tasks):
                    # make tmp_model for inner loop step ################################
                    # tmp_model = Set_tabnet_network(
                    #                 m=Tabnet_architecture(),
                    #                 x_train=sample_tr[num_wk][0].detach().cpu().numpy(),
                    #                 y_train=sample_tr[num_wk][1].detach().cpu().numpy(),
                    #                 x_eval=sample_val[num_wk][0].detach().cpu().numpy(),
                    #                 y_eval=sample_val[num_wk][1].detach().cpu().numpy() )

                    tmp_model = TabNetRegressor()
                    # tmp_model.__update__
                    tmp_model.input_dim = sample_tr[0][0].shape[1]  # dim of X
                    tmp_model.output_dim = sample_tr[0][1].shape[1] # dim of y
                    tmp_model._set_network()
                    # make tmp_model for inner loop step ################################
     
                    self.sample_tr = sample_tr[num_wk]
                    self.samle_val = sample_val[num_wk]  #########    

                    # inner loop
                    meta_loss = self.inner_loop(tmp_model, self.sample_tr, self.samle_val)
                    
                    wk_loss.append(meta_loss)
                    meta_loss_sum += meta_loss   # i: task                               
                # print(f'meta_loss_sum : {meta_loss_sum}')   ####
                # meta_loss_sum /= self.num_meta_tasks    ####
                # print(f'meta_loss_mean : {meta_loss_sum}')  ####
                total_meta_loss_tr += meta_loss_sum.item()
                meta_loss_tr = total_meta_loss_tr/len(self.train_dataloaders)
                
                # compute meta gradient of loss with respect to maml weights
                meta_loss_sum.backward()
                self.meta_optimizer.step()
            # loss_te, te_outputs, r2_res = self.validate(model)  ###
            loss_te, te_outputs, r2_res = self.validate(self.model)  ###


            # self.meta_losses_tr +=[meta_loss_sum]
            self.meta_losses_tr +=[meta_loss_tr]
            self.meta_losses_te += [loss_te]       
            self.r2_score += [r2_res]      

            time_ = datetime.datetime.today().strftime('%y%m%d/%H:%M:%S')
            print(f"{time_}[{e:02d}/{self.num_epochs}] meta_loss: {meta_loss_sum:.4f} loss_te: {loss_te:.4f}, r2_res = {r2_res:.4f}")

            # print(f"{time_}[{e:02d}/{self.num_epochs}] meta_loss: {meta_loss_sum:.4f} loss_te: {loss_te:.4f}, r2_res = {r2_res:.4f} / trrigger times : {trigger_times}")

            # if loss_te < self.best_loss:
            #     self.best_loss = loss_te
            #     self.best_ouputs = te_outputs
            #     self.best_r2_score = r2_res
            #     self.best_model = self.model
            #     self.best_epoch_num = e

            ###############################################################
            # early stopping
            if loss_te > self.best_loss:
                trigger_times += 1
                # print('Trigger Times:' , trigger_times)
                if trigger_times >= self.patience:
                    print('Early stopping! \n training step finish')
                    breaker = True
            else:
                self.best_loss = loss_te
                self.best_ouputs = te_outputs
                self.best_r2_score = r2_res
                self.best_model = self.model
                self.best_epoch_num = e
                print('Trigger Times: 0')
                trigger_times = 0  

            print(f"{time_}[{e:02d}/{self.num_epochs}] meta_loss: {meta_loss_sum:.4f} loss_te: {loss_te:.4f}, r2_res = {r2_res:.4f} / trrigger times : {trigger_times}")  

            if breaker == True:
                break
            ###############################################################


        self.name = get_filename('model_save', 'MAML_pretrain', '.pt')
        torch.save(self.best_model.state_dict(), os.path.join('model_save', self.name))  # save only state_dict of model
        print(f'Saved model state dict! name : {self.name}')
        # torch.save(self.best_model, os.path.join('model_save', name))             # save entire model


    def validate(self, model):  ###
        model.eval()

        total_loss = 0
        total_dot_loss = 0        
        outputs = torch.Tensor().cuda()
        r2_res = 0
        with torch.no_grad():
            for wk_valid_loader in self.test_dataloaders:
                for data, target in wk_valid_loader:
                    output = model(data)
                    # if self.dot:
                    #     d = torch.bmm(model.res_x, model.res_x.transpose(1, 2))
                    #     dot_loss = F.mse_loss(d, torch.eye(d.size(1)).repeat(data.shape[0], 1, 1).cuda())
                    #     loss = (1-self.lamb)*F.mse_loss(output, target) + self.lamb*dot_loss
                    # else:
                    #     dot_loss = 0
                    #     loss = F.mse_loss(output, target)

                    loss = F.mse_loss(output, target)
                    true = target.cpu().detach().numpy().squeeze()
                    pred = output.cpu().detach().numpy().squeeze()
                    r2_res += r2_score(true, pred)
                    total_loss += loss.item()
                    # total_dot_loss += dot_loss.item()
                    outputs = torch.cat((outputs, output))
        total_loss /= len(wk_valid_loader) * len(self.test_dataloaders)
        # total_dot_loss /= len(wk_valid_loader) * len(self.test_dataloaders)
        r2_res /= len(wk_valid_loader) * len(self.test_dataloaders)

        return total_loss, outputs, r2_res