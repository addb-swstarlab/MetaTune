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

from pytorch_tabnet.tab_model import TabNetRegressor

# interpreter : py3.8


'''ReshapeNet - activation : ReLU'''
class ReshapeNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, group_dim=32, wk_vec_dim=4, output_dim=1):
        super(ReshapeNet, self).__init__()
        self.input_dim = input_dim - wk_vec_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.group_dim = group_dim
        self.wk_vec_dim = wk_vec_dim

        self.embedding = nn.Linear(self.wk_vec_dim, self.hidden_dim)
        #self.knob_fc = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim*self.group_dim), nn.Sigmoid()) # (22, 1) -> (group*hidden, 1)
        self.knob_fc = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim*self.group_dim), nn.ReLU()) # (22, 1) -> (group*hidden, 1)
        self.attention = nn.MultiheadAttention(self.hidden_dim, 1)
        #self.active = nn.Sigmoid()
        self.activate = nn.ReLU()
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        wk = x[:, -self.wk_vec_dim:] # only workload information
        x = x[:, :-self.wk_vec_dim] # only knobs
        
        self.embed_wk = self.embedding(wk) # (batch, 4) -> (batch, dim)
        self.embed_wk = self.embed_wk.unsqueeze(1) # (batch, 1, dim)
        self.x = self.knob_fc(x) # (batch, 22) -> (batch, group*hidden)
        self.res_x = torch.reshape(self.x, (-1, self.group_dim, self.hidden_dim)) # (batch, group, hidden)
        
        # attn_ouptut = (1, batch, hidden), attn_weights = (batch, 1, group)
        self.attn_output, self.attn_weights = self.attention(self.embed_wk.permute((1,0,2)), self.res_x.permute((1,0,2)), self.res_x.permute((1,0,2)))
        self.attn_output = self.activate(self.attn_output.squeeze())
        outs = self.attn_output
        self.outputs = self.fc(outs)  
        return self.outputs
    
    def parameterised(self, x, weights):
        # like forward, but uses ``weights`` instead of ``model.parameters()``
        # it'd be nice if this could be generated automatically for any nn.Module...
        # https://easy-going-programming.tistory.com/11
        wk = x[:, -self.wk_vec_dim:] # only workload information
        x = x[:, :-self.wk_vec_dim] # only knobs
        embed_wk = F.linear(wk, weights[0], weights[1])
        embed_wk = embed_wk.unsqueeze(1)
        x = F.linear(x, weights[2], weights[3] )    # (22, 1) -> (group*hidden, 1)
        #x = F.sigmoid(x)
        x = F.relu(x)
        self.p_res_x = torch.reshape(x, (-1, self.group_dim, self.hidden_dim)) # (batch, group, hidden)
        # attn_output, self.attn_weights = self.attention(embed_wk.permute((1,0,2)), res_x.permute((1,0,2)), res_x.permute((1,0,2)))
        attn_output, _ = F.multi_head_attention_forward(embed_wk.permute((1,0,2)), self.p_res_x.permute((1,0,2)), self.p_res_x.permute((1,0,2)), self.hidden_dim, 1, 
                                                        weights[4], weights[5], 
                                                        None, None, False, 0,
                                                        weights[6], weights[7]) # self.attention(embed_wk.permute((1,0,2)), res_x.permute((1,0,2)), res_x.permute((1,0,2)))
        #attn_output = F.sigmoid(attn_output.squeeze())
        attn_output = F.relu(attn_output.squeeze())
        outputs = F.linear(attn_output, weights[8], weights[9])

        return outputs                           


class MAML_one_batch():
    def __init__(self, model, train_dataloaders, val_dataloaders, test_dataloaders, num_epochs, inner_lr, meta_lr, inner_steps=1, dot=True, lamb=0.6):
    # def __init__(self, model, train_dataloaders, test_dataloaders, meta_tasks, inner_lr, meta_lr, K=K, inner_steps=1):   # K : number of sample data of task
        
        #############################################
        self.patience = 10  # for early stopping 
        #############################################

        # important objects
        self.model = model        
        self.train_dataloaders = train_dataloaders
        self.val_dataloaders = val_dataloaders
        self.test_dataloaders = test_dataloaders

        # hyperparameters
        self.num_epochs = num_epochs
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.dot = dot
        self.lamb = lamb

        # self.meta_tasks = meta_tasks    # list of using workload number for MAML
        self.num_meta_tasks = len(self.train_dataloaders)    # len(train_dataloaders) = len(test_dataloaders) 
        self.criterion = nn.MSELoss()
        self.weights = list(self.model.parameters()) # the maml weights we will be meta-optimising
        # ######################################################
        # # self.weights = list(torch.nn.init.xavier_uniform_(self.model.parameters())) # the maml weights we will be meta-optimising
        # self.weights = list(self.model.parameters()) # the maml weights we will be meta-optimising
        # for i in range(len(self.weights)):
        #     self.weights[i]=torch.nn.init.kaiming_normal_(self.weights[i])
        # ######################################################
        self.meta_optimizer = torch.optim.Adam(self.weights, self.meta_lr)
        
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

    def inner_loop(self, model, iter):     # i: task , iteration : iteration
        # reset inner model to current maml weights
        tmp_model = TabNetRegressor()
        tmp_model.load_state_dict(model.state_dict())
        # perform training on data sampled from task
        
        X, y = self.sample_tr[0], self.sample_tr[1]
        inner_loss = self.criterion(self.model.parameterised(X, temp_weights), y)
        grad = torch.autograd.grad(inner_loss, temp_weights)
        temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]

        temp_pred = self.model.parameterised(X, temp_weights)
        # calculate loss for update maml weight (with update inner loop weight)
        if self.dot:
            d = torch.bmm(self.model.p_res_x, self.model.p_res_x.transpose(1, 2))
            dot_loss = F.mse_loss(d, torch.eye(d.size(1)).repeat(X.shape[0], 1, 1).cuda())
            meta_loss = (1-self.lamb)*self.criterion(temp_pred, y) + self.lamb*dot_loss
            # meta_loss = (1-self.lamb)*F.mse_loss(self.model.parameterised(X, temp_weights), y) + self.lamb*dot_loss
        else:
            meta_loss = self.criterion(self.model.parameterised(X, temp_weights), y)
        
        return inner_loss, meta_loss
    

    def main_loop(self):
        # epoch_loss = 0 ####
        trigger_times = 0
        breaker = False ####
        for e in range(1, self.num_epochs+1):    # epoch
            sampler_tr = Sampler(dataloaders=self.train_dataloaders)
            # sampler_val = Sampler(dataloaders=self.train_dataloaders)        
            for iter in range(1, len(self.train_dataloaders[0])+1):    # iteration       
                total_meta_loss_tr = 0
   
                self.meta_optimizer.zero_grad()
                sample_tr = sampler_tr.get_sample()
                meta_loss_sum = 0
                wk_loss =[]                     
                for num_wk in range(self.num_meta_tasks):
                    self.sample_tr = sample_tr[num_wk]
                    # self.samle_val = sample_val[num_wk]  ###                  
                    _, meta_loss = self.inner_loop(iter)
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
            loss_te, te_outputs, r2_res = self.validate(model)  ###

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
                    if self.dot:
                        d = torch.bmm(model.res_x, model.res_x.transpose(1, 2))
                        dot_loss = F.mse_loss(d, torch.eye(d.size(1)).repeat(data.shape[0], 1, 1).cuda())
                        loss = (1-self.lamb)*F.mse_loss(output, target) + self.lamb*dot_loss
                    else:
                        dot_loss = 0
                        loss = F.mse_loss(output, target)
                    true = target.cpu().detach().numpy().squeeze()
                    pred = output.cpu().detach().numpy().squeeze()
                    r2_res += r2_score(true, pred)
                    total_loss += loss.item()
                    total_dot_loss += dot_loss.item()
                    outputs = torch.cat((outputs, output))
        total_loss /= len(wk_valid_loader) * len(self.test_dataloaders)
        total_dot_loss /= len(wk_valid_loader) * len(self.test_dataloaders)
        r2_res /= len(wk_valid_loader) * len(self.test_dataloaders)

        return total_loss, outputs, r2_res