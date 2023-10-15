# Train and validation step of predictive model

import torch
import torch.optim as optim
import torch.nn.functional as F
from utils import Sampler
import numpy as np

class MAMLTrainer():
    def __init__(self, model, opt, inner_steps=1):
        self.model = model
        self.inner_steps = inner_steps
        self.inner_lr = opt.maml_in_lr
        self.meta_lr = opt.maml_lr
        
        self.enc_weigths = list(self.model.encoder.parameters())
        self.dec_weigths = list(self.model.decoder.parameters())
                
        self.criterion = getattr(F, opt.loss + '_loss')
        self.meta_optimizer = optim.AdamW(self.model.parameters(), lr=self.meta_lr)
        
    def train(self, train_loader, valid_loader):        
        batch_num = len(train_loader[0])
        task_num = len(train_loader)
                
        self.model.train()
        tr_sampler = Sampler(dataloaders=train_loader)
        val_sampler = Sampler(dataloaders=valid_loader)
        
        for _ in range(batch_num):
            self.meta_optimizer.zero_grad()
            meta_loss = 0.
            
            # get batch data
            tr_samples = tr_sampler.get_sample()
            val_samples = val_sampler.get_sample()
            
            loss_weight = 1 / task_num
            for task_idx in range(task_num):
                tmp_enc_weights = [w.clone() for w in self.enc_weigths]
                tmp_dec_weights = [w.clone() for w in self.dec_weigths]
                
                X_tr, y_tr = tr_samples[task_idx]
                for s in range(self.inner_steps):
                    adaptation_loss = self.criterion(self.model.parameterized(X_tr, y_tr, tmp_enc_weights, tmp_dec_weights), y_tr)
                    
                    tmp_weights = tmp_enc_weights + tmp_dec_weights
                    grad = torch.autograd.grad(adaptation_loss, tmp_weights)
                    tmp_weights = [w - self.inner_lr*g for w, g in zip(tmp_weights, grad)]
                    
                    tmp_enc_weights = tmp_weights[:len(tmp_enc_weights)]
                    tmp_dec_weights = tmp_weights[len(tmp_enc_weights):]
                
                X_val, y_val = val_samples[task_idx]
                evaluation_loss = self.criterion(self.model.parameterized(X_val, y_val, tmp_enc_weights, tmp_dec_weights), y_val) * loss_weight
                meta_loss += evaluation_loss            
            meta_loss.backward()
            self.meta_optimizer.step()
        return meta_loss
    
    def valid(self, test_loader):
        task_num = len(test_loader)
        loss_weight = 1 / task_num
        
        self.model.eval()
        total_loss = 0.
        
        with torch.no_grad():
            for i in range(task_num):    # check each task_meta_data
                total_task_loss = 0.
                self.trues = []
                self.preds = []
                for data, target in test_loader[i]:  # iteration
                    output = self.model(data, target, train=False)

                    task_loss = self.criterion(output, target)

                    total_task_loss += task_loss.item()

                    pred = output.detach().cpu().numpy().squeeze()   
                    true = target.detach().cpu().numpy().squeeze()
                    self.preds.append(pred)
                    self.trues.append(true)
                total_task_loss /= len(test_loader[i])
                total_loss += total_task_loss * loss_weight
                self.trues = np.concatenate(self.trues)
                self.preds = np.concatenate(self.preds)

            return total_loss



def train(model, train_loader, lr, loss_f='mse'):
    loss_f += '_loss'
    ## Construct optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    ## Set phase
    model.train()
    
    ## Train start
    total_loss = 0.
    for data, target in train_loader:
        ## initilize gradient
        optimizer.zero_grad()
        ## predict
        output = model(data, target)
        ## loss
        loss = getattr(F, loss_f)(output, target)
        ## backpropagation
        loss.backward()
        optimizer.step()
        ## Logging
        total_loss += loss.item()
    total_loss /= len(train_loader)
    return total_loss


def valid(model, valid_loader, loss_f='mse'):
    loss_f += '_loss'
    ## Set phase
    model.eval()
    
    ## Valid start    
    total_loss = 0.
    outputs = torch.Tensor().cuda()
    with torch.no_grad():
        for data, target in valid_loader:
            output = model(data, target, train=False)
            loss = getattr(F, loss_f)(output, target)
            total_loss += loss.item()
            outputs = torch.cat((outputs, output))
    total_loss /= len(valid_loader)
    return total_loss, outputs