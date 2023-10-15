# Train and validation step of predictive model

import torch
import torch.optim as optim
import torch.nn.functional as F
from utils import Sampler
import numpy as np

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