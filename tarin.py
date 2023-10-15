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