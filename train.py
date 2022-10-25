import torch
import torch.optim as optim
import torch.nn.functional as F
from utils import Sampler
from sklearn.metrics import r2_score
import numpy as np
from datetime import datetime

def train(model, train_loader, lr):
    ## Construct optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    ## Set phase
    model.train()
    
    ## Train start
    total_loss = 0.
    for data, target in train_loader:
        ## data.shape = (batch_size, 22)
        ## target.shape = (batch_size, 1)
        ## initilize gradient
        optimizer.zero_grad()
        ## predict
        output = model(data) # output.shape = (batch_size, 1)
        ## loss
        loss = F.mse_loss(output, target)
        ## backpropagation
        loss.backward()
        optimizer.step()
        ## Logging
        total_loss += loss.item()
    total_loss /= len(train_loader)
    return total_loss

def valid(model, valid_loader):
    ## Set phase
    model.eval()
    
    ## Valid start    
    total_loss = 0.
    outputs = torch.Tensor().cuda()
    with torch.no_grad():
        for data, target in valid_loader:
            output = model(data)
            loss = F.mse_loss(output, target) # mean squared error
            total_loss += loss.item()
            outputs = torch.cat((outputs, output))
    total_loss /= len(valid_loader)
    return total_loss, outputs


# def wmaml_train(maml, maml_dl_tr, maml_dl_val, optimizer):    
#     cnt = 0
#     print(f'wmaml_train_{cnt}, {datetime.now()}') ################################################
#     cnt += 1    ################################################
#     batch_num = len(maml_dl_tr[0])  
#     task_num = len(maml_dl_tr)
    
#     maml.train()
#     tr_sampler = Sampler(dataloaders=maml_dl_tr)
#     val_sampler = Sampler(dataloaders=maml_dl_val)
#     # te_sampler = Sampler(dataloaders=maml_dl_te)
#     # print('-'*10,f'epoch: {epoch_idx}','-'*10)

#     for iter in range(batch_num):   #batch_num = len(maml_dl_tr[0]) / task_num = len(maml_dl_tr)
#         # print('-'*10,{iter},'-'*10)
#         optimizer.zero_grad()
#         meta_loss = 0.
#         print(f'wmaml_train_{cnt}, {datetime.now()}') ################################################
#         cnt += 1    ################################################
#         # get batch datas
#         tr_samples = tr_sampler.get_sample()
#         val_samples = val_sampler.get_sample()
#         # te_samples = te_sampler.get_sample()
#         print(f'wmaml_train_{cnt}, {datetime.now()}') ################################################
#         cnt += 1    ################################################
#         for task_idx in range(task_num):
#             task_model = maml.clone()
#             print(f'wmaml_train_{cnt}, {datetime.now()}') ################################################
#             cnt += 1    ################################################
#             X_tr, y_tr = tr_samples[task_idx]
#             pred, M_loss = task_model(X_tr)
#             print(f'wmaml_train_{cnt}, {datetime.now()}') ################################################
#             cnt += 1    ################################################
#             adaptation_loss = F.mse_loss(pred, y_tr)
#             task_model.adapt(adaptation_loss,
#                                 allow_nograd=True,
#                                 allow_unused=True)
#             print(f'wmaml_train_{cnt}, {datetime.now()}') ################################################
#             cnt += 1    ################################################
#             X_val, y_val = val_samples[task_idx]
#             pred, M_loss = task_model(X_val)
#             evaluation_loss = F.mse_loss(pred, y_val)
#             print(f'wmaml_train_{cnt}, {datetime.now()}') ################################################
#             cnt += 1    ################################################
#             meta_loss += evaluation_loss
#             meta_loss /= task_num
#         meta_loss.backward()
#         optimizer.step()
#         print(f'wmaml_train_{cnt}, {datetime.now()}') ################################################
#         cnt += 1    ################################################
#     # del(tr_sampler)
#     # del(val_sampler)
#     return meta_loss # task별로 pred가 나올텐데 pred 합치는 것 어떻게 할지 

# def wmaml_valid(maml, maml_dl_te):
#     ## Set phase
#     maml.eval()
#     cnt = 0
#     print(f'wmaml_valid_{cnt}, {datetime.now()}') ################################################
#     cnt += 1    ################################################
#     ## Valid start 
#     total_loss = 0.
#     # outputs = torch.Tensor().cuda()
#     r2_res_list = []
#     with torch.no_grad():
#         for i in range(len(maml_dl_te)):    # check each task_meta_data
#             print(f'wmaml_valid_{cnt}, {datetime.now()}') ################################################
#             cnt += 1    ################################################
#             total_task_loss = 0.
#             # task_output = torch.Tensor().cuda()           
#             trues = []
#             preds = []
#             for data, target in maml_dl_te[i]:  # iteration
#                 print(f'wmaml_valid_{cnt}, {datetime.now()}') ################################################
#                 cnt += 1    ################################################
#                 output, M_loss = maml(data)
#                 print(f'wmaml_valid_{cnt}, {datetime.now()}') ################################################
#                 cnt += 1    ################################################
#                 task_loss = F.mse_loss(output, target)

#                 total_task_loss += task_loss.item()
#                 # task_output = torch.cat((task_output, output))     

#                 pred = output.detach().cpu().numpy().squeeze()   
#                 true = target.detach().cpu().numpy().squeeze()
#                 preds.append(pred)
#                 trues.append(true)
#             # outputs = torch.stack(outputs, task_output)        
#             total_task_loss /= len(maml_dl_te[i])
#             total_loss += total_task_loss
#             r2_res = r2_score(true, pred)
#             r2_res_list.append(r2_res)

#         total_loss /= len(maml_dl_te)  
#         r2_sum = np.mean(r2_res_list)
#         r2_mean = r2_sum/len(r2_res_list)
#         return total_loss, r2_mean


def wmaml_train(maml, maml_dl_tr, maml_dl_val, optimizer):
    batch_num = len(maml_dl_tr[0])  
    task_num = len(maml_dl_tr)
    
    maml.train()
    tr_sampler = Sampler(dataloaders=maml_dl_tr)
    val_sampler = Sampler(dataloaders=maml_dl_val)
    # te_sampler = Sampler(dataloaders=maml_dl_te)
    # print('-'*10,f'epoch: {epoch_idx}','-'*10)

    for iter in range(batch_num):   #batch_num = len(maml_dl_tr[0]) / task_num = len(maml_dl_tr)
        # print('-'*10,{iter},'-'*10)
        optimizer.zero_grad()
        meta_loss = 0.
        
        # get batch datas
        tr_samples = tr_sampler.get_sample()
        val_samples = val_sampler.get_sample()
        # te_samples = te_sampler.get_sample()

        for task_idx in range(task_num):
            task_model = maml.clone()

            X_tr, y_tr = tr_samples[task_idx]
            pred, M_loss = task_model(X_tr)
            adaptation_loss = F.mse_loss(pred, y_tr)
            task_model.adapt(adaptation_loss,
                                allow_nograd=True,
                                allow_unused=True)

            X_val, y_val = val_samples[task_idx]
            pred, M_loss = task_model(X_val)
            evaluation_loss = F.mse_loss(pred, y_val)
            meta_loss += evaluation_loss
            meta_loss /= task_num
        meta_loss.backward()
        optimizer.step()
    # del(tr_sampler)
    # del(val_sampler)
    return meta_loss # task별로 pred가 나올텐데 pred 합치는 것 어떻게 할지 

def wmaml_valid(maml, maml_dl_te):
    ## Set phase
    maml.eval()

    ## Valid start 
    total_loss = 0.
    # outputs = torch.Tensor().cuda()
    r2_res_list = []
    with torch.no_grad():
        for i in range(len(maml_dl_te)):    # check each task_meta_data
            total_task_loss = 0.
            # task_output = torch.Tensor().cuda()           
            trues = []
            preds = []
            for data, target in maml_dl_te[i]:  # iteration
                output, M_loss = maml(data)

                task_loss = F.mse_loss(output, target)

                total_task_loss += task_loss.item()
                # task_output = torch.cat((task_output, output))     

                pred = output.detach().cpu().numpy().squeeze()   
                true = target.detach().cpu().numpy().squeeze()
                preds.append(pred)
                trues.append(true)
            # outputs = torch.stack(outputs, task_output)        
            total_task_loss /= len(maml_dl_te[i])
            total_loss += total_task_loss
            r2_res = r2_score(true, pred)
            r2_res_list.append(r2_res)

        total_loss /= len(maml_dl_te)  
        r2_sum = np.mean(r2_res_list)
        r2_mean = r2_sum/len(r2_res_list)
        return total_loss, r2_mean