import torch
import torch.optim as optim
import torch.nn.functional as F

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



def wmaml_train():
    pass

def wmaml_valid(maml, maml_dl_te):
    # check loss
    maml.eval()
    with torch.no_grad():
        total_loss = 0.
        for i in range(len(maml_dl_te)):
            total_task_loss = 0.
            for data, target in maml_dl_te[i]:
                output, M_loss = maml(data)

                task_loss = F.mse_loss(output, target)

                total_task_loss += task_loss.item()
            total_task_loss /= len(maml_dl_te[i])

            total_loss += total_task_loss
        total_loss /= len(maml_dl_te)

        true = target.cpu().detach().numpy().squeeze()
        pred = output.cpu().detach().numpy().squeeze()
        r2_res = r2_score(true, pred)
        return total_loss, outputs