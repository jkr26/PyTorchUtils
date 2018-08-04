import torch
import numpy as np
import tensorboardX
import datetime
import pdb
import math
import torch.nn as nn
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()

def to_np(x):
    return x.data.cpu().numpy()

def setOptimizerLr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def sgdr(period, batch_idx):
    radians = math.pi*float(batch_idx)/period
    return 0.5*(1.0 + math.cos(radians))

def plot_lr_finder(model, criterion, data, batch_size):
    lr_list = []
    loss_list = []
    lr = 1e-15
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    m = np.Inf
    train, train_response = data[0], data[1]
    n = len(train)
    t = 0
    n_iters = int(1e4)
    while t < 2*m and lr < 100:
        total_loss = 0
        for b in range(0, n_iters, batch_size):
            batch =  (torch.Tensor(train.loc[b % n:(b+batch_size) %  n, :].values), torch.Tensor(train_response.loc[b % n:(b+batch_size) % n].values))
            if batch[0].size()[0] > 0:
                if use_cuda:
                    batch = (batch[0].cuda(), batch[1].cuda())
                preds = model(batch[0].unsqueeze(1))
                loss = criterion(preds.squeeze(1), batch[1])
                total_loss += loss.item()
                loss.backward()
                opt.step()
        t = total_loss/(n_iters/batch_size)
        print(str(t))
        lr_list.append(np.log10(lr))
        loss_list.append(t)
        lr = lr*2
        if t < m:
            m = t
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(np.array(lr_list), np.array(loss_list))
    plt.show()

def batchedTrainIters(name, n_iters, batch_size, print_every, model, optimizer, criterion,  data, text, use_sgdr=False, period=100):
    """Reusable trainiters function for convNet.
    Args:
    n_iters, int, num iterations
    batch_size, int, batch size
    model, subclass of torch.nn.Module
    optimizer, subclass of torch.nn.Optim
    criterion, loss function
    data, 3-tuple of 2-tuples pd.DataFrames, consisting of:
        (train, train_response), (val, val_response),  (test, test_response)
    """
    if use_cuda:
        model = model.cuda()
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    writer = tensorboardX.SummaryWriter('/media/jkr/hdd1/TensorboardLogs/'+name+'/'+now)
    train, train_response = data[0][0], data[0][1]
    val, val_response = data[1][0], data[1][1]
    test, test_response = data[2][0], data[2][1]
    min_val_loss = np.Inf
    assc_test_loss = np.Inf
    n = len(train)
    total = 0
    orig_lr = optimizer.param_groups[0]['lr']
    for b in range(0, n_iters, batch_size):
        if use_sgdr:
            if b % period == 0:
                setOptimizerLr(optimizer,orig_lr*sgdr(period, b))
        if (b+batch_size) %  n > b % n:
            optimizer.zero_grad()
            variables = (torch.Tensor(train.loc[b % n:(b+batch_size) %  n, :].values), torch.Tensor(train_response.loc[b % n:(b+batch_size) % n].values))
            if use_cuda:
                variables = (variables[0].cuda(), variables[1].cuda())
            output = model(variables[0].unsqueeze(1))
            loss = criterion(output.squeeze(1), variables[1])
            total += loss.item()
            loss.backward()
            optimizer.step()
            if b % print_every == 0 and b > 0:
                print('Loss total for the past '+str(print_every)+' examples is '+str(total))
                writer.add_scalar('train_loss', total, b)
                writer.add_text('hyperparams', text)
                for tag, value in model.named_parameters():
                    try:
                        tag = tag.replace('.', '/')
                        writer.add_histogram(tag, to_np(value), b)
                        writer.add_histogram(tag+'/grad', to_np(value.grad), b)
                    except AttributeError:
                        pdb.set_trace()
                total = 0
                total_val_loss = 0
                for k in range(0, len(val), batch_size):
                    val_vars = (torch.Tensor(val.loc[k:k+batch_size, :].values), torch.Tensor(val_response.loc[k:k+batch_size].values))
                    if use_cuda:
                        val_vars = (val_vars[0].cuda(), val_vars[1].cuda())
                    val_output = model(val_vars[0].unsqueeze(1))
                    val_loss = criterion(val_output.squeeze(1), val_vars[1])
                    total_val_loss += val_loss.item()
                print('Val loss is '+str(total_val_loss))
                writer.add_scalar('val_loss', total_val_loss, b)
                # Embedding a little hard to pull off
#                writer.add_embedding(val_output, metadata=val_vars[1])
                if total_val_loss < min_val_loss:
                    min_val_loss = total_val_loss
                    total_test_loss = 0
                    for j in range(0, len(test), batch_size):
                        test_vars = (torch.Tensor(test.loc[j:j+batch_size, :].values), torch.Tensor(test_response.loc[j:j+batch_size].values))
                        if use_cuda:
                            test_vars = (test_vars[0].cuda(), test_vars[1].cuda())
                        test_output = model(test_vars[0].unsqueeze(1))
                        test_loss = criterion(test_output.squeeze(1), test_vars[1])
                        total_test_loss += test_loss.item()
                    print('Test loss is '+str(total_test_loss))
                    assc_test_loss = total_test_loss
                    writer.add_scalar('test_loss', assc_test_loss, b)
    return total_test_loss