import torch
import torch.nn as nn
import numpy as np
import tensorboardX
import datetime

use_cuda = torch.cuda.is_available()


def batchedTrainIters(name, n_iters, batch_size, print_every, model, optimizer, criterion,  data):
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
    writer = tensorboardX.SummaryWriter('~/Documents/TensorboardLogs/'+name+'/'+now)
    train, train_response = data[0][0], data[0][1]
    val, val_response = data[1][0], data[1][1]
    test, test_response = data[2][0], data[2][1]
    min_val_loss = np.Inf
    assc_test_loss = np.Inf
    n = len(train)
    for b in range(0, n_iters, batch_size):
        if (b+batch_size) %  n > b % n:
            optimizer.zero_grad()
            variables = (torch.Tensor(train.loc[b % n:(b+batch_size) %  n, cols].values), torch.Tensor(train_response.loc[b % n:(b+batch_size) % n].values))
            if use_cuda:
                variables = (variables[0].cuda(), variables[1].cuda())
            output = model(variables[0].unsqueeze(1))
            loss = criterion(output.squeeze(1).squeeze(1), variables[1])
            total += loss.item()
            loss.backward()
            optimizer.step()
            if b % print_every == 0 and b > 0:
                print('Loss total for the past thousand examples is '+str(total))
                writer.add_scalar('train_loss', total)
                total = 0
                total_val_loss = 0
                for k in range(0, len(val), batch_size):
                    val_vars = (torch.Tensor(val.loc[k:k+batch_size, cols].values), torch.Tensor(val_response.loc[k:k+batch_size].values))
                    if use_cuda:
                        val_vars = (val_vars[0].cuda(), val_vars[1].cuda())
                    val_output = model(val_vars[0].unsqueeze(1))
                    val_loss = criterion(val_output.squeeze(1).squeeze(1), val_vars[1])
                    total_val_loss += val_loss.item()
                print('Val loss is '+str(total_val_loss))
                writer.add_scalar('val_loss', total_val_loss)
                writer.add_embedding(val_output, metadata=val_vars[1])
                if total_val_loss < min_val_loss:
                    for j in range(0, len(test), batch_size):
                        test_vars = (torch.Tensor(test.loc[j:j+batch_size, cols].values), torch.Tensor(test_response.loc[j:j+batch_size].values))
                        if use_cuda:
                            test_vars = (test_vars[0].cuda(), test_vars[1].cuda())
                        test_output = model(test_vars[0].unsqueeze(1))
                        test_loss = criterion(test_output.squeeze(1).squeeze(1), test_vars[1])
                        total_test_loss += test_loss.item()
                    print('Test loss is '+str(total_test_loss))
                    assc_test_loss = total_test_loss
                    writer.add_scalar(total_test_loss)