#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import torch
from datetime import datetime
from logger import Logger
from torch import optim
from numpy import random
import torch.nn as nn
import pdb
import gc
import numpy as np

"""
Created on Mon May  7 19:02:45 2018

@author: jkr
"""

use_cuda = torch.cuda.is_available()
torch.backends.cudnn.enabled = True
now = datetime.now()

def to_np(x):
    return x.data.cpu().numpy()

def saveModel(problem_name, model, iterate):
    pass


def batchedTrainIters(problem_name, pairs, model, criterion,
                      n_iters, batch_size=128, print_every=1000,
                      learning_rate=1e-2, text='', scheduler=False, scheduler_patience=100):
    """Function to train general models with batching.
    Args:
        pairs, list of tuples of Variables (Tensors in PyTorch 0.4.0+)--
        input first, target second.
        model, the model to be trained
        criterion, the loss function to be used
        n_iters,
    """
    start = time.time()
    n_examples = len(pairs)
    print_loss_total = 0  # Reset every print_every
    # Set the logger
    logdir = '/home/jkr/TensorboardLogs/'+problem_name+'/' + now.strftime('%Y%m%d-%H%M%S')
    logger = Logger(logdir)
    if text != '':
        logger.text_summary(text)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, ams_grad=True)
    if scheduler:
        schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                        patience=scheduler_patience)

    random.shuffle(pairs)
    val_pairs = pairs[int(.9*len(pairs)):]
    training_pairs = pairs[:int(.9*len(pairs))]

    min_val_loss = np.Inf

    for iter in range(0, n_iters, batch_size):
        if iter%n_examples<(iter+batch_size)%n_examples:
            training_batch = training_pairs[iter%n_examples:(iter+batch_size)%n_examples]
            
        else:
            list1 = training_pairs[iter%n_examples:]
            list2 = training_pairs[:(iter+batch_size)%n_examples]
            training_batch = list1+list2
            
        if training_batch:
            input_variables = [example[0] for example in training_batch]
            target_variables = [example[1] for example in training_batch]
            
            output = model(input_variables)

            loss = criterion(target_variables, output)
            
            schedule.step(loss)

            print_loss_total += loss.item()

        gc.collect()
        if iter % print_every == 0 and iter > 0:
            model.eval()
            val_batch = val_pairs[:batch_size]
            if val_batch and training_batch:
                model
                print('%s (%d %d%%) %.4f' % (time.time()-start,
                                             iter, iter / (n_iters)*1.0 , print_loss_total))
                print_loss_total = 0
                val_input = [example[0] for example in val_batch]
                val_target = [example[1] for example in val_batch]
                val_output = model(val_input)
                val_loss = criterion(val_output, val_target)
                if float(val_loss.item()) < min_val_loss:
                    min_val_loss = float(loss)
                    saveModel(problem_name, model, iter)
                #============ TensorBoard logging ============#
                # (1) Log the scalar values
                info = {
                    'loss': print_loss_total,
                    'val_loss': val_loss
                }
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, iter+1)
        
                # (2) Log values and gradients of the parameters (histogram)
                for tag, value in model.named_parameters():
                    try:
                        tag = tag.replace('.', '/')
                        logger.histo_summary(tag, to_np(value), iter+1)
                        logger.histo_summary(tag+'/grad', to_np(value.grad), iter+1)
                    except AttributeError:
                        pdb.set_trace()
                model.train()