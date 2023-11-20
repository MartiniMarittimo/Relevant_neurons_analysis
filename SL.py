import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, floor
import random
import time



def loss_mse(output, target, mask):
    """
    Mean squared error loss
    :param output: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param target: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :return: float
    """
    
    # Compute loss for each trial & timestep (average accross output neurons)
    loss_tensor = (mask * (target - output)).pow(2).mean(dim=-1)
    #print("loss_tensor: ", loss_tensor.shape)

    # Compute loss for each trial (average across timesteps)
    # and account also for different number of masked values per trial
    loss_by_trial = loss_tensor.sum(dim=-1) / mask[:, :, 0].sum(dim=-1)
    #print("loss_by_trial: ", loss_by_trial.shape)
    
    #Compute the final loss (average across trials)
    loss = loss_by_trial.mean()
    #print("loss: ", loss.shape)
    
    return loss



def train(net, input, target, mask, n_epochs, lr=1e-2, batch_size=100, plot_learning_curve=False,\
          plot_gradient=False, mask_gradients=False, clip_gradient=None, keep_best=False, cuda=False,\
          resample=False, lambda1=None, lambda2=None, save_loss=False):
    """
    Train a network
    :param net: nn.Module
    :param input: torch tensor of shape (num_trials, num_timesteps, input_dim)
    :param target: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :param n_epochs: int
    :param lr: float, learning rate
    :param batch_size: int
    :param plot_learning_curve: bool
    :param plot_gradient: bool
    :param clip_gradient: None or float, if not None the value at which gradient norm is clipped
    :param keep_best: bool, if True, model with lower loss from training process will be kept (for this option, the network has to implement a method clone())
    :param resample: for SupportLowRankRNNs, set True
    :return: nothing
    """
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    num_samples = input.shape[0] # NUMBER of trials, NUMERO totale di campionamenti i (samples) nel dataset D
    all_losses = []
    mean_losses = []
            
    with torch.no_grad():
        output = net(input)  # forward (in this first case using to the whole input dataset)
        initial_loss = loss_mse(output, target, mask)
        mean_losses.append(initial_loss.item()) 
        print("Initial loss: %.3f." % (initial_loss.item()))        
        
    for epoch in range(n_epochs):
        
        losses = []
        
        for mb in range(num_samples // batch_size): # mb numera i mini-batch --> in un'epoca ciclo su tutti i mini-batch
            optimizer.zero_grad()
            random_batch_idx = random.sample(range(num_samples), batch_size)
            batch = input[random_batch_idx]
            output = net(batch) # forward
            loss = loss_mse(output, target[random_batch_idx], mask[random_batch_idx])             
            
            # qui sto calcolando i gradienti quindi dL/dw_previous, con L funzione di costo: 
            # ho un aggiornamento per ogni mini-batch, per ogni epoca
            loss.backward() 
            
            # qui invece sto facendo l'update dei parametri, con i gradienti calcolati precedentemente:
            # ho un aggiornamento per ogni mini-batch, per ogni epoca
            optimizer.step()
            
            # Two important lines to prevent memory leaks
            loss.detach_()
            output.detach_()
            
            losses.append(loss.item())
            all_losses.append(loss.item())
        
        mean_losses.append(np.mean(losses))
        print("Loss: %.3f" % (mean_losses[epoch]))
        
    return mean_losses

    