import sys
sys.path.append('../')
import os

import numpy as np
from sklearn import svm
from sklearn.linear_model import Perceptron
import copy
import random

import matplotlib.pyplot as plt
import matplotlib 
import pandas as pd
import seaborn as sbn
from contextlib import redirect_stdout
import json

import Reinforce as rln


def tuning_curves(relevant_neurons, relevant_weights, X, stimuli, network, label, size):
    
    #frates = X.T   
    #frates_rid = frates[relevant_neurons]
    #x_values = np.zeros(stimuli.shape[0])
    #stimuli_rid = stimuli
    
    indices = np.lexsort(stimuli.T)
    stimuli = stimuli[indices]
    frates = X[indices].T 
    frates_rid = frates[relevant_neurons]
    
    split_indices = np.where(np.any(stimuli[:-1] != stimuli[1:], axis=1))[0] + 1
    split_stimuli = np.split(stimuli, split_indices)
    split_frates_rid = np.split(frates_rid.T, split_indices)

    block_chain = np.zeros(len(relevant_neurons))    
    for block in split_frates_rid:
        block = block.T
        block = np.mean(block, axis=1)
        block_chain = np.vstack((block_chain, block))
        
    block_chain = block_chain[1:].T
    frates_rid = block_chain
    
    stimuli_rid = np.zeros((len(split_stimuli),4))    
    for i, segment in enumerate(split_stimuli):
        stimuli_rid[i] = segment[0]
        
    x_values = np.zeros(stimuli_rid.shape[0])
            
    if network == "ACTOR":
        color = "purple"
    else:
        color = "green"
    
    if label == "ACTIONS" or label == "ACTIONS_random":
        for i in range(len(x_values)):
            x_values[i] = stimuli_rid[i][0]*stimuli_rid[i][1] - stimuli_rid[i][2]*stimuli_rid[i][3]
        x_label = "v1p1-v2p2"
        saving_path = "tuning_curves_"+size+"/"+label+"/"
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        saving_path = "tuning_curves_"+size+"/"+label+"/"+network+" network.png"
        title = "Firing rates of the most relevant neurons\nencoding for "+\
                label+"' values in the "+network+" network"
        if label == "ACTIONS_random":
            color = "black"
            saving_path = "tuning_curves_"+size+"/actions/"+network+" network - random.png"
            title = "Firing rates of "+str(len(relevant_neurons))+\
                    " random neurons\nsorted by actions' values in the "+network+" network"
   
    #----------------------------------------------------------------------------------------#            
    
    if label == "RIGHT VALUES" or label == "RIGHT VALUES_random":
        for i in range(len(x_values)):
            x_values[i] = stimuli_rid[i][0]*stimuli_rid[i][1]
        x_label = "v1p1"
        saving_path = "tuning_curves_"+size+"/"+label+"/"
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        saving_path = "tuning_curves_"+size+"/"+label+"/"+network+" network.png"
        title = "Firing rates of the most relevant neurons\nencoding for "+label+" in the "+network+" network"
        if label == "RIGHT VALUES_random":
            color = "black"
            saving_path = "tuning_curves_"+size+"/right_values/"+network+" network - random.png"
            title = "Firing rates of "+str(len(relevant_neurons))+\
                    " random neurons\nsorted by right_values in the "+network+" network"
    
    #----------------------------------------------------------------------------------------#            
    
    if label == "LEFT VALUES" or label == "LEFT VALUES_random":
        for i in range(len(x_values)):
            x_values[i] = stimuli_rid[i][2]*stimuli_rid[i][3]
        x_label = "v2p2"
        saving_path = "tuning_curves_"+size+"/"+label+"/"
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        saving_path = "tuning_curves_"+size+"/"+label+"/"+network+" network.png"
        title = "Firing rates of the most relevant neurons\nencoding for "+\
                label+" in the "+network+" network"
        if label == "LEFT VALUES_random":
            color = "black"
            saving_path = "tuning_curves_"+size+"/left_values/"+network+" network - random.png"
            title = "Firing rates of "+str(len(relevant_neurons))+\
                    " random neurons\nsorted by left_values in the "+network+" network"
                
    #----------------------------------------------------------------------------------------#
    
    if label == "GLOBAL VALUES" or label == "GLOBAL VALUES_random":
        for i in range(len(x_values)):
            x_values[i] = np.max((stimuli_rid[i][0]*stimuli_rid[i][1], stimuli_rid[i][2]*stimuli_rid[i][3]))
        x_label = "max(v1p1, v2p2)"
        saving_path = "tuning_curves_"+size+"/"+label+"/"
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        saving_path = "tuning_curves_"+size+"/"+label+"/"+network+" network.png"
        title = "Firing rates of the most relevant neurons\nencoding for "+\
                label+" in the "+network+" network"
        if label == "GLOBAL VALUES_random":
            color = "black"
            saving_path = "tuning_curves_"+size+"/"+label+"/"+network+" network - random.png"
            title = "Firing rates of "+str(len(relevant_neurons))+\
                    " random neurons\nsorted by global_values in the "+network+" network"
     
    #----------------------------------------------------------------------------------------#
    
    indices = np.argsort(x_values)
    x_values_ordered = x_values[indices]
    frates_rid_ordered = frates_rid.T[indices].T
    
    split_indices = np.where(x_values_ordered[:-1] != x_values_ordered[1:])[0] + 1
    split_x_values_ordered = np.split(x_values_ordered, split_indices)
    split_frates_ordered = np.split(frates_rid_ordered.T, split_indices)
    
    
    tc_mean = np.zeros(len(relevant_neurons))
    tc_std = np.zeros(len(relevant_neurons))
    x_mean = np.zeros(len(split_frates_ordered))
    for b, block in enumerate(split_frates_ordered):
        tc_mean = np.vstack((tc_mean, np.mean(block, axis=0)))
        tc_std = np.vstack((tc_std, np.std(block, axis=0)))
        x_mean[b] = np.mean(split_x_values_ordered[b])
        #print("\nhere\n", block, "\nhere\n")
    #print(tc_mean)
    tc_mean = tc_mean[1:, :]
    tc_std = tc_std[1:, :]
    
    fig, axx = plt.subplots(int(len(relevant_neurons)/2), 2, figsize=(15, 20))
    axx = axx.reshape(-1)
    for n, ax in enumerate(axx):
        for i in range(len(x_values)):
            ax.plot(x_values_ordered[i], frates_rid_ordered[n, i], "o", markersize=5, color=color, zorder=0, alpha=0.7)
            #ax.text(x_values[i]+0.03, frates_rid[n, i], str(stimuli_rid[i,:]), fontsize=10)
            ax.set_title("neuron %i" %(relevant_neurons[n]), size=20)
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)
            ax.set_xlabel(x_label, size=15)
            ax.set_ylabel("firing rate", size=15)
        #for i in range(len(tc_mean)):
        ax.plot(x_mean, tc_mean[:, n], "D-", markersize=5, zorder=1, color="black")
        ax.vlines(x_mean, tc_mean[:, n] - tc_std[:, n], tc_mean[:, n] + tc_std[:, n], color="black", linewidth=2, zorder=2)
    plt.tight_layout()      
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(title, size=25)
    plt.savefig(saving_path)
   
    for n in range(frates_rid.shape[0]):
        frates_rid[n,:] *= relevant_weights[n]
    lin_comb = np.sum(frates_rid, axis=0)
    lin_comb_ordered = lin_comb[indices]
    split_lin_comb_ordered = np.split(lin_comb_ordered, split_indices)
    
    lin_comb_mean = np.zeros(len(split_lin_comb_ordered))
    lin_comb_std = np.zeros(len(split_lin_comb_ordered))
    for b, block in enumerate(split_lin_comb_ordered):
        lin_comb_mean[b] = np.mean(block)
        lin_comb_std[b] = np.std(block)
    
    plt.figure(figsize=(16,6))
    for i in range(len(x_values)):
        plt.plot(x_values[i], lin_comb[i], "o", markersize=5, color=color, zorder=0, alpha=0.7)
    plt.plot(x_mean, lin_comb_mean, "D-", markersize=5, zorder=1, color="black")
    plt.vlines(x_mean, lin_comb_mean - lin_comb_std, lin_comb_mean + lin_comb_std, color="black", linewidth=2, zorder=1)
    plt.title("Average over all relevant neurons\n"+network+" network on "+label, size=20)   
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.xlabel(x_label, size=15)
    plt.ylabel("firing rate", size=15)
    
    if label[-7:] != "_random":
        plt.savefig("tuning_curves_"+size+"/"+label+"/"+network+" network_LinComb.png")

    if label[-7:] == "_random":
        label = label[:-7]
        plt.savefig("tuning_curves_"+size+"/"+label+"/"+network+" network - random_LinComb.png")        
    
#############################################################################################################
#============================================================================================================    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     

def new_tuning_curves(relevant_neurons, relevant_weights, X, stimuli, network, label):
    
    indices = np.lexsort(stimuli.T)
    stimuli = stimuli[indices]
    frates = X[indices].T 
    frates_rid = frates[relevant_neurons]
    
    split_indices = np.where(np.any(stimuli[:-1] != stimuli[1:], axis=1))[0] + 1
    split_stimuli = np.split(stimuli, split_indices)
    split_frates_rid = np.split(frates_rid.T, split_indices)

    block_chain = np.zeros(len(relevant_neurons))    
    for block in split_frates_rid:
        block = block.T
        block = np.mean(block, axis=1)
        block_chain = np.vstack((block_chain, block))
        
    block_chain = block_chain[1:].T
    frates_rid = block_chain
    
    stimuli_rid = np.zeros((len(split_stimuli), 4))    
    for i, segment in enumerate(split_stimuli):
        stimuli_rid[i] = segment[0]
        
    x_values = np.zeros(stimuli_rid.shape[0])
    color = []
    
    for i in range(len(x_values)):
        x_values[i] = (stimuli_rid[i][1]) / (stimuli_rid[i][3])
        if stimuli_rid[i][0] == 1:
            color.append("violet")
        elif stimuli_rid[i][0] == 3:
            color.append("lime")
        else:
            color.append("black")
    x_label = "v1p1/v2p2"
    
    saving_path = "tuning_curves/global_values/"
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    saving_path += network+" network_special.png"
    title = "Firing rates of the most relevant neurons\nencoding for "+\
            label+" in the "+network+" network"
    if label == "global_values_random":
        color = "black"
        saving_path += network+" network - random_special.png"
        title = "Firing rates of "+str(len(relevant_neurons))+\
                " random neurons\nsorted by global_values in the "+network+" network"
         
    #indices = np.argsort(x_values)
    #x_values_ordered = x_values[indices]
    #frates_rid_ordered = frates_rid.T[indices].T
    
    #split_indices = np.where(x_values_ordered[:-1] != x_values_ordered[1:])[0] + 1
    #split_x_values_ordered = np.split(x_values_ordered, split_indices)
    #split_frates_ordered = np.split(frates_rid_ordered.T, split_indices)
    #
    #tc_mean = np.zeros(len(relevant_neurons))
    #tc_std = np.zeros(len(relevant_neurons))
    #x_mean = np.zeros(len(split_frates_ordered))
    #for b, block in enumerate(split_frates_ordered):
    #    tc_mean = np.vstack((tc_mean, np.mean(block, axis=0)))
    #    tc_std = np.vstack((tc_std, np.std(block, axis=0)))
    #    x_mean[b] = np.mean(split_x_values_ordered[b])
    #    
    #tc_mean = tc_mean[1:, :]
    #tc_std = tc_std[1:, :]
        
    fig, axx = plt.subplots(int(len(relevant_neurons)/2), 2, figsize=(15, 200))
    axx = axx.reshape(-1)
    for n, ax in enumerate(axx):
        for i in range(len(x_values)):
            ax.plot(x_values[i], frates_rid[n, i], "o", markersize=5, color=color[i], zorder=0, alpha=0.7)
            #ax.plot(x_values_ordered[i], frates_rid_ordered[n, i], "o", markersize=5, color=color, zorder=0, alpha=0.7)
            #ax.text(x_values[i]+0.03, frates_rid[n, i], str(stimuli_rid[i,:]), fontsize=10)
            ax.set_title("neuron %i" %(relevant_neurons[n]), size=20)
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)
            ax.set_xlabel(x_label, size=15)
            ax.set_ylabel("firing rate", size=15)
        #for i in range(len(tc_mean)):
        #ax.plot(x_mean, tc_mean[:, n], "D-", markersize=5, zorder=1, color="black")
        #ax.vlines(x_mean, tc_mean[:, n] - tc_std[:, n], tc_mean[:, n] + tc_std[:, n], color="black", linewidth=2, zorder=2)
    plt.tight_layout()      
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(title, size=25)
    plt.savefig(saving_path)  
    
    #for n in range(frates_rid.shape[0]):
    #    frates_rid[n,:] *= relevant_weights[n]
    #lin_comb = np.sum(frates_rid, axis=0)
    #lin_comb_ordered = lin_comb[indices]
    #split_lin_comb_ordered = np.split(lin_comb_ordered, split_indices)
    #
    #lin_comb_mean = np.zeros(len(split_lin_comb_ordered))
    #lin_comb_std = np.zeros(len(split_lin_comb_ordered))
    #for b, block in enumerate(split_lin_comb_ordered):
    #    lin_comb_mean[b] = np.mean(block)
    #    lin_comb_std[b] = np.std(block)
    
    #plt.figure(figsize=(16,6))
    #for i in range(len(x_values)):
    #    plt.plot(x_values[i], lin_comb[i], "o", markersize=5, color=color, zorder=0, alpha=0.7)
    #plt.plot(x_mean, lin_comb_mean, "D-", markersize=5, zorder=1, color="black")
    #plt.vlines(x_mean, lin_comb_mean - lin_comb_std, lin_comb_mean + lin_comb_std, color="black", linewidth=2, zorder=1)
    #plt.title("Average over all relevant neurons\n"+network+" network on "+label, size=20)   
    #plt.tick_params(axis='x', labelsize=15)
    #plt.tick_params(axis='y', labelsize=15)
    #plt.xlabel(x_label, size=15)
    #plt.ylabel("firing rate", size=15)
    #
    #if label[-7:] != "_random":
    #    plt.savefig("tuning_curves/"+label+"/"+network+" network_LinComb.png")
#
    #if label[-7:] == "_random":
    #    label = label[:-7]
    #    plt.savefig("tuning_curves/"+label+"/"+network+" network - random_LinComb.png")  
        
#############################################################################################################
#============================================================================================================    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     

def critic_tuning_curves(timeav_values, overall_values):
    
    global_values = overall_values
    lista1 = []
    lista2 = []
    lista3 = []
    lista4 = []
    lista5 = []
    lista6 = []
    
    plt.figure(figsize=(25,7))
    
    for i in range(len(global_values)):
        if global_values[i] == 0.25:
            color="yellow"
            lista1.append(timeav_values[i])
        if global_values[i] == 0.5:
            color="orange"
            lista2.append(timeav_values[i])
        if global_values[i] == 0.75:
            color="red"
            lista3.append(timeav_values[i])
        if global_values[i] == 1:
            color="purple"
            lista4.append(timeav_values[i])
        if global_values[i] == 1.5:
            color="blue"
            lista5.append(timeav_values[i])
        if global_values[i] == 2.25:
            color="green"
            lista6.append(timeav_values[i])
            
        plt.plot(global_values[i], timeav_values[i], "o", color=color, zorder=1)
        
    plt.axhline(0.25, color="yellow", zorder=0)
    plt.axhline(0.5, color="orange", zorder=0)
    plt.axhline(0.75, color="red", zorder=0)
    plt.axhline(1, color="purple", zorder=0)
    plt.axhline(1.5, color="blue", zorder=0)
    plt.axhline(2.25, color="green", zorder=0)
    
    media1 = np.mean(lista1)
    std1 = np.std(lista1)
    media2 = np.mean(lista2)
    std2 = np.std(lista2)
    media3 = np.mean(lista3)
    std3 = np.std(lista3)
    media4 = np.mean(lista4)
    std4 = np.std(lista4)
    media5 = np.mean(lista5)
    std5 = np.std(lista5)
    media6 = np.mean(lista6)
    std6 = np.std(lista6)
    x_values = [0.25, 0.5, 0.75, 1, 1.5, 2.25]
    medie = [media1, media2, media3, media4, media5, media6]
    stds = [std1, std2, std3, std4, std5, std6]
    
    for i in range(6):
        plt.plot(x_values[i], medie[i], "D", color="black", markersize=10, zorder=2)
        plt.vlines(x_values[i]+0.02, medie[i] - stds[i], medie[i] + stds[i], color="black", linewidth=3, zorder=2)
    
    plt.xlabel("overall value", size=20)
    plt.ylabel("critic output", size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.title("critic output vs overall trials' values", size=25);
    plt.savefig("tuning_curves/critic_readout_neuron.png")
    
#############################################################################################################
#============================================================================================================    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    

def neurons_population(Xa, Xc, Y, label):
    
    saving_path = "neurons_population/"+label+"/"
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    
    X_col = [Xa, Xc]
    Y_original = copy.deepcopy(np.asarray(Y))

    dcs_final = np.zeros((128,2))

    network = "network"
    
    for net in range(2):
        
        if net == 0:
            network = "actor"
        elif net == 1: 
            network = "critic"
            
        X = np.asarray(X_col[net])
        Y = Y_original
        
        indices = np.argsort(Y, kind='mergesort')
        Y = Y[indices]
        frates = X[indices]

        split_indices = np.where(Y[:-1] != Y[1:])[0] + 1
        split_Y = np.split(Y, split_indices)
        split_frates = np.split(frates, split_indices)

        averages = np.zeros((128,2))
        stds = np.zeros((128,2))
        dcs = np.zeros(128)

        colors = ["red", "green"]

        for i in range(128):

            plt.figure(figsize=(6,4))

            for j in range(len(split_frates)):
                average = np.mean(split_frates[j][:,i])
                std = np.std(split_frates[j][:,i])
                plt.hist(split_frates[j][:,i], label="%i" %(split_Y[j][0]), alpha=0.7, color=colors[j])
                plt.hlines((j+1)*10, average - 2*std / 2, average + 2*std / 2, color="dark"+colors[j])
                plt.axvline(average, linewidth=2, color="dark"+colors[j])
                #plt.text(j, j, "$\mu$: %f\n$\sigma$: %f" %(average, std))
                averages[i,j] = average
                stds[i,j] = std
            plt.title("neuron %i - %s network" %(i+1, network), size=20)           
            plt.tick_params(axis='x', labelsize=15)
            plt.tick_params(axis='y', labelsize=15)
            plt.xlabel("firing rate", size=15)
            plt.ylabel("occurrences", size=15)
            plt.legend(fontsize=15, loc="upper right")
            plt.savefig(saving_path+"neuron "+str(i))
            plt.close()

            if averages[i,0] == 0 and averages[i,1] == 0:
                dcs[i] = 0
                dcs_final[i,net] = 0
            else:
                dcs[i] = (averages[i,0]-averages[i,1]) / (stds[i,0]+stds[i,1])
                dcs_final[i,net] = (averages[i,0]-averages[i,1]) / (stds[i,0]+stds[i,1])
        
        #df = pd.DataFrame(data={'':dcs})
        #display(df)
        #df = df.sort_values(by='')
        #display(df)
                
        #nan_mask = np.isnan(dcs)
        #dcs[nan_mask] = -7
        array1_list = averages.tolist()
        array2_list = stds.tolist()
        array3_list = dcs.tolist()
        array4_list = dcs_final.tolist()
        data = {
            "averages": array1_list,
            "stds": array2_list,
            "dcs": array3_list,
            "dcs_final": array4_list
        }
        with open(saving_path+'neurons_population_'+network+'.json', 'w') as json_file:
            json.dump(data, json_file)    