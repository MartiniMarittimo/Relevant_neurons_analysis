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


def small_dataset_gen(iterations):
    
    input_values = np.array([1, 2, 3])
    input_probabilities = np.array([0.25, 0.5, 0.75])
    
    new_frates_actor = np.zeros(128)
    new_frates_critic = np.zeros(128)
    new_stimuli = np.zeros(4)
    fin_actions = []
    right_values = []
    left_values = []
    new_overall_values = []
    global_values = []
    new_timeav_values = []
    
    for i, v1 in enumerate(input_values):
        for j, p1 in enumerate(input_probabilities):
            for k, v2 in enumerate(input_values):
                for l, p2 in enumerate(input_probabilities):
                    
                    v1_array = np.array([v1])
                    p1_array = np.array([p1])
                    v2_array = np.array([v2])
                    p2_array = np.array([p2])
                    
                    reinforce = rln.REINFORCE(name_load_actor="models/RL_actor_network_good.pt",
                                              name_load_critic="models/RL_critic_network_good.pt",
                                              v1s=v1_array, p1s=p1_array, v2s=v2_array, p2s=p2_array)
    
                    observations, rewards, actions,\
                    log_action_probs, entropies, values,\
                    trial_begins, errors, frates_actor, frates_critic,\
                    timeav_values, final_actions, overall_values, stimuli = reinforce.experience(iterations)
                            
                    frates_actor = np.mean(frates_actor, axis=1)
                    frates_critic = np.mean(frates_critic, axis=1)
                    stimuli = np.mean(stimuli, axis=0)
                    overall_values = np.mean(overall_values)
                    timeav_values = np.mean(timeav_values)
                    
                    if  stimuli[0]*stimuli[1] < stimuli[2]*stimuli[3]:
                        fin_actions.append(-1)
                    elif stimuli[0]*stimuli[1] > stimuli[2]*stimuli[3]:
                        fin_actions.append(1)
                    else:
                        a = random.choice([1, -1])
                        fin_actions.append(a)
                    
                    if  stimuli[0]*stimuli[1] <= 1:
                        right_values.append(-1)
                    else:
                        right_values.append(1)
                    
                    if  stimuli[2]*stimuli[3] <= 1:
                        left_values.append(-1)
                    else:
                        left_values.append(1)
                        
                    if overall_values <= 1:
                        global_values.append(-1) 
                    else:
                        global_values.append(1) 
                    
                    new_frates_actor = np.vstack((new_frates_actor, frates_actor))
                    new_frates_critic = np.vstack((new_frates_critic, frates_critic))
                    new_stimuli = np.vstack((new_stimuli, stimuli))
                    new_overall_values.append(overall_values)
                    new_timeav_values.append(timeav_values)
    
    new_frates_actor = new_frates_actor[1:, :]
    new_frates_critic = new_frates_critic[1:, :]
    new_stimuli = new_stimuli[1:, :]
    fin_actions = np.asarray(fin_actions)
    right_values = np.asarray(right_values)
    left_values = np.asarray(left_values)
    new_overall_values = np.asarray(new_overall_values)
    global_values = np.asarray(global_values)
    new_timeav_values = np.asarray(new_timeav_values)
   
    array1_list = new_frates_actor.tolist()
    array2_list = new_frates_critic.tolist()
    array3_list = fin_actions.tolist()
    array4_list = right_values.tolist()
    array5_list = left_values.tolist()
    array6_list = new_overall_values.tolist()
    array7_list = global_values.tolist()
    array8_list = new_stimuli.tolist()
    array9_list = new_timeav_values.tolist()
    
    data = {
        "frates_actor": array1_list,
        "frates_critic": array2_list,
        "final_actions": array3_list,
        "right_values": array4_list,
        "left_values": array5_list,
        "overall_values": array6_list,
        "global_values": array7_list,    
        "stimuli": array8_list,
        "timeav_values": array9_list
    }
    
    with open('small_dataset.json', 'w') as json_file:
        json.dump(data, json_file)

#############################################################################################################
#============================================================================================================    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    

def big_dataset_gen(iterations):
    
    reinforce = rln.REINFORCE(name_load_actor="models/RL_actor_network_good.pt",
                              name_load_critic="models/RL_critic_network_good.pt")
    
    observations, rewards, actions,\
    log_action_probs, entropies, values,\
    trial_begins, errors, frates_actor, frates_critic,\
    timeav_values, final_actions, overall_values, stimuli = reinforce.experience(iterations)
    
    right_values = np.zeros(len(overall_values))
    left_values = np.zeros(len(overall_values))
    global_values = copy.deepcopy(overall_values)
    
    for i in range (len(global_values)):
        if  stimuli[i,0]*stimuli[i,1] < stimuli[i,2]*stimuli[i,3]:
            final_actions[i] = -1
        elif stimuli[i,0]*stimuli[i,1] > stimuli[i,2]*stimuli[i,3]:
            final_actions[i] = 1
        
        if  stimuli[i,0]*stimuli[i,1] <= 1:
            right_values[i] = -1
        else:
            right_values[i] = 1
        
        if  stimuli[i,2]*stimuli[i,3] <= 1:
            left_values[i] = -1
        else:
            left_values[i] = 1
        
        if global_values[i] <= 1:
            global_values[i] = -1
        else:
            global_values[i] = 1
    
    array1_list = frates_actor[:, 1:].T.tolist()
    array2_list = frates_critic[:, 1:].T.tolist()
    array3_list = final_actions[1:].tolist()
    array4_list = right_values[1:].tolist()
    array5_list = left_values[1:].tolist()
    array6_list = overall_values[1:].tolist()
    array7_list = global_values[1:].tolist()
    array8_list = stimuli[1:].tolist()
    array9_list = timeav_values[1:].tolist()
    
    data = {
        "frates_actor": array1_list,
        "frates_critic": array2_list,
        "final_actions": array3_list,
        "right_values": array4_list,
        "left_values": array5_list,
        "overall_values": array6_list,
        "global_values": array7_list,    
        "stimuli": array8_list,
        "timeav_values": array9_list
    }
    
    with open('big_dataset.json', 'w') as json_file:
        json.dump(data, json_file)

#############################################################################################################
#============================================================================================================    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    

def special_dataset_gen(iterations=100):
    
    input_values = np.array([1, 3])
    input_probabilities = np.array([0.25, 0.5, 0.75])
    
    new_frates_actor = np.zeros(128)
    new_frates_critic = np.zeros(128)
    new_stimuli = np.zeros(4)
    global_values = []
    
    for i, v1 in enumerate(input_values):
        for j, p1 in enumerate(input_probabilities):
            for k, v2 in enumerate(input_values):
                for l, p2 in enumerate(input_probabilities):
                    
                    if v1 != v2:
                        
                        v1_array = np.array([v1])
                        p1_array = np.array([p1])
                        v2_array = np.array([v2])
                        p2_array = np.array([p2])
                        
                        reinforce = rln.REINFORCE(name_load_actor="models/RL_actor_network_good.pt",
                                                  name_load_critic="models/RL_critic_network_good.pt",
                                                  v1s=v1_array, p1s=p1_array, v2s=v2_array, p2s=p2_array)
        
                        observations, rewards, actions,\
                        log_action_probs, entropies, values,\
                        trial_begins, errors, frates_actor, frates_critic,\
                        timeav_values, final_actions, overall_values, stimuli = reinforce.experience(iterations)
                                
                        #frates_actor = np.mean(frates_actor, axis=1)
                        #frates_critic = np.mean(frates_critic, axis=1)
                        #stimuli = np.mean(stimuli, axis=0)
                        
                        for i in range(len(overall_values)):
                            if overall_values[i] <= 1:
                                global_values.append(-1) 
                            else:
                                global_values.append(1) 
                                
                        print(frates_actor.shape, frates_critic.shape)
                        new_frates_actor = np.vstack((new_frates_actor, frates_actor.T))
                        new_frates_critic = np.vstack((new_frates_critic, frates_critic.T))
                        new_stimuli = np.vstack((new_stimuli, stimuli))
                        
    new_frates_actor = new_frates_actor[1:, :]
    new_frates_critic = new_frates_critic[1:, :]
    new_stimuli = new_stimuli[1:, :]
    global_values = np.asarray(global_values)
   
    array1_list = new_frates_actor.tolist()
    array2_list = new_frates_critic.tolist()
    array7_list = global_values.tolist()
    array8_list = new_stimuli.tolist()
    
    data = {
        "frates_actor": array1_list,
        "frates_critic": array2_list,
        "global_values": array7_list,    
        "stimuli": array8_list
    }
    
    with open('special_dataset.json', 'w') as json_file:
        json.dump(data, json_file)

#############################################################################################################
#============================================================================================================    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    

def tuning_curves(relevant_neurons, relevant_weights, X, stimuli, network, label):
    
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
            
    if network == "actor":
        color = "purple"
    else:
        color = "green"
    
    if label == "actions" or label == "actions_random":
        for i in range(len(x_values)):
            x_values[i] = stimuli_rid[i][0]*stimuli_rid[i][1] - stimuli_rid[i][2]*stimuli_rid[i][3]
        x_label = "v1p1-v2p2"
        saving_path = "tuning_curves/actions/"
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        saving_path = "tuning_curves/actions/"+network+" network.png"
        title = "Firing rates of the most relevant neurons\nencoding for "+\
                label+"' values in the "+network+" network"
        if label == "actions_random":
            color = "black"
            saving_path = "tuning_curves/actions/"+network+" network - random.png"
            title = "Firing rates of "+str(len(relevant_neurons))+\
                    " random neurons\nsorted by actions' values in the "+network+" network"
   
    #----------------------------------------------------------------------------------------#            
    
    if label == "right_values" or label == "right_values_random":
        for i in range(len(x_values)):
            x_values[i] = stimuli_rid[i][0]*stimuli_rid[i][1]
        x_label = "v1p1"
        saving_path = "tuning_curves/right_values/"
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        saving_path = "tuning_curves/right_values/"+network+" network.png"
        title = "Firing rates of the most relevant neurons\nencoding for "+\
                label+" in the "+network+" network"
        if label == "right_values_random":
            color = "black"
            saving_path = "tuning_curves/right_values/"+network+" network - random.png"
            title = "Firing rates of "+str(len(relevant_neurons))+\
                    " random neurons\nsorted by right_values in the "+network+" network"
    
    #----------------------------------------------------------------------------------------#            
    
    if label == "left_values" or label == "left_values_random":
        for i in range(len(x_values)):
            x_values[i] = stimuli_rid[i][2]*stimuli_rid[i][3]
        x_label = "v2p2"
        saving_path = "tuning_curves/left_values/"
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        saving_path = "tuning_curves/left_values/"+network+" network.png"
        title = "Firing rates of the most relevant neurons\nencoding for "+\
                label+" in the "+network+" network"
        if label == "left_values_random":
            color = "black"
            saving_path = "tuning_curves/left_values/"+network+" network - random.png"
            title = "Firing rates of "+str(len(relevant_neurons))+\
                    " random neurons\nsorted by left_values in the "+network+" network"
                
    #----------------------------------------------------------------------------------------#
    
    if label == "global_values" or label == "global_values_random":
        for i in range(len(x_values)):
            x_values[i] = np.max((stimuli_rid[i][0]*stimuli_rid[i][1], stimuli_rid[i][2]*stimuli_rid[i][3]))
        x_label = "max(v1p1, v2p2)"
        saving_path = "tuning_curves/global_values/"
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        saving_path = "tuning_curves/global_values/"+network+" network.png"
        title = "Firing rates of the most relevant neurons\nencoding for "+\
                label+" in the "+network+" network"
        if label == "global_values_random":
            color = "black"
            saving_path = "tuning_curves/global_values/"+network+" network - random.png"
            title = "Firing rates of "+str(len(relevant_neurons))+\
                    " random neurons\nsorted by global_values in the "+network+" network"
     
    #----------------------------------------------------------------------------------------#
    
    indices = np.argsort(x_values)
    x_values_ordered = x_values[indices]
    frates_rid_ordered = frates_rid.T[indices].T
    
    split_indices = np.where(x_values_ordered[:-1] != x_values_ordered[1:])[0] + 1
    split_x_values_ordered = np.split(x_values_ordered, split_indices)
    split_frates_ordered = np.split(frates_rid_ordered.T, split_indices)
    
    
    tc_mean = np.zeros(10)
    tc_std = np.zeros(10)
    x_mean = np.zeros(len(split_frates_ordered))
    for b, block in enumerate(split_frates_ordered):
        tc_mean = np.vstack((tc_mean, np.mean(block, axis=0)))
        tc_std = np.vstack((tc_std, np.std(block, axis=0)))
        x_mean[b] = np.mean(split_x_values_ordered[b])
        #print("\nhere\n", block, "\nhere\n")
    #print(tc_mean)
    tc_mean = tc_mean[1:, :]
    tc_std = tc_std[1:, :]
    
    fig, axx = plt.subplots(5, 2, figsize=(15, 20))
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
        plt.savefig("tuning_curves/"+label+"/"+network+" network_LinComb.png")

    if label[-7:] == "_random":
        label = label[:-7]
        plt.savefig("tuning_curves/"+label+"/"+network+" network - random_LinComb.png")        
    
#############################################################################################################
#============================================================================================================    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    

def neurons_population(X, Y, network, label):
    
    saving_path = "neurons_population/"+label+"/"+network+" network/"
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    
    Y = np.asarray(Y)
    
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
        else:
            dcs[i] = (averages[i,0]-averages[i,1]) / (stds[i,0]+stds[i,1])
    
    #nan_mask = np.isnan(dcs)
    #dcs[nan_mask] = -7
    array1_list = averages.tolist()
    array2_list = stds.tolist()
    array3_list = dcs.tolist()
    
    
    data = {
        "averages": array1_list,
        "stds": array2_list,
        "dcs": array3_list
    }
    
    with open(saving_path+'neurons_population.json', 'w') as json_file:
        json.dump(data, json_file)
    
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
    
    
def new_tuning_curves(relevant_neurons, relevant_weights, X, stimuli, network, label):
    
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
            
    if network == "actor":
        color = "purple"
    else:
        color = "green"

                
    #----------------------------------------------------------------------------------------#
    
    if label == "global_values" or label == "global_values_random":
        for i in range(len(x_values)):
            x_values[i] = np.max((stimuli_rid[i][0]*stimuli_rid[i][1], stimuli_rid[i][2]*stimuli_rid[i][3]))
        x_label = "max(v1p1, v2p2)"
        saving_path = "tuning_curves/global_values/"
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        saving_path = "tuning_curves/global_values/"+network+" network.png"
        title = "Firing rates of the most relevant neurons\nencoding for "+\
                label+" in the "+network+" network"
        if label == "global_values_random":
            color = "black"
            saving_path = "tuning_curves/global_values/"+network+" network - random.png"
            title = "Firing rates of "+str(len(relevant_neurons))+\
                    " random neurons\nsorted by global_values in the "+network+" network"
     
    #----------------------------------------------------------------------------------------#
    
    indices = np.argsort(x_values)
    x_values_ordered = x_values[indices]
    frates_rid_ordered = frates_rid.T[indices].T
    
    split_indices = np.where(x_values_ordered[:-1] != x_values_ordered[1:])[0] + 1
    split_x_values_ordered = np.split(x_values_ordered, split_indices)
    split_frates_ordered = np.split(frates_rid_ordered.T, split_indices)
    
    
    tc_mean = np.zeros(10)
    tc_std = np.zeros(10)
    x_mean = np.zeros(len(split_frates_ordered))
    for b, block in enumerate(split_frates_ordered):
        tc_mean = np.vstack((tc_mean, np.mean(block, axis=0)))
        tc_std = np.vstack((tc_std, np.std(block, axis=0)))
        x_mean[b] = np.mean(split_x_values_ordered[b])
        #print("\nhere\n", block, "\nhere\n")
    #print(tc_mean)
    tc_mean = tc_mean[1:, :]
    tc_std = tc_std[1:, :]
    
    fig, axx = plt.subplots(5, 2, figsize=(15, 20))
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
        plt.savefig("tuning_curves/"+label+"/"+network+" network_LinComb.png")

    if label[-7:] == "_random":
        label = label[:-7]
        plt.savefig("tuning_curves/"+label+"/"+network+" network - random_LinComb.png")  