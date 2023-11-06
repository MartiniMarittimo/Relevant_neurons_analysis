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
    new_overall_values = []
    new_timeav_values = []
    
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
                            new_overall_values.append(overall_values[i])
                            new_timeav_values.append(timeav_values[i])
                                
                        new_frates_actor = np.vstack((new_frates_actor, frates_actor.T))
                        new_frates_critic = np.vstack((new_frates_critic, frates_critic.T))
                        new_stimuli = np.vstack((new_stimuli, stimuli))
                        
                        
    new_frates_actor = new_frates_actor[1:, :]
    new_frates_critic = new_frates_critic[1:, :]
    new_stimuli = new_stimuli[1:, :]
    new_overall_values = np.asarray(new_overall_values)
    global_values = np.asarray(global_values)
    new_timeav_values = np.asarray(new_timeav_values)
    
    
    all_indeces = np.arange(0, len(global_values))
    np.random.shuffle(all_indeces)
    new_frates_actor = new_frates_actor[all_indeces, :]
    new_frates_critic = new_frates_critic[all_indeces, :]
    new_stimuli = new_stimuli[all_indeces, :]
    global_values = global_values[all_indeces]
    new_overall_values = new_overall_values[all_indeces]
    new_timeav_values = new_timeav_values[all_indeces]
   
    array1_list = new_frates_actor.tolist()
    array2_list = new_frates_critic.tolist()   
    array6_list = new_overall_values.tolist()
    array7_list = global_values.tolist()
    array8_list = new_stimuli.tolist()
    array9_list = new_timeav_values.tolist()
    
    data = {
        "frates_actor": array1_list,
        "frates_critic": array2_list,
        "overall_values": array6_list,
        "global_values": array7_list,    
        "stimuli": array8_list,
        "timeav_values": array9_list
    }
    
    with open('special_dataset.json', 'w') as json_file:
        json.dump(data, json_file)

#############################################################################################################
#============================================================================================================    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  