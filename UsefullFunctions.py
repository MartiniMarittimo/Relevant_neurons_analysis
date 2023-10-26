import sys
sys.path.append('../')
import os

import numpy as np
from sklearn import svm
from sklearn.linear_model import Perceptron

import matplotlib.pyplot as plt
import matplotlib 
import pandas as pd
import seaborn as sbn
from contextlib import redirect_stdout
import json

import Reinforce as rln





def frates_labels(iterations):
    
    reinforce = rln.REINFORCE(name_load_actor="models/RL_actor_network_good.pt",
                              name_load_critic="models/RL_critic_network_good.pt")
    
    observations, rewards, actions,\
    log_action_probs, entropies, values,\
    trial_begins, errors, frates_actor, frates_critic,\
    timeav_values, final_actions, global_values, stimuli = reinforce.experience(iterations)
    
    right_values = np.zeros(len(global_values))
    left_values = np.zeros(len(global_values))

    for i in range (len(global_values)):
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
    array6_list = global_values[1:].tolist()
    array7_list = stimuli[1:].tolist()
    
    data = {
        "frates_actor": array1_list,
        "frates_critic": array2_list,
        "final_actions": array3_list,
        "right_values": array4_list,
        "left_values": array5_list,
        "global_values": array6_list,    
        "stimuli": array7_list
    }
    
    with open('frates_labels.json', 'w') as json_file:
        json.dump(data, json_file)

#############################################################################################################
#============================================================================================================    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    

def rel_nurons(X, Y, model, C, network, label, noise_mag=0):
    
    saving_path = 'clf_data/'+label
    if not os.path.exists(saving_path):
            os.makedirs(saving_path)
            
    model = model
            
    if model == 'perceptronL1':
        C_perc = C
    elif model == 'svm':
        C_svm = C
        
    original_X = X
    mean = np.mean(X)
    list_test_scores = []
    average_test_scores = []
    list_test_random_scores = []
    
    for k in range(len(noise_mag)):
        
        X = original_X
        std = mean * noise_mag[k]
        noise = np.random.normal(0, std, X.shape)
        X += noise
                
        nb_epochs = 10
    
        test_scores = np.zeros(nb_epochs)
        
        nb_trials = X.shape[0]
        percentage_training_set = 0.8
        nb_indeces_training = int(nb_trials*percentage_training_set)
        
        for i in range(nb_epochs):
            
            all_indeces = np.arange(0, nb_trials)
            
            if i == 0:
                indeces_train = all_indeces[0:nb_indeces_training]
                indeces_test = all_indeces[nb_indeces_training:]
            else:
                np.random.shuffle(all_indeces)
                indeces_train = all_indeces[0:nb_indeces_training]
                indeces_test = all_indeces[nb_indeces_training:]
        
            X_train_trial = X[indeces_train,:]
            Y_train_trial = Y[indeces_train]
            X_test_trial = X[indeces_test,:]
            Y_test_trial = Y[indeces_test]
            
            if model=='perceptron':
                clf = Perceptron(tol=1e-3, random_state=0)
            elif model == 'perceptronL1':
                clf = Perceptron(tol=1e-3, random_state=0, penalty='l1', alpha=C_perc)
            elif model == 'svm':
                clf = svm.LinearSVC(penalty='l1', C=C_svm, dual = False, max_iter=1000)
        
            clf.fit(X_train_trial, Y_train_trial)
            test_score = clf.score(X_test_trial, Y_test_trial)
        
            test_scores[i] = test_score
            
        list_test_scores.append(test_scores)
        average_test_scores.append(np.mean(test_scores))
        
        #----------------------------------------------------------------------------------------#
    
        test_random_scores = np.zeros(nb_epochs)
    
        for i in range(nb_epochs):
            
            all_indeces = np.arange(0, nb_trials)
            
            if i == 0:
                indeces_train = all_indeces[0:nb_indeces_training]
                indeces_test = all_indeces[nb_indeces_training:]
            else:
                np.random.shuffle(all_indeces)
                indeces_train = all_indeces[0:nb_indeces_training]
                indeces_test = all_indeces[nb_indeces_training:]
        
            X_train_trial = X[indeces_train,:]
            Y_train_trial = Y[indeces_train]
            X_test_trial = X[indeces_test,:]
            Y_test_trial = 2*np.random.binomial(size=200, n=1, p=0.5)-1 
            
            if model=='perceptron':
                clf = Perceptron(tol=1e-3, random_state=0)
            elif model == 'perceptronL1':
                clf == Perceptron(tol=1e-3, random_state=0, penalty='l1', alpha=C_perc)
            elif model == 'svm':
                clf = svm.LinearSVC(penalty='l1', C=C_svm, dual=False, max_iter=1000)
            
            clf.fit(X_train_trial, Y_train_trial)
            test_score = clf.score(X_test_trial, Y_test_trial)
            
            test_random_scores[i] = test_score
            
        list_test_random_scores.append(test_random_scores)
        
        print("average over 10 epochs of test scores: %.3f" %(np.mean(test_scores)))
        print("average over 10 epochs of test random scores: %.3f" %(np.mean(test_random_scores)))        
     
    fig, axx = plt.subplots(len(noise_mag), 1, figsize=(6, 20))
    axx = axx.reshape(-1)
    for i, ax in enumerate(axx):
        bin_edges = np.linspace(0.2, 1, 50)
        ax.hist(list_test_scores[i], bins=bin_edges, label="test scores", edgecolor="k")
        ax.hist(list_test_random_scores[i], bins=bin_edges, label="test random scores", edgecolor="k")
        ax.set_title("noise: %.2f$\mu$" %(noise_mag[i]), fontsize=15)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.set_xlabel("mean accuracy", size=15)
        ax.set_ylabel("occurences", size=15)
        ax.legend(fontsize=15, loc="upper left")
    plt.tight_layout()      
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("test scores for the "+network+" network on "+label+\
                 "\n(average neuronal activity $\mu$=%.2f)" %(mean), fontsize=20)
    plt.savefig('clf_data/'+label+'/hist: '+network+' - '+label+'.png')
    
    plt.figure(figsize=(10,5))
    plt.plot(noise_mag, average_test_scores, "-^", markersize=10)
    plt.title("mean test scores over noise magnitude for the "+network+" network on "+label, fontsize=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.xlabel("noise magnitude [$\mu$/$\mu_0$]", size=15)
    plt.ylabel("mean test score", size=15)
    plt.savefig(saving_path+'/'+network+' network mean test scores.png')      
    
    #----------------------------------------------------------------------------------------#

    all_indeces = np.arange(0, nb_trials)
    np.random.shuffle(all_indeces)
    indeces_train = all_indeces[0:nb_indeces_training]
    indeces_test = all_indeces[nb_indeces_training:]
    X = original_X
    X_train_trial = X[indeces_train,:]
    Y_train_trial = Y[indeces_train]
    X_test_trial = X[indeces_test,:]
    Y_test_trial = Y[indeces_test]
    
    if model=='perceptron':
        clf = Perceptron(tol=1e-3, random_state=0)
    elif model == 'perceptronL1':
        clf == Perceptron(tol=1e-3, random_state=0, penalty='l1', alpha=C_perc)
    elif model == 'svm':
        clf = svm.LinearSVC(penalty='l1', C=C_svm, dual=False, max_iter=1000)
        
    clf.fit(X_train_trial, Y_train_trial)
    training_score = clf.score(X_train_trial, Y_train_trial)
    test_score = clf.score(X_test_trial, Y_test_trial)
    print("----------\ntraining score: %.3f" %(training_score))
    print("test score: %.3f" %(test_score), "\n----------")
    
    w = clf.coef_
    b = clf.intercept_
    
    df = pd.DataFrame(w[0,:])
    df.to_csv("clf_data/"+label+"/perceptron_wo_"+network+".csv", index=False)
     
    relevant_neurons = []
    relevant_neurons_values = []
    plt.figure(figsize=(15,7))
    plt.plot(w[0,:])
    for i in range(len(w[0,:])):
        if w[0,i] != 0:
            plt.text(i, w[0,i], str(i), style='italic', fontsize=15)
            relevant_neurons_values.append(np.abs(w[0,i]))
            relevant_neurons.append(i)
    plt.title(network+" network relevant neurons for "+label, fontsize=20);
    plt.xticks(size=15)
    plt.yticks(size=15)
    
    sorted_pairs = sorted(zip(relevant_neurons_values, relevant_neurons))
    relevant_neurons = [pair[1] for pair in sorted_pairs]
    relevant_neurons.reverse()
    relevant_size = 10
    relevant_neurons = relevant_neurons[:relevant_size]
    
    if network == "actor":
        with open('clf_data/'+label+'/relevant_neurons_actor.txt', 'w') as f:
            with redirect_stdout(f):
                print(relevant_neurons)
    else:
        with open('clf_data/'+label+'/relevant_neurons_critic.txt', 'w') as f:
            with redirect_stdout(f):
                print(relevant_neurons)    

#############################################################################################################
#============================================================================================================    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    

def tuning_curves(relevant_neurons, X, stimuli, network, label):
    
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
    
    fig, axx = plt.subplots(5, 2, figsize=(15, 20))
    axx = axx.reshape(-1)
    for n, ax in enumerate(axx):
        for i in range(len(x_values)):
            ax.plot(x_values[i], frates_rid[n, i], "o", markersize=5, color=color)
            #ax.text(x_values[i]+0.03, frates_rid[n, i], str(stimuli_rid[i,:]), fontsize=10)
            ax.set_title("neuron %i" %(relevant_neurons[n]), size=20)
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)
            ax.set_xlabel(x_label, size=15)
            ax.set_ylabel("firing rate", size=15)
    plt.tight_layout()      
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(title, size=25)
    plt.savefig(saving_path)
    
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

