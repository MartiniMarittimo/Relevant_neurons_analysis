import sys
sys.path.append('../')
import os

import numpy as np
from sklearn import svm
from sklearn.linear_model import Perceptron
import copy

import matplotlib.pyplot as plt
import matplotlib 
import pandas as pd
import seaborn as sbn
from contextlib import redirect_stdout
import json

import Reinforce as rln



def noise_effect(X, Y, model, param_mag, network, label, noise_mag, size):
    
    saving_path = 'clf_data_'+size+'/'+label
    if not os.path.exists(saving_path):
            os.makedirs(saving_path)
            
    model = model
    
    if model == 'perceptronL1':
        C_perc = param_mag
    elif model == 'svm':
        C_svm = param_mag
        
    original_X = copy.deepcopy(X)
    mean = np.mean(original_X)
    list_test_scores = []
    average_test_scores = []
    list_test_random_scores = []
    
    for k in range(len(noise_mag)):
        
        X = copy.deepcopy(original_X)
        std = mean * noise_mag[k]
        #noise = np.random.normal(0, std, X.shape)
        #X += noise

        nb_epochs = 250

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
            
            noise = np.random.normal(0, std, X_train_trial.shape)
            X_train_trial += noise

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
            Y_test_trial = 2*np.random.binomial(size=X_test_trial.shape[0], n=1, p=0.5)-1 
            
            noise = np.random.normal(0, std, X_train_trial.shape)
            X_train_trial += noise

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

        print("average over "+str(nb_epochs)+" epochs of test scores: %.2f" %(np.mean(test_scores)))
        print("average over "+str(nb_epochs)+" epochs of test random scores: %.2f" %(np.mean(test_random_scores)))        
     
    fig, axx = plt.subplots(int(len(noise_mag)/2), 2, figsize=(12, 20))
    axx = axx.reshape(-1)
    for i, ax in enumerate(axx):
        bin_edges = np.linspace(0.2, 1, 50)
        ax.hist(list_test_scores[i], bins=bin_edges, label="test scores", edgecolor="k")
        ax.hist(list_test_random_scores[i], bins=bin_edges, label="test random scores", edgecolor="k")
        ax.set_title("%i) noise: %.2f$\mu$" %(i+1, noise_mag[i]), fontsize=15)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.set_xlabel("mean accuracy", size=15)
        ax.set_ylabel("occurences", size=15)
        ax.legend(fontsize=15, loc="upper left")
    plt.tight_layout()      
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.suptitle("TEST SCORES\n"+network+" network on "+label+\
                 "\n(mean neuronal activity $\mu$=%.2f)" %(mean), fontsize=20)
    plt.savefig(saving_path+'/hist_noise: '+network+' - '+label+'.png')
    plt.close()
    
    plt.figure(figsize=(10,5))
    plt.plot(noise_mag, average_test_scores, "-^", markersize=10, color="tomato")
    for i in range(len(noise_mag)):
        plt.text(noise_mag[i], average_test_scores[i], "%.3f" %(average_test_scores[i]), fontsize=10)
    plt.title("Mean test scores over noise magnitude\n("+network+" network on "+label+", C = %.2f)" %(param_mag), fontsize=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.xlabel("noise magnitude [$\mu$]", size=15)
    plt.ylabel("mean test score", size=15)
    plt.savefig(saving_path+'/'+network+' network mean test scores over noise.png')

#############################################################################################################
#============================================================================================================    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
    
def regularization_intensity(X, Y, model, param_mag, network, label, noise_mag, size):
    
    saving_path = 'clf_data_'+size+'/'+label
    if not os.path.exists(saving_path):
            os.makedirs(saving_path)
            
    model = model
    
    original_X = copy.deepcopy(X)
    mean = np.mean(original_X)
    std = mean * noise_mag
    X = copy.deepcopy(original_X)
    #noise = np.random.normal(0, std, X.shape)
    #X += noise
        
    list_test_scores = []
    average_test_scores = []
    list_test_random_scores = []
    
    for k in range(len(param_mag)):
        
        if model == 'perceptronL1':
            C_perc = param_mag[k]
        elif model == 'svm':
            C_svm = param_mag[k]

        nb_epochs = 50

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
            
            noise = np.random.normal(0, std, X_train_trial.shape)
            X_train_trial += noise

            if model=='perceptron':
                clf = Perceptron(tol=1e-3, random_state=0)
            elif model == 'perceptronL1':
                clf = Perceptron(tol=1e-5, random_state=0, penalty='l1', alpha=C_perc, max_iter=10000, n_iter_no_change=100)
            elif model == 'svm':
                clf = svm.LinearSVC(penalty='l1', C=C_svm, dual = False, max_iter=1000)

            clf.fit(X_train_trial, Y_train_trial)
            train_score = clf.score(X_train_trial, Y_train_trial)
            test_score = clf.score(X_test_trial, Y_test_trial)

            #test_scores[i] = test_score
            test_scores[i] = train_score

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
            Y_test_trial = 2*np.random.binomial(size=X_test_trial.shape[0], n=1, p=0.5)-1 
            
            noise = np.random.normal(0, std, X_train_trial.shape)
            X_train_trial += noise

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

        print("average over "+str(nb_epochs)+" epochs of test scores: %.2f" %(np.mean(test_scores)))
        print("average over "+str(nb_epochs)+" epochs of test random scores: %.2f" %(np.mean(test_random_scores)))        
     
    fig, axx = plt.subplots(int(len(param_mag)/2), 2, figsize=(12, 20))
    axx = axx.reshape(-1)
    for i, ax in enumerate(axx):
        bin_edges = np.linspace(0.2, 1, 50)
        ax.hist(list_test_scores[i], bins=bin_edges, label="test scores", edgecolor="k")
        ax.hist(list_test_random_scores[i], bins=bin_edges, label="test random scores", edgecolor="k")
        ax.set_title("%i) C= %.2f" %(i+1, param_mag[i]), fontsize=15)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.set_xlabel("mean accuracy", size=15)
        ax.set_ylabel("occurences", size=15)
        ax.legend(fontsize=15, loc="upper left")
    plt.tight_layout()      
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.suptitle("TEST SCORES\n"+network+" network on "+label+\
                 "\n(mean neuronal activity $\mu$=%.2f)" %(mean), fontsize=20)
    plt.savefig(saving_path+'/hist_regularization: '+network+' - '+label+'.png')
    plt.close()
    
    plt.figure(figsize=(10,5))
    plt.plot(param_mag, average_test_scores, "-^", markersize=10, color="lightseagreen")
    for i in range(len(param_mag)):
        plt.text(param_mag[i], average_test_scores[i], "%.3f" %(average_test_scores[i]), fontsize=10)
    plt.xscale("log")
    plt.title("Mean test scores over regularization parameter magnitude\n("+network+" network on "+label+", noise = %.2f)" %(noise_mag), fontsize=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.xlabel("regularization parameter magnitude", size=15)
    plt.ylabel("mean test score", size=15)
    plt.savefig(saving_path+'/'+network+' network mean test scores over regularization.png')
    
#############################################################################################################
#============================================================================================================    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    
def rel_neurons(X, Y, model, C, network, label, noise_mag, size):

    saving_path = 'clf_data_'+size+'/'+label
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
            
    model = model
            
    if model == 'perceptronL1':
        C_perc = C
    elif model == 'svm':
        C_svm = C
    
    mean = np.mean(X)
    std = mean * noise_mag
    #noise = np.random.normal(0, std, X.shape)
    #X += noise
    
    nb_trials = X.shape[0]
    percentage_training_set = 0.8
    nb_indeces_training = int(nb_trials*percentage_training_set)
    all_indeces = np.arange(0, nb_trials)
    np.random.shuffle(all_indeces)
    
    indeces_train = all_indeces[0:nb_indeces_training]
    indeces_test = all_indeces[nb_indeces_training:]
    X_train_trial = X[indeces_train,:]
    Y_train_trial = Y[indeces_train]
    X_test_trial = X[indeces_test,:]
    Y_test_trial = Y[indeces_test]
    
    noise = np.random.normal(0, std, X_train_trial.shape)
    X_train_trial += noise
    
    if model=='perceptron':
        clf = Perceptron(tol=1e-3, random_state=0)
    elif model == 'perceptronL1':
        clf = Perceptron(tol=1e-3, random_state=0, penalty='l1', alpha=C_perc)
    elif model == 'svm':
        clf = svm.LinearSVC(penalty='l1', C=C_svm, dual=False, max_iter=1000)
        
    clf.fit(X_train_trial, Y_train_trial)
    training_score = clf.score(X_train_trial, Y_train_trial)
    test_score = clf.score(X_test_trial, Y_test_trial)
    print("----------\ntraining score: %.3f" %(training_score))
    print("test score: %.3f" %(test_score), "\n----------")
    
    w = clf.coef_
    b = clf.intercept_
    
    relevant_neurons = []
    relevant_neurons_values = []
    relevant_neurons_values_abs = []
    plt.figure(figsize=(15,7))
    plt.plot(w[0,:])
    for i in range(len(w[0,:])):
        if w[0,i] != 0:
            plt.text(i, w[0,i], "(%i, %.3f)" %(i, w[0,i]), style='italic', fontsize=15)
            relevant_neurons_values.append(w[0,i])
            relevant_neurons_values_abs.append(np.abs(w[0,i]))
            relevant_neurons.append(i)
    plt.title(network+" network relevant neurons for "+label, fontsize=20);
    plt.xticks(size=15)
    plt.yticks(size=15)
    
    relevant_size = 10
    sorted_pairs = sorted(zip(relevant_neurons_values_abs, relevant_neurons))
    relevant_neurons = [pair[1] for pair in sorted_pairs]
    relevant_neurons.reverse()
    relevant_neurons = relevant_neurons[:relevant_size]
    sorted_pairs = sorted(zip(relevant_neurons_values_abs, relevant_neurons_values))
    relevant_neurons_values = [pair[1] for pair in sorted_pairs]
    relevant_neurons_values.reverse()
    relevant_neurons_values = relevant_neurons_values[:relevant_size]
    
    check = True
    random_neurons = np.zeros(len(relevant_neurons))
    while check is True:
        random_neurons = np.random.randint(0, 128, 10)
        bool_array = np.isin(random_neurons, relevant_neurons)
        check = any(bool_array)
    random_weights = w[0, random_neurons]
    
    array1_list = np.asarray(relevant_neurons).tolist()
    array2_list = np.asarray(relevant_neurons_values).tolist()
    array3_list = random_neurons.tolist()
    array4_list = random_weights.tolist()
    
    data = {
        "relevant_neurons": array1_list,
        "relevant_weights": array2_list,
        "random_neurons": array3_list,
        "random_weights": array4_list
    }
    
    if network == "actor":
        with open(saving_path+'/relevant_neurons_actor.json', 'w') as json_file:
            json.dump(data, json_file)
    else:
        with open(saving_path+'/relevant_neurons_critic.json', 'w') as json_file:
            json.dump(data, json_file)   

#############################################################################################################
#============================================================================================================    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
