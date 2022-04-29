#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed October 2020

@author: juanjosealcaraz

"""
import numpy as np
import matplotlib.pyplot as plt
import os

titles = ['Scenario 1', 'Scenario 2', 'Scenario 3']
scenarios = [0,1,2]
algo_names = ['KBRL_97', 'KBRL_99']
labels = ['KBRL 0.97', 'KBRL 0.99']
WINDOW = 400
SPAN = 20000

# titles = ['Scenario 3']
# scenarios = [3]
# WINDOW = 40
# SPAN = 1000

def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

# subplot
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 2), constrained_layout=True)
# fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(4, 6.0), constrained_layout=True)

for i, (j, title) in enumerate(zip(scenarios,titles)):
    # iterate over algorithms
    for algo, label in zip(algo_names, labels):
        data = False
        path = './results/scenario_{}/{}/'.format(j,algo)
        runs = 0

        # iterate over files
        for filename in os.listdir(path):
            if filename.endswith(".npz"):
                runs += 1
                histories = np.load(path + filename)
                # load data for each run
                if runs == 1:
                    adjustment = movingaverage(histories['adjusted'], WINDOW)
                else: # store the history of each run
                    adjustment = np.vstack((adjustment, movingaverage( histories['adjusted'], WINDOW)))
                            
        # average over different runs
        adjustment_mean = np.mean(adjustment, axis=0)
        adjustment_std = np.std(adjustment, axis=0)
            
        # plot results
        steps = np.arange(SPAN)

        axs[i].set_title(title)
        axs[i].plot(steps, adjustment_mean[0:SPAN], label = label)
        axs[i].fill_between(steps, adjustment_mean[0:SPAN] - 1.95 * adjustment_std[0:SPAN] / np.sqrt(runs), 
                            adjustment_mean[0:SPAN] + 1.95 * adjustment_std[0:SPAN] / np.sqrt(runs), color = '#DDDDDD')

        if algo == algo_names[-1]:
            axs[i].set_ylim((0.,0.5))
            axs[i].set_xlabel('Stages')  # Add an x-label to the axes.
            axs[i].set_ylabel('adjustments')
            axs[i].legend(loc='best')
            axs[i].grid()

fig.savefig('./figures/adjustments.png', format='png')          
