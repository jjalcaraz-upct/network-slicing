#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed October 2020

@author: juanjosealcaraz

"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# trainning results
WINDOW = 400
START = 0
END = 20000 # up to 39900
algo_names = ['A2C', 'PPO1', 'PPO2', 'TRPO', 'SAC', 'TD3', 'NAF', 'KBRL_97','KBRL_99']
labels = ['A2C', 'PPO1', 'PPO2', 'TRPO', 'SAC', 'TD3', 'NAF', 'KBRL 0.97', 'KBRL 0.99']

SPAN = END - START

prbs_values = [200, 150, 100]
scenarios = [0,1,2]

def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

if __name__=='__main__':
    try:
        scenario = int(sys.argv[1])
    except IndexError:
        scenario = 0

    if scenario not in scenarios:
        scenario = 0

    dir_path = './results/scenario_{}/'.format(scenario)
    prbs = prbs_values[scenario]

    # subplot
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 3.5), constrained_layout=True)

    # iterate over algorithms
    for algo, label in zip(algo_names, labels):
        violations = np.empty([1])
        actions = np.empty([1])
        regret = np.empty([1])
        data = False
        proposal = False
        path = './results/scenario_{}/{}/'.format(scenario, algo)
        runs = 0

        # iterate over files
        for filename in os.listdir(path):
            if filename.endswith(".npz"):
                histories = np.load(path + filename)
                _violations = histories['violation']
                _resources = histories['resources']
                if len(_violations) < END:
                    continue
                _violations = _violations[START:END]
                _resources = _resources[START:END]
                runs += 1
                # load data for each run
                if not data:
                    violations = movingaverage(_violations, WINDOW)
                    regret = movingaverage(_violations.cumsum(), WINDOW)
                    actions = movingaverage(_resources, WINDOW)
                    if proposal:
                        accuracy = movingaverage(np.mean(histories['hits'], axis=0), WINDOW)
                    data = True
                else: # store the history of each run
                    violations = np.vstack((violations, movingaverage(_violations, WINDOW)))
                    regret = np.vstack((regret, movingaverage(_violations.cumsum(), WINDOW)))
                    actions = np.vstack((actions, movingaverage(_resources, WINDOW)))
                    if proposal:
                        accuracy = np.vstack((accuracy, movingaverage(np.mean(histories['hits'], axis=0), WINDOW)))
    
        print('Algorithm {}'.format(algo))
        
        # average over different runs

        actions_mean = np.mean(actions, axis=0)
        actions_std = np.std(actions, axis=0)

        violations_mean = np.mean(violations, axis=0)
        violations_std = np.std(violations, axis=0)

        regret_mean = np.mean(regret, axis=0)
        regret_std = np.std(regret, axis=0)

        if proposal:
            accuracy_mean = np.mean(accuracy, axis=0)
            accuracy_std = np.std(accuracy, axis=0)

        # plot results
        steps = np.arange(len(actions_mean[0:SPAN]))

        axs[2].set_title('Resource allocation')
        axs[2].plot(steps, actions_mean[0:SPAN])
        # ax2.fill_between(steps, actions_mean - actions_std, actions_mean + actions_std, color = '#DDDDDD')
        axs[2].fill_between(steps, actions_mean[0:SPAN] - 1.697 * actions_std[0:SPAN] / np.sqrt(runs), 
                        actions_mean[0:SPAN] + 1.697 * actions_std[0:SPAN] / np.sqrt(runs), color = '#DDDDDD')
        if algo == algo_names[-1]:
            axs[2].set_ylim((0,prbs))
            axs[2].set_xlabel('stages')  # Add an x-label to the axes.
            axs[2].set_ylabel('PRBs')
            # axs[2].legend(loc='best')
            axs[2].grid()

        axs[0].set_title('SLA violations')
        axs[0].plot(steps, violations_mean[0:SPAN], label = label)
        # ax3.fill_between(steps, violations_mean - violations_std, violations_mean + violations_std, color = '#DDDDDD')
        axs[0].fill_between(steps, violations_mean[0:SPAN] - 1.697 * violations_std[0:SPAN] / np.sqrt(runs), 
                        violations_mean[0:SPAN] + 1.697 * violations_std[0:SPAN] / np.sqrt(runs), color = '#DDDDDD')
        if algo == algo_names[-1]:
            axs[0].set_xlabel('stages')  # Add an x-label to the axes.
            axs[0].set_ylabel('SLA violations')
            axs[0].legend(loc='best')
            axs[0].grid()

        axs[1].set_title('Cumulative SLA violations')
        axs[1].plot(steps, regret_mean[0:SPAN], label = label)
        # ax4.fill_between(steps, regret_mean - regret_std, regret_mean + regret_std, color = '#DDDDDD')
        axs[1].fill_between(steps, regret_mean[0:SPAN] - 1.697 * regret_std[0:SPAN] / np.sqrt(runs), 
                        regret_mean[0:SPAN] + 1.697 * regret_std[0:SPAN] / np.sqrt(runs), color = '#DDDDDD')
        if algo == algo_names[-1]:
            axs[1].set_xlabel('stages')  # Add an x-label to the axes.
            axs[1].set_ylabel('cumulative SLA violations')
            axs[1].set_ylim((0,15000))
            # axs[1].legend(loc='best')
            axs[1].grid()        
        
        if START > 0:
            fig.savefig('./figures/_trained_subplots_{}.png'.format(scenario), format='png')
        else:
            # fig.savefig('./figures/subplots_{}.svg'.format(scenario), format='svg')
            fig.savefig('./figures/subplots_{}.png'.format(scenario), format='png')
        # fig.savefig('_subplots_' + scenario + '.svg', format='svg')       
