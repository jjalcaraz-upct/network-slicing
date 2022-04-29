#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: juanjosealcaraz

This script evaluates the NAF algorithm, provided by keras-rl, in 3 network-slicing scenarios. 

For each scenario the script launches 30 simulation runs.

Each run is divided into a learning phase of 40000 steps and an inference phase of 10000 steps
(except for scenario 3)

The results of the K-th run on scenario N, are stored in:

./results/scenario_N/NAF/history_K.npz

'''

import os
import concurrent.futures as cf
from numpy.random import default_rng
from scenario_creator import create_env
from naf_agent_creator import create_naf_agent
from wrapper import ReportWrapper

RUNS = 30
PROCESSES = 4 # 30 if enough threads 
TRAIN_STEPS = 39936
EVALUATION_STEPS = 10500
# TRAIN_STEPS = 20000 # for scenario 3
# EVALUATION_STEPS = 5000

CONTROL_STEPS = 60000
PENALTY = 1000
SLOTS_PER_STEP = 50
run_list = list(range(RUNS))
scenarios = [0,1,2]
# scenarios = [3]

class NAFEvaluator():
    def __init__(self, scenario):
        self.scenario = scenario
        self.train_path = './results/scenario_{}/NAF/'.format(scenario)
        self.test_path = './results/scenario_{}/NAF_t/'.format(scenario)

        if not os.path.isdir(self.train_path):
            try:
                os.makedirs(self.train_path)
            except OSError:
                print ('Creation of the directory {} failed'.format(self.train_path))
            else:
                print ('Successfully created the directory {}'.format(self.train_path))
        
        if not os.path.isdir(self.test_path):
            try:
                os.makedirs(self.test_path)
            except OSError:
                print ('Creation of the directory {} failed'.format(self.test_path))
            else:
                print ('Successfully created the directory {}'.format(self.test_path))
    
    def evaluate(self, i):
        print('----------------------------------------------------------------------')
        print('Run {}:'.format(i))
        print('Performing NAF evaluation...')
        rng = default_rng(seed = i)
        env = create_env(rng, self.scenario, penalty = PENALTY)
        node_env = ReportWrapper(env, steps = TRAIN_STEPS, 
                        control_steps = CONTROL_STEPS,
                        env_id = i, 
                        path = self.train_path,
                        verbose = False)
        print('wrapped environment created')
        agent = create_naf_agent(node_env)
        print('NAF agent created...')
        agent.fit(node_env, TRAIN_STEPS, log_interval = TRAIN_STEPS)
        print('NAF agent trained!')
        node_env.save_results()
        print('NAF train results saved.')
        print('Test starts...')
        agent.training = False
        
        # ################  for scenarios 0,1,2  ################
        node_env.set_evaluation(EVALUATION_STEPS) 
        obs = node_env.obs # in scenarios 0,1,2
        # #######################################################
        
        # ################   for scenario 3    ################
        # env = create_env(rng, self.scenario, penalty = PENALTY) 
        # node_env = ReportWrapper(env, steps = EVALUATION_STEPS,
        #                 control_steps = CONTROL_STEPS,
        #                 env_id = i, 
        #                 path = self.test_path,
        #                 verbose = False)
        # print('wrapped TEST environment created')
        # obs = node_env.reset() # for scenario 3
        # #######################################################       

        for i in range(EVALUATION_STEPS):
            action = agent.forward(obs)
            obs, _, _, _ = node_env.step(action)

        print('evaluation done!')
        node_env.save_results()
        print('NAF test results saved.')

if __name__=='__main__':
    for scenario in scenarios:
        evaluator = NAFEvaluator(scenario)
        # ################################################################
        # # use this code for sequential execution
        # for run in run_list:
        #     evaluator.evaluate(run)
        # ################################################################

        # ################################################################
        # use this code for parallel execution
        with cf.ProcessPoolExecutor(PROCESSES) as E:
            results = E.map(evaluator.evaluate, run_list)
        # ################################################################