#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanjosealcaraz

This script evaluates the RL baselines algorithms provided by Stable Baselines 
(PPO1, PPO2, TRPO, SAC, A2C, TD3, DDPG) in 3 network-slicing scenarios. 

For each scenario, and each algorithm, the script launches 30 simulation runs.

Each run is divided into a learning phase of 40000 steps and an inference phase of 10000 steps.

The results of the K-th run of algorithm ALG on scenario N, are stored in:

./results/scenario_N/ALG/history_K.npz

"""

import os
from numpy.random import default_rng
from itertools import product
import concurrent.futures as cf
from scenario_creator import create_env
from wrapper import ReportWrapper
from stable_baselines import PPO1, PPO2, TRPO, SAC, A2C, TD3, DDPG
from stable_baselines.common.cmd_util import make_vec_env
from tensorflow import set_random_seed

RUNS = 30
PROCESSES = 4 # 30 if enough threads 
TRAIN_STEPS = 39936 # must be a multiple of 256
EVALUATION_STEPS = 10500
CONTROL_STEPS = 60000
PENALTY = 1000
VERBOSE = False

run_list = list(range(RUNS))
scenarios = [0,1,2]

algorithms = {
    'SAC':SAC,
    'PPO1':PPO1, 
    'PPO2':PPO2, 
    'TRPO':TRPO,
    'A2C': A2C,
    'TD3': TD3,
    'DDPG': DDPG
}

deterministic = {
    'SAC':True,
    'PPO1':True, 
    'PPO2':False, 
    'TRPO':False,
    'A2C':False,
    'TD3':False,
    'DDPG':False
}

class RLEvaluator():
    def __init__(self, scenario, algo_name, algorithm):
        self.scenario = scenario
        self.algo_name = algo_name
        self.algorithm = algorithm
        self.path = './results/scenario_{}/{}/'.format(scenario, algo_name)
        if not os.path.isdir(self.path):
            try:
                os.makedirs(self.path)
            except OSError:
                print ("Creation of the directory %s failed" % self.path)
            else:
                print ("Successfully created the directory %s " % self.path)
        self.model_path = './trained_models/scenario_{}/{}/'.format(scenario, algo_name)
        if not os.path.isdir(self.model_path):
            try:
                os.makedirs(self.model_path)
            except OSError:
                print ("Creation of the directory %s failed" % self.model_path)
            else:
                print ("Successfully created the directory %s " % self.model_path)

    
    def evaluate(self, i):
        print('start evaluation of scenario {} run {} algorithm {}'.format(self.scenario, i, self.algo_name))
        rng = default_rng(seed = i) # environment seed
        set_random_seed(i) # tensorflow seed
        node_env = create_env(rng, self.scenario, penalty = PENALTY)
        print('environment created')
        node_env = ReportWrapper(node_env, steps = TRAIN_STEPS, 
                            control_steps = CONTROL_STEPS, 
                            env_id = i, 
                            path = self.path,
                            verbose = VERBOSE)
        print('wrapped environment created')
        env = make_vec_env(lambda: node_env, n_envs=1)
        print('vectorised environment created')
        model = self.algorithm('MlpPolicy', env, verbose=0)
        print('scenario {}: run {} of algorithm {} ... '.format(self.scenario, i, self.algo_name))
        model.learn(total_timesteps = TRAIN_STEPS)
        print('trainning done!')
        node_env.save_results()
        model_path = '{}{}_agent_{}'.format(self.model_path, self.algo_name, i)
        model.save(model_path)
        print('model saved')
        node_env.set_evaluation(EVALUATION_STEPS)
        obs = node_env.obs
        det = deterministic[self.algo_name]
        action, state = model.predict(obs, deterministic = det)
        for i in range(EVALUATION_STEPS):
            action, state = model.predict(obs, state = state, deterministic = det)
            obs, _, _, _ = node_env.step(action)
        print('evaluation done')
        node_env.save_results()
        print('results saved')

if __name__=='__main__':
    for scenario, (alg_name, alg) in product(scenarios, algorithms.items()):
        evaluator = RLEvaluator(scenario, alg_name, alg)
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
