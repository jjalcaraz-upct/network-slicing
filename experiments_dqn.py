#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: juanjosealcaraz

This script evaluates DQN, provided by stable-baselines, in 3 network-slicing scenarios. 

For each scenario the script launches 30 simulation runs.

The learning phase lasts 20000 steps, and the inference phase lasts 5000 steps

The results of the K-th run on scenario N, are stored in:

./results/scenario_N/DQN/history_K.npz (learning phase)
./results/scenario_N/DQN_t/history_K.npz (inference phase)

'''

import os
import concurrent.futures as cf
from numpy.random import default_rng
from scenario_creator import create_env
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines import DQN
from wrapper import DQNWrapper

SCENARIO = 3
RUNS = 30
PROCESSES = 4 # 30 if enough threads 
TRAIN_STEPS = 20000
EVALUATION_STEPS = 5000
CONTROL_STEPS = 30000
PENALTY = 1000
SLOTS_PER_STEP = 50
PRBS = [200, 150, 100, 70]

run_list = list(range(RUNS))

class Evaluator():
    def __init__(self, scenario = SCENARIO):
        self.scenario = scenario
        self.train_path = './results/scenario_{}/DQN/'.format(scenario)
        self.test_path = './results/scenario_{}/DQN_t/'.format(scenario)
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
        rng = default_rng(seed = i)
        env = create_env(rng, self.scenario, penalty = PENALTY)
        node_env = DQNWrapper(env, steps = TRAIN_STEPS, 
                        control_steps = CONTROL_STEPS, 
                        env_id = i, 
                        path = self.train_path,
                        verbose = False)
        print('wrapped environment created')
        env = make_vec_env(lambda: node_env, n_envs=1)
        print('vectorised environment created')
        agent = DQN('MlpPolicy', env, verbose=True)
        print('DQN agent created...')
        agent.learn(total_timesteps = TRAIN_STEPS)
        print('trainning done!')
        node_env.save_results()
        print('DQN results saved.')
        print('Test starts...')
        env = create_env(rng, self.scenario, penalty = PENALTY)
        node_env = DQNWrapper(env, steps = EVALUATION_STEPS,
                        control_steps = CONTROL_STEPS,
                        env_id = i, 
                        path = self.test_path,
                        verbose = False)
        print('wrapped TEST environment created')
        obs = node_env.reset()
        action, state = agent.predict(obs, deterministic = True)
        for i in range(EVALUATION_STEPS):
            action, state = agent.predict(obs, state = state, deterministic = True)
            obs, _, _, _ = node_env.step(action)
        print('evaluation done')
        node_env.save_results()
        print('results saved')

if __name__=='__main__':
    evaluator = Evaluator()
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