#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class generates a wrapper for the slice environment with the OpenAI gym environment

@author: juanjosealcaraz

Classes:

ReportWrapper
DQNWrapper
TimerWrapper

"""

import numpy as np
import gym
from gym import spaces
from itertools import product
import time

PENALTY = 1000
SLICES = 5

# SLICES = 2 # scenario 3

class ReportWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    this environment holds the history of the env variables
    - self.violation_history
    - self.reward_history
    - self.action_history 
    done = True if the number of steps is reached
    """
    def __init__(self, env, steps = 2000, control_steps = 500, env_id = 1, extra_samples = 10, path = './logs/', verbose = False):
        # Call the parent constructor, so we can access self.env later
        super(ReportWrapper, self).__init__(env)
        self.action_space = spaces.Box(low=0, high = 1,
                                        shape=(self.n_slices + 1,), dtype=np.float)
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(self.n_variables,), dtype=np.float)
        self.steps = steps
        self.step_counter = 0
        self.control_steps = control_steps
        self.env_id = env_id
        self.verbose = verbose
        self.path = path
        self.file_path = '{}history_{}.npz'.format(path, env_id)
        self.extra_samples = extra_samples # for safety
        self.reset_history()

        print('n_prbs = {}'.format(self.n_prbs))
        print('n_slices = {}'.format(self.n_slices))
    
    def reset_history(self):
        self.violation_history = np.zeros((self.steps), dtype = np.int16)
        self.reward_history = np.zeros((self.steps), dtype = np.float)
        self.action_history = np.zeros((self.steps), dtype = np.int16)
  
    def reset(self):
        """
        Reset the environment (but only when it is created)
        """
        self.step_counter = 0
        self.obs = self.env.reset()
        if self.verbose:
            print('Environment {} RESET'.format(self.env_id))
        return self.obs

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        # this works with actions like [0.5, 0.2, 0.3]
        if len(action) > self.n_slices: # action = [0.5, 0.2, 0.3]
            action = abs(action) # no negative values allowed
            t_action = action.sum()
            if t_action == 0:
                t_action = 1
            action = np.array([np.floor(self.n_prbs * action[i]/t_action) for i in range(self.n_slices)], dtype=np.int)
            # action = np.array([np.floor(self.n_prbs * action[i]/t_action) + 1 for i in range(self.n_slices)], dtype=np.int)

        obs, reward, done, info = self.env.step(action)

        # RL algorithms work better with normalized observations between -1 and 1
        obs = np.clip(obs,-0.5,1.5) 
        obs = obs - 0.5
        self.obs = obs

        # # (uncomment for NAF and TD3)
        # # this normalizes the return [-1., 1.]
        # if reward < 0:
        # #     reward = reward / (PENALTY * SLICES)
        #     reward = -1
        # else:
        #     reward = reward / self.n_prbs

        # collect historical data
        violations = info['total_violations']

        if self.step_counter < self.steps:
            self.violation_history[self.step_counter] = violations
            self.reward_history[self.step_counter] = reward
            self.action_history[self.step_counter] = action.sum()

        # increment counter
        self.step_counter += 1

        if self.step_counter % self.control_steps == 0:
            self.save_results()
        
        if self.verbose:
            print('Environment {}: {}/{} steps, reward: {}, violations: {}'.format(self.env_id, self.step_counter, self.steps, reward, info['total_violations']))

        # return obs, reward, done, info
        return obs, reward, done, {0:0} # for keras rl this avoids problems

    def save_results(self):
        np.savez(self.file_path, violation = self.violation_history, 
                                reward = self.reward_history,
                                resources = self.action_history)
    
    def set_evaluation(self, eval_steps, new_path = None, change_name = False):
        self.step_counter = self.steps
        self.steps += eval_steps
        self.violation_history = np.pad(self.violation_history, [(0, eval_steps)])
        self.reward_history = np.pad(self.reward_history, [(0, eval_steps)])
        self.action_history = np.pad(self.action_history, [(0, eval_steps)])
        if new_path:
            self.path = new_path
        if change_name:
            self.file_path = '{}evaluation_{}.npz'.format(self.path, self.env_id)

class DQNWrapper(ReportWrapper):
    '''
    Variation for DQN
    '''
    def __init__(self, env, steps = 2000, control_steps = 500, env_id = 1, extra_samples = 10, path = './logs/', verbose = False):
        # Call the parent constructor, so we can access self.env later
        super().__init__(env, steps = steps, control_steps = control_steps, env_id = env_id, extra_samples = extra_samples, path = path, verbose = verbose)
        g_eMBB = 2 # ganularity
        max_eMBB = 51 # max prbs for a single slice
        self.actions = []
        a = list(range(0,max_eMBB,g_eMBB))
        for (a1,a2) in product(a,a):
            if a1 + a2 <= self.n_prbs:
                self.actions.append(np.array([a1, a2], dtype = np.int16))
        self.action_space = spaces.Discrete(len(self.actions))
    
    def step(self, action):
        a = self.actions[action]
        return super(DQNWrapper, self).step(a)

class TimerWrapper(gym.Wrapper):
    '''
    Auxiliary wrapper for time measurement
    '''
    def __init__(self, env, steps = 2000):
        # Call the parent constructor, so we can access self.env later
        super(TimerWrapper, self).__init__(env)
        self.action_space = spaces.Box(low=0, high = 1,
                                        shape=(self.n_slices + 1,), dtype=np.float)
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(self.n_variables,), dtype=np.float)
        self.steps = steps
        self.step_counter = 0
        self.simtime = 0
        self.time_samples = np.zeros((self.steps), dtype = np.float)
        print('n_prbs = {}'.format(self.n_prbs))
        print('n_slices = {}'.format(self.n_slices))
  
    def reset(self):
        """
        Reset the environment 
        """
        self.step_counter = 0
        self.simtime = 0
        self.obs = self.env.reset()
        return self.obs
    
    def get_simtime(self):
        return self.simtime

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """

        # this should operate well with actions like [0.5, 0.2, 0.3]
        if len(action) > self.n_slices: # action = [0.5, 0.2, 0.3]
            action = abs(action) # no negative values allowed
            t_action = action.sum()
            if t_action == 0:
                t_action = 1
            action = np.array([np.floor(self.n_prbs * action[i]/t_action) for i in range(self.n_slices)], dtype=np.int)
        
        # measure simulation time
        t1 = time.time()
        obs, reward, _, _ = self.env.step(action)
        self.simtime += t1 - time.time()
        
        # RL algorithms work better with normalized observations between -0.5 and 0.5
        obs = np.clip(obs,-0.5,1.5) 
        obs = obs - 0.5
        self.obs = obs

        # increment counter
        self.step_counter += 1

        # return obs, reward, done, info
        return obs, reward, False, {0:0} # for keras rl this avoids problems