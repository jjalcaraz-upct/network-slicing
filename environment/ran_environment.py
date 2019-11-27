#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class generates an environment for a single node with the OpenAI gym environment

Created on November 26, 2019

@author: juanjosealcaraz
"""

from channel_models import ChannelGenerator
from slice_ran import SliceRANeMBB
from schedulers import ProportionalFair
from slice_l1 import SliceL1
from node_b import NodeB
from backlog import HistoryLog

CBR_description = {}
CBR_description['lambda'] = 1.0/60.0
CBR_description['t_mean'] = 30.0
CBR_description['bit_rate'] = 500000
CBR_description['q_target'] = 10000

VBR_description = {}
VBR_description['lambda'] = 1.0/60.0
VBR_description['t_mean'] = 30.0
VBR_description['p_size'] = 1000
VBR_description['b_size'] = 500
VBR_description['b_rate'] = 1
VBR_description['q_target'] = 15000

n_prbs = 50
slots_per_step = 10

channel_generator = ChannelGenerator(type = 'eMBB')

# create a list with slices RAN
slices_ran = [SliceRANeMBB(id, CBR_description, VBR_description, channel_generator) for id in range(3)]

# create SLAs for the slices ran
SLA_l1 = {(0,0):{'cbr_th': 10e6, 'cbr_prb': 30, 'cbr_queue': 10e4, 'vbr_th': 10e6, 'vbr_prb': 40, 'vbr_queue': 15e4},\
          (0,1):{'cbr_th': 10e6, 'cbr_prb': 30, 'cbr_queue': 10e4, 'vbr_th': 10e6, 'vbr_prb': 40, 'vbr_queue': 15e4},\
          (0,2):{'cbr_th': 10e6, 'cbr_prb': 30, 'cbr_queue': 10e4, 'vbr_th': 10e6, 'vbr_prb': 40, 'vbr_queue': 15e4}}

# create the L1 slices: WARNING: each one must have its own scheduler!
slices_l1 = [SliceL1(n_prbs, slices_ran, ProportionalFair(debug = False))]

# create the node with the L1 slices
node = NodeB(slices_l1, slots_per_step, n_prbs, SLA_l1, output_msg = True, summary_state = True)

# class SingleNodeEnvironment(gym.Env): # uncoment to install the environment as an OpenAI Gym
class SingleNodeEnvironment():
    def __init__(self, node = node):
        self.node = node

    def reset(self):
        self.node.reset()
        return self.node.state

    def step(self, action):
        state, reward = node.step(action)
        return state, reward, False, {}

    def render(self, mode='human'):
        pass

if __name__ == '__main__':
    STEPS = 1000
    # create an object for generating figures
    # logger = HistoryLog(len(slices_l1), slots_per_step, STEPS)
    node_env = SingleNodeEnvironment()
    S = node_env.reset()
    for i in range(STEPS):
        A = [n_prbs]
        state, reward, _, _ = node_env.step(A)
        # logger.sample(state)
        if i % 100 == 0 and i > 0:
            print(' {}: state = {}, reward = {}'.format(i, state, reward))
    # logger.plot()
