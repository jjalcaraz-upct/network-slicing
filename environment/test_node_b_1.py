#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script creates a node b with several slices and runs it

Created on November 14, 2019

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
steps = 1000 # simulation time

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
node = NodeB(slices_l1, slots_per_step, n_prbs, SLA_l1, output_msg = True)

# create an object for generating figures
logger = HistoryLog(len(slices_l1), slots_per_step, steps)

node.reset()
for i in range(steps):
    action = [n_prbs]
    state, reward = node.step(action)
    logger.sample(state)
    if i % 100 == 0 and i > 0:
        print(' {}: {}'.format(i, reward))

logger.plot()
