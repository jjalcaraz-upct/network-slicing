#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 14, 2019

@author: juanjosealcaraz
"""
import numpy as np
from collections import Counter

class NodeB():
    def __init__(self, slices_l1, slots_per_step, n_prbs, SLAs, output_msg = False, slot_length = 1e-3, summary_state = False):
        self.slices_l1 = slices_l1
        self.slots_per_step = slots_per_step
        self.n_prbs = n_prbs
        self.output = output_msg
        self.summary_state = summary_state
        self.SLAs = SLAs
        self.slot_length = slot_length
        self.l1_ran_pairs =[]
        for i, slice_l1 in enumerate(self.slices_l1):
            for slice_ran in slice_l1.slices_ran:
                self.l1_ran_pairs.append((i, slice_ran.id))
        self.reset()

    def slot(self):
        slot_state = []
        info = []
        for slice_l1 in self.slices_l1:
            for slice_ran in slice_l1.slices_ran:
                # generate arrivals and departures for each slice ran
                arrivals, departures = slice_ran.slot()
                slice_l1.extract_users(departures)
                slice_l1.add_users(arrivals)
                if self.output and (len(arrivals) > 0 or len(departures) > 0):
                    print('>>> arrivals : {} departures : {}'.format(arrivals, departures))
            # advance the l1 slice one slot in time
            state_l1, info_l1 = slice_l1.slot()
            slot_state.append(state_l1)
            info.append(info_l1)
        return slot_state, info

    def reset(self):
        self.steps = 0
        self.reset_state()
        self.reset_info()
        for slice_l1 in self.slices_l1:
            slice_l1.reset()

    def reset_state(self):
        ''' The node state is a list of dictionaries, one per l1 slice.
            Each entry of the dictionary contains a list
            with as many elements as time slots in a step.
        '''
        self.state = []
        for _ in range(len(self.slices_l1)):
            if self.summary_state:
                self.state.append({'queues': 0, 'queues_diff': 0,'sinr': 0, 'arrivals': 0, 'delivered': 0})
            else:
                self.state.append({'queues': [], 'sinr': [], 'queue_obj': [], 'arrivals': [], \
                                    'delivered': [], 'slice_id': [], 'type': []})

    def reset_info(self):
        ''' The node info is a list of dictionaries, one per l1 slice.
            Each dictionary contains one dictionary per RAN slice.
            Each element is a dictionary with SLA information.'''
        self.info = {}
        for l1_ran_pair in self.l1_ran_pairs:
            self.info[l1_ran_pair] = Counter({'cbr_th': 1, 'cbr_prb': 1,\
                                  'cbr_queue': 1,'vbr_th': 1,\
                                  'vbr_prb': 1, 'vbr_queue': 1}) # 1s will be removed later

    def update_info(self, slot_info):
        for (i,j) in self.l1_ran_pairs:
            self.info[(i,j)] += slot_info[i][j] # adds two counters

    def compute_reward(self):
        '''checks if the SLA is fulfilled for each RAN slice'''
        violations = 0
        observation_time = self.slots_per_step * self.slot_length
        for l1_ran_pair in self.l1_ran_pairs:
            info_ran = self.info[l1_ran_pair]
            SLA = self.SLAs[l1_ran_pair]
            cbr_th = (info_ran['cbr_th'] - 1)/observation_time > SLA['cbr_th']
            cbr_prb = (info_ran['cbr_prb'] - 1)/observation_time > SLA['cbr_prb']
            cbr_queue = (info_ran['cbr_queue'] - 1)/observation_time < SLA['cbr_queue']
            vbr_th = (info_ran['vbr_th'] - 1)/observation_time > SLA['vbr_th']
            vbr_prb = (info_ran['vbr_prb'] - 1)/observation_time > SLA['vbr_prb']
            vbr_queue = (info_ran['vbr_queue'] - 1)/observation_time < SLA['vbr_queue']
            cbr_fulfilled = cbr_th or cbr_prb or cbr_queue
            vbr_fulfilled = vbr_th or vbr_prb or vbr_queue
            SLA_fulfilled = cbr_fulfilled and vbr_fulfilled
            violations += not(SLA_fulfilled)
        return -1*violations

    def update_state(self, slot_state):
        for l1_state, l1_slot_state in zip(self.state, slot_state):
            if self.summary_state and len(l1_slot_state['queues']>0): # {'queues_diff': 0, 'sinr': 0, 'arrivals': 0, 'delivered': 0}
                queue_diff_array = l1_slot_state['queue_obj'] - l1_slot_state['queues']
                queue_diff = queue_diff_array.mean()
                sinr = l1_slot_state['sinr'].mean()
                queues = l1_slot_state['queues'].sum()
                arrivals = l1_slot_state['arrivals'].sum()
                delivered = l1_slot_state['delivered'].sum()
                l1_state['queues'] += queues
                l1_state['queues_diff'] += queue_diff
                l1_state['sinr'] += sinr
                l1_state['arrivals'] += arrivals
                l1_state['delivered'] += delivered
                return
            else:
                l1_state['queues'].append(l1_slot_state['queues'])
                l1_state['sinr'].append(l1_slot_state['sinr'])
                l1_state['queue_obj'].append(l1_slot_state['queue_obj'])
                l1_state['arrivals'].append(l1_slot_state['arrivals'])
                l1_state['delivered'].append(l1_slot_state['delivered'])
                l1_state['slice_id'].append(l1_slot_state['slice_id'])
                l1_state['type'].append(l1_slot_state['type'])

    def step(self, action):
        ''' move a step forward using the selected action
            each step consists of a number of time slots
        '''
        self.reset_state()
        self.reset_info()

        if len(action)!=len(self.slices_l1):
            print('The action must contain as many elements as slices!')
            return self.state, 0

        # configure slices
        for slice_l1, prbs in zip(self.slices_l1, action):
            slice_l1.n_prbs = prbs

        # run a step
        for _ in range(self.slots_per_step):
            slot_state, info = self.slot()
            self.update_state(slot_state)
            self.update_info(info)

        # obtain reward checking the slas
        reward = self.compute_reward()

        self.steps += 1

        return self.state, reward
