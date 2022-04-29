#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanjosealcaraz
"""
import numpy as np

class NodeB():
    def __init__(self, slices_l1, slots_per_step, n_prbs, slot_length = 1e-3):
        self.slices_l1 = slices_l1
        self.n_slices_l1 = len(self.slices_l1)
        self.slots_per_step = slots_per_step
        self.n_prbs = n_prbs
        self.slot_length = slot_length
        self.reset()

    def reset(self):
        self.steps = 0
        for slice_l1 in self.slices_l1:
            slice_l1.reset()
        state = self.get_state()
        return state

    def get_n_variables(self):
        n_variables = 0
        for slice_l1 in self.slices_l1:
            n_variables += slice_l1.get_n_variables()
        return n_variables

    def reset_info(self):
        ''' Reset the info of the l1 slices for SLA assessment'''
        for l1 in self.slices_l1:
            l1.reset_info()

    def slot(self):
        ''' runs the system just for one time-slot '''
        for slice_l1 in self.slices_l1:
            slice_l1.slot()

    def get_state(self):
        state = np.array([], dtype = np.float32)
        for l1 in self.slices_l1:
            state = np.concatenate((state, l1.get_state()), axis=None)
        return state
    
    def get_info(self, violations = 0, SLA_labels = 0):
        info = {'l1_info': [l1.get_info() for l1 in self.slices_l1], 'SLA_labels': SLA_labels, \
                'violations': violations, 'n_prbs': [l1.n_prbs for l1 in self.slices_l1]}
        return info

    def compute_reward(self):
        '''checks if the SLA is fulfilled for each slice'''
        SLA_labels = np.zeros(self.n_slices_l1, dtype = np.int)
        violations = np.zeros(self.n_slices_l1, dtype = np.int)
        for i, l1 in enumerate(self.slices_l1):
            SLA_labels[i], violations[i] = l1.compute_reward()
        return SLA_labels, violations

    def step(self, action):
        ''' 
        move a step forward using the selected action
        each step consists of a number of time slots
        '''
        self.reset_info()

        if len(action)!=len(self.slices_l1):
            print('The action must contain as many elements as slices!')
            return self.get_state, self.get_info()

        # configure slices
        i_prb = 0
        for slice_l1, prbs in zip(self.slices_l1, action):
            slice_l1.set_prbs(i_prb, prbs)
            i_prb += prbs

        # run a step
        for _ in range(self.slots_per_step):
            self.slot()

        # get the node state
        state = self.get_state()

        # check the SLAs of each slice_l1
        SLA_labels, violations = self.compute_reward()

        # the info is a dict
        info = self.get_info(SLA_labels=SLA_labels, violations=violations)

        self.steps += 1

        return state, info
