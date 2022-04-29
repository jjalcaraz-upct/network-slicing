#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanjosealcaraz

"""

import numpy as np

DEBUG = True

class Learner:
    '''
    Auxiliary class with elements and variables for the KBRL agent
    '''
    def __init__(self, algorithm, indexes, initial_action, security_factor):
        self.algorithm = algorithm
        self.indexes = indexes
        self.initial_action = initial_action
        self.security_factor = security_factor
        self.step = 1

class KBRL_Control:
    '''
    KBRL_Control: Kernel Model-Based RL for online learning and control.
    Its objective is to assign spectrum resources (RBs) among several ran slices
    '''
    def __init__(self, learners, n_prbs, alfa = 0.05, accuracy_range = [0.99, 0.999]):
        self.learners = learners # there must be one learner instance per slice
        self.accuracy_range = accuracy_range
        self.n_slices = len(learners)
        self.n_prbs = n_prbs
        self.alfa = alfa
        self.adjusted = 0
        self.action = np.array([h.initial_action for h in learners], dtype = np.int16)
        self.security_factors = np.array([h.security_factor for h in learners], dtype = np.int16)        
        self.margins = np.array([0]*self.n_slices, dtype = np.int16)
        intial_value = (self.accuracy_range[0] + self.accuracy_range[1])/2
        self.accuracies = np.full((self.n_slices, self.n_prbs), intial_value, dtype = float)

    def select_action(self, state):
        action = np.zeros((self.n_slices), dtype = np.int16)
        adjusted = 0
        for i, h in enumerate(self.learners):
            algorithm = h.algorithm
            _i_ = h.indexes
            l1_state = state[_i_]
            min_prbs = 0
            max_prbs = self.n_prbs

            # we check the prediction for assignment "offset" prbs below the action
            offset = self.security_factors[i]
            margin = 0
            for l1_prbs in range(max(min_prbs - offset,0), max_prbs+1):
                x = np.append(l1_state, l1_prbs/self.n_prbs)
                prediction = algorithm.predict(x)
                if prediction == 1:
                    a = min(self.n_prbs, l1_prbs + offset)
                    margin = a - l1_prbs
                    l1_prbs = a
                    break
            action[i] = l1_prbs
            self.margins[i] = margin

        assigned_prbs = action.sum()
        if assigned_prbs > self.n_prbs: # not enough resources
            adjusted = 1
            action, diff = self.adjust_action(action, assigned_prbs, self.n_prbs)
            self.margins = self.margins - diff
        
        self.action = action

        return action, adjusted

    def adjust_action(self, action, assigned_prbs, n_prbs):
        relative_p = action / assigned_prbs
        new_action = np.array([np.floor(n_prbs * p) for p in relative_p], dtype=np.int16)
        return new_action, action - new_action

    def update_control(self, state, action, reward):
        hits = np.zeros((self.n_slices), dtype = np.int16)

        for i, h in enumerate(self.learners):
            algorithm = h.algorithm
            _i_ = h.indexes
            l1_state = state[_i_]
            l1_action = action[i]
            x = np.append(l1_state, l1_action/self.n_prbs)
            y_pred = algorithm.predict(x)
            y = reward[i]
            hit = y == y_pred
            margin = max(0, self.margins[i])
            if y_pred == 1:
                if hit == 0: # with the same or less margin we would have made the same mistake
                    self.accuracies[i,0:margin+1] = (1 - self.alfa) * self.accuracies[i,0:margin+1]
                else: # with the same or more margin we would have succeeded as well
                    self.accuracies[i,margin:] = (1 - self.alfa) * self.accuracies[i,margin:] + self.alfa
            if not self.adjusted: # if the action was not adjusted then update security_factor
                self.security_factors[i] = np.argmax(self.accuracies[i,:] > self.accuracy_range[0])

            hits[i] = hit
            # sample augmentation
            if y == 1: # fulfilled
                for a in range(l1_action, self.n_prbs + 1): # same or more prbs would obtain the same y
                    new_x = np.append(l1_state, a/self.n_prbs)
                    y_pred = algorithm.predict(new_x)
                    algorithm.update(new_x, y)
            else: # not fulfilled (y = -1)
                for a in range(0, l1_action + 1): #  same or fewer prbs would obtain the same y
                    new_x = np.append(l1_state, a/self.n_prbs)
                    y_pred = algorithm.predict(new_x)
                    algorithm.update(new_x, y)

        return hits

    def run(self, system, steps, learning_time = -1):
        action = self.action

        SLA_history = np.zeros((steps), dtype = np.int16)
        reward_history = np.zeros((steps), dtype = np.float)
        violation_history = np.zeros((steps), dtype = np.int16)
        adjusted_actions = np.zeros((steps), dtype = np.int16)
        resources_history = np.zeros((steps), dtype = np.int16)
        hits_history = np.zeros((len(action),steps), dtype = np.int16)

        state = system.reset()

        for i in range(steps):
            new_state, reward, _, info = system.step(action)
            SLA_labels = info['SLA_labels']
            if learning_time < steps:
                hits = self.update_control(state, action, SLA_labels)
            action, self.adjusted = self.select_action(new_state)
            state = new_state

            SLA_history[i] = SLA_labels.sum()
            reward_history[i] = reward
            violation_history[i] = info['total_violations']
            resources_history[i] = action.sum()
            adjusted_actions[i] = self.adjusted
            hits_history[:,i] = hits

        print('mean resources = {}'.format(resources_history.mean()))
        print('total violations = {}'.format(violation_history.sum()))
        print('mean adjusted = {}'.format(adjusted_actions.mean()))
        print('mean accuracy = {}'.format(hits_history.mean(axis=1)))

        output = {
            'reward': reward_history, 
            'resources': resources_history, 
            'hits': hits_history,
            'adjusted': adjusted_actions,
            'SLA': SLA_history,
            'violation': violation_history
        }

        return output