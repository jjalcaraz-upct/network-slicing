#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code for sampling and ploting the variables of the slices

Created on November 19, 2019

@author: juanjosealcaraz
"""
import matplotlib.pyplot as plt
import numpy as np

class HistoryLog():
    def __init__(self, n_slices, slots_per_step, steps, sample_period = 1e-3, window_length= 100):
        samples = int(np.floor(slots_per_step * steps / window_length))
        # sliding window arrays
        self.mean_queue_window = np.zeros((n_slices,window_length))
        self.max_queue_window = np.zeros((n_slices,window_length))
        self.mean_snr_window = np.zeros((n_slices,window_length))
        self.users_window = np.zeros((n_slices,window_length))
        self.rate_window = np.zeros((n_slices,window_length))

        # history backlog arrays
        self.mean_queue = np.zeros((n_slices,samples))
        self.max_queue = np.zeros((n_slices,samples))
        self.mean_snr = np.zeros((n_slices,samples))
        self.users = np.zeros((n_slices,samples))
        self.rate = np.zeros((n_slices,samples))

        self.sample_period = sample_period
        self.slots_per_step = slots_per_step
        self.window_length = window_length
        self.n_slices = n_slices
        self.sample_counter = np.zeros((n_slices,), dtype=int)
        self.log_counter = np.zeros((n_slices,), dtype=int)

    def sample(self, state):
        for i, l1_state in enumerate(state):
            # measures are done slice by slice
            queues = l1_state['queues']
            sinr = l1_state['sinr']
            delivered = l1_state['delivered']

            # we first fill the observation window
            for slot in range(self.slots_per_step):
                k = self.sample_counter[i]
                if len(queues[slot])>0:
                    self.mean_queue_window[i, k] = queues[slot].mean()
                    self.max_queue_window[i, k] = queues[slot].max()
                    self.mean_snr_window[i, k] = sinr[slot].mean()
                    self.rate_window[i, k] = delivered[slot].sum()/self.sample_period
                else:
                    self.mean_queue_window[i, k] = 0
                    self.max_queue_window[i, k] = 0
                    self.mean_snr_window[i, k] = 0
                    self.rate_window[i, k] = 0
                self.users_window[i, k] = len(queues[slot])
                self.sample_counter[i] += 1

                # if the window full, then store the measures
                if self.sample_counter[i] == self.window_length:
                    t = self.log_counter[i]
                    self.mean_queue[i, t] = self.mean_queue_window[i,:].mean()
                    self.max_queue[i, t] = self.max_queue_window[i,:].max()
                    self.mean_snr[i, t] = self.mean_snr_window[i,:].mean()
                    self.users[i, t] = self.users_window[i,:].mean()
                    self.rate[i, t] = self.rate_window[i,:].mean()
                    # increase the log counter
                    self.log_counter[i] += 1
                    # reset the sample counter for this slice
                    self.sample_counter[i] = 0


    def plot(self, name_code = ''):
        plt.figure(1)
        ax = plt.subplot(111)
        for i in range(self.n_slices):
            plt.plot(self.mean_queue[i,:], label='mean queue slice {}'.format(i+1))
            plt.plot(self.max_queue[i,:], label='max queue slice {}'.format(i+1))
        ax.legend(loc='upper left')
        plt.grid()
        plt.savefig('queue_{}.png'.format(name_code))

        plt.figure(2)
        ax = plt.subplot(111)
        for i in range(self.n_slices):
            plt.plot(self.rate[i,:], label='throughput slice {}'.format(i+1))
        ax.legend(loc='upper left')
        plt.grid()
        plt.savefig('rate_{}.png'.format(name_code))

        plt.figure(3)
        ax = plt.subplot(111)
        for i in range(self.n_slices):
            plt.plot(self.users[i,:], label='users slice {}'.format(i+1))
        ax.legend(loc='upper left')
        plt.grid()
        plt.savefig('users_{}.png'.format(name_code))
