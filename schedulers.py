#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 3, 2021

@author: juanjosealcaraz
"""

import numpy as np

'''proportional fair scheduler for average cqi reports'''
class ProportionalFair:
    def __init__(self, mcs_codeset, granularity = 2, slot_length = 1e-3, window = 50, sym_per_prb = 158):
        self.granularity = granularity
        self.mcs_codeset = mcs_codeset
        self.b = 1/window
        self.a = 1 - self.b
        self.sym_per_prb = sym_per_prb
        self.slot_length = slot_length

    def allocate(self, ues, n_prb, error_bound = 0.1):
        '''
        Updates the following variables of the ues:
        - ue.bits : assigned bits in this subframe
        - ue.prbs : assigned prbs in this subframe
        - ue.p : reception probability
        '''
        # create auxiliary data structures
        n_ues = len(ues)
        ue_rbs = np.zeros(n_ues, dtype = np.int)
        ue_mcs = np.zeros(n_ues, dtype = np.int)
        ue_queue = np.zeros(n_ues, dtype = np.int)
        ue_rate = np.zeros(n_ues, dtype = np.int)
        ue_bits = np.zeros(n_ues, dtype = np.int)
        ue_th = np.zeros(n_ues)

        # extract ue information
        for i, ue in enumerate(ues):
            ue_th[i] = max(ue.th, 1) # to avoid division by zero
            ue_queue[i] = ue.queue
            # determine the mcs given the objective and the estimated snr
            ue_mcs[i], bits_per_sym = self.mcs_codeset.mcs_rate_vs_error(ue.e_snr, error_bound)
            # achievable rate for the ue
            ue_rate[i] = self.sym_per_prb * bits_per_sym
        
        # loop over the resources
        for r in range(0, n_prb, self.granularity):
            # prbs to be allocated in this iteration
            prbs = min(n_prb - r, self.granularity)

            # selected user for this resource (remove users without data)
            index = np.argmax(ue_rate * (ue_queue > 0)/ ue_th)
            
            # assign the resource to this ue
            ue_rbs[index] += prbs

            # update queue and throughput of this user
            tx_bits = min(prbs * ue_rate[index], ue_queue[index])
            ue_queue[index] -= tx_bits
            ue_bits[index] += tx_bits

            # update the estimated throughput with current allocation
            ue_th[index] = self.a * max(ue.th, 1) + self.b * ue_bits[index] / self.slot_length
    
        # update ues
        prb_i = 0
        for i, ue in enumerate(ues):
            prbs = ue_rbs[i]
            ue.prbs = prbs
            ue.bits = ue_bits[i]
            if prbs:
                snr_values = ue.snr[prb_i: prb_i + prbs]
                ue.p = self.mcs_codeset.response(ue_mcs[i], snr_values)
            else:
                ue.p = 0
            prb_i += prbs
