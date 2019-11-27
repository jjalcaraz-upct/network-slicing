#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 15, 2019

@author: juanjosealcaraz
"""
import numpy as np
from physical_layer import SNRtoTBS

class ProportionalFair():
    def __init__(self, alpha = 1.1, granularity = 6, allocation_period = 1e-3, debug = False):
        self.ue_ids = np.array([], dtype = np.int64)
        self.rates = np.array([], dtype = np.float32)
        self.alpha = alpha
        self.granularity = granularity
        self.allocation_period = allocation_period
        self.map = SNRtoTBS()
        self.debug = debug

    def add_user(self, ue):
        self.ue_ids = np.append(self.ue_ids, ue.id)
        self.rates = np.append(self.rates, 0.0)

    def extract_user(self, ue_id):
        self.rates = self.rates[self.ue_ids != ue_id]
        self.ue_ids = self.ue_ids[self.ue_ids != ue_id]

    def allocate(self, sinr_array, queues, n_prbs, weights = None):
        allocation = np.full_like(self.ue_ids, 0, dtype = np.int16)
        allocated_prbs = np.full_like(self.ue_ids, 0, dtype = np.int16)

        if (sinr_array.size == 0):
            return allocation, allocated_prbs

        if not weights:
            weights = np.full_like(self.ue_ids, 1.0, dtype = np.float32)
        tbs_array = np.array([], dtype = np.int16)

        # determine the bits per prb for each sinr
        # this is arough approximation but enough for modeling
        for snr in sinr_array:
            tbs = self.map.snr_to_tbs(snr)
            tbs_array = np.append(tbs_array, tbs)

        # now we allocate groups of prbs in chuncks of granularity
        # following proportional fair algorithm (7.4.1 Communication Networks, Srikant)
        iters = np.int(np.ceil(n_prbs/self.granularity))
        provisional_rates = self.rates
        provisional_queues = np.copy(queues)
        available_prbs = n_prbs
        for iter in range(iters):
            prbs_to_allocate = min(available_prbs, self.granularity)
            potential_alocation = np.minimum(provisional_queues, prbs_to_allocate*tbs_array)
            attainable_rates = potential_alocation / self.allocation_period
            factors = weights * attainable_rates / provisional_rates # some may be Inf
            if self.debug:
                print(potential_alocation)
                print(factors)
            #ties broken at random
            if np.isnan(factors).all():
                i = 0 # whatever
            else:
                i = np.random.choice([i_ for i_, f_ in enumerate(factors) if f_ == np.nanmax(factors)])
            # allocate
            allocation[i] += potential_alocation[i]
            allocated_prbs[i] += prbs_to_allocate
            available_prbs -= prbs_to_allocate

            #update queues
            provisional_queues[i] -= potential_alocation[i]

            # update provisional rate for i
            provisional_rates[i] = (1.0 - 1.0/self.alpha)*self.rates[i] + (1.0/self.alpha) * allocation[i]/self.allocation_period

        # update rates
        self.rates =  (1.0 - 1.0/self.alpha)*self.rates + (1.0/self.alpha) * allocation/self.allocation_period

        return allocation, allocated_prbs

if __name__ == '__main__':
    scheduler = ProportionalFair()
    for i in range(4):
        scheduler.ue_ids = np.append(scheduler.ue_ids, i)
        scheduler.rates = np.append(scheduler.rates, 0.0)

    sinr_array = np.array([12, 2, 5, 24], dtype = np.float32)
    queues = np.array([5000, 5000, 5000, 5000])
    n_prbs = 50

    for n in range(10):
        allocation = scheduler.allocate(sinr_array, queues, n_prbs)
        queues -= allocation
        print(sinr_array)
        print(allocation)
        print(queues)
        sinr_array += np.random.uniform(-2,2,4)
