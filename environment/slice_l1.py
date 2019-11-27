#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sept 28, 2019

@author: juanjosealcaraz
"""
import numpy as np
from collections import Counter

CBR = 0
VBR = 1

class SliceL1:
    def __init__(self, n_prbs, slices_ran, scheduler, SLAs = []):
        self.n_prbs = n_prbs
        self.slices_ran = slices_ran
        self.scheduler = scheduler
        self.SLAs = SLAs
        self.reset()

    def reset(self):
        self.queues = np.array([], dtype = np.int64)
        self.queue_objectives = np.array([], dtype = np.int64)
        self.ue_ids = np.array([], dtype = np.int64)
        self.slice_ran_ids = np.array([], dtype = np.int16)
        self.types = np.array([], dtype = np.int16)
        self.ues = []
        for slice_ran in self.slices_ran:
            slice_ran.reset()

    def add_users(self, ue_list):
        for ue in ue_list:
            self.ue_ids = np.append(self.ue_ids, ue.id)
            self.queue_objectives= np.append(self.queue_objectives, ue.q_target)
            self.queues = np.append(self.queues, 0)
            self.slice_ran_ids = np.append(self.slice_ran_ids, ue.slice_ran_id)
            self.types = np.append(self.types, ue.type)
            self.ues.append(ue)
            self.scheduler.add_user(ue)

    def extract_users(self, ue_id_list):
        for ue_id in ue_id_list:
            # delete all entries for this ue_id
            remain = self.ue_ids != ue_id
            self.queues = self.queues[remain]
            self.queue_objectives = self.queue_objectives[remain]
            self.ues = [ue for ue in self.ues if ue.id != ue_id]
            self.types = self.types[remain]
            self.slice_ran_ids = self.slice_ran_ids[remain]
            self.ue_ids = self.ue_ids[remain]
            self.scheduler.extract_user(ue_id)

    def slot(self):
        # data arrival
        arrivals = [ue.traffic_source.step() for ue in self.ues]
        arrival_array = np.array(arrivals, dtype = np.int64)
        self.queues = self.queues + arrival_array

        # channel step
        sinr = [ue.channel_model.step() for ue in self.ues]
        sinr_array = np.array(sinr, dtype = np.float32)

        # scheduling
        tbs_array, prbs_array = self.scheduler.allocate(sinr_array, self.queues, self.n_prbs)

        # transmission
        p_samples = np.random.random_sample((len(self.ues),))
        received = p_samples > 0.1
        tbs_eff = tbs_array * received

        # update queues
        self.queues = self.queues - tbs_eff
        self.queues[self.queues < 0] = 0

        # build the state
        state = {'queues': self.queues, 'sinr': sinr_array, 'queue_obj': self.queue_objectives,\
                'arrivals': arrival_array, 'delivered': tbs_eff, 'slice_id': self.slice_ran_ids, 'type': self.types}

        # create a info summary for SLA assessment
        info = {}
        for slice_ran in self.slices_ran:
            slice_indexes = self.slice_ran_ids == slice_ran.id
            slice_ran_summary = Counter({})
            if CBR in self.types[slice_indexes]:
                cbr_indexes = np.logical_and(slice_indexes, CBR == self.types)
                cbr_th = tbs_eff[cbr_indexes]
                slice_ran_summary['cbr_th'] = cbr_th.sum()
                cbr_prbs = prbs_array[cbr_indexes]
                slice_ran_summary['cbr_prb'] = cbr_prbs.sum()
                cbr_queue = self.queues[cbr_indexes]
                slice_ran_summary['cbr_queue'] = cbr_queue.max()
            else:
                slice_ran_summary['cbr_th'] = 0
                slice_ran_summary['cbr_prb'] = 0
                slice_ran_summary['cbr_queue'] = False

            if VBR in self.types[slice_indexes]:
                vbr_indexes = np.logical_and(slice_indexes, VBR == self.types)
                vbr_th = tbs_eff[vbr_indexes]
                slice_ran_summary['vbr_th'] = vbr_th.sum()
                vbr_prbs = prbs_array[vbr_indexes]
                slice_ran_summary['vbr_prb'] = vbr_prbs.sum()
                vbr_queue = self.queues[vbr_indexes]
                slice_ran_summary['vbr_queue'] = vbr_queue.max()
            else:
                slice_ran_summary['vbr_th'] = 0
                slice_ran_summary['vbr_prb'] = 0
                slice_ran_summary['vbr_queue'] = False

            # store the info of the slice ran
            info[slice_ran.id] = slice_ran_summary

        return state, info
