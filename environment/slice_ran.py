#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sept 28, 2019

@author: juanjosealcaraz
"""
DEBUG = True
CBR = 0
VBR = 1

import numpy as np
from traffic_generators import VbrSource, CbrSource

class UE:
    def __init__(self, id, traffic_source, channel_model, type):
        self.id = id
        self.traffic_source = traffic_source
        self.channel_model = channel_model
        self.type = type
    def __repr__(self):
        return str(self.id)

class SliceRANeMBB:
    def __init__(self, id, CBR_description, VBR_description, channel_generator, slot_length = 1e-3):
        self.id = id
        self.slot_length = slot_length

        self.cbr_arrival_rate = CBR_description['lambda']
        self.cbr_mean_time = CBR_description['t_mean']
        self.cbr_bit_rate = CBR_description['bit_rate']
        self.cbr_queue_target = CBR_description['q_target']

        self.vbr_arrival_rate = VBR_description['lambda']
        self.vbr_mean_time = VBR_description['t_mean']
        self.vbr_packet_size = VBR_description['p_size']
        self.vbr_burst_size= VBR_description['b_size']
        self.vbr_burst_rate= VBR_description['b_rate']
        self.vbr_queue_target = VBR_description['q_target']

        self.channel_generator = channel_generator

        self.reset()

    def reset(self):
        self.step_counter = 0
        self.user_counter = 0
        self.remaining_time = {}
        self.cbr_steps_next_arrival = 0
        self.vbr_steps_next_arrival = 0

    def cbr_arrivals(self):
        if self.cbr_steps_next_arrival == 0:
            self.user_counter += 1
            source = CbrSource(bit_rate = self.cbr_bit_rate)
            channel = self.channel_generator.generate_channel()
            ue = UE(self.user_counter, source, channel, CBR)
            ue.slice_ran_id = self.id
            ue.q_target = self.cbr_queue_target

            # generate holding time
            holding_time = np.random.exponential(self.cbr_mean_time)
            holding_time = np.rint(holding_time / self.slot_length)
            self.remaining_time[self.user_counter] = holding_time

            # generate next arrival
            inter_arrival_time = np.random.exponential(1.0 / self.cbr_arrival_rate)
            inter_arrival_time = np.rint(inter_arrival_time / self.slot_length)
            self.cbr_steps_next_arrival = inter_arrival_time

            return [ue]
        else:
            self.cbr_steps_next_arrival -= 1
            return []

    def vbr_arrivals(self):
        if self.vbr_steps_next_arrival == 0:
            self.user_counter += 1
            source = VbrSource(packet_size = self.vbr_packet_size, burst_size = self.vbr_burst_size, \
                               burst_rate = self.vbr_burst_rate)
            channel = self.channel_generator.generate_channel()
            ue = UE(self.user_counter, source, channel, VBR)
            ue.slice_ran_id = self.id
            ue.q_target = self.vbr_queue_target

            # generate holding time
            holding_time = np.random.exponential(self.vbr_mean_time)
            holding_time = np.rint(holding_time / self.slot_length)
            self.remaining_time[self.user_counter] = holding_time

            # generate next arrival
            inter_arrival_time = np.random.exponential(1.0 / self.vbr_arrival_rate)
            inter_arrival_time = np.rint(inter_arrival_time / self.slot_length)
            self.vbr_steps_next_arrival = inter_arrival_time
            return [ue]
        else:
            self.vbr_steps_next_arrival -= 1
            return []

    def departures(self):
        departures = []
        for id in self.remaining_time.keys():
            self.remaining_time[id] -= 1
            if self.remaining_time[id] == 0:
                departures.append(id)

        for i in departures:
            del self.remaining_time[i] # delete user

        return departures

    def slot(self):
        self.step_counter += 1
        arrivals = self.cbr_arrivals()
        arrivals.extend(self.vbr_arrivals())
        departures = self.departures()
        return arrivals, departures

if __name__ == '__main__':
    from channel_models import ChannelGenerator

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

    channel_generator = ChannelGenerator(type = 'eMBB')

    slice = SliceRANeMBB(1, CBR_description, VBR_description, channel_generator)
    total_arrivals = 0
    total_departures = 0

    for t in range(100000):
        arrivals, departures = slice.slot()
        if len(arrivals) > 0 or len(departures) > 0:
            print('t = {}: arrivals = {}, departures = {}'.format(t, arrivals, departures))
