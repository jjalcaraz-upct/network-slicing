#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanjosealcaraz

Classes:

PeriodicSource
OnOffSource
CbrSource
VbrSource

"""
import numpy as np

SLOT_LENGTH = 1e-3

class PeriodicSource:
    def __init__(self, packet_size = 640, period = 10):
        self.packet_size = packet_size
        self.period = period
        self.counter = self.period

    def step(self):
        self.counter = max(self.counter - 1, 0)
        if self.counter == 0:
            self.counter = self.period
            return self.packet_size
        else:
            return 0

class OnOffSource:
    def __init__(self, packet_size = 1000, period = 2, T_on = 500, T_off = 1000, initial_state = 1):
        self.T_on = T_on
        self.T_off = T_off
        self.state = initial_state
        self.periodic_source = PeriodicSource(packet_size, period)
        self.time_to_change = np.random.geometric(p = 1/T_off)

    def step(self):
        if self.time_to_change == 0:
            if self.state == 1:
                self.state = 0
                self.time_to_change = np.random.geometric(p = 1/self.T_on)
            else:
                self.state = 1
                self.time_to_change = np.random.geometric(p = 1/self.T_off)

        self.time_to_change = max(self.time_to_change - 1, 0)

        if self.state == 1:
            return self.periodic_source.step()
        else:
            return 0

class CbrSource(PeriodicSource):
    def __init__(self, bit_rate = 1000000, step_length = SLOT_LENGTH):
        packet_size = bit_rate * step_length
        super().__init__(packet_size = packet_size, period = 1)

class VbrSource:
    def __init__(self, packet_size = 1000, burst_size = 500, burst_rate = 1, step_length = SLOT_LENGTH):
        self.burst_size = burst_size
        self.packet_size = packet_size
        self.inter_arrival_steps = (1/burst_rate)/step_length
        self.steps_to_next_arrival = np.rint(np.random.exponential(self.inter_arrival_steps))
        self.active_bursts = []
        self.steps_to_go = []

    def step(self):
        bits = 0
        ending = []

        # active bursts
        for i, source in enumerate(self.active_bursts):
            if i >= len(self.steps_to_go):
                print(self.steps_to_go)
                print(self.active_bursts)
            self.steps_to_go[i] -= 1
            if self.steps_to_go[i] == 0:
                ending.append(i)
            else:
                bits += source.step()

        # ending bursts
        if len(ending) > 0:
            # self.steps_to_go = [steps for steps in self.steps_to_go if steps > 0]
            self.steps_to_go = [self.steps_to_go[i] for i, _ in enumerate(self.active_bursts) if i not in ending]
            self.active_bursts = [self.active_bursts[i] for i, _ in enumerate(self.active_bursts) if i not in ending]

        # arriving bursts
        self.steps_to_next_arrival -= 1
        if self.steps_to_next_arrival == 0:
            # new arrival
            self.active_bursts.append(PeriodicSource(packet_size = self.packet_size, period = 1))
            self.steps_to_go.append(np.rint(np.random.exponential(self.burst_size)))
            self.steps_to_next_arrival = np.rint(np.random.exponential(self.inter_arrival_steps))

        return bits

if __name__ == '__main__':
    source = VbrSource(burst_rate = 5)
    total_bits = 0
    for t in range(10000):
        bits = source.step()
        total_bits += bits
        if (t%100) == 0:
            print('VBR: t = {}: steps to next burst = {}, arriving bits = {}, total_bits = {}'.format(t, source.steps_to_next_arrival, bits, total_bits))
