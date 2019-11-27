#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sept 8, 2019

@author: juanjosealcaraz
"""

import numpy as np

# [1.95, 4, 6, 8, 10, 11.95, 14.05, 16, 17.9, 19.9, 21.5, 23.45, 25.0, 27.30, 29

snr_array = np.array([1.0, 4, 6, 8, 10, 14, 18, 20, 22, 24, 26, 28, 30, 32])

class iidChannel:
    def __init__(self, elements = None):
        array_length = len(snr_array[:-1])
        if elements and elements < array_length:
            first_index = np.random.randint(array_length - elements)
            self.snr_array = snr_array[first_index:first_index + elements]
            self.elements = elements
        else:
            self.snr_array = snr_array[:-1]
            self.elements = array_length

    def step(self):
        index = np.random.randint(self.elements)
        return self.snr_array[index]

class ChannelGenerator:
    def __init__(self, type):
        self.type = type

    def generate_channel(self):
        return iidChannel()


if __name__ == '__main__':
    channel_1 = iidChannel(elements = 4)
    channel_2 = iidChannel(elements = 8)
    channel_3 = iidChannel()
    for t in range(100):
        print('t = {}: ber_1 = {}, ber_2 = {}, ber_3 = {}'.format(t, channel_1.step(), channel_2.step(), channel_3.step()))
