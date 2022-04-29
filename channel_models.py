#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanjosealcaraz

Classes:

NominalSINR
SINRSelectiveFading
SNRGenerator
MCSCodeset

"""

import pandas as pd
import numpy as np
from numpy.random import default_rng
from math import log
import matplotlib.pyplot as plt

IN = -110 # Interference plus noise per RB (in dBm) 
MCL = 70 # dB
Gmax = 15 #dBi
Tx_pw = 30 # dBm
Rmax = 2 # Km (cell range)
F = 9 # Noise Figure in dB
radius = 1/2

FILESNAMES = [
    './datasets/fading_trace_EPA_3kmph.csv', 
    './datasets/fading_trace_ETU_3kmph.csv', 
    './datasets/fading_trace_EVA_60kmph.csv'
    ]

def sigmoid(x, x0 = 0, k = 1):
    y = 1 / (1 + np.exp(-k*(x-x0)))
    return (y)

def inv_sigmoid(y, x0, k):
    x = -(1/k) * np.log(1/y - 1) + x0
    return (x)

'''auxiliar functions defining a hexagonal cell'''
def lower_left(x):
    return np.sqrt(3)*radius / 2 - np.sqrt(3)*x

def lower_right(x):
    return -1*3*np.sqrt(3)*radius / 2 + np.sqrt(3)*x

def upper_left(x):
    return np.sqrt(3)*radius / 2 + np.sqrt(3)*x

def upper_right(x):
    return 5*np.sqrt(3)*radius / 2 - np.sqrt(3)*x

def location(x, y):
    distance = np.sqrt((x - 1/3)**2 + y**2)
    x_t = x-1/3
    cos_theta = x_t*1/3 /(np.sqrt(x_t**2 + y**2) * 1/3)
    theta = np.arccos(cos_theta)
    return distance, theta

def generate_xy(rng):
    in_cell = False
    x, y = 0.1, 0.1
    while not in_cell:
        [x, y] = rng.random(2) 
        in_cell = (y > lower_left(x)) and (y > lower_right(x)) and (y < upper_left(x)) and (y < upper_right(x))
    return x, y

def antenna_pattern(theta):
    # provides the gain of the annena according to TS 36.942 section 4.2 Antenna models
    return -1*min(12*(theta/65)**2, 20)

def macro_cell(rng, A = 128.1, B = 37.6):
    # TS 36.942 section 4.5 Propagation conditions and channel models
    x, y = generate_xy(rng)
    d, theta = location(x, y)
    R = max(d*Rmax, 0.1)
    G = Gmax + antenna_pattern(theta)
    LogF = rng.normal(0,10)
    L = A + B * np.log10(R)
    gamma = 2.6
    FSPL = 20*np.log10(R) + 20*np.log10(2) + 93.45 + gamma*10*np.log10(R) # Free Space Path Loss (R in Km, f in GHz)
    L = max(L, FSPL)
    Rx_pw = Tx_pw - max(L + LogF - G, MCL)
    SINR = Rx_pw - IN - F
    return SINR

def free_space(rng, gamma = 2.6):
    # TS 36.942 section 4.5 Propagation conditions and channel models
    x, y = generate_xy(rng)
    d, theta = location(x, y)
    R = max(d*Rmax, 0.1)
    G = Gmax + antenna_pattern(theta)
    LogF = rng.normal(0,10)
    FSPL = 20*np.log10(R) + 20*np.log10(2) + 93.45 + gamma*10*np.log10(R) # Free Space Path Loss (R in Km, f in GHz)
    Rx_pw = Tx_pw - max(FSPL + LogF - G, MCL)
    SINR = Rx_pw - IN - F
    return SINR

class NominalSINR():
    '''
    Class that computes the nominal sinr using a strategy pattern
    '''
    def __init__(self, rng, name):
        self.rng = rng
        self.list_of_functions = {'macro_cell_urban_2GHz': macro_cell,
                                    'macro_cell_urban_900MHz': macro_cell,
                                    'macro_cell_rural': macro_cell
                                    }
        self.list_of_parameters = {'macro_cell_urban_2GHz': {'A': 120.9, 'B': 37.6},
                                    'macro_cell_urban_900MHz': {'A': 128.1, 'B': 37.6},
                                    'macro_cell_rural': {'A': 95.5, 'B': 34.1}
                                    }       
        self.sinr_function = self.list_of_functions[name]
        self.parameters = self.list_of_parameters[name]
        
    def generate(self):
        return self.sinr_function(self.rng, **self.parameters)

class SINRSelectiveFading:
    ''' 
    Class that generates a sequence of SINR samples with selective frequency fading from a dataset
    '''
    def __init__(self, rng, model_name, n_prbs = 100, filenames = FILESNAMES, user_ids = None):
        # random number generator
        self.rng = rng
        self.nominal_sinr = NominalSINR(self.rng, model_name)

        # extract samples from file
        self.samples = []
        for filename in filenames:
            df = pd.read_csv(filename, header = None)
            if n_prbs > 100:
                sample_matrix = df.to_numpy()
                extension = n_prbs - 100
                sample_matrix = np.vstack((sample_matrix, sample_matrix[0:extension,:]))
                self.samples.append(sample_matrix)
            else:
                self.samples.append(df.to_numpy())
        
        # create dictionary of users
        self.users = {}
        
        # insert users
        if user_ids:
            for u_id in user_ids:
                self.insert_user(u_id)

    def reset(self):
        self.users = {}

    def insert_user(self, user_id):
        fading_type = self.rng.integers(len(self.samples))
        n_samples = self.samples[fading_type].shape[1]
        index = self.rng.integers(n_samples)
        step = self.rng.choice([-1,1])
        sinr = self.nominal_sinr.generate()
        self.users[user_id] = {'fading_type': fading_type, 'index': index, 'step': step, 'nominal_sinr': sinr, 'n_samples': n_samples}
        
    def get_snr(self, user_id):
        is_nan = True
        
       # beware of nans 
        while is_nan:
            # iterate one step
            self.users[user_id]['index'] += self.users[user_id]['step']

            # if limit is reached jump to a random location
            if self.users[user_id]['index'] >= self.users[user_id]['n_samples'] or self.users[user_id]['index'] < 0:
                self.users[user_id]['index'] = self.rng.integers(self.users[user_id]['n_samples'])
                self.users[user_id]['step'] = self.rng.choice([-1,1])        
            
            f = self.users[user_id]['fading_type']
            i = self.users[user_id]['index']

            # Fading gain per RB
            fading_vector = self.samples[f][:,i]
            is_nan = np.isnan(np.sum(fading_vector))

        return fading_vector + self.users[user_id]['nominal_sinr'] # this is a column array
        
    def extract_user(self, user_id):
        self.users.pop(user_id)


class SNRGenerator:
    '''
    Generates sequence of SNR values from a dataset considering average fading over the spectrum
    '''
    def __init__(self, rng, filename = './datasets/srslte_v19.03.csv', user_ids = None, powers = None):
        # random number generator
        self.rng = rng

        # extract the samples from file
        df = pd.read_csv(filename)
        self.norm_snr_array = df[["mean_snr"]].to_numpy().flatten() - df[["txpower"]].to_numpy().flatten()
        self.n_samples = len(self.norm_snr_array)

        # create dictionary of users
        self.users = {}
        
        # insert users
        if user_ids:
            self.insert_user_list(user_ids, powers)

    def reset(self):
        self.users = {}
        
    def get_snr(self, user_id, power = None):
        # update power:
        if power:
            self.users[user_id]['power'] = power

        # iterate one step
        self.users[user_id]['index'] += self.users[user_id]['step']

        # if limit is reached jump to a random location
        if self.users[user_id]['index'] >= self.n_samples or self.users[user_id]['index'] < 0:
            self.users[user_id]['index'] = self.rng.integers(self.n_samples)
            self.users[user_id]['step'] = self.rng.choice([-1,1])
        
        snr = self.norm_snr_array[self.users[user_id]['index']] + self.users[user_id]['power']

        return [snr]

    def insert_user_list(self, user_id_list, powers = None):
        # tx power
        if not powers:
            powers = np.array(len(user_id_list) * [0.0], dtype=float)
        
        for u_id, power in zip(user_id_list, powers):
            self.insert_user(u_id, power)

    def insert_user(self, user_id, power = None):
        if not power:
            power = 0.0
        index = self.rng.integers(self.n_samples)
        step = self.rng.choice([-1,1])
        self.users[user_id] = {'index': index, 'step': step, 'power': power}

    def extract_user(self, user_id):
        self.users.pop(user_id)

    
class MCSCodeset:
    '''
    Generates the response of the Modulation and Coding Scheme under a given SNR
    '''
    def __init__(self, filename = './datasets/mcs_codeset.csv'):
        df = pd.read_csv(filename)
        self.rate = df[["rate"]].to_numpy().flatten()
        self.snr = df[["snr"]].to_numpy().flatten()
        self.order = df[["order"]].to_numpy().flatten()
        self.modulation = df[["modulation"]].squeeze()
        self.n_mcs = len(self.snr)
        self.A, self.B = self.compute_factors(0.1)
        self.MIparameters = {'qpsk': [-0.25040431, 0.31591749],
                             '16qam': [5.12440916, 0.25423209],
                             '64qam': [9.16962738, 0.22298101]}
    
    def compute_factors(self, Delta):
        # fits the factors A, B in rx_ptob(x) = A *(snr - snr_ref) + B
        # such that rx_prob(snr_ref) = 0.9
        # and rx_prob(snr_ref - Delta) = 0.1
        A = 1.0/Delta
        A = A * (log(1/sigmoid(0.1) - 1) - log(1/sigmoid(0.9) - 1))
        B = - log(1/sigmoid(0.9) - 1)
        return A, B

    def estimate_rx_prob(self, mcs, snr):
        # returns the packet reception probability of a given mcs under a given sinr
        # snr_ref is the one providing a 0.9 reception probability for the given mcs
        snr_ref = self.snr[mcs]
        x = self.A*(snr - snr_ref) - self.B
        return sigmoid(x)

    def mcs_rate_vs_error(self, snr, error_upper_bound):
        # returns the highest mcs whose estimated error is below the given bound
        # and the corresponding achievable rate in bits per symbol
        rx_prob = 1.0 - error_upper_bound
        for mcs in range(self.n_mcs):
            if self.estimate_rx_prob(mcs, snr) < rx_prob:
                return max(mcs-1, 0), self.rate[mcs] * self.order[mcs]
        return mcs, self.rate[mcs] * self.order[mcs]

    def response(self, mcs, snr):
        # returns the response of the channel for a given MCS
        # the channel is characterized by the SINR values of each RB
        # inputs:
        #  - mcs: selected Modulation and Coding Scheme
        #  - snr: list with the SINRs of each RB
        # outputs:
        #  - error probabilty
        if len(snr) > 1: # the UE measures one sinr per RB
            # compute the average mutual information of each RB
            modulation = self.modulation.iloc[mcs]
            params = self.MIparameters[modulation]
            MIvalues = sigmoid(snr, *params)
            averageMI = np.mean(MIvalues)
            snr = inv_sigmoid(averageMI, *params)
        rx_prob = self.estimate_rx_prob(mcs, snr)
        return rx_prob

    def nominal_rate(self, mcs):
        return self.rate[mcs] * self.order[mcs]


if __name__ == '__main__':

    SAMPLES = 400
    PRBS = 150
    SEED = 3547879

    rng = default_rng(seed = SEED)

    generator = SINRSelectiveFading(rng, 'macro_cell_urban_2GHz', n_prbs = PRBS)
    for i in range(3):
        generator.insert_user(i)
        samples = np.empty((PRBS,SAMPLES))
        for t in range(SAMPLES):
            samples[:,t] = generator.get_snr(i)

        print('user: {}, mean sinr = {}'.format(i, samples.mean()))
        # Make data.
        X = np.arange(SAMPLES)
        Y = np.arange(PRBS)
        X, Y = np.meshgrid(X, Y)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # Plot the surface.
        surf = ax.plot_surface(X, Y, samples, cmap='autumn', linewidth=0.5, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(-20,50)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        fig.savefig('snr_sf_trace_{}'.format(i))

    generator = SNRGenerator(rng)
    for i in range(3):
        generator.insert_user(i)
        samples = [generator.get_snr(i, power = -5)[0] for _ in range(SAMPLES)]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(samples, color='tab:blue')
        ax.set_xlabel('samples')  # Add an x-label to the axes.
        ax.set_ylabel('SNR')
        ax.grid()
        fig.savefig('snr_trace_{}'.format(i))

    mcs_codeset = MCSCodeset()
    SNR_vector = np.linspace(-5,25,PRBS)

    mcs_values = []
    bits_p_sym_values = []
    for s in SNR_vector:
        mcs, bits_p_sym = mcs_codeset.mcs_rate_vs_error(s, 0.1)
        mcs_values.append(mcs)
        bits_p_sym_values.append(bits_p_sym)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(SNR_vector, mcs_values)
    ax.set_xlabel('SNR')  # Add an x-label to the axes.
    ax.set_ylabel('MCS')
    ax.grid()
    fig.savefig('selected_mcs')
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(SNR_vector, bits_p_sym_values)
    ax.set_xlabel('SNR')  # Add an x-label to the axes.
    ax.set_ylabel('Throughput (symbols/PRB)')
    ax.grid()
    fig.savefig('throughput')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for mcs in range(26):
        response = []
        for s in SNR_vector:
            th = mcs_codeset.nominal_rate(mcs)
            p = mcs_codeset.response(mcs, [s])
            response.append(p*th)
        ax.plot(SNR_vector, response)
    ax.set_xlabel('SNR')  # Add an x-label to the axes.
    ax.set_ylabel('throughput (bps/Hz)')
    ax.grid()
    fig.savefig('MCS_response')

    mcs_codeset = MCSCodeset()
    SNR_vector = np.linspace(-5,25,PRBS)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for mcs in range(26):
        response = []
        for s in SNR_vector:
            snr_array = np.array([s-2, s-2, s, s+1, s+2])
            th = mcs_codeset.nominal_rate(mcs)
            p = mcs_codeset.response(mcs, snr_array)
            response.append(p*th)
        ax.plot(SNR_vector, response)
    ax.set_xlabel('SNR')  # Add an x-label to the axes.
    ax.set_ylabel('throughput (bps/Hz)')
    ax.grid()
    fig.savefig('MCS_response_MI')