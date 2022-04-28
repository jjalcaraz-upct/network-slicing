# Network slicing environment

## Description

Source code of a network slicing environment and a control algorithm that allocates time-frequency resources (radio bearers, RBs, in the radio frames) among several network slices. The environtment implements the OpenAI Gym https://github.com/openai/gym interface and interacts with Stable-Baselines RL agents https://github.com/hill-a/stable-baselines and Keras-RL agents https://github.com/keras-rl/keras-rl. This code was developed for the paper "[Model-Based Reinforcement Learning with Kernels for Resource Allocation in RAN Slices](paper/manuscript.pdf)", where the control algorithm, referred to as KBRL, is presented.

<img src="img/general_diagram.png" align="center" width="40%"/>  

## Acknowledgements

This work was supported by project grant PID2020-116329GB-C22 funded by MCIN / AEI / 10.13039/501100011033  

<img src="img/MICINN_Gob_Web_AEI_2.jpg" align="right" width="40%"/>

## How to use it

### Requirements

The enviroment requires Open-AI gym, Numpy and Pandas packages. The RL agents are provided by stable-baselines (version 2, which uses TensorFlow), and the scripts for plotting results use scipy and matplotlib. The following versions of these packages are known to work fine with the environment:  

gym==0.15.3  
numpy==1.19.1  
pandas==0.25.2  
stable-baselines==2.10.1  
tensorflow==1.9.0  
scipy==1.5.4  
matplotlib==3.3.4  

To run the NAF agent, Keras and Keras-RL are also required. The tested versions are:  
Keras==2.2.1  
keras-rl==0.4.2  

It is recommended to use a python virtual environment to install the above packages.

### Instalation

1. Clone or download the repository in your local machine

2. Open a terminal window and (optionally) activate the virtual environment

3. Go to the gym-ran_slice folder in the terminal window 

4. Once in the gym-ran_slice folder run:
```python
pip install -e .
```

### Experiment scripts

There are four scripts for launching simulation experiments:

- experiments_rl.py: runs the experiments with the RL agents of stable-baselines  
- experiments_kbrl.py: runs the experiments with the proposed KBRL algorithm  
- experiments_naf.py: runs the experiments with the NAF algorithm provided by keras-rl  
- experiment_dqn.py: runs the experiments with the DQN algorithm provided by stable-baselines  

And four scripts for plotting results:  

- plot_results.py: plots the learning curves of the algorithms in the scenario given as a input (e.g. "python plot_results.py 0" plots paper's figure 3)
- plot_trained_results.py: plots the performance metrics during the inference phase of the MBRL algorithms (paper's figure 6)  
- plot_adjustment_results.py: plots the adjustment rate of KBRL (paper's figure 7)  
- plot_accuracy_results.py: plots the accuracy of KBRL (paper's figure 8)  

## Project structure

The following files implement the environment:  

- node_b.py  
- slice_ran.py  
- slice_l1.py  
- channel_models.py  
- traffic_generators.py  
- schedulers.py  
- ./gym-ran_slice/gym_ran_slice/ran_slice.py  

The KBRL agent is implemented in:

- kbrl_control.py
- ./algorithms/kernel.py
- ./algorithms/projectron.py

The following files are required to build the experiments:

- scenario_creator.py: creates the environments and the KBRL agents  
- wrapper.py: allows the interaction with stable-baselines  
- naf_agent_creator.py: creates NAF agents   

## How to cite this work

The code:

@misc{net_slice,  
    title={Network slicing environment},  
    author={Juan J. Alcaraz},  
    howpublished = {\url{https://github.com/jjalcaraz-upct/network-slicing/}},  
    year={2022}  
}

The paper:

@misc{alcaraz2022,
  author = {Alcaraz, Juan J. and Losilla, Fernando and Zanella, Andrea and Zorzi, Michele},  
  title = {Model-Based Reinforcement Learning with Kernels for Resource Allocation in RAN Slices},  
  year = {2022},  
  publisher = {IEEE},  
  journal = {IEEE Transactions on Wireless Communications},  
  note = {under review},  
}

## Licensing information

This code is released under MIT lisence.
