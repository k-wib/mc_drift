# mcdrift

This repository contains a Python implementation of the P-CDM and NP-CDM algorithms introduced in the paper Learning Discrete Time Markov Chains
under Concept Drift by Roveri (2019). The link to the paper is https://re.public.polimi.it/bitstream/11311/1113395/1/FINAL%20VERSION.pdf.

## Directory

- The file [mcdrift.py](https://github.com/k-wib/mc_drift/blob/main/mcdrift.py) contains the implementation of P-CDM and NP-CDM, as well as some helpful intermediate functions.
- The notebook [toy_example.ipynb](https://github.com/k-wib/mc_drift/blob/main/toy_example.ipynb) shows how to run P-CDM and NP-CDM on a toy dataset.
- The notebook [Experiments/experiment1.ipynb](https://github.com/k-wib/mc_drift/blob/main/Experiments/experiment1.ipynb) contains the code to reproduce Section 3.1 in the report.
- The notebook [Experiments/experiment2.ipynb](https://github.com/k-wib/mc_drift/blob/main/Experiments/experiment2.ipynb) contains the code to reproduce Section 3.2 in the report.
- The notebook [Experiments/experiment3.ipynb](https://github.com/k-wib/mc_drift/blob/main/Experiments/experiment3.ipynb) contains the code to reproduce Section 3.3 in the report.
- The notebook [Experiments/experiment4.ipynb](https://github.com/k-wib/mc_drift/blob/main/Experiments/experiment4.ipynb) contains the code to reproduce Section 3.4 in the report.


## Usage example

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mcdrift import *

# Setting a seed
seed = 2023

# Parameters
N = 3 # number of states
p0 = np.array([[0.2, 0.3, 0.5], [0.1, 0.8, 0.1], [0.3, 0.4, 0.3]]) # initial transition matrix
p1 = np.array([[0.8, 0.1, 0.1], [0.6, 0.2, 0.2], [0.4, 0.1, 0.5]]) # modified transition matrix
T = 10000 # sequence length
tstar = 2000 # time when abrupt change occurs
L = 1000 # number of observations guaranteed to come from p0
K = 3 # detection threshold
W = 5 # subsequence length
pi = np.array([1/3, 1/3, 1/3]) # initial distribution

# Generate a sequence of states
np.random.seed(seed)
seq_sim = simulate_mc(pi, p0, tstar) # generate observations before shift
init_vec = np.zeros(N)
init_vec[seq_sim[-1]] = 1
seq_sim_2 = simulate_mc(init_vec, p1, T - tstar + 1) # generate observations after shift
seq_comb = seq_sim + seq_sim_2[1:] # combine observations into a list

# P-CDM
pcdm(seq_comb, W, p0, p1, K) # returns the subsequence index at which the change is predicted to occur

# NP-CDM
npcdm(seq_comb, W, N, L, K) # returns the subsequence index at which the change is predicted to occur
```

