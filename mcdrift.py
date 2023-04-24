#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


### Get asymptotic distribution given a transition matrix

def get_asymp_dist(tm):
    evals, evecs = np.linalg.eig(tm.T)
    evec1 = evecs[:,np.isclose(evals, 1)]
    evec1 = evec1[:,0]
    stationary = evec1 / evec1.sum()
    stationary = stationary.real
    return stationary


# In[9]:


#### Simulate a Markov chain given initial distribution and transition matrix

def simulate_mc(init, tm, length):
    n_states = len(init)
    seq = [np.random.choice(n_states, p = init)]
    for _ in range(length - 1):
        cur_state = seq[-1]
        cur_trans = tm[cur_state,:]
        next_state = np.random.choice(n_states, p = cur_trans)
        seq.append(next_state)
    return seq


# In[4]:


#### Calculate log likelihood of a sequence given initial distribution and transition matrix

def get_loglik(init, tm, seq):
    ll = np.log(init[seq[0]])
    for i in range(1, len(seq)):
        ll += np.log(tm[seq[i-1], seq[i]])
    return ll


# In[5]:


#### Estimate transition matrix from sequence data

def est_tm(seq, n_states):
    M = [[0]*n_states for _ in range(n_states)]

    for (i,j) in zip(seq,seq[1:]):
        M[i][j] += 1

    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return np.array(M)


# In[6]:


#### Algorithm 1 (P-CDM)

## Input: seq (T), subseq_length (W), tm0, tm1, K
## Output: the subseq index where the change is detected

def pcdm(seq, subseq_len, tm0, tm1, K):
    seq_len = len(seq)
    n_subseq = int(seq_len/subseq_len)
    m = 0
    asy0 = get_asymp_dist(tm0)
    asy1 = get_asymp_dist(tm1)
    for i in range(n_subseq):
        cur_seq = seq[subseq_len * i: subseq_len * (i+1)]
        ll0 = get_loglik(asy0, tm0, cur_seq)
        ll1 = get_loglik(asy1, tm1, cur_seq)
        l = ll1 - ll0
        m = np.max([0, m + np.sign(l)])
        if m >= K:
            return i
            break
    return 'No change detected'


# In[7]:


#### Algorithm 2 (NP-CDM)
## Input: seq (T), subseq_length (W), L, K
## Output: the subseq where the change is detected

def npcdm(seq, subseq_len, n_states, L, K):
    seq_len = len(seq)
    n_subseq = int(seq_len/subseq_len)
    m = 0
    tm0 = est_tm(seq[:L], n_states)
    tm1 = est_tm(seq[:L], n_states)
    asy0, asy1 = get_asymp_dist(tm0), get_asymp_dist(tm1)
    for i in range(n_subseq):
        cur_seq = seq[subseq_len * i: subseq_len * (i+1)]
        ll0 = get_loglik(asy0, tm0, cur_seq)
        ll1 = get_loglik(asy1, tm1, cur_seq)
        l = ll1 - ll0
        if subseq_len * (i+1) > L:
            m = np.max([0, m + np.sign(l)])
        if m >= K:
            return i
            break
        if subseq_len * (i+1) > L:
            tm1 = est_tm(seq[(subseq_len * (i+1) - L) : subseq_len * (i+1)], n_states)
            asy1 = get_asymp_dist(tm1)
    return 'No change detected'

