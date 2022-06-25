#!/usr/bin/env python
# coding: utf-8

# In[1]:


# this is just for the N node problem

import numpy as np
from scipy.optimize import minimize, LinearConstraint
from math import log
import scipy.optimize as opt
import matplotlib.pyplot as plt


# define the constants
import itertools
def objective_function(z): 
    # exploit probability
    v = np.array([.5, .5, .5])
    #v = np.array([v1,0.5])
    
    
    # propagation prbability
    alpha = np.array([[0, .5, .5],[.5, 0, 0],[.5, 0, 0]])
    #alpha = np.array([[0, .5],[.5, 0]])
    
    # theta and gamma
    th = 2
    ga = 0.7

    try:
        # after we obtain v_dash, we compute the log v_cap
        log_v_cap = np.ones(v.size)
        for k in range(v.size):
            log_v_cap[k] *= (log(v[k]) - th*log(ga*z[k] + 1))

        print('Soln of log_v_cap ', log_v_cap)

        # we compute now the v_bar, whose summation we want to minimize
        v_bar = np.ones(v.size)

        for k in range(v.size):
            do = log_v_cap[k]
            for t in range(v.size):
                if alpha[k][t] != 0 and t!=k:
                    do *= (v_bar[t]*(alpha[t][k] - 1) + 1)
            v_bar[k] = 1 + do
        return np.sum(v_bar)
    except:
        pass
    return None
    #return np.sum(v_bar)
                


# In[2]:


# Say 3 solutions
z_solns = 3
W = 30.0
res = None

results_z1=[]
results_z2=[]
results_z3=[]
v_s = []
for W in range(20,220,20):
    #while res is None:
    constraint = LinearConstraint(np.ones(z_solns), lb=0, ub=W)
    try:
        res = minimize(
            objective_function,
            bounds = [(0, W) for i in range(z_solns)],
            #args=(float(v/100.0),),
            x0=np.abs(np.random.normal(0.0, W, z_solns)),
            constraints=constraint
        )
    except:
        pass

    #print(res)
    results_z1.append(res.x[0])
    results_z2.append(res.x[1])
    results_z3.append(res.x[2])
    v_s.append(W)
    
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(v_s, results_z1, s=10, c='b', marker="s", label='z1')
ax1.scatter(v_s,results_z2, s=10, c='r', marker="o", label='z2')
ax1.scatter(v_s,results_z3, s=10, c='m', marker="*", label='z3')
# plt.scatter(v_s,results)
plt.xlabel('W')
# plt.ylabel('z1')
plt.legend(loc='upper left');
plt.show()


# In[ ]:




