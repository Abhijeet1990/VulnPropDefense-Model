#!/usr/bin/env python

# this is just for the N node problem

import numpy as np
from scipy.optimize import minimize, LinearConstraint
from math import log
import scipy.optimize as opt



# define the constants
import itertools
def objective_function(z): 
    # exploit probability
    v = np.array([0.8, 0.5, 0.3])
    
    # propagation prbability
    alpha = np.array([[0, .8, .3],[.4, 0, 0],[.6, 0, 0]])
    
    # theta and gamma
    th = 2
    ga = 0.7
    
    # first we will compute the v_dash from v and alpha
    
    # it will be a set of N non-linear equation solved using Newton method
    def compute_vdash(vdash):
        exp = vdash
        # compute the equations for each n node
        for i in range(v.size):
            p = log(v[i])
            for j in range(v.size):
                if alpha[j][i]!=0:
                    p*=(vdash[j]*(alpha[j][i] - 1) + 1)
            exp[i] = p + 1
        #print(exp)
        return exp
    
    # compute the jacobian for the same..it computes the derivative of the above equations having relationship of 
    # vdash and v
    def J_compute_vdash(vdash):
        j_exp = np.zeros((v.size,v.size))
        for i in range(v.size):
            for j in range(v.size):
                if i != j:
                    k = (alpha[j][i]-1)
                    l = 1
                    var=''
                    N = v.size - 2
                    for m in range(v.size):                                       
                        if (m!=i and m!=j):
                            var+=str(m) 

                    # possible combination
                    # print(N)... will discuss in the meeting how these expressions were obtained
                    for f in range(N):
                        combo = list(itertools.combinations(var,f+1))  
                        #print('combo ',combo)
                        for com in combo:
                            exp_t = 1
                            for item in com:
                                exp_t *= v[int(item)] * (alpha[int(item)][i] - 1)
                            l += exp_t
                    j_exp[i][j] = k*l
        #print(j_exp)
        return j_exp
                            
    sol = opt.root(compute_vdash, x0 = np.ones(v.size), jac = J_compute_vdash)
    v_dash = sol.x
    
    #print('Soln of vdash ', v_dash)
    try:
        # after we obtain v_dash, we compute the log v_cap
        log_v_cap = np.ones(v.size)
        for k in range(v.size):
            log_v_cap[k] *= (log(v_dash[k]) - th*log(ga*z[k] + 1))

        #print('Soln of log_v_cap ', log_v_cap)

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

# Say 3 solutions
z_solns = 3
W = 30.0
res = None
while res is None:
    constraint = LinearConstraint(np.ones(z_solns), lb=0, ub=W)
    try:
        res = minimize(
            objective_function,
            bnds = [(0, W) for i in range(z_solns)],
            x0=np.abs(np.random.normal(0.0, W, z_solns)),
            constraints=constraint
        )
    except:
        pass

print(res)




