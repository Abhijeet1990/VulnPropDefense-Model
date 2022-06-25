#!/usr/bin/env python

# this is just for the two node problem
import networkx as nx
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from math import log
import matplotlib.pyplot as plt

# define the constants

def objective_function(z,arg): 
    # exploit probability
    v2 = 0.5
    v1 = arg[0]

    # propagation prbability
    ratio = arg[1]
    #a12 = 0.5
    #a21 = 0.5
    a12 = ratio/(1+ratio)
    a21 = 1-a12
    
    # theta and gamma
    th = 2
    ga = 0.7
    try:
        a = log(v1)- th*(log(ga*z[0]) + 1)
        b = log(v2)- th*(log(ga*z[1]) + 1)
        
        num = 2 + a*(a21 + b*(a21-1)) + b*(a12 + a*(a12 - 1))
        den = 1 - a*b*(a12-1)*(a21-1)
        return float(num/den)
    
    except Exception as e:
        pass
      
    return 1.0

def objective_function_W(z): 
    # exploit probability
    v2 = 0.2
    v1 = 0.8

    # propagation prbability
    a12 = 0.5
    a21 = 0.5
    
    # theta and gamma
    th = 2
    ga = 0.7
    try:
        a = log(v1)- th*(log(ga*z[0]) + 1)
        b = log(v2)- th*(log(ga*z[1]) + 1)
        
        num = 2 + a*(a21 + b*(a21-1)) + b*(a12 + a*(a12 - 1))
        den = 1 - a*b*(a12-1)*(a21-1)
        return float(num/den)
    
    except Exception as e:
        pass
      
    return 1.0


# two solns z1 and z2 and W the budget
z_solns = 2
W = 20
constraint = LinearConstraint(np.ones(z_solns), lb=0, ub=W)
#cnsts =[]

results_all=[]
v_s = []
# for v in range(20,220,20):
#     constraint = LinearConstraint(np.ones(z_solns), lb=0, ub=v)
#     #cnsts.append(constraint)
#     #cnsts.append({'type':'ineq', 'fun': lambda x: x})
#     bnds = [(0, v) for i in range(2)] 
#     res = minimize(
#         objective_function,
#         x0=np.abs(np.random.normal(0.0, v, z_solns)),
#         #args=(v,),
#         bounds=bnds,
#         constraints=constraint
#     )
#     print(res)
#     results.append(res.x[1])
#     v_s.append(v)

# plt.scatter(v_s,results)
# plt.xlabel('W')
# plt.ylabel('z2')
# plt.show()
#print(res)
markers_list=['s','o','*','v','^']
colors=['b','k','g','y','r']

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set(ylim=(5,12),autoscale_on=False)


# plt.scatter(v_s,results)
plt.xlabel('v1',fontsize =14)
plt.ylabel('z1',fontsize = 14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

ctr=0
for r in range(20,120,20):
    results=[]
    v_s = []
    for v in range(1,99,1):
        res = minimize(
            objective_function,
            x0=np.abs(np.random.normal(0.0, W, z_solns)),
            args=([float(v/100.0),float(r/100.0)],),
            bounds = [(0, W) for i in range(2)],
            constraints=constraint
        )
        print(res)
        results.append(res.x[0])
        v_s.append(float(v/100.0))
    results_all.append(results)
    ax1.scatter(v_s, results, s=10, c=colors[ctr], marker=markers_list[ctr], label='r='+str(float(r/100.0)))
    ctr+=1
    
plt.legend(loc='upper left');
plt.show()

# plt.scatter(v_s,results)
# plt.xlabel('v1')
# plt.ylabel('z1')
# plt.show()

results=[]
results2=[]
w_s = []
for w in range(20,400,1):
    constraint = LinearConstraint(np.ones(z_solns), lb=0, ub=w)
    #cnsts.append(constraint)
    #cnsts.append({'type':'ineq', 'fun': lambda x: x})
    bnds = [(0, w) for i in range(2)] 
    res = minimize(
        objective_function_W,
        x0=np.abs(np.random.normal(0.0, w, z_solns)),
        #args=(v,),
        bounds=bnds,
        constraints=constraint
    )
    print(res)
    results.append(res.x[0])
    results2.append(res.x[1])
    w_s.append(w)
    
fig = plt.figure()

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

ax1 = fig.add_subplot(111)
ax1.scatter(w_s, results, s=10, c='b', marker="s", label='z1')
ax1.scatter(w_s,results2, s=10, c='r', marker="o", label='z2')

plt.xlabel('W',fontsize =14)
# plt.ylabel('z1')
plt.legend(loc='upper left');
plt.show()





