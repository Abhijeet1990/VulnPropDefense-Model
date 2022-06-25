#!/usr/bin/env python
# coding: utf-8

# this is just for the N node problem
import networkx as nx
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from math import log
import scipy.optimize as opt
import matplotlib.pyplot as plt

# define the constants
import itertools

def set_v(G):
    N = len(list(G.nodes))
    v = np.zeros(N)
    i=0
    for n in enumerate(list(G.nodes)):
        #v[i] = np.random.uniform(0,1)
        #print(n)
        #v[i] = float(G.degree(n[1]))/100.0
        v[i] = 0.5
        i+=1
    return v


# In[52]:


import json
# Generic Solution Test
# construct the Graph from the hierarchical model
with open('Utility_0.json') as f:
    json_data = json.loads(f.read())
G = nx.Graph()

G.add_nodes_from(elem['label'].split('.')[-1] if '.' in elem['label'] else elem['label'] for elem in json_data['nodes']['$values'])

G.add_edges_from((elem['src'].split('.')[-1], elem['dest'].split('.')[-1]) if ('.' in elem['src'] or '.' in elem['dest']) else (elem['src'], elem['dest']) for elem in json_data['links']['$values'])

nx.draw(G, pos=nx.kamada_kawai_layout(G),with_labels=True, node_size=2, linewidths=10,node_color='#00b4d9',font_size=9)
plt.savefig('util0.png',dpi=150)
plt.show()
v_prob = set_v(G)

# select the node with highest degree for evaluation of the optimal costs

for n in enumerate(list(G.nodes)):
    print('Node: '+str(n[1])+', degree: '+str(G.degree(n[1])))

def compute_adjacency_alpha(G):
    edge_tuples = []
    for i,sd_pair in enumerate(list(G.edges)):
        edge_tuples.append(sd_pair)
    
    N = len(list(G.nodes))
    alpha = np.zeros((N, N))
    
    #print(edge_tuples)
    # assign alpha based on the edges
    for i in list(G.nodes):
        for j in list(G.nodes):
            s_i = list(G.nodes).index(i)
            d_i = list(G.nodes).index(j)
            if (i,j) in edge_tuples:
                #alpha[i][j] = np.random.uniform(0,1)
                alpha[s_i][d_i] = 0.5
                alpha[d_i][s_i] = 0.5
    return alpha
    

def objective_function_gen(z,args): 
    # exploit probability
    v = args[0]

    # propagation prbability
    alpha = args[1]
 
    # theta and gamma
    th = 2
    ga = 0.7

    try:
        # after we obtain v_dash, we compute the log v_cap
        log_v_cap = np.ones(v.size)
        for k in range(v.size):
            log_v_cap[k] *= (log(v[k]) - th*log(ga*z[k] + 1))

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

alpha = compute_adjacency_alpha(G)
N = len(list(G.nodes))
z_solns = len(list(G.nodes))
W = 2000.0
res = None

def solve_optimization():
    v_prob = set_v(G)
    markers_list=['s','o','*','v','^','s','o','*','v','^','s','o','*','v','^','s','o','*','v','^']
    colors=['b','k','g','y','r','b','k','g','y','r','b','k','g','y','r','b','k','g','y','r']
    results = [[] for i in range(N)]
    v_s = []

    for v in range(1,99,1):
        #while res is None:
        v_prob[nodes_to_change_v] = float(v/100.0)
        constraint = LinearConstraint(np.ones(z_solns), lb=0, ub=W)
        try:
            res = minimize(
                objective_function_gen,
                bounds = [(0, W) for i in range(z_solns)],
                args=([v_prob,alpha],),
                x0=np.abs(np.random.normal(0.0, W, z_solns)),
                constraints=constraint
            )
        except:
            pass

        #print(res)
        for i in range(len(res.x)):
            results[i].append(res.x[i])
        v_s.append(float(v/100.0))


    fig = plt.figure()
    ax1 = fig.add_subplot(111)


    ctrx = 0
    for i in nodes_to_probe:  
        ax1.scatter(v_s, results[i], s=10, c=colors[ctrx], marker=markers_list[ctrx], label='z'+str(int(i)))
        ctrx+=1


    # ax1.scatter(v_s, results_z1, s=10, c='b', marker="s", label='z1')
    # ax1.scatter(v_s,results_z2, s=10, c='r', marker="o", label='z2')
    # ax1.scatter(v_s,results_z3, s=10, c='m', marker="*", label='z3')
    # plt.scatter(v_s,results)
    # plt.xlabel('v1')
    # plt.ylabel('optimal z')
    plt.xlabel('v'+str(nodes_to_change_v),fontsize =14)
    plt.ylabel('opt. z nearest nbr',fontsize = 14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='upper left');
    plt.show()


nodes_to_change_v = 4
nodes_to_probe = [j for j in range(16,34,1)]
nodes_to_probe.append(nodes_to_change_v)
solve_optimization()


nodes_to_change_v = 3
nodes_to_probe = [34,35]
nodes_to_probe.append(nodes_to_change_v)
solve_optimization()

nodes_to_change_v = 1
nodes_to_probe = [4,5,6]
nodes_to_probe.append(nodes_to_change_v)
solve_optimization()


nodes_to_change_v = 2
nodes_to_probe = [6,7,8]
nodes_to_probe.append(nodes_to_change_v)
solve_optimization()

nodes_to_change_v = 0
nodes_to_probe = [3,4,9]
nodes_to_probe.append(nodes_to_change_v)
solve_optimization()


nodes_to_change_v = 5
nodes_to_probe = [0,1,10,11]
nodes_to_probe.append(nodes_to_change_v)
solve_optimization()


nodes_to_change_v = 6
nodes_to_probe = [1,12,13]
nodes_to_probe.append(nodes_to_change_v)
solve_optimization()
