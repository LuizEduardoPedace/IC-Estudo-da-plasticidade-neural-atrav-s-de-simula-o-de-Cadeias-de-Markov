# Stochastic
# Leak
# Plasticity
# Graph and BoxPlot
# Numpy
# Inhibitory neurons
# Fix the plasticity (p matrix and Standart)

from random import random, sample
from math import e
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns


# Create phi()
def phi(potential):
    a = -2  # Here I choose the a float(input('Choose the value of a'))
    prob = 1 / (1 + e ** (-0.2 * potential + a))
    return prob


# Graph
def graph(axe, colormap, minimum):
    global limit, phantom_edge, ntype
    G = nx.DiGraph()
    cw = list()  # List for colormap
    for gi in range(n):  # Add nodes
        G.add_node(f'N{gi + 1}')
    for gi, ni in enumerate(w):  # Add edges
        for gj, nj in enumerate(ni):
            if abs(nj) > minimum:  # Put the minimum weight
                G.add_edge(f'N{gi + 1}', f'N{gj + 1}', weight=nj)
                cw.append(nj)
    if not colormap:
        limit = max(abs(np.array(cw)))
        phantom_edge = limit
        if max(cw) == limit: phantom_edge = -limit
    cw.append(phantom_edge)
    G.add_edge(f'N{n}', f'N{n}', weight=phantom_edge, width=0)
    if colormap:
        nx.draw(G, nx.circular_layout(G), edge_color=cw, width=4, arrowsize=15, node_size=500, font_size=15,
                edge_cmap=plt.cm.RdGy, vmin=-abs(limit), vmax=abs(limit),
                connectionstyle='Arc3, rad=0.15', with_labels=True)
        nx.draw_networkx_nodes([g for ng, g in enumerate(G.nodes) if ntype[ng] == -1], nx.circular_layout(G),
                               node_size=500, node_color='r')  # Red node
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdGy, norm=plt.Normalize(vmin=-abs(limit), vmax=abs(limit)))
        sm._A = []
        plt.colorbar(sm)
    else:
        nx.draw_networkx_labels(G, nx.circular_layout(G), font_size=15, ax=axe)
        nx.draw_networkx_nodes([g for ng, g in enumerate(G.nodes) if ntype[ng] == 1], nx.circular_layout(G),
                               node_size=500, ax=axe)
        nx.draw_networkx_nodes([g for ng, g in enumerate(G.nodes) if ntype[ng] == -1], nx.circular_layout(G),
                               node_size=500, ax=axe, node_color='r')  # Red node
        rep = list()  # Optimize the plot
        for gi in w:
            for gj in gi:
                if gj not in rep:  # Optimize the plot
                    weighted_edges = [(node1, node2) for (node1, node2, edge_attr) in G.edges(data=True) if
                                      edge_attr['weight'] == gj]
                    if gj < 0:
                        nx.draw_networkx_edges(G, nx.circular_layout(G), edgelist=weighted_edges, arrowsize=15,
                                               width=abs(gj) / 5, connectionstyle='Arc3, rad=0.15', ax=axe,
                                               edge_color='r')
                    else:
                        nx.draw_networkx_edges(G, nx.circular_layout(G), edgelist=weighted_edges, arrowsize=15,
                                               width=gj / 5, connectionstyle='Arc3, rad=0.15', ax=axe)
                    rep.append(gj)
        nx.draw(nx.DiGraph(), ax=axe)


n = int(input('Digit the number of neurons: '))
t = int(input('Digit the max time: '))

# Initial condition for the membrane potential(Vector with n coordinates)
v = np.ones(n) * float(input('Insert V(0): '))  # Here I choose the potential
gv = [v.tolist()]  # Vector for graphic of potential

# Initial condition for w(Matrix of n x n)
w = np.ones((n, n)) * float(input('Insert W: '))  # Here I choose the W
np.fill_diagonal(w, 0)

# Type of neuron
theta = float(input('Choose the theta (0 <): '))
ntype = np.ones(n)  # Vector type of neurons
for nwi in range(n):
    if sample([-1, -1, 1, 1, 1, 1, 1, 1, 1, 1], 1)[0] == -1:  # Inhibitory neuron
        w[nwi] *= -theta
        ntype[nwi] = -1

# Graph
fig, (axe1, axe2) = plt.subplots(1, 2)  # Subplot 1x2
graph(axe1, False, 0)  # Initial graph

# Create matrix x (Neurons in line and time+1 in column)
x = np.zeros((n, t + 1))

# Create L (Vector with n coordinates)
L = np.zeros(n)

# Create p (Vector with n coordinates)
p = np.zeros((n, n), dtype=bool)

# Interaction between neurons
alpha = float(input('Choose the value of the leak in (0, 1): '))  # Here I put the leak
beta = float(input('Choose the beta (0 <): '))  # Here I put the beta
c = float(input('Choose the value of C (0 <): '))  # Here I put the C
for time in range(1, t + 1):
    for j in range(n):  # Update the x
        if random() <= phi(v[j]):  # Spike
            x[j][time] = 1
            v[j] = -40
    for j in range(n):  # Update the w
        for i in range(n):
            if ntype[j] == 1 and i != j:  # Plasticity for excitatory neurons
                w[j][i] *= e ** (-beta)
                if p[j][i] == True and x[i][time] == 1:
                    w[j][i] += c
                    if w[j][i] < 0: w[j][i] = 0  # Floor for excitatory neurons
                    p[j][i] = False
            if ntype[j] == -1 and i != j:  # Plasticity for inhibitory neurons
                w[j][i] *= e ** beta
                if p[j][i] == True and x[i][time] == 1:
                    w[j][i] += 4 * c
                    if w[j][i] > 0: w[j][i] = 0  # Roof for inhibitory neurons
                    p[j][i] = False
            if w[j][i] < -25: w[j][i] = -25  # Floor
            if w[j][i] > 25: w[j][i] = 25  # Roof
    for j in range(n):  # Update the L
        if x[j][time] == 1:
            L[j] = time
            p[j] = np.ones(n, dtype=bool)
    for j in range(n):  # Update the v
        if x[j][time] == 0:  # No Spike
            v[j] = v[j] * alpha
            for i in range(n):
                v[j] += w[i][j] * x[i][time]
    gv.append(v.tolist())  # Graphic of potential

# Print
print('~' * (2 * t + 10))
for i, neuron in enumerate(x):
    print(f'Neuron {i + 1:>2}:', end=' ')
    for time, val in enumerate(neuron[1:len(neuron)]):
        if val == 1:
            print(f'\033[31m{val:.0f}\033[m', end=' ')
        else:
            print(f'{val:.0f}', end=' ')
    print()
print('~' * (2 * t + 10))

# Graph
graph(axe2, False, 0)  # Final graph
axe1.set_title('Begin')
axe2.set_title('End')
plt.show()

# Visualization
gv = np.transpose(np.array(gv))
gt = np.arange(t + 1)
gN = int(input('What do you want to see? '))
while gN != -1:
    if gN == -2:
        print(np.around(w, 1))
        graph(axe1, True, float(input('Insert the minimum weight: ')))
    elif gN == -3:
        sns.set_theme(style='whitegrid')
        np.fill_diagonal(w, limit + 1)  # Exclude principal diagonal  of plot
        sns.boxplot(data=[val for boxi in w for val in boxi if val != limit + 1], width=0.5)
    elif gN == -4:
        np.fill_diagonal(w, limit + 1)  # Exclude principal diagonal  of plot
        sns.histplot(data=[val for boxi in w for val in boxi if val != limit + 1], binwidth=1, stat='density', kde=True,
                     color='b')
    else:
        if gN == 0:
            for g0 in range(n):
                plt.plot(gt, gv[g0], label=f'Neuron {g0 + 1}')
        else:
            plt.plot(gt, gv[gN - 1], label=f'Neuron {gN}')
        plt.xlabel('Time')
        plt.ylabel('Potential')
        plt.legend()
        plt.title('Potential x Time')
    plt.show()
    np.fill_diagonal(w, 0)  # Return w to normal
    gN = int(input('What do you want to see? '))
