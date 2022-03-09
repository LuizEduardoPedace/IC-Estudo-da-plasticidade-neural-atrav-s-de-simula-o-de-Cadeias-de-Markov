# Deterministic

from random import random


# Create phi()
def phi(potential, threshold):
    if potential >= threshold:
        prob = 1
    else:
        prob = 0
    return prob


n = int(input('Digit the number of neurons: '))
t = int(input('Digit the max time: '))
x = list()
v = list()
w = list()
aux = list()

# Initial condition for the membrane potential
for i in range(n):
    v.append(float(input(f'Insert the potential in time t=0 for neuron {i + 1}: ')))  # Here I choose the potential

# Initial condition for w(Matrix of n x n with i in line and j in column)
for i in range(n):
    for j in range(n):
        if i == j:
            aux.append(0)
        else:
            aux.append(float(input(f'Insert the w({i + 1})->({j + 1}): ')))  # Here I choose the W
    w.append(aux[:])
    aux.clear()

# Create matrix x(neurons in line and time in column)
for i in range(n):
    for time in range(t + 1):
        aux.append(0)
    x.append(aux[:])
    aux.clear()

# Interaction between neurons
thresh = 10  # or float(input('Insert the threshold: '))
for time in range(1, t+1):
    for j in range(n):
        rand = random()
        if rand <= phi(v[j], thresh):  # Spike
            x[j][time] = 1
            v[j] = 0
    for j in range(n):
        if x[j][time] == 0:  # No Spike
            for i in range(n):
                v[j] += w[i][j] * x[i][time]

# Print
print('~'*17)
print('Final Potentials:')
for i, V in enumerate(v):
    print(f'Neuron {i+1}: {V:.1f}')
print('~' * 17)
print('~' * (2*t+10))
for i, neuron in enumerate(x):
    print(f'Neuron {i+1}:', end=' ')
    for time, val in enumerate(neuron[1:len(neuron)]):
        if val == 1:
            print(f'\033[31m{val}\033[m', end=' ')
        else:
            print(f'{val}', end=' ')
    print()
print('~' * (2*t+10))
