#I CAN'T TELL IF YOU ARE WORKING OR NOT!!!!!

import cirq
from matplotlib import pyplot as plt
import math
import random

#Basic implementation of a quantum random walk on an undirected, cycle graph

n = 4

number_q = int(math.log(n)/math.log(2))
number_q = number_q
q = []

for i in range(0, number_q+1):
    q.append(cirq.GridQubit(0, i))

print(q)

def create_work_qubits(n):
    qubit_store = []
    for i in range(0, n-1):
        qubit_store.append(cirq.GridQubit(0, n+i+1))
    return qubit_store

qubit_store = create_work_qubits(number_q)

def apply_n_qubit_tof(n, input_target):
    yield cirq.CCX.on(q[input_target[0]], q[input_target[1]], qubit_store[0])
    for i in range (2, n):
        yield cirq.CCX.on(q[input_target[i]], qubit_store[i-2], qubit_store[i-1])

    yield cirq.CNOT.on(qubit_store[n-2], q[input_target[n]])
    counter = n
    for i in range (2, n):
        yield cirq.CCX.on(q[input_target[counter-1]], qubit_store[counter-3], qubit_store[counter-2])
        counter = counter - 1
    yield cirq.CCX.on(q[input_target[0]], q[input_target[1]], qubit_store[0])

circuit = cirq.Circuit()

'''
for i in range(1, number_q+1):
    cirq.H.on(cirq.GridQubit(0, i))
'''

def run_iter_walk():

    circuit.append(cirq.H.on(q[0]))


    for i in range(0, number_q-1):
        holder = []
        for v in range (0, number_q-i+1):
            holder.append(v)
        circuit.append(apply_n_qubit_tof(number_q-i, holder), strategy=cirq.InsertStrategy.NEW)
    circuit.append(cirq.CNOT.on(q[0], q[1]))

    for h in range (0, len(q)):
        circuit.append(cirq.X.on(cirq.GridQubit(0, h)), strategy=cirq.InsertStrategy.INLINE)

    for i in range(0, number_q-1):
        holder = []
        for v in range (0, number_q-i+1):
            holder.append(v)
        circuit.append(apply_n_qubit_tof(number_q-i, holder), strategy=cirq.InsertStrategy.NEW)
    circuit.append(cirq.CNOT.on(q[0], q[1]))

    for h in range (0, len(q)):
        circuit.append(cirq.X.on(cirq.GridQubit(0, h)), strategy=cirq.InsertStrategy.INLINE)


for i in range(0, 10):
    run_iter_walk()
'''
for k in range(1, number_q+1):
    circuit.append(cirq.measure(cirq.GridQubit(0, k), key=str(k)))
'''

circuit.append(cirq.measure(*q, key='x'))
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=2000)
print(result)
final = str(result)[2:].split(", ")

occurence_generator = []
x = []
for i in range(0, n):
    occurence_generator.append(0)
    x.append(i)


#Might have to switch order based on qubit endian-ness (I'm not sure what it is for Cirq yet)

print(circuit)

for i in range(0, len(final[0])):
    string_record = ""
    for v in range (1, len(final)):
        string_record = string_record+final[v][i]
    string_record = string_record[::-1]
    print(string_record)
    occurence_generator[int(string_record, 2)] = occurence_generator[int(string_record, 2)] + 1

plt.bar(x, occurence_generator)
plt.show()


#Classical Random Walk (to compare with quantum one)


'''
x = []
col = []
for b in range (0, 32):
    x.append(b)
    col.append(0)

for v in range(0, 2000):
    walker = 0
    for i in range(900):
        sel = random.randint(0, 1)
        if (sel == 0):
            walker = walker - 1
            walker = walker%32
        else:
            walker = walker + 1
            walker = walker%32
    col[walker] = col[walker]+1
    print("Done: "+str(v))

plt.bar(x, col)
plt.show()
'''
