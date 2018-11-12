import cirq
import numpy as np
import random
import tensorflow as tf
import time
import timeit
from cirq.google import ExpWGate, Exp11Gate, XmonMeasurementGate
from cirq.google import XmonSimulator
from matplotlib import pyplot as plt
from itertools import combinations

one_sum = 0

#for xl in range(0, 100):

qubit_count = 3

bias = 1

threshold_value = bias + 1

#Control qubits

q = []
for l in range(0, qubit_count):
    q.append(cirq.GridQubit(0, l))

#Target qubit

target_q = cirq.GridQubit(0, qubit_count)

qubit_store = []
for k in range(0, qubit_count-1):
    qubit_store.append(cirq.GridQubit(0, qubit_count+1+k))
circuit = cirq.Circuit()

def controlled_n_qubit(n, qubit_sub):

    for v in qubit_sub:

        if (n == 1):
            cnot = cirq.CNOT.on(v[0], target_q)
            circuit.append(cnot)
        if (n == 2):
            ccx = cirq.CCX.on(v[0], v[1], target_q)
            circuit.append(ccx)

        if (n > 2):

            def apply_n_qubit_tof(f):

                yield cirq.CCX.on(v[0], v[1], qubit_store[0])

                for i in range (2, f):
                    yield cirq.CCX.on(v[i], qubit_store[i-2], qubit_store[i-1])
                yield cirq.CNOT.on(qubit_store[f-2], target_q)
                counter = f-1
                for i in range (2, f):
                    yield cirq.CCX.on(v[counter], qubit_store[counter-2], qubit_store[counter-1])
                    counter = counter - 1
                yield cirq.CCX.on(v[0], v[1], qubit_store[0])

            circuit.append(apply_n_qubit_tof(n))

def apply_h():
    yield cirq.H.on(q[0])
    yield cirq.H.on(q[1])
    yield cirq.H.on(q[2])

circuit.append(apply_h())

count = qubit_count
for y in range (threshold_value-1, qubit_count):
    controlled_n_qubit(count, list(combinations(q, count)))
    count = count - 1

def apply_after():
    yield cirq.H.on(target_q)

def circuit_init(meas=True):
    if meas:
        yield XmonMeasurementGate(key='q')(target_q)

circuit.append(circuit_init())

print(" ")
print(" ")
print(circuit)
print(" ")
print(" ")
    #simulator = XmonSimulator()
    #result = simulator.run(circuit)
    #if (str(result)[2] == "1"):
    #    one_sum = one_sum + 1
