# QUANTUM PERCEPTRON SIMULATION WITH 3 QUBITS AND 1 BIAS

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

# Input qubits

q = []
for l in range(0, 3):
    q.append(cirq.GridQubit(0, l))

# Target qubits

target = []
for o in range(3, 6):
    target.append(cirq.GridQubit(0, o))

# Work qubits
work = []
for y in range (6, 9):
    work.append(cirq.GridQubit(0, y))

# Final target qubit

final = cirq.GridQubit(0, 9)

moment0 = cirq.Moment([cirq.H.on(q[0]), cirq.H.on(q[1]), cirq.H.on(q[2])])
moment1 = cirq.Moment([cirq.CCX(q[0], q[1], target[0])])
moment2 = cirq.Moment([cirq.CCX(q[0], q[2], target[1])])
moment3 = cirq.Moment([cirq.CCX(q[2], q[1], target[2])])

circuit = cirq.Circuit((moment0, moment1, moment2, moment3))

def apply_n_qubit_tof(f):

    yield cirq.CCX.on(target[0], target[1], work[0])

    for i in range (2, f):
        yield cirq.CCX.on(target[i], work[i-2], work[i-1])
    yield cirq.CNOT.on(work[f-2], final)
    counter = f-1
    for i in range (2, f):
        yield cirq.CCX.on(target[counter], work[counter-2], work[counter-1])
        counter = counter - 1
    yield cirq.CCX.on(target[0], target[1], work[0])

circuit.append(apply_n_qubit_tof(3))

def apply_h():
    yield cirq.CCX.on(target[0], target[1], final)
    yield cirq.CCX.on(target[0], target[2], final)
    yield cirq.CCX.on(target[1], target[2], final)
    yield cirq.CNOT.on(target[0], final)
    yield cirq.CNOT.on(target[1], final)
    yield cirq.CNOT.on(target[2], final)
    #yield cirq.H.on(final)

circuit.append(apply_h())

def circuit_init_again(meas=True):
    if meas:
        yield XmonMeasurementGate(key='qubit')(final)

#circuit.append()
circuit.append(circuit_init_again())

print(" ")
print(" ")
print(circuit)
print(" ")
print(" ")

simulator = XmonSimulator()
result = simulator.run(circuit, repetitions=20)
print(result)
