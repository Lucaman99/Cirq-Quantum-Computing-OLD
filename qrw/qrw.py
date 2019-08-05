import cirq
import random
import numpy as np
import copy
import sympy
import itertools
from matplotlib import pyplot as plt

number_qubits = 7
iterator = 30

#A basic, one-dimensional quantum random walk implemented in Cirq

#There will be a qubit allocated which will be flipped by the "quantum coin", and will tell the random walk whether to jump left or right
#The position vector for a 2^N node random walk will be encoded using N qubits

qubits = []
for i in range(0, number_qubits):
    qubits.append(cirq.GridQubit(0, i))

ancilla = []
for i in range(0, number_qubits+2):
    ancilla.append(cirq.GridQubit(1, i))


def apply_n_qubit_tof(ancilla, args):

    if (len(args) == 2):
        yield cirq.CNOT.on(args[0], args[1])

    elif (len(args) == 3):
        yield cirq.CCX.on(args[0], args[1], args[2])

    else:

        yield cirq.CCX.on(args[0], args[1], ancilla[0])
        for k in range(2, len(args)-1):
            yield cirq.CCX(args[k], ancilla[k-2], ancilla[k-1])

        yield cirq.CNOT.on(ancilla[len(args)-3], args[len(args)-1])

        for k in range(len(args)-2, 1, -1):
            yield cirq.CCX(args[k], ancilla[k-2], ancilla[k-1])
        yield cirq.CCX.on(args[0], args[1], ancilla[0])

def initial_state():

    yield cirq.X.on(cirq.GridQubit(0, 1))
    #Make the even initial state for the coin-flip qubit


    yield cirq.X.on(cirq.GridQubit(0, number_qubits))
    #yield cirq.X.on(cirq.GridQubit(0, number_qubits))

    #yield cirq.H.on(cirq.GridQubit(0, number_qubits))
    #yield cirq.S.on(cirq.GridQubit(0, number_qubits))

def walk_step():
    
    #Start by applying the coin operator to the flip qubit

    #Implement the Addition Operator

    yield cirq.H.on(cirq.GridQubit(0, number_qubits))

    yield cirq.X.on(cirq.GridQubit(0, number_qubits))

    for i in range(number_qubits, 0, -1):

        yield apply_n_qubit_tof(ancilla, [cirq.GridQubit(0, v) for v in range(number_qubits, i-2, -1)])
        yield cirq.X.on(cirq.GridQubit(0, i-1))

    for i in range(number_qubits+1, 1, -1):
        yield cirq.X.on(cirq.GridQubit(0, i-1))

    #Implement the Substraction Operator

    yield cirq.X.on(cirq.GridQubit(0, number_qubits))

    for i in range(number_qubits+1, 1, -1):
        yield cirq.X.on(cirq.GridQubit(0, i-1))

    for i in range(1, number_qubits+1):

        yield apply_n_qubit_tof(ancilla, [cirq.GridQubit(0, v) for v in range(number_qubits, i-2, -1)])
        yield cirq.X.on(cirq.GridQubit(0, i-1))


circuit = cirq.Circuit()

circuit.append(initial_state())

for j in range(0, iterator):
    circuit.append(walk_step())
circuit.append(cirq.measure(*qubits, key='x'))

simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=100)
final = result.histogram(key='x')

x_arr = [j for j in dict(final).keys()]
y_arr = [dict(final)[j] for j in dict(final).keys()]

x_arr_final = []
y_arr_final = []

while (len(x_arr) > 0):

    x_arr_final.append(min(x_arr))
    y_arr_final.append(y_arr[x_arr.index(min(x_arr))])
    holder = x_arr.index(min(x_arr))
    del x_arr[holder]
    del y_arr[holder]

plt.plot(x_arr_final, y_arr_final)
plt.scatter(x_arr_final, y_arr_final)
plt.show()
