import cirq
import numpy as np
import random
import tensorflow as tf
import time
import timeit
from cirq.google import ExpWGate, Exp11Gate, XmonMeasurementGate
from cirq.google import XmonSimulator
from matplotlib import pyplot as plt

qubit_count = 5

zero_graph = []

figure = plt.figure()
x = figure.add_subplot(211)

units = np.arange(0, 32, 1)

for a in range(0, 2):
    for j in range(0, 2):
        for k in range(0, 2):
            for h in range (0, 2):
                for l in range (0, 2):
                    '''
                    .
                    .
                    .
                    .
                    n nested for loops
                    '''

                    initial = [a, j, k, h, l]

                    print(initial)

                    zero_store = 0
                    one_store = 0

                    for l in range(0, 30):

                        #Control/target qubits

                        q = []

                        for i in range(0, qubit_count):
                            q.append(cirq.GridQubit(0, i))

                        #Work qubits

                        #print(q)

                        def create_work_qubits(n):
                            qubit_store = []
                            for i in range(0, n-2):
                                qubit_store.append(cirq.GridQubit(0, n+i))
                            return qubit_store

                        qubit_store = create_work_qubits(qubit_count)

                        #print(qubit_store)

                        def apply_n_qubit_tof(n):
                            yield cirq.CCX.on(q[0], q[1], qubit_store[0])
                            for i in range (2, n-1):
                                yield cirq.CCX.on(q[i], qubit_store[i-2], qubit_store[i-1])
                            yield cirq.CNOT.on(qubit_store[n-3], q[n-1])
                            counter = n-2
                            for i in range (2, n-1):
                                yield cirq.CCX.on(q[counter], qubit_store[counter-2], qubit_store[counter-1])
                                counter = counter - 1
                            yield cirq.CCX.on(q[0], q[1], qubit_store[0])

                        qubits_arr = [q[0], q[1], q[2], q[3], q[4]]

                        moment_arr = []
                        for u in range(0, len(initial)):
                            if (initial[u] == 1):
                                moment_arr.append(cirq.X.on(qubits_arr[u]))

                        moment0 = cirq.Moment(moment_arr)
                        #moment0 = cirq.Moment([cirq.X.on(q[0])])
                        moment1_arr = []
                        for i in q:
                            moment1_arr.append(cirq.H.on(i))
                        moment1 = cirq.Moment(moment1_arr)
                        moment_half = cirq.Moment([cirq.CNOT.on(q[0], q[1])])
                        moment2 = cirq.Moment([cirq.CCX.on(q[0], q[1], q[2])])

                        circuit = cirq.Circuit((moment0, moment1, moment_half, moment2))

                        if (len(q) > 3):
                            for i in range (4, qubit_count+1):
                                circuit.append(apply_n_qubit_tof(i))

                        def apply_z():
                            for i in q:
                                yield cirq.Z.on(i)

                        def apply_h():
                            yield cirq.H.on(q[0])
                            yield cirq.H.on(q[1])
                            yield cirq.H.on(q[2])

                        circuit.append(apply_h())

                        def circuit_init(meas=True):
                            if meas:
                                yield XmonMeasurementGate(key='q2')(q[2])

                        circuit.append(circuit_init())

                        #print(" ")
                        #print(" ")
                        #print(circuit)
                        #print(" ")
                        #print(" ")

                        simulator = XmonSimulator()

                        result = simulator.run(circuit)

                        measured = str(result)[len(str(result))-1]
                        if (int(measured) == 0):
                            zero_store = zero_store + 1
                        if (int(measured) == 1):
                            one_store = one_store + 1
                    print("Measured 0s: "+str(zero_store))
                    print("Measured 1s: "+str(one_store))

                    zero_graph.append(zero_store)



x.plot(units, zero_graph)
plt.show()


#The following part is simply for prototyping the products of different unitary matrices (gate composition)

'''

matrix_cxx = np.around(cirq.CCX.matrix())
matrix_cnot = np.around(cirq.CNOT.matrix())

matrix_pad = np.zeros(matrix_cxx.shape)


def identity(width, height, coordinate_1):
    coordinate_1 = coordinate_1 - 1
    full = []
    counter = 0
    for i in range (0, height):
        temporary = []
        for o in range (0, width):
            if (o != counter):
                temporary.append(0)
            else:
                temporary.append(1)
        full.append(temporary)
        counter = counter + 1
    full[coordinate_1][coordinate_1+1] = 1
    full[coordinate_1][coordinate_1] = 0
    full[coordinate_1+1][coordinate_1] = 1
    full[coordinate_1+1][coordinate_1+1] = 0
    return full

'''


#a = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
#[0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
#[0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]]

#b = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0],
#[0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
#[0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0]]

#print(a)
#print(b)
#print("-----------------")
#print(np.dot(b, a))
#print("-----------------")
#print(np.dot(a, b))
