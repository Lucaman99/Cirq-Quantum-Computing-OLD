import cirq
import numpy as np
import random
import tensorflow as tf
import time
import timeit
from cirq.google import ExpWGate, Exp11Gate, XmonMeasurementGate
from cirq.google import XmonSimulator
from matplotlib import pyplot as plt

qubit_count = 4

zero_graph = []

full_graph = []

figure = plt.figure()
x = figure.add_subplot(331)
y = figure.add_subplot(332)
z = figure.add_subplot(333)
w = figure.add_subplot(334)
xy = figure.add_subplot(335)
xz = figure.add_subplot(336)
xw = figure.add_subplot(337)
yz = figure.add_subplot(338)
yw = figure.add_subplot(339)

units = np.arange(0, 16, 1)

'''
for a in range(0, 2):
    for j in range(0, 2):
        for k in range(0, 2):
            for h in range (0, 2):
'''

#initial = [a, j, k, h]

init_arr = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0],
[0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0]]

for gh in range (0, 9):

    graphing = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    initial = init_arr[gh]

    #print(initial)

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

        qubits_arr = [q[0], q[1], q[2], q[3]]

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
            yield cirq.H.on(q[3])

        circuit.append(apply_h())

        def circuit_init(meas=True):
            if meas:
                yield XmonMeasurementGate(key='q0')(q[0])
                yield XmonMeasurementGate(key='q1')(q[1])
                yield XmonMeasurementGate(key='q2')(q[2])
                yield XmonMeasurementGate(key='q3')(q[3])

        circuit.append(circuit_init())

        print(" ")
        print(" ")
        print(circuit)
        print(" ")
        print(" ")

        simulator = XmonSimulator()

        result = simulator.run(circuit)

        measured = int(str(result)[3])



        print(str(result))
        print(str(result)[3])
        print(str(result)[8])
        print(str(result)[13])


        if ((str(result)[3]+str(result)[8]+str(result)[13]+str(result)[18]) == "0000"):
            graphing[0] = graphing[0] + 1
        if ((str(result)[3]+str(result)[8]+str(result)[13]+str(result)[18]) == "0001"):
            graphing[1] = graphing[1] + 1
        if ((str(result)[3]+str(result)[8]+str(result)[13]+str(result)[18]) == "0010"):
            graphing[2] = graphing[2] + 1
        if ((str(result)[3]+str(result)[8]+str(result)[13]+str(result)[18]) == "0011"):
            graphing[3] = graphing[3] + 1
        if ((str(result)[3]+str(result)[8]+str(result)[13]+str(result)[18]) == "0100"):
            graphing[4] = graphing[4] + 1
        if ((str(result)[3]+str(result)[8]+str(result)[13]+str(result)[18]) == "0101"):
            graphing[5] = graphing[5] + 1
        if ((str(result)[3]+str(result)[8]+str(result)[13]+str(result)[18]) == "0110"):
            graphing[6] = graphing[6] + 1
        if ((str(result)[3]+str(result)[8]+str(result)[13]+str(result)[18]) == "0111"):
            graphing[7] = graphing[7] + 1

        if ((str(result)[3]+str(result)[8]+str(result)[13]+str(result)[18]) == "1000"):
            graphing[8] = graphing[8] + 1
        if ((str(result)[3]+str(result)[8]+str(result)[13]+str(result)[18]) == "1001"):
            graphing[9] = graphing[9] + 1
        if ((str(result)[3]+str(result)[8]+str(result)[13]+str(result)[18]) == "1010"):
            graphing[10] = graphing[10] + 1
        if ((str(result)[3]+str(result)[8]+str(result)[13]+str(result)[18]) == "1011"):
            graphing[11] = graphing[11] + 1
        if ((str(result)[3]+str(result)[8]+str(result)[13]+str(result)[18]) == "1100"):
            graphing[12] = graphing[12] + 1
        if ((str(result)[3]+str(result)[8]+str(result)[13]+str(result)[18]) == "1101"):
            graphing[13] = graphing[13] + 1
        if ((str(result)[3]+str(result)[8]+str(result)[13]+str(result)[18]) == "1110"):
            graphing[14] = graphing[14] + 1
        if ((str(result)[3]+str(result)[8]+str(result)[13]+str(result)[18]) == "1111"):
            graphing[15] = graphing[15] + 1

        '''

        if (int(measured) == 0):
            zero_store = zero_store + 1
        if (int(measured) == 1):
            one_store = one_store + 1
    zero_graph.append(zero_store)

    '''
    full_graph.append(graphing)



x.plot(units, full_graph[0])
y.plot(units, full_graph[1])
z.plot(units, full_graph[2])
w.plot(units, full_graph[3])
xy.plot(units, full_graph[4])
xz.plot(units, full_graph[5])
xw.plot(units, full_graph[6])
yz.plot(units, full_graph[7])
yw.plot(units, full_graph[8])
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
#print("---------------
