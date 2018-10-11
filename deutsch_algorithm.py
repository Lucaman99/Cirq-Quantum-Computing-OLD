import cirq
import numpy as np
import random
import tensorflow as tf
import time
import timeit
from cirq.google import ExpWGate, Exp11Gate, XmonMeasurementGate
from cirq.google import XmonSimulator

begin_execution = timeit.default_timer()

#This is going to be an attempt to simulate Deutsch's Algorithm using 3 qubits



#length = 1

q0 = cirq.GridQubit(0, 0)
q1 = cirq.GridQubit(0, 1)
q2 = cirq.GridQubit(0, 2)

#qubits = [cirq.GridQubit(i, j) for i in range(length) for j in range(length)]
#print(qubits)

moment0 = cirq.Moment([cirq.X.on(q0), cirq.X.on(q2)])
moment1 = cirq.Moment([cirq.H.on(q2), cirq.H.on(q1)])

circuit = cirq.Circuit((moment0, moment1))

#We can set passing_function to any f where f:{0, 1} --> {0, 1}
function_array = [0, 1]

def passing_function(qubit_x_measure):
    if (qubit_x_measure == 0):
        return function_array[0]
    if (qubit_x_measure == 1):
        return function_array[1]

if (function_array[0] == function_array[1]):
    tof_gate = cirq.CCX.on(q0, q1, q2)
    circuit.append(tof_gate)
    cnot_gate = cirq.CNOT(q1, q2)
    circuit.append(cnot_gate)
    if (function_array[0] == 1):
        x_gate = cirq.X.on(q2)
        circuit.append(x_gate)

else:
    cnot_gate = cirq.CNOT.on(q1, q2)
    circuit.append(cnot_gate)
    if (function_array[0] == 1):
        x_gate = cirq.X.on(q2)
        circuit.append(x_gate)

def deutsch_circuit_init(meas=True):
    yield cirq.H.on(q1)
    if meas:
        yield XmonMeasurementGate(key='q1')(q1)

circuit.append(deutsch_circuit_init())

print(circuit)

simulator = XmonSimulator()
result = simulator.run(circuit)

print(result)


end_execution = timeit.default_timer() - begin_execution
print(end_execution)
