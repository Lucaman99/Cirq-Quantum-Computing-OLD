'''
MAXCUT QAOA (With random sampling --> Inefficient but good for first implementation)
Define qubit registers and p-value
Define objective function
Define U(C) and U(B) operators
Build the quantum circuit
Measure the state
Use the state to calculate the value of the objective function
TODO above: Repeat process h times, find the largest value (Maybe implement Grover's???)
TODO above: Repeat the whole process while varying the angles of the two operators
'''

import cirq
import itertools

n = 2

qubits = []
for i in range(0, n):
    for j in range(0, n):
        qubits.append(cirq.GridQubit(i, j))

work_qubit = cirq.GridQubit(n, n)
qubits.append(work_qubit)

def apply_h_gates(length):
    for i in range (0, length):
        for j in range (0, length):
            yield cirq.H.on(cirq.GridQubit(i, j))

def apply_C_unitary(length, gamma):
    # Apply the compute --> rotate --> uncompute  method to change the phase of a specific computational basis state

    #Apply to horizontally adjacent qubits
    for i in range(0, n):
        for j in range(0, n-1):
            yield cirq.CNOT.on(cirq.GridQubit(i, j), cirq.GridQubit(i, j+1))
            yield cirq.ZPowGate(exponent=(-1*gamma)).on(cirq.GridQubit(i, j+1))
            yield cirq.CNOT.on(cirq.GridQubit(i, j), cirq.GridQubit(i, j+1))
            yield cirq.ZPowGate(exponent=(-1*float(0.5)*gamma)).on(work_qubit)

    #Apply to vertically adjacent qubits
    for i in range(0, n-1):
        for j in range(0, n):
            yield cirq.CNOT.on(cirq.GridQubit(i, j), cirq.GridQubit(i+1, j))
            yield cirq.ZPowGate(exponent=(-1*gamma)).on(cirq.GridQubit(i+1, j))
            yield cirq.CNOT.on(cirq.GridQubit(i, j), cirq.GridQubit(i+1, j))
            yield cirq.ZPowGate(exponent=(-1*float(0.5)*gamma)).on(work_qubit)

def apply_B_unitary(length, beta):
    for i in range(0, n):
        for j in range(0, n):
            yield cirq.XPowGate(exponent=(2*beta)).on(cirq.GridQubit(i, j))

def apply_everything(length, gamma, beta):
    circuit.append(apply_C_unitary(length, gamma))
    circuit.append(apply_B_unitary(length, beta))


def objective_calc(values):
    total = 0
    #Gets the values of horizontally adjacent qubits
    for i in range(0, n):
        for j in range(0, n-1):
            multi = -1*values[(i*(n))+j]*values[(i*(n))+j+1]
            total = total + (multi+1)*0.5

    #Gets the value of horizontally adjacent qubits
    for i in range(0, n):
        for j in range(0, n-1):
            multi = -1*values[(j*(n))+i]*values[(((j+1)*n))+i]
            total = total + (multi+1)*0.5

    return total



values_for_rotation = [0.1, 0.3, 0.5, 0.7, 0.9]
new_rotation = list(itertools.permutations(values_for_rotation, 3))

maxcut_value = 0
maxcut_state = 0

#Pick 2 to be used in the set
for g in range (0, len(new_rotation)):
    for h in range(0, len(new_rotation)):

        gamma_matrix = new_rotation[g]
        beta_matrix = new_rotation[h]

        circuit = cirq.Circuit()
        circuit.append(apply_h_gates(n))
        apply_everything(n, float(gamma_matrix[0]), float(beta_matrix[0]))
        apply_everything(n, float(gamma_matrix[1]), float(beta_matrix[1]))
        apply_everything(n, float(gamma_matrix[2]), float(beta_matrix[2]))

        circuit.append(cirq.measure(*qubits, key='x'))
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=100)

        processed_results = str(result)[2:].split(", ")

        sum_total = 0

        for j in range(0, len(processed_results[0])):
            trial_holder = []
            for k in range(0, len(processed_results)-1):
                if (int(processed_results[k][j]) == 0):
                    trial_holder.append(-1)
                else:
                    trial_holder.append(int(processed_results[k][j]))

            sum_total = sum_total + objective_calc(trial_holder)

        sum_total = sum_total/100

        print(result.histogram(key='x'))
        print(sum_total)
        print([gamma_matrix, beta_matrix])
        print("------------------------------------------")

        if (sum_total > maxcut_value):
            maxcut_value = sum_total
            maxcut_state = [gamma_matrix, beta_matrix]

print("Max-Cut Value: "+str(maxcut_value))
print("------------------------------------------")
print("Max-Cut Parameters: "+str(maxcut_state))


circuit = cirq.Circuit()
circuit.append(apply_h_gates(n))
apply_everything(n, maxcut_state[0][0], maxcut_state[1][0])
apply_everything(n, maxcut_state[0][1], maxcut_state[1][1])
apply_everything(n, maxcut_state[0][2], maxcut_state[1][2])
circuit.append(cirq.measure(*qubits, key='x'))
print(circuit)
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=1000)
print(result.histogram(key='x'))
final = str(result)[2:].split(", ")
