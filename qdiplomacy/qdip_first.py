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

n = 3

qubits = []
for i in range(0, n):
    for j in range(0, n):
        qubits.append(cirq.GridQubit(i, j))

work_qubit = cirq.GridQubit(n, n)
extra_qubit = cirq.GridQubit(n, n+1)
extra_qubit2 = cirq.GridQubit(n, n+2)

qubits.append(work_qubit)
qubits.append(extra_qubit)
qubits.append(extra_qubit2)


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

    #Apply to horizontal triples
    for i in range(0, n):
        for j in range(0, n-2):
            yield cirq.CNOT.on(cirq.GridQubit(i, j), cirq.GridQubit(i, j+1))
            yield cirq.CNOT.on(cirq.GridQubit(i, j+1), cirq.GridQubit(i, j+2))
            yield cirq.ZPowGate(exponent=(-1*gamma)).on(cirq.GridQubit(i, j+2))
            yield cirq.CNOT.on(cirq.GridQubit(i, j), cirq.GridQubit(i, j+1))
            yield cirq.CNOT.on(cirq.GridQubit(i, j+1), cirq.GridQubit(i, j+2))
            yield cirq.ZPowGate(exponent=(-1*float(0.5)*gamma)).on(work_qubit)

    #Apply to vertical triples
    for i in range(0, n-1):
        for j in range(0, n):
            yield cirq.CNOT.on(cirq.GridQubit(i, j), cirq.GridQubit(i+1, j))
            yield cirq.CNOT.on(cirq.GridQubit(i+1, j), cirq.GridQubit(i+2, j))
            yield cirq.ZPowGate(exponent=(-1*gamma)).on(cirq.GridQubit(i+2, j))
            yield cirq.CNOT.on(cirq.GridQubit(i, j), cirq.GridQubit(i+1, j))
            yield cirq.CNOT.on(cirq.GridQubit(i+1, j), cirq.GridQubit(i+2, j))
            yield cirq.ZPowGate(exponent=(-1*float(0.5)*gamma)).on(work_qubit)

def apply_B_unitary(length, beta):
    for i in range(0, n):
        for j in range(0, n):
            yield cirq.XPowGate(exponent=(2*beta)).on(cirq.GridQubit(i, j))

#ASSUMES 3x3 QUBITS FOR A SPECIFIC CASE (AS SHOWN BELOW)

def apply_everything(length, gamma, beta):
    circuit.append(apply_C_unitary(length, gamma))
    circuit.append(apply_B_unitary(length, beta))



def objective_calc(values, extra):
    coefficient = 1

    matrix_horiz = [[-1, -1, 1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, 1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1, 1]]

    if (values not in matrix_horiz):
        coefficient = -1
    else:
        coefficient = 4

    total = 0
      #Gets the values of horizontally adjacent qubits
    for i in range(0, n):
        for j in range(0, n-1):
            multi = -1*values[(i*(n))+j]*values[(i*(n))+j+1]
            total = total + (multi+1)*0.5

      #Gets the value of vertically adjacent qubits
    for i in range(0, n):
        for j in range(0, n-1):
            multi = -1*values[(j*(n))+i]*values[(((j+1)*n))+i]
            total = total + (multi+1)*0.5

      #Gets vertical triples
    for i in range(0, n):
        for j in range(0, n-2):
            multi = values[(j*(n))+i]*values[(((j+1)*n))+i]*values[(((j+2)*n))+i]
            total = total + (multi+1)*0.5

      #Gets horizontal triples
    for i in range(0, n):
        for j in range(0, n-2):
            multi = values[(i*(n))+j]*values[(i*(n))+j+1]*values[(i*(n))+j+2]
            total = total + (multi+1)*0.5


    return coefficient*total



values_for_rotation = [0.1, 0.3, 0.5, 0.7]
new_rotation = list(itertools.permutations(values_for_rotation, 4))

maxcut_value = -10000
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
        apply_everything(n, float(gamma_matrix[3]), float(beta_matrix[3]))
        #circuit.append(apply_check_gates())

        circuit.append(cirq.measure(*qubits, key='x'))
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=100)

        processed_results = str(result)[2:].split(", ")

        sum_total = 0

        for j in range(0, len(processed_results[0])):
            trial_holder = []
            for k in range(0, len(processed_results)-3):
                if (int(processed_results[k][j]) == 0):
                    trial_holder.append(-1)
                else:
                    trial_holder.append(int(processed_results[k][j]))

            #print(trial_holder)
            extra = int(processed_results[len(processed_results)-1][j])


            sum_total = sum_total + objective_calc(trial_holder, extra)

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
print(maxcut_state)
apply_everything(n, maxcut_state[0][0], maxcut_state[1][0])
apply_everything(n, maxcut_state[0][1], maxcut_state[1][1])
apply_everything(n, maxcut_state[0][2], maxcut_state[1][2])
apply_everything(n, maxcut_state[0][3], maxcut_state[1][3])
circuit.append(cirq.measure(*qubits, key='x'))
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=100000)
print("64: "+str(dict(result.histogram(key='x'))[64]))
print("8: "+str(dict(result.histogram(key='x'))[8]))
print("512: "+str(dict(result.histogram(key='x'))[512]))
final = str(result)[2:].split(", ")
