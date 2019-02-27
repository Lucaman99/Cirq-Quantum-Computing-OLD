import cirq
import numpy
import scipy
import sympy
import random

length = 3

qubits = []
for i in range (length):
    for j in range (length):
        qubits.append(cirq.GridQubit(i, j))


def x_rotation(length, turns):
    rot = cirq.XPowGate(exponent=turns)
    for i in range (length):
        for j in range (length):
            yield rot.on(cirq.GridQubit(i, j))

def random_spin_generation(rows, cols):
    final_matrix = []
    for i in range (rows):
        column = []
        for j in range (cols):
            spin = random.choice([+1, -1])
            column.append(spin)
        final_matrix.append(column)
    return final_matrix

def get_matrices(length):
    h = random_spin_generation(length, length)
    row = random_spin_generation(length-1, length)
    col = random_spin_generation(length, length-1)
    return h, row, col

par = [[[-1, 1, -1], [1, -1, -1], [-1, 1, -1]], [[1, 1, -1], [1, -1, 1]], [[1, -1], [-1, 1], [-1, 1]]]
#par = get_matrices(3)

def rot_z(h, turns, length):
    for i in range (length):
        for j in range (length):
            if (h[i][j] == +1):
                rotz = cirq.ZPowGate(exponent=turns)
                yield rotz.on(cirq.GridQubit(i, j))

def apply_cz(row, col, turns, length):
    for i in range(length-1):
        for j in range(length):
            if (row[i][j] == -1):
                yield cirq.X.on(cirq.GridQubit(i, j))
                yield cirq.X.on(cirq.GridQubit(i+1, j))
            yield cirq.CZPowGate(exponent=turns).on(cirq.GridQubit(i, j), cirq.GridQubit(i+1, j))
            if (row[i][j] == -1):
                yield cirq.X.on(cirq.GridQubit(i, j))
                yield cirq.X.on(cirq.GridQubit(i+1, j))
    for i in range(length):
        for j in range(length-1):
            if (col[i][j] == -1):
                yield cirq.X.on(cirq.GridQubit(i, j))
                yield cirq.X.on(cirq.GridQubit(i, j+1))
            yield cirq.CZPowGate(exponent=turns).on(cirq.GridQubit(i, j), cirq.GridQubit(i, j+1))
            if (col[i][j] == -1):
                yield cirq.X.on(cirq.GridQubit(i, j))
                yield cirq.X.on(cirq.GridQubit(i, j+1))

def apply_everything(length, a, b, c):
    circuit.append(x_rotation(length, a))
    circuit.append(rot_z(par[0], b, length))
    circuit.append(apply_cz(par[1], par[2], c, length))

k = [0.1, 0.3, 0.5, 0.7, 0.9]
recorder = []

for a in k:
    for b in k:
        for c in k:

            circuit = cirq.Circuit()
            apply_everything(3, a, b, c)

            circuit.append(cirq.measure(*qubits, key='x'))

            simulator = cirq.Simulator()

            result = simulator.run(circuit, repetitions=100)

            final = str(result)[2:].split(", ")
            print(result.histogram(key='x'))

            def calculate_energy(h, row, col, num, result):
                total_energy_value = 0
                for i in range(length):
                    for j in range(length):
                        total_energy_value += (1 - (2*int(result[(3*i)+j][num])))*h[i][j]
                for i in range(length-1):
                    for j in range(length):
                        total_energy_value += row[i][j]*(1 - (2*int(result[(3*i)+j][num])))*(1 - (2*int(result[(3*(i+1))+j][num])))
                for j in range(length-1):
                    for i in range(length):
                        total_energy_value += col[i][j]*(1 - (2*int(result[(3*i)+j][num])))*(1 - (2*int(result[(3*i)+j+1][num])))
                return total_energy_value

            counter = []
            counter2 = []



            for f in range (100):
                res_eng = calculate_energy(par[0], par[1], par[2], f, final)
                if (res_eng not in counter):
                    counter.append(res_eng)
                    counter2.append(1)
                else:
                    counter2[counter.index(res_eng)] += 1

            print(counter)
            print(counter2)

            expectation = 0
            for i in range (len(counter)):
                expectation += counter[i]*counter2[i]/100

            print(expectation)
            recorder.append([a, b, c, expectation])

print(recorder[1][3])
minimum = False
a = False
b = False
c = False
for r in recorder:
    if (minimum == False or r[3] < minimum):
        minimum = r[3]
        a = r[0]
        b = r[1]
        c = r[2]
print("The minimum value of the energy eigenvalue for the Hamiltonian with h = "+str(par[0])+", J(rows) = "+str(par[1])+", and J(cols) = "+str(par[2])+" is "+str(minimum)+" with parameters on the ansatz as a = "+str(a)+", b = "+str(b)+", and c = "+str(c))
