'''
---------------------------------
SIMONS'S ALGORITHM - OVERVIEW
---------------------------------

Simon's Algorithm solves the problem of finding a particular value of s when some function f:{0, 1}^n --> {0, 1}^n
is inputted into the program that follows this rule: "f(x) = f(y) if and only if x (+) y is in the set {0^n, s}"

---------------------------------
STEPS OF THE ALGORITHM
---------------------------------

1. Begin with two n-qubit registers, each in the |0> state
2. Apply a Hadamard transform to the first n-qubit register, therefore creating an even superposition of states
3. The oracle (which encodes values of the function) is queried B_f |x>|y> = |x>|y (+) f(x)>, therefore mapping |0^n> --> |f(x)>
4. Apply a Hadamard transform to the first n-qubit register

---------------------------------
MEASUREMENT
---------------------------------

It can be found that for an output string y, then y (dot mod 2) s is always equal to 0, as we can calculate y (dot mod 2) s = 1 occuring with probability = 0
We get a system of eqautions, which we can use to solve for s (provided y_1, ..., y_(n-1) are linearlly independent)! We measure the string y to be the first n-qubit register

'''

import cirq
import random
import numpy as np

# Qubit preparation

number_qubits = 3

def main(number_qubits):

    circuit_sampling = number_qubits-1

    #Create the qubits which are used within the circuit

    first_qubits = [cirq.GridQubit(i, 0) for i in range(number_qubits)]
    second_qubits = [cirq.GridQubit(i, 0) for i in range(number_qubits, 2*number_qubits)]

    the_activator = cirq.GridQubit(2*number_qubits, 0)

    #Create the qubits that can be used for large-input Toffoli gates
    ancilla = []
    for v in range(2*number_qubits+1, 3*number_qubits):
        ancilla.append(cirq.GridQubit(v, 0))

    #Create the function that is inputted into the algorithm (secret!)

    domain = []
    selector = []
    co_domain = []
    fixed = []

    for k in range(0, 2**number_qubits):
        domain.append(k)
        selector.append(k)
        co_domain.append(False)
        fixed.append(k)

    #Create the "secret string"
    s = domain[random.randint(0, len(domain)-1)]

    #Create the "secret function"
    for g in range(0, int((2**number_qubits)/2)):
        v = random.choice(selector)
        x = random.choice(domain)
        co_domain[x] = v
        co_domain[x^s] = v
        del selector[selector.index(v)]
        del domain[domain.index(x)]
        if (s != 0):
            del domain[domain.index(x^s)]

    secret_function = [fixed, co_domain]

    oracle = make_oracle(ancilla, secret_function, first_qubits, second_qubits, s, the_activator)

    c = make_simon_circuit(first_qubits, second_qubits, oracle)

    #Sampling the circuit

    simulator = cirq.Simulator()
    result = simulator.run(c, repetitions=number_qubits-1)
    final = result.histogram(key='y')
    print("Secret String: "+str(s))
    print("Secret Function (Domain and Co-Domain): "+str(secret_function))
    final = str(result)[str(result).index("y")+2:len(str(result))].split(", ")
    last = []
    for i in range(0, number_qubits-1):
        holder = []
        for j in final:
            holder.append(int(j[i]))
        last.append(holder)

    print("Results: "+str(last))

    #Classical post-processing --> XOR Satisfiability Problem


    matrix = [[1, 1, 1, 1, 0], [1, 1, 0, 1, 0], [0, 1, 0, 0, 0], [1, 0, 1, 1, 1]]

#Elimination into triangular form (mod 2)

for i in range(0, len(matrix)):
  for j in range(i+1, len(matrix)):
    x = -1*matrix[j][i]/matrix[i][i]
    iterator = map(lambda y: y*int(x), matrix[i])
    new = [sum(z)%2 for z in zip(matrix[j], iterator)]
    matrix[j] = new

    print(matrix)
  print("--------")

print(matrix)

for i in range(len(matrix)-1, 0, -1):
  for j in range(i, len(matrix)):
    print([i, j])




def make_oracle(ancilla, secret_function, first_qubits, second_qubits, s, the_activator):

    #Hard-code oracle on a case-by-case basis


    for o in range(0, len(secret_function[0])):
        counter = 0
        for j in list(str(format(secret_function[0][o], "0"+str(number_qubits)+"b"))):
            if (int(j) == 0):
                yield cirq.X.on(first_qubits[counter])
            counter = counter+1
        yield apply_n_qubit_tof(ancilla, first_qubits+[the_activator])
        counter = 0
        for j in list(str(format(secret_function[0][o], "0"+str(number_qubits)+"b"))):
            if (int(j) == 0):
                yield cirq.X.on(first_qubits[counter])
            counter = counter+1

        counter = 0
        for j in list(str(format(secret_function[1][o], "0"+str(number_qubits)+"b"))):
            if (int(j) == 1):
                yield cirq.CNOT.on(the_activator, second_qubits[counter])
            counter = counter+1

        counter = 0
        for j in list(str(format(secret_function[0][o], "0"+str(number_qubits)+"b"))):
            if (int(j) == 0):
                yield cirq.X.on(first_qubits[counter])
            counter = counter+1
        yield apply_n_qubit_tof(ancilla, first_qubits+[the_activator])
        counter = 0
        for j in list(str(format(secret_function[0][o], "0"+str(number_qubits)+"b"))):
            if (int(j) == 0):
                yield cirq.X.on(first_qubits[counter])
            counter = counter+1

def apply_n_qubit_tof(ancilla, args):

    if (len(args) == 3):
        yield cirq.CCX.on(args[0], args[1], args[2])

    else:

        yield cirq.CCX.on(args[0], args[1], ancilla[0])
        for k in range(2, len(args)-1):
            yield cirq.CCX(args[k], ancilla[k-2], ancilla[k-1])

        yield cirq.CNOT.on(ancilla[len(args)-3], args[len(args)-1])

        for k in range(len(args)-2, 1, -1):
            yield cirq.CCX(args[k], ancilla[k-2], ancilla[k-1])
        yield cirq.CCX.on(args[0], args[1], ancilla[0])


def make_simon_circuit(first_qubits, second_qubits, oracle):

    circuit = cirq.Circuit()

    #Apply the first set of Hadamard gates

    for i in range(0, number_qubits):
        circuit.append(cirq.H.on(first_qubits[i]))

    #Apply the oracle

    circuit.append(oracle)

    #Apply the second set of Hadamard gates

    for i in range(0, number_qubits):
        circuit.append(cirq.H.on(first_qubits[i]))

    #Perform measurements upon the qubits

    circuit.append(cirq.measure(*second_qubits, key='x'))
    circuit.append(cirq.measure(*first_qubits, key='y'))

    return circuit

main(number_qubits)
