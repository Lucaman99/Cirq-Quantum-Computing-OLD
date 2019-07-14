'''
MAXCUT QAOA (With random sampling --> Inefficient but good for first implementation)
'''

print("Hello")

import cirq
import itertools
import random

weight_on_first_ham = 1
weight_on_second_ham = 2

# Graph objects for creating the graphs that will be used in the optimization algorithm

class Edge:
    def __init__(self, start_node, end_node):
        self.start_node = start_node
        self.end_node = end_node


class Graph:
    def __init__(self, edges_set):
        self.edges_set = edges_set
        self.node_set = []
        for i in edges_set:
            if (i.start_node not in self.node_set):
                self.node_set.append(i.start_node)
            if (i.end_node not in self.node_set):
                self.node_set.append(i.end_node)

#Connects nodes with an edge
    def connect_nodes(self, edge):
        if (edge not in self.edges_set and edge.start_node in self.node_set and edge.end_node in self.node_set):
            self.edges_set.append(edge)

#Adds a node to the graph
    def add_nodes(self, node):
        if (node not in self.node_set):
            self.node_set.append(node)

#Removes nodes from the graph. If a node is removed, all edges connected to that node are removed as well

    def remove_nodes(self, node):
        if (node in self.node_set):
            del self.node_set[self.node_set.index(node)]
            new = []
            for i in range (0, len(self.edges_set)):
                if (node != self.edges_set[i].start_node and node != self.edges_set[i].end_node):
                    new.append(self.edges_set[i])
            self.edges_set = new

#Disconnects nodes, thereby removing an edge
    def disconnect_nodes(self, edge):
        if (edge in self.edges_set):
            del self.edges_set[self.edges_set.index(edge)]



#Define the problem graph
set_edges = [Edge(0, 1), Edge(1, 2)]

graph = Graph(set_edges)

n = len(set_edges)+1

def find_loops():
    last_list = []
    for i in set_edges:
        last_list.append([i.start_node, i.end_node])
    return last_list


the_search = find_loops()


#Generate and Store the qubits in an array

qubits = []
for i in range(0, n):
    qubits.append(cirq.GridQubit(0, i))

qubit_store = []
for i in range(0, n):
    qubits.append(cirq.GridQubit(1, i))

print(qubits)

#Creating extra/work qubits to be used during calculations
work_qubit = cirq.GridQubit(n+5, n+5)


def apply_h_gates(length):
    for i in range (0, length):
        yield cirq.H.on(cirq.GridQubit(0, i))


def apply_n_qubit_tof(number, input_target):
    yield cirq.CCX.on(qubits[input_target[0]], qubits[input_target[1]], qubit_store[0])
    for i in range (2, number):
        yield cirq.CCX.on(qubits[input_target[i]], qubit_store[i-2], qubit_store[i-1])

    yield cirq.CNOT.on(qubit_store[number-2], qubits[input_target[number]])
    counter = number
    for i in range (2, number):
        yield cirq.CCX.on(qubits[input_target[counter-1]], qubit_store[counter-3], qubit_store[counter-2])
        counter = counter - 1
    yield cirq.CCX.on(qubits[input_target[0]], qubits[input_target[1]], qubit_store[0])

'''
def apply_other_c(length, gamma):
    for j in graph.node_set:
        yield cirq.CZPowGate(exponent=(-1*gamma)).on(cirq.GridQubit(0, j), work_qubit)
'''

def apply_C_unitary(length, gamma):
    # Apply the compute --> rotate --> uncompute  method to change the phase of a specific computational basis state

    for i in the_search:
        for j in range (0, len(i)-1):
            yield cirq.CNOT.on(cirq.GridQubit(0, i[j]), cirq.GridQubit(0, i[j+1]))
            yield cirq.Rz((-1*gamma)).on(cirq.GridQubit(0, i[(j+1)]))
            yield cirq.CNOT.on(cirq.GridQubit(0, i[j]), cirq.GridQubit(0, i[(j+1)]))
            yield cirq.Rz((-1*gamma)).on(work_qubit)

def apply_B_unitary(length, beta):
    for i in range(0, n):
        yield cirq.Rx((2*beta)).on(cirq.GridQubit(0, i))

def apply_everything(length, gamma, beta):
    circuit.append(apply_C_unitary(length, gamma))
    circuit.append(apply_B_unitary(length, beta))


def objective_calc(values, extra):

    coefficient = 1

    total = 0
      #Gets the values of horizontally adjacent qubits
    for i in the_search:
        for j in range(0, len(i)-1):
            multi = -1*values[i[j]]*values[i[(j+1)]]
            total = total + (multi+1)*0.5

    return float(coefficient)*total

def new_calc(values, extra):
    coefficient = 1
    total = 0
    for i in graph.node_set:
        multi = -1*values[i]
        total = total + (multi+1)*0.5

    return float(coefficient)*total


values_for_rotation = [0.2, 0.4, 0.6, 0.8]
number_of_steps = 4
choice = 100

new_rotation = list(itertools.permutations(values_for_rotation, number_of_steps))
holding = []
for j in range (0, choice):
    holding.append(new_rotation[random.randint(0, len(new_rotation)-1)])
new_rotation = holding


maxcut_value = -10000
maxcut_state = 0

#Pick 2 to be used in the set

for g in range (0, len(new_rotation)):
    print("Complete Trial: "+str(g)+"/"+str(len(new_rotation)))
    for h in range(0, len(new_rotation)):

        gamma_matrix = new_rotation[g]
        beta_matrix = new_rotation[h]

        circuit = cirq.Circuit()

        circuit.append(apply_h_gates(n))

        for g in range(0, number_of_steps):
            apply_everything(n, float(gamma_matrix[g]), float(beta_matrix[g]))
        #circuit.append(apply_check_gates())

        circuit.append(cirq.measure(*qubits, key='x'))

        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=100)
        print("Done")

        processed_results = str(result)[2:].split(", ")

        sum_total = 0

        for j in range(0, len(processed_results[0])):
            trial_holder = []
            for k in range(0, len(processed_results)):
                if (int(processed_results[k][j]) == 0):
                    trial_holder.append(-1)
                else:
                    trial_holder.append(int(processed_results[k][j]))

            #print(trial_holder)
            extra = int(processed_results[len(processed_results)-1][j])


            sum_total = sum_total + objective_calc(trial_holder, extra)

        sum_total = sum_total/100
        #print(result.histogram(key='x'))
        #print([gamma_matrix, beta_matrix])
        #print("------------------------------------------")

        if (sum_total > maxcut_value):
            maxcut_value = sum_total
            maxcut_state = [gamma_matrix, beta_matrix]

'''
print("Max-Cut Value: "+str(maxcut_value))
print("------------------------------------------")
print("Max-Cut Parameters: "+str(maxcut_state))
'''


circuit = cirq.Circuit()
circuit.append(apply_h_gates(n))

for g in range(0, number_of_steps):
    apply_everything(n, maxcut_state[0][g], maxcut_state[1][g])

print(qubits)
circuit.append(cirq.measure(*qubits, key='x'))

simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=5000)
final = result.histogram(key='x')

print(circuit)

print(final)
