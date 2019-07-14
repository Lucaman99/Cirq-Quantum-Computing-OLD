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
#Figure out how to penalize multiple qubits set to 1 within the circuit
'''

import cirq
import itertools
import networkx as nx
from matplotlib import pyplot as plt

#You must change the weights based on the number of steps that the simulator takes

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

set_edges = [Edge(0, 1), Edge(1, 2), Edge(0, 2), Edge(2, 3), Edge(2, 4), Edge(3, 4)]
#set_edges = [Edge(0, 1), Edge(1, 2), Edge(0, 2), Edge(2, 3), Edge(2, 4), Edge(3, 4)]

#set_edges = [Edge(0, 1), Edge(1, 2)]

G = nx.Graph()

for z in set_edges:
    G.add_edge(str(z.start_node), str(z.end_node))

nx.draw(G)
plt.savefig('graph.png')

graph = Graph(set_edges)

n = len(graph.node_set)


connection = [0, 1]

matrix_horiz = []
for i in connection:
    holder = []
    for g in range (0, n):
        if (g == i):
            holder.append(1)
        else:
            holder.append(-1)
    matrix_horiz.append(holder)


#Generate and Store the qubits in an array

qubits = []
for i in range(0, n):
    qubits.append(cirq.GridQubit(0, i))

qubit_store = []
for i in range(0, n):
    qubits.append(cirq.GridQubit(1, i))

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

def apply_other_c(length, gamma):
    for j in graph.node_set:
        yield cirq.CZPowGate(exponent=(-1*gamma)).on(cirq.GridQubit(0, j), work_qubit)

def apply_C_unitary(length, gamma):
    # Apply the compute --> rotate --> uncompute  method to change the phase of a specific computational basis state
    for j in set_edges:
        yield cirq.CNOT.on(cirq.GridQubit(0, j.start_node), cirq.GridQubit(0, j.end_node))
        yield cirq.Rz((-1*gamma)).on(cirq.GridQubit(0, j.end_node))
        yield cirq.CNOT.on(cirq.GridQubit(0, j.start_node), cirq.GridQubit(0, j.end_node))
        yield cirq.Rz((-1*gamma)).on(work_qubit)

def apply_B_unitary(length, beta):
    for i in range(0, n):
        yield cirq.Rx((2*beta)).on(cirq.GridQubit(0, i))

def apply_everything(length, gamma, beta):
    circuit.append(apply_C_unitary(length, gamma))
    circuit.append(apply_B_unitary(length, beta))

def other_apply(length, gamma, beta):
    circuit2.append(apply_other_c(length, gamma))
    circuit2.append(apply_B_unitary(length, beta))



def objective_calc(values, extra):

    coefficient = 1
    total = 0

    for i in set_edges:
        multi = -1*values[i.start_node]*values[i.end_node]
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

new_rotation = list(itertools.permutations(values_for_rotation, number_of_steps))

maxcut_value = -10000
maxcut_state = 0

#Pick 2 to be used in the set

for g in range (0, len(new_rotation)):
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

        processed_results = str(result)[2:].split(", ")

        sum_total = 0

        for j in range(0, len(processed_results[0])):
            trial_holder = []
            for k in range(0, len(processed_results)-3):
                if (int(processed_results[k][j]) == 0):
                    trial_holder.append(-1)
                else:
                    trial_holder.append(int(processed_results[k][j]))
            extra = int(processed_results[len(processed_results)-1][j])


            sum_total = sum_total + objective_calc(trial_holder, extra)

        sum_total = sum_total/100

        if (sum_total > maxcut_value):
            maxcut_value = sum_total
            maxcut_state = [gamma_matrix, beta_matrix]


circuit = cirq.Circuit()
circuit.append(apply_h_gates(n))

for g in range(0, number_of_steps):
    apply_everything(n, maxcut_state[0][g], maxcut_state[1][g])

circuit.append(cirq.measure(*qubits, key='x'))

values_for_rotation = [0.1, 0.4, 0.7]
number_of_steps = 3

new_rotation = list(itertools.permutations(values_for_rotation, number_of_steps))

maxcut_value = -10000
maxcut_state = 0

for g in range (0, len(new_rotation)):
    for h in range(0, len(new_rotation)):

        gamma_matrix = new_rotation[g]
        beta_matrix = new_rotation[h]

        circuit2 = cirq.Circuit()

        circuit2.append(apply_h_gates(n))
        circuit2.append(cirq.X.on(work_qubit))

        for g in range(0, number_of_steps):
            other_apply(n, float(gamma_matrix[g]), float(beta_matrix[g]))

        circuit2.append(cirq.measure(*qubits, key='x'))

        simulator = cirq.Simulator()
        result = simulator.run(circuit2, repetitions=100)

        processed_results = str(result)[2:].split(", ")

        sum_total = 0

        for j in range(0, len(processed_results[0])):
            trial_holder = []
            for k in range(0, len(processed_results)-3):
                if (int(processed_results[k][j]) == 0):
                    trial_holder.append(-1)
                else:
                    trial_holder.append(int(processed_results[k][j]))
            extra = int(processed_results[len(processed_results)-1][j])


            sum_total = sum_total + new_calc(trial_holder, extra)

        sum_total = sum_total/100

        if (sum_total > maxcut_value):
            maxcut_value = sum_total
            maxcut_state = [gamma_matrix, beta_matrix]

circuit2 = cirq.Circuit()
circuit2.append(apply_h_gates(n))
circuit2.append(cirq.X.on(work_qubit))

for g in range(0, number_of_steps):
    other_apply(n, maxcut_state[0][g], maxcut_state[1][g])

circuit2.append(cirq.measure(*qubits, key='x'))

#NUMBER OF SHOTS
shots = 20

def test_circuit():

    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=5000)
    final = result.histogram(key='x')
    result2 = simulator.run(circuit2, repetitions=500)
    final2 = result2.histogram(key='x')

    print(final)
    print(final2)
    print(circuit)


    #Assigns a score to each of the outputted configurations

    entry_number = []
    entry_number2 = []
    score = []
    entry_score = []

    indexing = str(final).split(", ")
    indexing2 = str(final2).split(", ")

    entry_number.append(int(indexing[0][9:indexing[0].index(":")]))
    entry_number2.append(int(indexing2[0][9:indexing2[0].index(":")]))

    for i in range (1, len(indexing)):
        entry_number.append(int(indexing[i][0:indexing[i].index(":")]))
    for i in range (1, len(indexing2)):
        entry_number2.append(int(indexing2[i][0:indexing2[i].index(":")]))

    for i in entry_number:
        score.append(i)
        entry_score.append(0)
    for i in entry_number2:
        if (i not in score):
            score.append(i)
            entry_score.append(0)

    for i in range(0, len(entry_number)):
        entry_score[score.index(entry_number[i])] += weight_on_first_ham*(i+1)

    for i in range(0, len(entry_number2)):
        entry_score[score.index(entry_number2[i])] += weight_on_second_ham*(i+1)

    return [score, entry_score]

last_score = []
last_entry = []



for i in range(0, shots):
    v = test_circuit()
    for j in range (0, len(v[0])):
        if (v[0][j] not in last_score):
            last_score.append(v[0][j])
            last_entry.append(v[1][j])
        else:
            last_entry[last_score.index(v[0][j])] += v[1][j]

for b in range (0, len(last_entry)):
    last_entry[b] = last_entry[b]/shots

min_val = 10000
optimal_choice = 0
for i in range(0, len(last_entry)):
    if (last_entry[i] < min_val):
        min_val = last_entry[i]
        optimal_choice = last_score[i]
optimal_choice = float(optimal_choice)

print(optimal_choice)

plt.show()
