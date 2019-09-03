import cirq
import random
import numpy as np
import copy
import sympy
import itertools
import time
start_time = time.time()
import math

optimal_energy = math.inf
optimal_params = []


def qaoa_main(gamma_parameter, beta_parameter, testing_trials):

    global optimal_energy
    global optimal_params

    #Attempting to find eigenstate corresponding to the minimal eigenvalue of the objective Hamiltonian

    #Amino acid qubit preparation

    #One-Hot Encoding Key
    #0 qubit index --> Up
    #1 qubit index --> Down
    #2 qubit index --> Left
    #3 qubit index --> Right

    ############################### PARAMETERS ###############################

    turn_directions = 4
    dimension = turn_directions/2
    polypeptide = 6
    polypeptide = polypeptide-1

    overlap_penalty = 5

    circuit_depth = 1

    ##########################################################################

    amino_acid_qubits = []
    qubits = []
    for i in range(0, polypeptide):
        store = []
        for j in range (0, turn_directions):
            store.append(cirq.GridQubit(i, j))
            qubits.append(cirq.GridQubit(i, j))
        amino_acid_qubits.append(store)


    ancilla = []
    for i in range(0, polypeptide):
        ancilla.append(cirq.GridQubit(polypeptide, i))

    phase_kickback = cirq.GridQubit(polypeptide+1, 1)

    #Store information about energies based on the Miyazawa-Jirzan interaction energies
    #The specific qubit that we are interested in simulating is the PSVKMA polypeptide chain,
    #therefore we only need to consider energy stored in the following bonds: P - K, P - A, S - M, and V - A

    energy_matrix = [-0.97, -3.03, -4.04, -2.03]

    ############################### TESTING ###############################

    def apply_initial_test():
        yield cirq.X.on(cirq.GridQubit(0, 1))
        yield cirq.X.on(cirq.GridQubit(1, 2))
        yield cirq.X.on(cirq.GridQubit(2, 3))
        yield cirq.X.on(cirq.GridQubit(3, 0))

    #######################################################################

    def initial_conditions(amino_acid_qubits):

        for i in amino_acid_qubits:
            yield cirq.X.on(i[0])

    #Fix the first turn's position

    def fix_first_turn(amino_acid_qubits):
        yield cirq.X.on(amino_acid_qubits[0][0])

    #Apply initial superposition

    def apply_even_superposition(qubits):
        for i in qubits[4:]:
            yield cirq.H.on(i)

    #Creates the n-qubit Toffoli gate (last qubit in args list is the target, all others are controls)

    def apply_n_qubit_tof(ancilla, args, ham_modifier):

        if (len(args) == 2):
            yield cirq.CNOT.on(args[0], args[1])

        elif (len(args) == 3):
            yield cirq.CCX.on(args[0], args[1], args[2])

        else:

            yield cirq.CCX.on(args[0], args[1], ancilla[0])
            for k in range(2, len(args)-1):
                yield cirq.CCX(args[k], ancilla[k-2], ancilla[k-1])

            yield cirq.CNOT.on(ancilla[len(args)-3], args[len(args)-1])
            if (ham_modifier != False):
                yield cirq.ZPowGate(exponent=-1*ham_modifier).on(args[len(args)-1])
                yield cirq.CNOT.on(ancilla[len(args)-3], args[len(args)-1])

            for k in range(len(args)-2, 1, -1):
                yield cirq.CCX(args[k], ancilla[k-2], ancilla[k-1])
            yield cirq.CCX.on(args[0], args[1], ancilla[0])

    #Checks to see if an amino acid sub-chain overalps itself

    '''
    def check_sum_string_k(amino_acid_1_index, amino_acid_2_index, k_dir):

        for i in range(amino_acid_1_index, amino_acid_2_index+1):
            for j in range(amino_acid_1_index, amino_acid_2_index+1):
                if (i != j):
                    yield cirq.CNOT.on(cirq.GridQubit(i, k_dir), cirq.GridQubit(j, k_dir+1))
                    yield cirq.X.on(cirq.GridQubit(j, k_dir+1))
                    yield cirq.CNOT.on(cirq.GridQubit(j, k_dir+1), cirq.GridQubit(i, k_dir))
    '''

    #Future: Use multi-CZ gates for the loop overlap cost?????

    def check_4_overlap_cost(aa_1_index, aa_2_index, amino_acid_qubits, ancilla, phase_kickback, overlap_penalty):

        for i in range(aa_1_index, aa_2_index):
            for j in range(0, 4):
                holder = [amino_acid_qubits[i%4][j%4], amino_acid_qubits[(i+1)%4][(j+1)%4], amino_acid_qubits[(i+2)%4][(j+2)%4], amino_acid_qubits[(i+3)%4][(j+3)%4], phase_kickback]
                yield apply_n_qubit_tof(ancilla, holder, overlap_penalty)

    def check_2_overlap_cost(aa_1_index, aa_2_index, amino_acid_qubits, phase_kickback, overlap_penalty):

        yield cirq.CZPowGate(exponent=-1*overlap_penalty).on(amino_acid_qubits[aa_1_index][0], amino_acid_qubits[aa_2_index][1])
        yield cirq.CZPowGate(exponent=-1*overlap_penalty).on(amino_acid_qubits[aa_1_index][2], amino_acid_qubits[aa_2_index][3])

    def check_nearest_neighbours_3(amino_acid_qubits, phase_kickback, ancilla):

        for i in range(1, len(amino_acid_qubits)-2):

            yield apply_n_qubit_tof(ancilla, [amino_acid_qubits[i][0], amino_acid_qubits[i+1][2], amino_acid_qubits[i+1][3], amino_acid_qubits[i+2][1], phase_kickback], gamma_parameter*energy_matrix[i])
            yield apply_n_qubit_tof(ancilla, [amino_acid_qubits[i][1], amino_acid_qubits[i+1][2], amino_acid_qubits[i+1][3], amino_acid_qubits[i+2][0], phase_kickback], gamma_parameter*energy_matrix[i])
            yield apply_n_qubit_tof(ancilla, [amino_acid_qubits[i][2], amino_acid_qubits[i+1][1], amino_acid_qubits[i+1][0], amino_acid_qubits[i+2][3], phase_kickback], gamma_parameter*energy_matrix[i])
            yield apply_n_qubit_tof(ancilla, [amino_acid_qubits[i][3], amino_acid_qubits[i+1][1], amino_acid_qubits[i+1][0], amino_acid_qubits[i+2][2], phase_kickback], gamma_parameter*energy_matrix[i])

        yield apply_n_qubit_tof(ancilla, [amino_acid_qubits[0][0], amino_acid_qubits[i+1][2], amino_acid_qubits[i+1][3], amino_acid_qubits[i+2][1], phase_kickback], gamma_parameter*energy_matrix[0])

    def check_nearest_neighbours_all(amino_acid_qubits, phase_kickback, ancilla):

        #"Hairpin" configuration
        yield apply_n_qubit_tof(ancilla, [amino_acid_qubits[1][0], amino_acid_qubits[2][2], amino_acid_qubits[3][1], amino_acid_qubits[4][1], phase_kickback], gamma_parameter*(energy_matrix[1]+energy_matrix[3]))
        yield apply_n_qubit_tof(ancilla, [amino_acid_qubits[1][0], amino_acid_qubits[2][3], amino_acid_qubits[3][1], amino_acid_qubits[4][1], phase_kickback], gamma_parameter*(energy_matrix[1]+energy_matrix[3]))

        #Horizontal "staple" configuration
        yield apply_n_qubit_tof(ancilla, [amino_acid_qubits[1][2], amino_acid_qubits[2][2], amino_acid_qubits[3][1], amino_acid_qubits[4][3], phase_kickback], gamma_parameter*(energy_matrix[2]+energy_matrix[3]))
        yield apply_n_qubit_tof(ancilla, [amino_acid_qubits[1][3], amino_acid_qubits[2][3], amino_acid_qubits[3][1], amino_acid_qubits[4][2], phase_kickback], gamma_parameter*(energy_matrix[2]+energy_matrix[3]))

        #Vertical "staple" configuration
        yield apply_n_qubit_tof(ancilla, [amino_acid_qubits[1][3], amino_acid_qubits[2][1], amino_acid_qubits[3][1], amino_acid_qubits[4][2], phase_kickback], gamma_parameter*(energy_matrix[0]+energy_matrix[3]))
        yield apply_n_qubit_tof(ancilla, [amino_acid_qubits[1][2], amino_acid_qubits[2][1], amino_acid_qubits[3][1], amino_acid_qubits[4][3], phase_kickback], gamma_parameter*(energy_matrix[0]+energy_matrix[3]))


    def apply_total_cost_hamiltonian(amino_acid_qubits, ancilla, phase_kickback):

        for i in range(0, len(amino_acid_qubits)-1):
            circuit.append(check_2_overlap_cost(i, i+1, amino_acid_qubits, phase_kickback, gamma_parameter*overlap_penalty))

        for i in range(0, len(amino_acid_qubits)-3):
            circuit.append(check_4_overlap_cost(i, i+3, amino_acid_qubits, ancilla, phase_kickback, gamma_parameter*overlap_penalty))

        circuit.append(check_nearest_neighbours_3(amino_acid_qubits, phase_kickback, ancilla))
        circuit.append(check_nearest_neighbours_all(amino_acid_qubits, phase_kickback, ancilla))

    def apply_mixer_hamiltonian(amino_acid_qubits):

        '''
        for i in range(1, len(amino_acid_qubits)):
            for j in range(0, len(amino_acid_qubits[i])):
                for k in range(0, len(amino_acid_qubits[i])):
                    if (j != k):
                        #Halved again to account for repetitions in the looping procedure
                        yield cirq.Rx(-0.5*beta_parameter).on(amino_acid_qubits[i][j])
                        yield cirq.Rx(-0.5*beta_parameter).on(amino_acid_qubits[i][k])
                        yield cirq.Ry(-0.5*beta_parameter).on(amino_acid_qubits[i][j])
                        yield cirq.Ry(-0.5*beta_parameter).on(amino_acid_qubits[i][k])
        '''

        one = [np.complex(math.cos(-0.5*beta_parameter),0), np.complex(0,0), np.complex(0,0), np.complex(0,math.sin(-0.5*beta_parameter))]
        two = [np.complex(0,0), np.complex(math.cos(-0.5*beta_parameter),0), np.complex(0,math.sin(-0.5*beta_parameter)), np.complex(0,0)]
        three = [np.complex(0,0), np.complex(0,math.sin(-0.5*beta_parameter)), np.complex(math.cos(-0.5*beta_parameter),0), np.complex(0,0)]
        four = [np.complex(0,math.sin(-0.5*beta_parameter)), np.complex(0,0), np.complex(0,0), np.complex(math.cos(-0.5*beta_parameter),0)]

        double_x_swap = np.array([one, two, three, four])

        one2 = [np.complex(math.cos(-0.5*beta_parameter),0), np.complex(0,0), np.complex(0,0), np.complex(0,-1*math.sin(-0.5*beta_parameter))]
        two2 = [np.complex(0,0), np.complex(math.cos(-0.5*beta_parameter),0), np.complex(0,1*math.sin(-0.5*beta_parameter)), np.complex(0,0)]
        three2 = [np.complex(0,0), np.complex(0,1*math.sin(-0.5*beta_parameter)), np.complex(math.cos(-0.5*beta_parameter),0), np.complex(0,0)]
        four2 = [np.complex(0,-1*math.sin(-0.5*beta_parameter)), np.complex(0,0), np.complex(0,0), np.complex(math.cos(-0.5*beta_parameter),0)]

        double_y_swap = np.array([one2, two2, three2, four2])

        for i in range(1, len(amino_acid_qubits)):
            for j in range(0, len(amino_acid_qubits[i])):
                for k in range(0, len(amino_acid_qubits[i])):
                    if (j < k):
                        yield cirq.TwoQubitMatrixGate(double_y_swap).on(amino_acid_qubits[i][j], amino_acid_qubits[i][k])
                        yield cirq.TwoQubitMatrixGate(double_x_swap).on(amino_acid_qubits[i][j], amino_acid_qubits[i][k])

    circuit = cirq.Circuit()

    #circuit.append(apply_initial_test())
    circuit.append(initial_conditions(amino_acid_qubits))
    #circuit.append(fix_first_turn(amino_acid_qubits))
    #circuit.append(apply_even_superposition(qubits))
    apply_total_cost_hamiltonian(amino_acid_qubits, ancilla, phase_kickback)
    circuit.append(apply_mixer_hamiltonian(amino_acid_qubits))

    #circuit.append(check_sum_string_k(0, 3, 2), strategy=cirq.InsertStrategy.NEW)
    #circuit.append(check_sum_string_k(0, 3, 2), strategy=cirq.InsertStrategy.NEW)

    circuit.append(cirq.measure(*qubits[4:], key='y'))

    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=testing_trials)
    #final = result.histogram(key='x')

    #Creates the measurement and sampling function --> One pass of the algorithm involves p applications of H_M H_C, each parametrized by different values

    final = str(result)[str(result).index("y")+2:len(str(result))].split(", ")
    last = []
    for i in range(0, testing_trials-1):
        holder = []
        for j in final:
            holder.append(int(j[i]))
        last.append(holder)

    last = [[1, 0, 0, 0]+i for i in last]

    #Find the most commonly occuring bitstring

    numbers = []
    bitstring = []

    for y in last:
        if (y in bitstring):
            numbers[bitstring.index(y)]+=1
        else:
            numbers.append(1)
            bitstring.append(y)

    common = bitstring[numbers.index(max(numbers))]

    #Calculates the energy of the protein structure for one pass of the algorithm --> Calculates new parameters for the QAOA algorithm

    def calculate_energy(state):

        total_energy = 0

        #Calculate the "penalty energy based on the protein overlaps"

        for r in range(0, polypeptide-3):
            if (((state[4*r]+state[4*(r+2)+1] == 2) or (state[(4*r)+1]+state[4*(r+2)] == 2)) and ((state[(4*(r+1))+2]+state[4*(r+3)+3] == 2) or (state[(4*(r+1))+3]+state[4*(r+3)+2] == 2))):
                total_energy = total_energy+overlap_penalty
            if (((state[4*(r+1)]+state[4*(r+3)+1] == 2) or (state[(4*(r+1))+1]+state[4*(r+3)] == 2)) and ((state[(4*r)+2]+state[4*(r+2)+3] == 2) or (state[(4*r)+3]+state[4*(r+2)+2] == 2))):
                total_energy = total_energy+overlap_penalty

        for r in range(0, polypeptide-1):
            if ((state[4*r]+state[4*(r+1)+1] == 2) or (state[(4*r)+2]+state[4*(r+1)+3] == 2)):
                total_energy = total_energy+overlap_penalty
            if ((state[(4*r)+1]+state[4*(r+1)] == 2) or (state[(4*r)+3]+state[4*(r+1)+2] == 2)):
                total_energy = total_energy+overlap_penalty

        travelled_list = [[0, 0]]
        holder = [0, 0]
        for r in range(0, int(len(state)/turn_directions)):
            if (state[4*r:4*(r+1)] == [1, 0, 0, 0]):
                holder[0] = holder[0]+1
            if (state[4*r:4*(r+1)] == [0, 1, 0, 0]):
                holder[0] = holder[0]-1
            if (state[4*r:4*(r+1)] == [0, 0, 1, 0]):
                holder[1] = holder[1]+1
            if (state[4*r:4*(r+1)] == [0, 0, 0, 1]):
                holder[1] = holder[1]-1
            travelled_list.append(copy.copy(holder))
        for counter1, j in enumerate(travelled_list):
            for counter2, k in enumerate(travelled_list):
                if ((abs(j[0]-k[0])+abs(j[1]-k[1]) == 1) and abs(counter1-counter2) > 1):
                    if (((counter1 == 0) and (counter2 == 3)) or ((counter1 == 3) and (counter2 == 0))):
                        total_energy+=energy_matrix[0]/2
                    if (((counter1 == 1) and (counter2 == 4)) or ((counter1 == 4) and (counter2 == 1))):
                        total_energy+=energy_matrix[1]/2
                    if (((counter1 == 2) and (counter2 == 5)) or ((counter1 == 5) and (counter2 == 2))):
                        total_energy+=energy_matrix[2]/2
                    if (((counter1 == 0) and (counter2 == 5)) or ((counter1 == 5) and (counter2 == 0))):
                        total_energy+=energy_matrix[3]/2

        return total_energy

    sum = 0
    e_hold = []
    for i in last:
        a = calculate_energy(i)
        sum = sum+a
        e_hold.append(a)
    sum = sum/len(last)

    if (sum < optimal_energy):
        optimal_energy = sum
        optimal_params = [gamma_parameter, beta_parameter]

    print("--- %s seconds ---" % (time.time() - start_time))

    return [last[e_hold.index(min(e_hold))], min(e_hold)]
'''
for i in range(1, 5):
    for j in range(1, 5):
        qaoa_main(0.2*i, 0.2*j, 30)

'''
print(optimal_params)
print(optimal_energy)

print(qaoa_main(optimal_params[0], optimal_params[1], 30))
#print(qaoa_main(0.4, 0.8, 30))


#Updates the parameters of the QAOA

#Output and visualization
