#Exhaustive search for TQC braidword translation to unitary matrices for qubit quantum computation

import math
import itertools
import numpy as np
import mpmath

#Selection options for braidword sequences

overall_loss = 10000000
winning_matrix = 0
seq = 0
other_seq = 0

iterations = 50000

approx_matrix = np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)]])

braidword_length = 34

golden = (1 + 5**0.5)/2

row1 = [golden**(-1)*complex(mpmath.expj((4/5)*math.pi)), golden**(-0.5)*complex(mpmath.expj(-1*(3/5)*math.pi))]

row2 = [golden**(-0.5)*complex(mpmath.expj(-1*(3/5)*math.pi)), -1*golden**(-1)]
complex_boi = np.array([row1, [1, 0]])

braiding_matrices = [np.array([[complex(mpmath.expj(-1*(4/5)*np.pi)), 0], [0, complex(mpmath.expj((3/5)*np.pi))]]), complex_boi]
exp_selector = [-1, 1]

for i in range(0, iterations):

    total = np.random.choice(len(braiding_matrices), braidword_length)
    exponents = np.random.choice(exp_selector, braidword_length)

    #Loss function for computing the 'best matching' braidword
    def distance(matrix1, matrix2):
        matrix = np.conjugate(np.transpose(matrix2))
        multi = np.matmul(matrix1, matrix)
        final = np.sqrt(1-(np.absolute(np.trace(multi))/2))

        return final

    product = np.eye(2)
    for i in range(0, braidword_length):
        if (exponents[i] == -1):
            braiding_matrices[total[i]] = np.linalg.inv(braiding_matrices[total[i]])
        product = np.matmul(product, braiding_matrices[total[i]])

    x = distance(approx_matrix, product)
    if (x < overall_loss):
        winning_matrix = product
        overall_loss = x
        seq = total
        other_seq = exponents

print(overall_loss)
print(winning_matrix)
print(seq)
print(other_seq)
