import numpy as np


class SLAE:
    def __init__(self, matrix, vector):
        self.initial_matrix = np.array(matrix)
        self.initial_vector = vector
        self.system = np.append(matrix, np.array(vector)[np.newaxis].T, axis=1)
        self.dimension = len(vector)
        self.determinant = 1
        inverse = np.zeros((self.dimension, self.dimension))
        for i in range(self.dimension):
            inverse[i][i] = 1
        self.system = np.append(self.system, inverse, axis=1)

    def __str__(self):
        return np.array2string(self.system, max_line_width=np.inf)

    def forward(self, index):
        terminator = self.system[index]
        divider = terminator[index]

        for i in range(index, self.dimension):
            row = self.system[i]
            if i != index:
                target = row[index]
                row = [item - t * target/divider for item, t in zip(row, terminator)]
                self.system[i] = row

    def reverse(self, index):
        terminator = self.system[index]
        leader = terminator[index]
        terminator = [item / leader for item in terminator]
        self.system[index] = terminator

        for i in range(index):
            row = self.system[i]
            target = row[index]
            row = [item - t * target for item, t in zip(row, terminator)]
            self.system[i] = row

    def extract_matrix(self):
        return self.system[:, :self.dimension]

    def extract_solutions(self):
        return self.system[:, self.dimension]

    def extract_inverse_matrix(self):
        return self.system[:, -self.dimension:]

    def gaussian_elimination(self):
        for i in range(self.dimension):
            self.forward(i)
            self.determinant *= self.system[i][i]

        for i in reversed(range(self.dimension)):
            self.reverse(i)

    def get_residual(self):
        solutions = self.extract_solutions()
        output = np.zeros(self.dimension)
        for i in range(self.dimension):
            row = self.initial_matrix[i]
            for row_item, column_item in zip(row, solutions):
                output[i] += row_item * column_item
        residual = [out_item - exp_item for out_item, exp_item in zip(output, self.initial_vector)]
        return residual


