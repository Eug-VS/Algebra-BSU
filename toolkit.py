import numpy as np


class SLAE:
    def __init__(self, matrix, vector):
        """Initialize System of Linear Algebraic Equations"""
        self.initial_matrix = np.array(matrix)
        self.initial_vector = vector
        self.dimension = len(vector)
        self.determinant = 1

        # General system - Matrix(N x N*2+1), includes:
        # - Working matrix (N x N)
        # - Vector of solutions (N x 1)
        # - Inverse matrix (N x N) - initialized with unit matrix
        self.system = np.append(matrix, np.array(vector)[np.newaxis].T, axis=1)
        self.system = np.append(self.system, np.identity(self.dimension), axis=1)

    def __str__(self):
        return np.array2string(self.system, max_line_width=np.inf)

    def extract_matrix(self):
        return self.system[:, :self.dimension]

    def extract_solutions(self):
        return self.system[:, self.dimension]

    def extract_inverse_matrix(self):
        return self.system[:, -self.dimension:]

    def preprocess(self, index):
        """Preprocess the matrix:
        -find an item with the greatest absolute value
        -swap rows & columns so that it lays onto index position of the diagonal
        """
        max_value, max_i, max_j = 0, 0, 0
        for i in range(index, self.dimension):
            for j in range(index, self.dimension):
                if abs(self.system[i][j]) > max_value:
                    max_value, max_i, max_j = abs(self.system[i][j]), i, j

        # Perform necessary swaps
        self.system[index], self.system[max_i] = self.system[max_i], self.system[index].copy()
        self.system[:, index], self.system[:, max_j] = self.system[:, max_j], self.system[:, index].copy()

        # Change the determinant if needed: use XOR to avoid extra check
        if max_i == index ^ max_j == index:
            self.determinant *= -1

    def forward(self, index):
        """
        Terminate all elements below index position on the diagonal
        by executing elementary row operations.
        """
        for i in range(index + 1, self.dimension):
            self.system[i] = self.system[i] - self.system[index] / self.system[index][index] * self.system[i][index]

    def backwards(self, index):
        """
        Process index row so its' only non-zero element is 1.0, and
        terminate all elements above it by executing elementary row operations.
        """
        self.system[index] /= self.system[index][index]
        for i in range(index):
            self.system[i] = self.system[i] - self.system[index] * self.system[i][index]

    def gaussian_elimination(self):
        """
        Perform Gaussian elimination algorithm.
        Use the diagonal elements to compute the determinant.
        At the same time, form inverse matrix from unit-matrix.
        """
        # Forward elimination
        for i in range(self.dimension):
            self.preprocess(i)
            self.forward(i)
            self.determinant *= self.system[i][i]
        # Back-substitution
        for i in reversed(range(self.dimension)):
            self.backwards(i)

    def residual(self):
        return np.dot(self.initial_matrix, self.extract_solutions()) - self.initial_vector

    def verification_matrix(self):
        return np.dot(self.extract_inverse_matrix(), self.initial_matrix)
