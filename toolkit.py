import numpy as np
import math


def sign(value):
    return 1 if value > 0 else -1


class SLAE:
    def __init__(self, matrix, vector):
        """Initialize System of Linear Algebraic Equations"""
        self.matrix = np.array(matrix)
        self.vector = vector
        self.dimension = len(vector)
        self.s = np.zeros((self.dimension, self.dimension))
        self.d = self.s.copy()
        self.y = np.zeros((self.dimension, 1))
        self.x = self.y.copy()
        self.determinant = 1

    def preprocess(self):
        transpose = self.matrix.T
        self.matrix = np.matmul(transpose, self.matrix)
        self.vector = np.matmul(transpose, self.vector)

    def forward(self):
        self.s = np.zeros((self.dimension, self.dimension))
        self.d = np.zeros((self.dimension, self.dimension))
        for i in range(self.dimension):
            value = self.matrix[i][i]
            for k in range(i):
                value -= self.s[k][i] * self.s[k][i] * self.d[k][k]
            self.d[i][i] = sign(value)
            self.s[i][i] = math.sqrt(value * self.d[i][i])
            self.determinant *= self.s[i][i] * self.s[i][i] * self.d[i][i]
            for j in range(i + 1, self.dimension):
                numerator = self.matrix[i][j]
                for k in range(i):
                    numerator -= self.s[k][i] * self.s[k][j] * self.d[k][k]
                self.s[i][j] = numerator / self.s[i][i] * self.d[i][i]
        self.determinant = math.sqrt(self.determinant)

        for i in range(self.dimension):
            numerator = self.vector[i]
            for k in range(i):
                numerator -= self.s[k][i] * self.y[k][0]
            self.y[i][0] = numerator / self.s[i][i]

    def backwards(self):
        for i in range(self.dimension-1, -1, -1):
            numerator = self.y[i][0]
            for k in range(i + 1, self.dimension):
                numerator -= self.s[i][k] * self.x[k][0] * self.d[i][i]
            self.x[i][0] = numerator / (self.s[i][i] * self.d[i][i])
