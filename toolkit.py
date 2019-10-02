import numpy as np


class SLAE:
    def __init__(self, matrix, vector):
        self.matrix = np.array(matrix)
        self.vector = np.array(vector).reshape(len(vector), 1)

    def __str__(self):
        nonhomogeneous = np.append(self.matrix, self.vector, axis=1)
        return str(nonhomogeneous)
