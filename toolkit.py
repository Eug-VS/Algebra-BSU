import numpy as np


class SLAE:
    def __init__(self, matrix, vector):
        """Initialize System of Linear Algebraic Equations"""

        # Store initial values
        self.initial_matrix = np.array(matrix)
        self.initial_vector = vector
        self.dimension = len(vector)

        # Initialize determinant with neutral element with
        # respect to multiplication operation
        self.determinant = 1

        # Setup the general system of equations:
        # Matrix of size (N x N*2+1), which consists of,
        # respecting the order:
        # - Working matrix (N x N)
        # - Vector of solutions (N x 1)
        # - Inverse matrix (N x N) - initialized with unit matrix
        self.system = np.append(matrix, np.array(vector)[np.newaxis].T, axis=1)
        inverse = np.zeros((self.dimension, self.dimension))
        for i in range(self.dimension):
            inverse[i][i] = 1
        self.system = np.append(self.system, inverse, axis=1)

    def __str__(self):
        """Return string representation of the object for debugging"""
        return np.array2string(self.system, max_line_width=np.inf)

    def preprocess(self, index):
        """
        Preprocess the matrix by finding an item with the greatest
        absolute value, and then swapping rows and columns in such
        way that it lays down on the diagonal.
        """

        # Find max element and its' position
        max_value, max_i, max_j = 0, 0, 0
        for i in range(index, self.dimension):
            for j in range(index, self.dimension):
                if abs(self.system[i][j]) > max_value:
                    max_value = abs(self.system[i][j])
                    max_i, max_j = i, j

        # Perform necessary swaps
        self.system[index], self.system[max_i] = self.system[max_i], self.system[index].copy()
        self.system[:, index], self.system[:, max_j] = self.system[:, max_j], self.system[:, index].copy()

        # Change the determinant if needed: use XOR to avoid extra check
        if max_i == index ^ max_j == index:
            self.determinant *= -1

    def forward(self, index):
        """
        Perform forward elimination - first part of the Gauss algorithm.
        Find an element on the diagonal according to the given index and
        use it to terminate all elements below it (set them to zero).
        """

        terminator = self.system[index]
        divider = terminator[index]

        # Terminate all rows below the index
        for i in range(index + 1, self.dimension):
            row = self.system[i]
            target = row[index]
            row = [item - t / divider * target for item, t in zip(row, terminator)]
            self.system[i] = row

    def backwards(self, index):
        """
        Perform back-substitution - second part of the Gauss algorithm.
        Find an element on the diagonal according to the given index and
        use it to terminate all elements above it (set them to zero).
        """

        terminator = self.system[index]
        divider = terminator[index]

        # Process the terminator row
        terminator = [item / divider for item in terminator]
        self.system[index] = terminator

        # Terminate all rows above the index
        for i in range(index):
            row = self.system[i]
            target = row[index]
            row = [item - t * target for item, t in zip(row, terminator)]
            self.system[i] = row

    def extract_matrix(self):
        """Extract the matrix from the general system"""
        return self.system[:, :self.dimension]

    def extract_solutions(self):
        """Extract the solutions vector from the general system"""
        return self.system[:, self.dimension]

    def extract_inverse_matrix(self):
        """Extract the inverse from the general system"""
        return self.system[:, -self.dimension:]

    def gaussian_elimination(self):
        """
        Perform Gaussian elimination algorithm.
        At the same time, use the diagonal elements to compute the
        determinant of the matrix.
        Also, since general system contains unit-matrix on the right
        edge, it happens to form inverse matrix after all operations.
        """

        # Forward elimination
        for i in range(self.dimension):
            self.preprocess(i)
            self.forward(i)
            self.determinant *= self.system[i][i]

        # Back-substitution
        for i in reversed(range(self.dimension)):
            self.backwards(i)

    def get_residual(self):
        """
        Assume that Gaussian elimination is already performed.
        Compute the residual vector for the approximated solution.
        """

        solutions = self.extract_solutions()
        output = np.zeros(self.dimension)

        # Multiply the matrix by an approximated vector
        for i in range(self.dimension):
            row = self.initial_matrix[i]
            for row_item, column_item in zip(row, solutions):
                output[i] += row_item * column_item

        # Compute the residual vector
        residual = [out_item - exp_item for out_item, exp_item in zip(output, self.initial_vector)]
        return residual

    def multiply_back(self):
        """
        Assume that Gaussian elimination is already performed.
        Multiply inverse matrix by the original one.
        The result is expected to be very close to unit-matrix.
        """

        output = np.zeros((self.dimension, self.dimension))
        inverse = self.extract_inverse_matrix()
        initial = self.initial_matrix.T

        # Multiply matrices
        for i in range(self.dimension):
            row = inverse[i]
            for j in range(self.dimension):
                column = initial[j]
                for row_item, column_item in zip(row, column):
                    output[i][j] += row_item * column_item
        return output

