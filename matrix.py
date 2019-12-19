import sys


class Matrix:
    @staticmethod
    def norm(matrix):
        if matrix:
            norm = sys.float_info.min
            for i in range(len(matrix)):
                somme = 0
                for j in range(len(matrix[0])):
                    somme += abs(matrix[i][j])
                if somme > norm:
                    norm = somme
            return norm
        else:
            return None

    @staticmethod
    def norm_for_vector(vector):
        norm = 0
        for el in vector:
            norm += el
        return norm

    @staticmethod
    def multiply(matrix1, matrix2):
        result = [[0 for _ in range(len(matrix2[0]))] for _ in range(len(matrix1))]
        for i in range(len(matrix1)):
            for j in range(len(matrix2[0])):
                result[i][j] = 0
                for k in range(len(matrix2)):
                    result[i][j] += matrix1[i][k] * matrix2[k][j]
        return result

    @staticmethod
    def multiply_matrix_column(matrix, column):
        result = [0] * len(column)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                result[i] += matrix[i][j] * column[j]
        return result

    @staticmethod
    def sub(m1, m2):
        return [[m1[i][j] - m2[i][j]
                 for j in range(len(m1[0]))]
                for i in range(len(m1))]

    @staticmethod
    def add(m1, m2):
        return [[m1[i][j] + m2[i][j]
                 for j in range(len(m1[0]))]
                for i in range(len(m1))]

    @staticmethod
    def add_v(v1, v2):
        return [v1[i] + v2[i] for i in range(len(v1))]

    @staticmethod
    def div(m, a):
        matr = [[m[i][j] / a
                 for j in range(len(m[0]))]
                for i in range(len(m))]
        return matr

    @staticmethod
    def transposed(matrix):
        return [[matrix[j][i] for j in range(len(matrix[0]))]
                for i in range(len(matrix))]

    @staticmethod
    def print(matrix, free_terms=None):
        free_terms_str = ["" for _ in range(len(matrix))] \
            if not free_terms else ["| " + str(f_t) + " " for f_t in free_terms]
        for i in range(len(matrix)):
            print("[ " + " ".join(list(map(str, matrix[i]))) + free_terms_str[i] + "]")


def max_difference(x_k1, x_k):
    diff_list = [abs(x_k1[i] - x_k[i]) for i in range(len(x_k1))]
    return max(diff_list)

