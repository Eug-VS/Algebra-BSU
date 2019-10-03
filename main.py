from toolkit import *

k = -3
k /= 2

A = [[3.81, 0.25, 1.28, 0.75 + k],
     [2.25, 1.32, 4.58 + k, 0.49],
     [5.31, 6.28 + k, 0.98, 1.04],
     [9.39 + k, 2.45, 3.35, 2.28]]

f = [4.21, 6.47 + k, 2.38, 10.48 + k]

S = SLAE(A, f)
print('Initial matrix: ')
print(S.extract_matrix())
S.gaussian_elimination()
print('\nVector of solutions: ')
print(S.extract_solutions())
print('\nDeterminant:', S.determinant)
print('\nInverse matrix: ')
print(S.extract_inverse_matrix())
print('\nResidual vector: ')
print(S.get_residual())
print('\nInverse matrix * original matrix: ')
print(S.multiply_back())
