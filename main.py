from toolkit import *


k = -3
alpha = k / 2

A = [[3.81, 0.25, 1.28, 0.75 + alpha],
     [2.25, 1.32, 4.58 + alpha, 0.49],
     [5.31, 6.28 + alpha, 0.98, 1.04],
     [9.39 + alpha, 2.45, 3.35, 2.28]]
f = [4.21, 6.47 + alpha, 2.38, 10.48 + alpha]

S = SLAE(A, f)
print('\nInitial matrix and vector: ')
print(S.matrix, '\n\n', S.vector)

S.preprocess()
print('\nSymmetrical matrix and corresponding vector: ')
print(S.matrix, '\n\n', S.vector)

S.forward()
print(f'\nDeterminant: {S.determinant}')
print('\nB = S * D')
print(np.matmul(S.s, S.d))
print('\nVector y: ')
print(S.y)

S.backwards()
print('\nVector x: ')
print(S.x)

