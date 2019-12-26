import time
from toolkit import *


if __name__ == '__main__':
    k = -3
    alpha = 0.5 * k
    matrix = [[3.81, 0.25, 1.28, 0.75 + alpha],
              [2.25, 1.32, 4.58 + alpha, 0.49],
              [5.31, 6.28 + alpha, 0.98, 1.04],
              [9.39 + alpha, 2.45, 3.35, 2.28]]
    f = [4.21, 6.47 + alpha, 2.38, 10.48 + alpha]

    diag_dominant = [[3.81, 0.25, 1.28, -0.75],
                     [-0.33, 3.65, 0.71, -0.75],
                     [0.81, 2.14, -5.18, 0.06],
                     [0.27, 1.95, 0.79, 3.78]]
    new_f = [4.21, -1.63, -7.56, 0.56]
    iterative_methods = SLAE(diag_dominant, new_f, 10 ** 5)

    print(f'Matrix:\n{iterative_methods.initial_matrix}')
    print(f'Vector: {iterative_methods.vector}')
    t1 = time.perf_counter()
    x1, n1, B, b = iterative_methods.simple_iterative_method()
    t2 = time.perf_counter()
    print(f'Simple iterative method:\nx vector = {x1}\niterations number: {n1}')
    print(f'incoherence:\n {iterative_methods.incoherence(x1)} B:\n')
    Matrix.print(B)
    print(f'b:{b}')
    print(f'Time: { t2 - t1}\n')
    t3 = time.perf_counter()
    x2, n2 = iterative_methods.jacobi_method()
    t4 = time.perf_counter()
    print(f'Jacobi method:\nTransformed matrix: ')
    Matrix.print(diag_dominant, new_f)
    print(f'B:')
    Matrix.print(B)
    print(f'b:{b}')
    print(f'||B||: {Matrix.norm(B)}')
    print(f'x vector = {x2}\niterations number: {n2}')
    print(f'incoherence: {iterative_methods.incoherence(x2)}')
    print(f'Time: { t4 - t3}\n')
    t5 = time.perf_counter()
    x4, n4 = iterative_methods.seidel_gauss_method()
    t6 = time.perf_counter()
    print(f'Seidel - Gauss method:\nx vector = {x4}\niterations number: {n4}')
    print(f'Incoherence: {iterative_methods.incoherence(x4)}')
    print(f'Time: {t6-t5}\n')
