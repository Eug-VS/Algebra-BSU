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

    print(f'Initial matrix:')
    Matrix.print(matrix, f)
    t_initial = [[7.6515, 2.2375, 2.262, 0.3675], [1.50, 7.53, -0.30, -1.21],
                 [2.25, 1.32, 6.08, 0.49], [3.81, 0.25, 1.28, 2.25]]
    t_free_terms = [8.4015, -1.83, 7.97, 4.21]

    iterative_methods = SLAE(t_initial, t_free_terms, 10**5)
    print()
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
    Matrix.print(t_initial, t_free_terms)
    print(f'B:')
    Matrix.print(B)
    print(f'b:{b}')
    print(f'||B||: {Matrix.norm(B)}')
    print(f'||b||: {Matrix.norm_for_vector(b)}')
    print(f'x vector = {x2}\niterations number: {n2}')
    print(f'incoherence: {iterative_methods.incoherence(x2)}')
    print(f'Time: { t4 - t3}\n')
    t5 = time.perf_counter()
    x4, n4 = iterative_methods.seidel_gauss_method()
    t6 = time.perf_counter()
    print(f'Seidel - Gauss method:\nx vector = {x4}\niterations number: {n4}')
    print(f'Incoherence: {iterative_methods.incoherence(x4)}')
    print(f'Time: {t6-t5}\n')
