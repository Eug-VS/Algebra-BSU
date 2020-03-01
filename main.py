from math import e, pow


def f(x):
    return 3*x - pow(e, x)

def fi(x):
    return pow(e, x)/3


interval = [0.6, 0.7]
x = (interval[0] + interval[1])/2
precision = pow(10, -5)

iteration = 0
previous = None

print(f'e = {precision}')

while (not previous or abs(x - previous) > precision / 2):
    print(f'x{iteration} = {x}')
    previous = x
    x = fi(x)
    iteration += 1

print(f'x{iteration} = {x}')
print(f'f(x{iteration}) = f({x}) = {f(x)} < {precision}')

