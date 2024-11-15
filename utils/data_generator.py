import numpy as np
import random

# n by m input and binary output

def generator(n, m, min_val, max_val, func, error=0):
    X = []
    y = []
    diff = max_val - min_val

    for i in range(n):
        x0 = [round(random.random() * diff + min_val, 2) for j in range(m)]
        X.append(x0)
        b = int(func(*x0))

        y.append(b if random.random() < error else 1-b)

    return np.array(X), np.array(y).reshape((-1,1))


if __name__ == '__main__':
    X, y = generator(10, 2, 0, 5, lambda x, y: x*x+y*y<16, 1)
    print(X)
    print(y)