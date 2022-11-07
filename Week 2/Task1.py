import numpy as np
from matplotlib import pyplot


def targetify(a): return a ** 3 - a ** 2 + 1


x = np.random.random_sample(100)
t = targetify(x)

print(x)
print(t)

pyplot.scatter(x, t)
pyplot.xlabel("x")
pyplot.ylabel("t")
pyplot.title("t = x^3 - x^2 + 1")
pyplot.show()
