import numpy as np
from math import exp
import matplotlib.pyplot as plt
N = 200000

arr = []
mat = np.empty((N, 20))
for i in range(N):
    for j in range(20):
        mat[i][j] = np.random.binomial(1, 0.5)

x = np.empty(N)
for i in range(N):
    x[i] = np.mean(mat[i])

epsilon = np.linspace(0,1, 50)
empx = np.empty(50)
hoeffding = np.empty(50)

count = 0
for val in epsilon:
    sum_x = 0
    hoeffding[count] = 2 * exp(-1 * 2 * 20 * pow(val, 2))
    for i in x:
        if abs(i - 0.5) > val:
            sum_x += 1
    empx[count] = sum_x / N
    count+=1

# print(empx[0:20])
plt.plot(epsilon, empx, label="Empirical Probability")
plt.plot(epsilon, hoeffding, label="Hoeffding")
plt.legend()
plt.show()
# print(mat[0:10, :])
