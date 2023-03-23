import math
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import numpy.random
import numpy as np

mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']

idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:1000], :].astype(int)
train_labels = labels[idx[:1000]]

test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]


def getLabel(images, labels_vec, query, k):
    distances = np.empty(labels_vec.size)
    labels = np.empty(k)
    imgindex = 0

    for img in images:
        distance = np.linalg.norm(query - img)
        distances[imgindex] = distance
        imgindex += 1
    for i in range(k):
        min = np.argmin(distances)
        labels[i] = labels_vec[min]
        distances[min] = math.inf
    labels = labels.astype(int)
    return np.bincount(labels).argmax()


def getPredAccForK(k):
    correct = 0
    total = 0
    i = 0
    for t in test:
        total += 1
        if getLabel(train, train_labels, t, k) == int(test_labels[i]):
            correct += 1
        i += 1

    return correct / total


def getPredAccForN(n):
    train_n = data[idx[:n], :].astype(int)
    train_labels_n = labels[idx[:n]]
    correct = 0
    total = 0
    i = 0
    for t in test:
        total += 1
        if getLabel(train_n, train_labels_n, t, 1) == int(test_labels[i]):
            correct += 1
        i += 1

    return correct / total


# ---- code for 2.b -----
acc = getPredAccForK(10)
print(acc)

# ---- code for 2.c ----
k_values = np.arange(1, 101, 1)
predAcc = np.zeros(100)

for k in k_values:
    predAcc[k - 1] = getPredAccForK(k)
    plt.scatter(k, predAcc[k - 1])

plt.show()

# ---- code for 2.d ----
n_values = np.arange(100, 5100, 100)
for n in n_values:
    plt.scatter(n, getPredAccForN(n))

plt.show()
