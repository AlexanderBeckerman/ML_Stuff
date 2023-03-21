import math

from sklearn.datasets import fetch_openml
import numpy.random
import numpy as np

mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']

idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:100], :].astype(int)
train_labels = labels[idx[:100]]

test = data[idx[10000:10050], :].astype(int)
test_labels = labels[idx[10000:10050]]


# print(train_labels[0])

def getLabel(images, labels_vec, query, k):
    nearest = np.empty(k)
    labels = np.empty(k)
    nearest.fill(math.inf)
    imgindex = 0
    for img in images:
        distance = np.linalg.norm(query - img)
        if (imgindex < k):
            nearest[imgindex] = distance
            labels[imgindex] = labels_vec[imgindex]
            imgindex += 1
            continue
        for index, d in np.ndenumerate(nearest):
            if (distance < d):
                nearest[index] = distance
                labels[index] = labels_vec[imgindex]
                break
        imgindex += 1

    return np.bincount(labels.astype(int)).argmax()


correct = 0
total = 0
i = 0
for t in test:
    total += 1
    if (getLabel(train, train_labels, t, 10) == test_labels[i]):
        print("got here")
        correct += 1
    i += 1

print(correct / total)
