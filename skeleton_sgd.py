#################################
# Your name: Alexander Beckerman
#################################
import math

import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    # train classifier
    w = np.zeros(data.shape[1])
    for t in range(1, T + 1):
        i = np.random.choice(data.shape[0], 1)[0]
        eta_t = eta_0 / t
        if labels[i] * (np.dot(w, data[i])) < 1:
            w = (1 - eta_t) * w + eta_t * C * labels[i] * data[i]
        else:
            w = (1 - eta_t) * w

    return w


def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """
    w = np.zeros(data.shape[1])
    for t in range(1, T + 1):
        i = np.random.choice(data.shape[0], 1)[0]
        eta_t = eta_0 / t
        w = w - eta_t * -(
                pow(math.e, -labels[i] * np.dot(w, data[i])) / (1 + pow(math.e, -labels[i] * np.dot(w, data[i])))) * labels[i] * data[i]

    return w
    # return w


#################################

# Place for additional code

#################################

def q1_find_best_eta_0(validation_data, validation_labels, train_data, train_labels, C, T):
    scale = [10 ** i for i in range(-5, 5)]
    avg_acc = []
    for eta_0 in scale:
        acc_sum = 0
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, C, eta_0, T)
            counter = 0
            for j in range(validation_data.shape[0]):
                x = validation_data[j]
                y = validation_labels[j]
                if np.dot(w, x) >= 0:
                    pred_y = 1
                else:
                    pred_y = -1
                if pred_y == y:
                    counter += 1
            acc = counter / validation_data.shape[0]
            acc_sum += acc

        avg_acc.append(acc_sum / 10)

    plt.xlabel("eta_0 Values")
    plt.ylabel("Avg Accuracy")
    plt.plot(scale, avg_acc)
    plt.legend()
    plt.show()

    best_eta_0 = scale[np.argmax(avg_acc)]
    print(f"best eta_0 is {best_eta_0} with accuracy of {avg_acc[np.argmax(avg_acc)]}")
    return best_eta_0


def q1b_find_best_c(validation_data, validation_labels, train_data, train_labels, eta_0, T):
    scale = [10 ** i for i in range(-5, 6)]
    avg_acc = []
    for C in scale:
        acc_sum = 0
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, C, eta_0, T)
            counter = 0
            for j in range(validation_data.shape[0]):
                x = validation_data[j]
                y = validation_labels[j]
                if np.dot(w, x) >= 0:
                    pred_y = 1
                else:
                    pred_y = -1
                if pred_y == y:
                    counter += 1
            acc = counter / validation_data.shape[0]
            acc_sum += acc

        avg_acc.append(acc_sum / 10)

    plt.xlabel("C Values")
    plt.ylabel("Avg Accuracy")
    plt.xscale("log")
    plt.plot(scale, avg_acc)
    plt.legend()
    plt.show()

    best_C = scale[np.argmax(avg_acc)]
    print(f"best C is {best_C} with accuracy of {avg_acc[np.argmax(avg_acc)]}")
    return best_C


def q1c_train(data, labels, C, eta_0, T):
    w = SGD_hinge(data, labels, C, eta_0, T)
    plt.imshow(np.reshape(w, (28, 28)), interpolation="nearest")
    plt.show()


def q1d_test(train_data, train_labels, test_data, test_labels, C, eta_0, T):
    w = SGD_hinge(train_data, train_labels, C, eta_0, T)
    counter = 0
    for j in range(test_data.shape[0]):
        x = test_data[j]
        y = test_labels[j]
        if np.dot(w, x) >= 0:
            pred_y = 1
        else:
            pred_y = -1
        if pred_y == y:
            counter += 1
    acc = counter / test_data.shape[0]
    print(f"best classifier accuracy is {acc}")
    return acc


def q2a_find_best_eta_0(validation_data, validation_labels, train_data, train_labels, T):
    scale = [10 ** i for i in range(-8, -2)]
    avg_acc = []
    for eta_0 in scale:
        acc_sum = 0
        for i in range(10):
            w = SGD_log(train_data, train_labels, eta_0, T)
            counter = 0
            for j in range(validation_data.shape[0]):
                x = validation_data[j]
                y = validation_labels[j]
                if np.dot(w, x) >= 0:
                    pred_y = 1
                else:
                    pred_y = -1
                if pred_y == y:
                    counter += 1
            acc = counter / validation_data.shape[0]
            acc_sum += acc

        avg_acc.append(acc_sum / 10)

    plt.xlabel("eta_0 Values")
    plt.ylabel("Avg Accuracy")
    plt.xscale("log")
    plt.plot(scale, avg_acc)
    plt.legend()
    plt.show()

    best_eta_0 = scale[np.argmax(avg_acc)]
    print(f"best eta_0 is {best_eta_0:.8f} with accuracy of {avg_acc[np.argmax(avg_acc)]}")
    return best_eta_0


def q2b(train_data, train_labels, test_data, test_labels, eta_0, T):
    w = SGD_log(train_data, train_labels, eta_0, T)
    plt.imshow(np.reshape(w, (28, 28)), interpolation="nearest")
    plt.show()
    counter = 0
    for j in range(test_data.shape[0]):
        x = test_data[j]
        y = test_labels[j]
        if np.dot(w, x) >= 0:
            pred_y = 1
        else:
            pred_y = -1
        if pred_y == y:
            counter += 1
    acc = counter / test_data.shape[0]
    print(f"best classifier accuracy is {acc}")
    return acc


def q2c(data, labels, eta_0, T):

    norms = []
    w = np.zeros(data.shape[1])
    for t in range(1, T + 1):
        i = np.random.choice(data.shape[0], 1)[0]
        eta_t = eta_0 / t
        w = w - eta_t * -(
                pow(math.e, -labels[i] * np.dot(w, data[i])) / (1 + pow(math.e, -labels[i] * np.dot(w, data[i])))) * \
            labels[i] * data[i]
        norms.append(np.linalg.norm(w))

    x_axis = np.arange(1, T + 1)
    plt.xlabel("iteration")
    plt.ylabel("norm")
    plt.plot(x_axis, np.array(norms))
    plt.legend()
    plt.show()

def main():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    # best_eta_0 = q1_find_best_eta_0(validation_data, validation_labels, train_data, train_labels, 1, 1000)
    # best_C = q1b_find_best_c(validation_data, validation_labels, train_data, train_labels, best_eta_0, 1000)
    # q1c_train(train_data, train_labels, best_C, best_eta_0, 20000)
    # q1d_test(train_data, train_labels, test_data, test_labels, best_C, best_eta_0, 20000)
    eta = q2a_find_best_eta_0(validation_data, validation_labels, train_data, train_labels, 1000)
    # acc = q2b(train_data, train_labels, test_data, test_labels, eta, 20000)
    q2c(train_data, train_labels, eta, 20000)
if __name__ == '__main__':
    main()
