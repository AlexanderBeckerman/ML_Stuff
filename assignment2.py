#################################
# Your name: Alexander Beckerman
#################################

import numpy as np
import matplotlib.pyplot as plt
from intervals import *


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        xs = np.random.uniform(0, 1, m)
        prob = []
        ys = np.zeros(m)
        xs.sort()
        for idx, x in enumerate(xs):
            if 0 <= x <= 0.2 or 0.4 <= x <= 0.6 or 0.8 <= x <= 1:
                prob.append([0.2, 0.8])
            else:
                prob.append([0.9, 0.1])
            ys[idx] = np.random.choice([0, 1], p=prob[idx])

        ys = np.array(ys)
        return np.column_stack((xs, ys))

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        avg_emp_err = []
        avg_true_err = []

        for m in range(m_first, m_last + 1, step):
            emp_err = []
            true_err = []
            for i in range(T):
                s = self.sample_from_D(m)
                emp_err.append(self.get_emp_err(m, k, s))
                true_err.append(self.get_true_err(m, k, s))

            avg_emp_err.append(sum(emp_err) / T)
            avg_true_err.append(sum(true_err) / T)

        avg_emp_err = np.array(avg_emp_err)
        avg_true_err = np.array(avg_true_err)
        x_axis = np.arange(m_first, m_last + 1, step)
        plt.xlabel("n Values")
        plt.ylabel("Average Error")
        plt.plot(x_axis, avg_emp_err, label="Empirical")
        plt.plot(x_axis, avg_true_err, label="True")
        plt.legend()
        plt.show()

        avg_errors = np.column_stack((avg_emp_err, avg_true_err))
        return avg_errors

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               k_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        s = self.sample_from_D(m)
        emp_err = []
        true_err = []
        for k in range(k_first, k_last + 1, step):
            emp_err.append(self.get_emp_err(m, k, s))
            true_err.append(self.get_true_err(m, k, s))


        x_axis = np.arange(k_first, k_last + 1, step)
        plt.plot(x_axis, emp_err, label="Empirical")
        plt.plot(x_axis, true_err, label="True")
        plt.xlabel("k Values")
        plt.ylabel("Error")
        plt.legend()
        plt.show()

        min_ind = np.argmin(emp_err)
        k_star = x_axis[min_ind]
        return k_star

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """

        emp_err = []

        s = self.sample_from_D(m)
        np.random.shuffle(s)
        train = s[:int(0.8 * m)]
        test = s[int(0.8 * m):]
        ind1 = np.argsort(train[:, 0])
        train = train[ind1]
        ind2 = np.argsort(test[:, 0])
        test = test[ind2]

        for k in range(1, 11):

            inter = find_best_interval(train[:, 0], train[:, 1], k)[0]
            loss = []
            for x, y in test:
                if self.contains_x(x, inter):
                    if y == 1:
                        loss.append(0)
                    else:
                        loss.append(1)
                else:
                    if y == 1:
                        loss.append(1)
                    else:
                        loss.append(0)

            emp_err.append(sum(loss) / len(test))


        x_axis = np.arange(1, 11)
        plt.xlabel("k Values")
        plt.ylabel("Error")
        plt.plot(x_axis, emp_err, label="Empirical")
        plt.legend()
        plt.show()
        return np.argmin(emp_err) + 1

    #################################
    # Place for additional methods

    #################################
    def get_emp_err(self, m, k, s):

        xs = s[:, 0]
        ys = s[:, 1]
        intervals, besterror = find_best_interval(xs, ys, k)
        return besterror / m

    def get_true_err(self, m, k, s):

        xs = s[:, 0]
        ys = s[:, 1]
        intervals, besterror = find_best_interval(xs, ys, k)
        return self.calc_true_err(intervals)

    def contains_x(self, x, intervals):
        for i in intervals:
            if i[0] <= x <= i[1]:
                return True
        return False

    def get_intersection(self, inter1, inter2):
        sum = 0
        i1 = 0
        i2 = 0

        while i1 < len(inter1) and i2 < len(inter2):
            left = max(inter1[i1][0], inter2[i2][0])
            right = min(inter1[i1][1], inter2[i2][1])
            if right - left > 0:
                sum += (right - left)
            if inter1[i1][1] == inter2[i2][1]:
                i1 += 1
                i2 += 1
            elif inter1[i1][1] < inter2[i2][1]:
                i1 += 1
            else:
                i2 += 1
        return sum

    def calc_true_err(self, intervals):

        prob1 = [(0, 0.2), (0.4, 0.6), (0.8, 1)]
        prob2 = [(0.2, 0.4), (0.6, 0.8)]
        prob1_intersect = self.get_intersection(intervals, prob1)
        prob2_intersect = self.get_intersection(intervals, prob2)

        return 0.8 * (0.6 - prob1_intersect) + 0.2 * prob1_intersect + 0.1 * (
                    0.4 - prob2_intersect) + 0.9 * prob2_intersect


if __name__ == '__main__':
    ass = Assignment2()
    # ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    # ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.cross_validation(1500)
