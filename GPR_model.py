import math

from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

class GPR:

    def __init__(self, optimize=True):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 0.5, "sigma_f": 0.2, "sigma_f0": 1}
        self.optimize = optimize

    def fit(self, X, y):
        # store train data
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)

         # hyper parameters optimization
        def negative_log_likelihood_loss(params):
            self.params["l"], self.params["sigma_f"], self.params["sigma_f0"] = params[0], params[1], params[2]
            Kyy = self.kernel(self.train_X, self.train_X) + params[2] * np.eye(len(self.train_X))
            loss = 0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) + 0.5 * np.linalg.slogdet(Kyy)[1] + 0.5 * len(self.train_X) * np.log(2 * np.pi)
            return loss.ravel()

        if self.optimize:
            res = minimize(negative_log_likelihood_loss,
                           [self.params["l"], self.params["sigma_f"], self.params["sigma_f0"]],
                   bounds=((1e-4, 1e4), (1e-4, 1e4), (1e-4, 1e4)),
                   method='L-BFGS-B')
            self.params["l"], self.params["sigma_f"], self.params["sigma_f0"] = res.x[0], res.x[1], res.x[2]

        self.is_fit = True

    def predict(self, X):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return

        X = np.asarray(X)
        Kff = self.kernel(self.train_X, self.train_X)  # (N, N)
        Kyy = self.kernel(X, X)  # (k, k)
        Kfy = self.kernel(self.train_X, X)  # (N, k)
        Kff_inv = np.linalg.inv(Kff + self.params["sigma_f0"] * np.eye(len(self.train_X)))  # (N, N)

        mu = Kfy.T.dot(Kff_inv).dot(self.train_y)
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)
        return mu, cov

    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix)

def get_mae(records_real, records_predict):
    """
    平均绝对误差
    """
    if len(records_real) == len(records_predict):
        return sum([abs(x - y) for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None

from pre_data import train_X, train_y, test_X, test_y
import time

t_start = time.time()
print('start training......')
gpr = GPR(optimize=True)
gpr.fit(train_X, train_y)
print('finishing training spending %.3f s' % (time.time() - t_start))

l, sigma_f, sigma_f0 = gpr.params['l'], gpr.params['sigma_f'], gpr.params['sigma_f0']
print('l, sigma_f, sigma_f0:', l, sigma_f, math.sqrt(sigma_f0))
mu, cov = gpr.predict(test_X)
pre_y = mu.ravel()
mae = get_mae(test_y, pre_y)
print('mae:', mae)
print('finishing predicting spending %.3f s' % (time.time() - t_start))
x = [i for i in range(50)]
# plt.plot(x, test_y, color='blue')
# plt.plot(x, pre_y, color='red')
# plt.show()

uncertainty = 1.96 * np.sqrt(np.diag(cov[:50]))
plt.figure()
plt.title("l=%.2f sigma_f=%.2f" % (l, sigma_f))
plt.fill_between(x, pre_y[:50] + uncertainty, pre_y[:50] - uncertainty, alpha=0.1)
plt.plot(x, pre_y[:50], label="predict")
plt.scatter(x, test_y[:50], label="train", c="red", marker="x")
plt.legend()
plt.show()
