import numpy as np
import random
from scipy.special import expit
from sklearn.model_selection import train_test_split


def generate_data(
    seed,
    n: int = 125,
    p: int = 200,
    testing_size: float = 0.2,
    cor: float = 0.6,
    dist: str = 'unif',
    response: str = 'bernoulli'
    ) -> [
        np.array, np.array, np.array, np.array]:
    """
    generate training and testing data with correlated features.
    :param seed: seed
    :param n: total sample size
    :param p: number of features
    :param testing_size: proportion of testing set size
    :param cor: feature correlation
    :param dist: distribution of design matrix, 'unif' for uniform(0,1), anything else for N(0,1)
    :param response: distribution of response, 'bernoulli' for binary classification, anything else for normal
    :return: x_train(, x_test), y_train(, y_test)
    """
    random.seed(seed)
    # initialize design matrix
    if dist == 'unif':
        x = np.random.uniform(-1, 1, (n, p))
        xu = np.random.uniform(-1, 1, n)
        xv = np.random.uniform(-1, 1, n)
    else:
        x = np.random.normal(0, 1, (n, p))
        xu = np.random.normal(0, 1, n)
        xv = np.random.normal(0, 1, n)
    t = np.sqrt(cor / (1 - cor))
    for j in range(4):
        x[:, j] = (x[:, j] + t * xu) / (1 + t)  # generate correlated features
    for j in range(4, p):
        x[:, j] = (x[:, j] + t * xv) / (1 + t)  # generate correlated features
    truefx = 6 * x[:, 0] + np.sqrt(84) * x[:, 1] ** 3 + np.sqrt(12 / (np.sin(6) / 12 + 1 / 2)) * np.sin(
        3 * x[:, 2]) + np.sqrt(48 / (np.exp(2) + np.exp(-1) - np.exp(-2) - np.exp(1))) * np.exp(x[:, 3])
    if response == 'bernoulli':
        prob = expit(truefx)  # compute probability
        y = np.random.binomial(1, p=prob)  # generate labels from Binomial distribution
    else:
        y = truefx + np.random.normal(0, 1, n)
    if testing_size > 0:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testing_size)
        return x_train, x_test, y_train, y_test
    else:
        return x, y
