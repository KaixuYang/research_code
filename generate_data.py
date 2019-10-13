import numpy as np
import random
from scipy.special import expit
from sklearn.model_selection import train_test_split


def generate_data_yfromx(
        seed,
        n: int = 125,
        p: int = 200,
        testing_size: float = 0.2,
        cor: float = 0.6,
        dist: str = 'unif',
        response: str = 'bernoulli'
) -> [np.array, np.array, np.array, np.array]:
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
    # generate true f(x)
    truefx = 6 * x[:, 0] + np.sqrt(84) * x[:, 1] ** 3 + np.sqrt(12 / (np.sin(6) / 12 + 1 / 2)) * np.sin(
        3 * x[:, 2]) + np.sqrt(48 / (np.exp(2) + np.exp(-1) - np.exp(-2) - np.exp(1))) * np.exp(x[:, 3])
    if response == 'bernoulli':
        prob = expit(truefx)  # compute probability
        y = np.random.binomial(1, p=prob)  # generate labels from Binomial distribution
    else:
        y = truefx + np.random.normal(0, 1, n)  # generate labels from normal distribution
    if testing_size > 0:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testing_size)
        return x_train, x_test, y_train, y_test
    else:
        return x, y


def generate_data_xfromy(
        seed,
        n: int = 125,
        p: int = 200,
        testing_size: float = 0.2,
        cor: float = 0.6,
        s: int = 10,
        sparse : bool = False
) -> [np.array, np.array, np.array, np.array]:
    """
    generate x from normal distribution conditioning on two classes of y
    :param seed: seed
    :param n: total sample size
    :param p: number of features
    :param testing_size: proportion of testing set size
    :param cor: feature correlation
    :param s: number of nonzero features, will be ignored if sparse is False
    :param sparse: whether x is sparse
    """
    random.seed(seed)
    if sparse:
        # generates expectation as 2 and -2 alternating for the first s elements and 0 for the rest
        mu0 = np.array([2 if i % 2 == 0 else -2 for i in range(s)] + [0] * (p-s))
        mu1 = np.array([2 if i % 2 == 1 else -2 for i in range(s)] + [0] * (p-s))
    else:
        # generates mean as range(p)
        mu0 = np.array(range(p))
        mu1 = np.array(range(p)[::-1])
    # generate covariance matrix from AR(1) structure
    sigma = np.array([[0.0] * p] * p)
    for i in range(p):
        for j in range(p):
            sigma[i][j] = cor ** abs(i - j)
    # generate y
    y = np.random.binomial(1, 0.5, n)
    # generate x
    x = np.array([[0.0] * p] * n)
    for i in range (n):
        if y[i] == 0:
            x[i, :] = np.random.multivariate_normal(mu0, sigma)
        else:
            x[i, :] = np.random.multivariate_normal(mu1, sigma)
    if testing_size > 0:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testing_size)
        return x_train, x_test, y_train, y_test
    else:
        return x, y