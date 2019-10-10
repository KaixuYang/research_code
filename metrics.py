import numpy as np


def mse(y_test: np.array, y_predict: np.array) -> float:
    """
    mean squared error
    :param y_test: observed y
    :param y_predict: predicted y
    :return: mse
    """
    return np.mean((y_test - y_predict) ** 2)


def rmse(y_test: np.array, y_predict: np.array) -> float:
    """
    rooted mean squared error
    :param y_test: observed y
    :param y_predict: predicted y
    :return: rmse
    """
    return np.sqrt(np.mean((y_test - y_predict) ** 2))


def mae(y_test: np.array, y_predict: np.array) -> float:
    """
    mean absolute error
    :param y_test: observed y
    :param y_predict: predicted y
    :return: mae
    """
    return np.mean(np.abs(y_test - y_predict))


def mape(y_test: np.array, y_predict: np.array) -> float:
    """
    mean absolute percentage error
    :param y_test: observed y
    :param y_predict: predicted y
    :return: mape
    """
    return np.mean(np.abs((y_test - y_predict) / y_test))


def pinball(y_test: np.array, y_predict: np.array, tau: float) -> float:
    """
    pinball loss
    :param tau: pinball parameter, between 0 and 1, smaller tau put more weight on more positive errors
    :param y_test: observed y
    :param y_predict: predicted y
    :return: pinball loss
    """
    return np.mean(np.abs(y_test - y_predict) * np.where(y_test - y_predict >= 0, tau, 1 - tau))


def binary_acc(y_test: np.array, y_predict: np.array) -> float:
    """
    binary accuracy
    :param y_test: observed y
    :param y_predict: predicted y
    :return: binary accuracy
    """
    y_predict = np.where(y_predict >= 0.5, 1, 0)
    return np.mean(y_test == y_predict)