from sklearn.decomposition import PCA
import numpy as np


def pca(x: np.array, variance: float = 0.9):
    """
    returns the principal components matrix of x
    :param x: the original feature matrix
    :param variance: what percentage of variance is preserved
    :return: the principal matrix
    """
    model = PCA(n_components=variance, svd_solver='full')
    model.fit(x)
    return model.transform(x)


def random_projection(d: int, x: np.array, B1: int, B2: int) -> (list, list):
    """
    generates B1 * B2 random projected training data
    :param d: the dimension projected in
    :param x: the training x
    :param B1: number of classifiers in ensemble
    :param B2: number of candidates in each classifier
    :return: two nested lists, B1 * B2 projection matrices and projected matrices
    """
    rp_x = []
    rp_p = []
    temp_x = []
    temp_p = []
    _, p = x.shape
    for b1 in range(B1):
        for b2 in range(B2):
            q = np.random.normal(0, 1, d * p).reshape(d, p)
            a, _, _ = np.linalg.svd(q.T, full_matrices=False)
            a = a.T
            temp_p.append(a)
            temp_x.append(np.matmul(a, x.T).T)
        rp_p.append(temp_p)
        rp_x.append(temp_x)
        temp_p = []
        temp_x = []
    return rp_p, rp_x
