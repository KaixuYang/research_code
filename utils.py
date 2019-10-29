import numpy as np

class orthogonalize:
    """
    orthogonalizes the original matrix, can do inverse transformation to the fitted parameters
    """
    def __init__(self):
        self.x = None
        self.q = None
        self.r = None

    def transform(self, x):
        """
        transforms the qr decomposition
        :param x: the matrix to be orthogonalized
        :return: orthorgonal matrix
        """
        self.x = x
        self.q, self.r = np.linalg.qr(x)
        return self.r

    def inverse_transform(self, w):
        """
        inverse transform the fitted parameters
        :param w: the fitted parameters to be inverse transformed
        :return: the inverse transformation
        """
        if self.r is None:
            print("model not fitted yet")
            return none
        else:
            return np.matmul(np.linalg.inv(self.r), w)