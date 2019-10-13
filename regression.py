from metrics import mse
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.model_selection import KFold


class Regression:
    """
    implements linear regression with/without penalization, for prediction or cv
    """
    def __init__(self, x: np.array, y: np.array):
        self.x = x
        self.y = y
        self.LinearModel = None
        self.LassoModel = None
        self.RidgeModel = None
        self.ElasticNetModel = None

    def lm_fit(self):
        """
        Fits the linear model.
        :return: None
        """
        self.LinearModel = LinearRegression().fit(self.x, self.y)

    def lm_predict(self, x: np.array) -> np.array:
        """
        predicts using the fitted linear model
        :param x: the design matrix to be predicted
        :return: the predicted y
        """
        if self.LinearModel is None:
            print('Linear Model not trained, please run linear_fit first!')
            return None
        else:
            return self.LinearModel.predict(x)

    def lm_cv(self, nsplits: int):
        """
        runs a cross validation on the data set and returns the cross validation performance
        :param nsplits: number of cv splits
        :return: the cross-validated mse
        """
        cv = KFold(n_splits=nsplits)
        mse_result = []
        for train, test in cv.split(self.x):
            x_train = self.x[train, :]
            x_test = self.x[test, :]
            y_train = self.y[train]
            y_test = self.y[test]
            model = LinearRegression().fit(x_train, y_train)
            y_predict = model.predict(x_test)
            mse_result.append(mse(y_test, y_predict))
        return np.mean(mse_result)

    def lasso_fit(self, lam: float):
        """
        fits the lasso model with a given lambda
        :param lam: tuning parameter
        :return: None
        """
        self.LassoModel = Lasso(alpha=lam).fit(self.x, self.y)

    def lassocv_fit(self, nsplits: int):
        """
        implements the lassoCV in sklearn to fit the model
        :param nsplits: the number of cv splits
        :return: None
        """
        self.LassoModel = LassoCV(cv=nsplits).fit(self.x, self.y)

    def lasso_predict(self, x: np.array) -> np.array:
        """
        predicts using the fitted lasso model
        :param x: the design matrix to be predicted
        :return: the predicted y
        """
        if self.LassoModel is None:
            print('Lasso Model not trained, please run lasso_fit first!')
            return None
        else:
            return self.LassoModel.predict(x)

    def lasso_cv(self, nsplits: int, lam: float = None):
        """
        runs a cross validation on the data set and returns the cross validation performance
        :param nsplits: number of cv splits
        :param lam: tuning parameter
        :return: the cross-validated mse
        """
        if lam is None:
            lam = LassoCV(cv=nsplits).fit(self.x, self.y).alpha_
        cv = KFold(n_splits=nsplits)
        mse_result = []
        for train, test in cv.split(self.x):
            x_train = self.x[train, :]
            x_test = self.x[test, :]
            y_train = self.y[train]
            y_test = self.y[test]
            model = Lasso(alpha=lam).fit(x_train, y_train)
            y_predict = model.predict(x_test)
            mse_result.append(mse(y_test, y_predict))
        return np.mean(mse_result)

    def ridge_fit(self, lam: float):
        """
        fits the ridge model with a given lambda
        :param lam: tuning parameter
        :return: None
        """
        self.RidgeModel = Ridge(alpha=lam).fit(self.x, self.y)

    def ridgecv_fit(self, nsplits: int):
        """
        implements the ridgeCV in sklearn to fit the model
        :param nsplits: the number of cv splits
        :return: None
        """
        self.RidgeModel = RidgeCV(cv=nsplits).fit(self.x, self.y)

    def ridge_predict(self, x: np.array) -> np.array:
        """
        predicts using the fitted ridge model
        :param x: the design matrix to be predicted
        :return: the predicted y
        """
        if self.RidgeModel is None:
            print('Ridge Model not trained, please run ridge_fit first!')
            return None
        else:
            return self.RidgeModel.predict(x)

    def ridge_cv(self, nsplits: int, lam: float = None):
        """
        runs a cross validation on the data set and returns the cross validation performance
        :param nsplits: number of cv splits
        :param lam: tuning parameter
        :return: the cross-validated mse
        """
        if lam is None:
            model = RidgeCV(cv=nsplits).fit(self.x, self.y)
            lam = model.alpha_
        cv = KFold(n_splits=nsplits)
        mse_result = []
        for train, test in cv.split(self.x):
            x_train = self.x[train, :]
            x_test = self.x[test, :]
            y_train = self.y[train]
            y_test = self.y[test]
            model = Ridge(alpha=lam).fit(x_train, y_train)
            y_predict = model.predict(x_test)
            mse_result.append(mse(y_test, y_predict))
        return np.mean(mse_result)

    def elasticnet_fit(self, lam: float, l1_ratio: float):
        """
        fits the elastic net model with a given lambda and a given l1_ratio
        :param lam: tuning parameter
        :param l1_ratio: balance l1 and l2 penalization, 0 means ridge, 1 means lasso
        :return: None
        """
        self.ElasticNetModel = ElasticNet(alpha=lam, l1_ratio=l1_ratio).fit(self.x, self.y)

    def elasticnetcv_fit(self, nsplits: int):
        """
        implements the elasticnetCV in sklearn to fit the model
        :param nsplits: the number of cv splits
        :return: None
        """
        self.ElasticNetModel = ElasticNetCV(
            cv=nsplits, l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.95, 0.99, 1]).fit(self.x, self.y)

    def elasticnet_predict(self, x: np.array) -> np.array:
        """
        predicts using the fitted elasticNet model
        :param x: the design matrix to be predicted
        :return: the predicted y
        """
        if self.ElasticNetModel is None:
            print('ElasticNet Model not trained, please run elasticnet_fit first!')
            return None
        else:
            return self.ElasticNetModel.predict(x)

    def elasticnet_cv(self, nsplits: int, lam: float = None, l1_ratio: float = None):
        """
        runs a cross validation on the data set and returns the cross validation performance
        :param nsplits: number of cv splits
        :param lam: tuning parameter
        :param l1_ratio: balance l1 and l2 penalization, 0 means ridge, 1 means lasso
        :return: the cross-validated mse
        """
        if lam is None or l1_ratio is None:
            model = ElasticNetCV(cv=nsplits, l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.95, 0.99, 1]).fit(self.x, self.y)
            if lam is None:
                lam = model.alpha_
            if l1_ratio is None:
                l1_ratio = model.l1_ratio_
        cv = KFold(n_splits=nsplits)
        mse_result = []
        for train, test in cv.split(self.x):
            x_train = self.x[train, :]
            x_test = self.x[test, :]
            y_train = self.y[train]
            y_test = self.y[test]
            model = ElasticNet(alpha=lam, l1_ratio=l1_ratio).fit(x_train, y_train)
            y_predict = model.predict(x_test)
            mse_result.append(mse(y_test, y_predict))
        return np.mean(mse_result)
