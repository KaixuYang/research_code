from metrics import binary_acc
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras import layers
import xgboost as xgb


class classification:
    """
    implements linear regression with/without penalization, for prediction or cv
    """
    def __init__(self, x: np.array, y: np.array):
        self.x = x
        self.y = y
        self.LogisticModel = None
        self.snnModel = None
        self.svmModel = None
        self.rfModel = None
        self.xgboostModel = None

    def logistic_fit(self, penalty: str = 'l2', c: float = 1.0):
        """
        Fits the logistic regression model.
        :param penalty: default 'l2', can use 'l1'.
        :param c: inverse tuning parameter
        :return: None
        """
        self.LogisticModel = LogisticRegression(solver='liblinear', penalty=penalty, C=c).fit(self.x, self.y)

    def logisticcv_fit(self, nsplits: int, penalty: str = 'l2'):
        """
        runs a cross validation on multiple penalty parameters and returns the best score
        :param nsplits: number of cv splits
        :param penalty: default 'l2', can use 'l1'.
        :return: the best c parameter
        """
        c_cand = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 8e-1, 1, 2, 5, 10, 20, 50, 100]
        self.LogisticModel = LogisticRegressionCV(
            solver='liblinear', Cs=c_cand, cv=nsplits, penalty=penalty).fit(self.x, self.y)

    def logistic_predict(self, x: np.array) -> np.array:
        """
        predicts using the fitted logistic regression model
        :param x: the design matrix to be predicted
        :return: the predicted y
        """
        if self.LogisticModel is None:
            print('Logistic Model not trained, please run logistic_fit first!')
            return None
        else:
            return self.LogisticModel.predict(x)

    def logistic_cv(self, nsplits: int = 5, penalty: str = 'l2') -> (float, float, float):
        """
        runs a cross validation on the data set and returns the cross validation performance
        :param nsplits: number of cv splits
        :param penalty: default 'l2', can use 'l1'.
        :return: the cross-validated mse
        """
        model = LogisticRegressionCV(solver='liblinear', Cs=50, cv=nsplits, penalty=penalty).fit(self.x, self.y)
        c = model.C_[0]
        cv = KFold(n_splits=nsplits)
        acc_result = []
        for train, test in cv.split(self.x):
            x_train = self.x[train, :]
            x_test = self.x[test, :]
            y_train = self.y[train]
            y_test = self.y[test]
            model = LogisticRegression(solver='liblinear', penalty=penalty, C=c).fit(x_train, y_train)
            y_predict = model.predict(x_test)
            acc_result.append(binary_acc(y_test, y_predict))
        return np.mean(acc_result), np.std(acc_result), c

    def svm_fit(self, c: float = 1.0):
        """
        implements the svm classifier
        :param c: tuning parameter
        :return: None
        """
        self.svmModel = SVC(C=c, gamma='auto').fit(self.x, self.y)

    def svm_predict(self, x) -> np.array:
        """
        predict the class of a feature matrix x
        :param x: the feature matrix
        :return: the predicted classes
        """
        if self.svmModel is None:
            print("svm not trained, please run svm_fit first!")
            return None
        else:
            return self.svmModel.predict(x)

    def svm_cv(self, nsplits: int = 5) -> (float, float, float):
        """
        runs a cross validation on the data set and returns the cross validation performance
        :param nsplits: number of cv splits
        :return: the cross-validated binary accuracy
        """
        c_cand = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]
        cv = KFold(n_splits=nsplits)
        acc_result = []
        for c in c_cand:
            acc_result_c = []
            for train, test in cv.split(self.x):
                x_train = self.x[train, :]
                x_test = self.x[test, :]
                y_train = self.y[train]
                y_test = self.y[test]
                model = SVC(C=c, gamma='auto').fit(x_train, y_train)
                y_predict = model.predict(x_test)
                acc_result_c.append(binary_acc(y_test, y_predict))
            acc_result.append(np.mean(acc_result_c))
        best_c = c_cand[acc_result.index(max(acc_result))]
        return max(acc_result), np.std(acc_result), best_c

    def randomforest_fit(self, n_estimators: int = 100, max_depth: int = None, min_samples_split: int = 2):
        """
        fits the random forest classifier
        :param n_estimators: number of trees
        :param max_depth: max tree depth
        :param min_samples_split: minimum sample size to split
        :return: None
        """
        self.rfModel = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split).fit(self.x, self.y)

    def randomforest_predict(self, x) -> np.array:
        """
        predicts the classes of x
        :param x: the feature  matrix
        :return: the predicted classes of x
        """
        if self.rfModel is None:
            print("random forest not trained, please run randomforest_fit first!")
            return None
        else:
            return self.rfModel.predict(x)

    def randomforest_cv(self, nsplits: int = 5) -> (float, float, float):
        """
        implements a cross validation on the data set and returns the best result
        :param nsplits: number of cross validation splits
        :return: the cv binary accuracy
        """
        params = {
            "n_estimators": [20, 50, 100, 200],
            "max_depth": [2, 3, 5, 8, 10, 15, 20],
        }
        model = RandomForestClassifier()
        gridcv = GridSearchCV(model, params, cv=nsplits)
        gridcv.fit(self.x, self.y)
        best_params = gridcv.best_params_
        cv = KFold(n_splits=nsplits)
        acc_result = []
        for train, test in cv.split(self.x):
            x_train = self.x[train, :]
            x_test = self.x[test, :]
            y_train = self.y[train]
            y_test = self.y[test]
            model = RandomForestClassifier(**best_params).fit(x_train, y_train)
            y_predict = model.predict(x_test)
            acc_result.append(binary_acc(y_test, y_predict))
        return np.mean(acc_result), np.std(acc_result), best_params

    def xgboost_fit(self, eta: float = 0.1, max_depth: int = 6, subsample: float = 1.0, colsample_bytree: float = 1.0,
                    lam: float = 1.0, alpha: float = 0.0, objective: str = 'binary:logistic',
                    eval_metric: str = 'error', num_round: int = 100, early_stop: bool = True):
        """
        fits the xgboost model
        :param eta: learning rate, between 0 and 1
        :param max_depth: max tree depth
        :param subsample: proportion of sample drawn
        :param colsample_bytree: proportion of features drawn
        :param lam: l2 tuning parameter
        :param alpha: l1 tuning parameter
        :param objective: objective of model, 'binary:logistic', etc.
        :param eval_metric: evaluation metric, 'auc', 'error', etc.
        :param num_round: number of boosting rounds
        :param early_stop: whether early stop or not
        :return: None
        """
        params = {'eta': eta, 'max_depth': max_depth, 'subsample': subsample, 'colsample_bytree': colsample_bytree,
                  'reg_lambda': lam, 'alpha': alpha, 'objective': objective, 'n_estimators': num_round}
        if early_stop:
            x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2)
            evallist = [(x_test, y_test)]
            self.xgboostModel = xgb.XGBClassifier(**params)
            self.xgboostModel.fit(x_train, y_train, eval_metric=eval_metric, eval_set=evallist,
                                  early_stopping_rounds=10)
        else:
            self.xgboostModel = xgb.XGBClassifier(**params)
            self.xgboostModel.fit(self.x, self.y)

    def xgboost_predict(self, x) -> np.array:
        """
        predicts the classes of x
        :param x: the feature matrix
        :return: the predicted classes of x
        """
        if self.xgboostModel is None:
            print("xgboost not trained, please run xgboost_fit first!")
            return None
        else:
            return self.xgboostModel.predict(x)

    def xgboost_cv(self, nsplits: int = 5) -> (float, float, float):
        """
        cross validation on xgboost model
        :param nsplits: number of cv splits
        :return: the cv result
        """
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2)
        params = {
                "max_depth": [2, 3, 5, 8],
                "eta": [0.01, 0.05, 0.1, 0.15, 0.2],
                "objective": ['binary:logistic'],
                "sumsample": [0.5, 0.7, 1],
                "colsample_bytree": [0.5, 0.7, 1],
                "n_estimators": [50, 100, 200, 500],
            }
        """
        fit_params = {
            "early_stopping_rounds": 20,
            "eval_metric": "error",
            "eval_set": [(x_test, y_test)]
        }
        """
        model = xgb.XGBClassifier()
        gridcv = GridSearchCV(model, params, cv=nsplits)
        gridcv.fit(x_train, y_train) # , **fit_params)
        best_params = gridcv.best_params_
        cv = KFold(n_splits=nsplits)
        acc_result = []
        for train, test in cv.split(self.x):
            x_train = self.x[train, :]
            x_test = self.x[test, :]
            y_train = self.y[train]
            y_test = self.y[test]
            model = xgb.XGBClassifier(**best_params).fit(x_train, y_train)
            """
            x_t, x_v, y_t, y_v = train_test_split(x_train, y_train, test_size=0.2)
            model = xgb.XGBClassifier(**best_params).fit(x_t, y_t, eval_metric="error", eval_set=[(x_v, y_v)],
                                                        early_stopping_rounds=20)
                                                        """
            y_predict = model.predict(x_test)
            acc_result.append(binary_acc(y_test, y_predict))
        return np.mean(acc_result), np.std(acc_result), best_params

    def shallownn_fit(self, hidden_size: int = 20, epochs: int = 20, batch_size: int = 20, validation: tuple = None):
        """
        fits a shallow neural network model
        :param hidden_size: number of nodes in the hidden layer
        :param epochs: number of epochs
        :param batch_size: number of obs in each batch
        :return: None
        """
        self.snnModel = tf.keras.Sequential()
        self.snnModel.add(layers.Dense(hidden_size, activation='relu', input_shape=(self.x.shape[1], )))
        self.snnModel.add(layers.Dense(2, activation='softmax'))
        self.snnModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        if validation is not None:
            self.snnModel.fit(self.x, tf.keras.utils.to_categorical(self.y), epochs=epochs, batch_size=batch_size)

    def shallownn_predict(self, x: np.array) -> np.array:
        """
        makes prediction for x
        :param x: the feature matrix
        :return: the predicted classes of x
        """
        if self.snnModel is None:
            print("neural network not trained, please run shallownn_fit first!")
            return None
        else:
            return np.argmax(self.snnModel.predict(x), axis=1)
