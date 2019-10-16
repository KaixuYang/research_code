from generate_data import generate_data_yfromx as gd
from classification import classification as cl
import numpy as np
from dimension_reduction import random_projection
from metrics import binary_acc
from sklearn.model_selection import KFold


def rp_nn(d: int, B1: int, B2: int, hidden_size: int, epochs: int, batch_size: int,
          x: np.array, y: np.array, nsplits: int):
    # extract the projection matrices and projected matrices
    rp_projection, rp_res = random_projection(d=d, x=x, B1=B1, B2=B2)
    rp_best_b2 = []  # create list to record best random projection matrix for each b1
    models = []  # create list to store models
    rp_model = []  # create list to store the models corresponding to b2's
    for b1 in range(B1):
        print(b1)
        temp_acc = []  # create list to record the cv accuracy for all b2 in b1
        for b2 in range(B2):
            # perform a cross validation to evaluate the performance for each b2
            cv = KFold(n_splits=nsplits)
            temp_acc_cv = []
            for train, test in cv.split(rp_res[b1][b2]):
                x_train = rp_res[b1][b2][train, :]
                x_test = rp_res[b1][b2][test, :]
                y_train = y[train]
                y_test = y[test]
                # for each cv fold, train a shallow neural network on the projected matrix
                model = cl(x_train, y_train)
                model.shallownn_fit(hidden_size=hidden_size, epochs=epochs, batch_size=batch_size)
                y_predict = model.shallownn_predict(x_test)
                temp_acc_cv.append(binary_acc(y_test, y_predict))  # record the accuracy for this cv fold
                models.append(model)
            temp_acc.append(np.mean(temp_acc_cv))  # record the cv accuracy for this b2
        # record the best b2 for this b1
        rp_best_b2.append(temp_acc.index(max(temp_acc)))  # length=B1, the best b2 indices
        rp_model.append(models[rp_best_b2[-1] * nsplits: rp_best_b2[-1] * nsplits + nsplits])
        models = []
        print('the best cv score for this b1 is', max(temp_acc))
    rp_projection_best = []  # create list to store the B1 projection matrices
    for b1 in range(B1):
        rp_projection_best.append(rp_projection[0][rp_best_b2[b1]])  # length=B1, the best projection matrices
        del rp_projection[0]
    return rp_projection_best, rp_model


def shallownn_rp_wrapper():
    d = 5
    B1 = 20
    B2 = 20
    hidden_size = 5
    epochs = 30
    batch_size = 5
    nsplits = 5
    x, y = gd(seed=1, testing_size=0, n=125, p=200)
    cv = KFold(n_splits=5)
    cv_res = []
    for train, test in cv.split(x):
        x_train = x[train, :]
        x_test = x[test, :]
        y_train = y[train]
        y_test = y[test]
        rp_projection_best, rp_model = rp_nn(d=d, B1=B1, B2=B2, hidden_size=hidden_size, epochs=epochs,
                                             batch_size=batch_size, x=x_train, y=y_train, nsplits=nsplits)
        y_predict = None
        for b1 in range(B1):
            for i in range(nsplits):
                model = rp_model[b1][i]
                if y_predict is None:
                    y_predict = model.shallownn_predict(np.matmul(x_test, rp_projection_best[b1].T))
                else:
                    y_predict = y_predict + model.shallownn_predict(np.matmul(x_test, rp_projection_best[b1].T))
            print(y_predict)
        y_predict = y_predict / B1 / nsplits
        y_final = np.where(y_predict > 0.5, 1, 0)
        cv_res.append(binary_acc(y_test, y_final))
    print('the cross validation result is', np.mean(cv_res), np.std(cv_res), cv_res)


shallownn_rp_wrapper()