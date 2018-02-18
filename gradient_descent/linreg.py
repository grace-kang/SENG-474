#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl 
mpl.use('TkAgg')
import matplotlib.pyplot as plt


class MyLinearRegressor():

    def __init__(self, kappa=0.1, lamb=0, max_iter=200, opt='batch'):
        self._kappa = kappa
        self._lamb = lamb # for bonus question
        self._opt = opt
        self._max_iter = max_iter

    def fit(self, X, y):
        X = self.__feature_rescale(X)
        X = self.__feature_prepare(X)
        error = []
        if self._opt == 'sgd':
            error = self.__stochastic_gradient_descent(X, y)
            self.plot_error(error)
        elif self._opt == 'batch':
            error = self.__batch_gradient_descent(X, y)
            self.plot_error(error)
        else:
            print('unknow opt')
        return error

    def predict(self, X):
        pass

    def __batch_gradient_descent(self, X, y):
        N, M = X.shape
        niter = 0
        error = []
        self._w = np.ones(X.shape[1])
        X_trans = X.transpose()

        for i in range(self._max_iter):
            hypothesis = np.dot(X, self._w)
            loss = y - hypothesis
            cost = np.sum(loss**2) / (2*N)
            error.append(cost)
            print("Iteration %d | Cost: %f" % (i, cost))
            gradient = np.dot(X_trans, loss) / N
            self._w = self._w + self._kappa * gradient

        return error

    def __stochastic_gradient_descent(self, X, y):
        N, M = X.shape
        niter = 0
        error = []
        self._w = np.ones(X.shape[1])
        X_trans = X.transpose()

        k = 0
        for i in range(self._max_iter):
            hypothesis = np.dot(X, self._w)
            loss = y - hypothesis
            cost = np.sum(loss**2) / (2*N)
            error.append(cost)
            print("Iteration %d | Cost: %f" % (i, cost))

            #shuffle X and y
            X_y = list(zip(X,y))
            np.random.shuffle(X_y)
            X, y = zip(*X_y)

            h = np.dot(X[k], self._w)
            l = y[k] - h
            gradient = np.dot(X[k], l)
            self._w = self._w + self._kappa * gradient
            k+=1

        return error

    def __total_error(self, X, y, w):
        ##############################
        #
        #  put your code here
        #
        ##############################
        return 1

    # add a column of 1s to X
    def __feature_prepare(self, X_):
        M, N = X_.shape
        X = np.ones((M, N+1))
        X[:, 1:] = X_
        return X

    # rescale features to mean=0 and std=1
    def __feature_rescale(self, X):
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        return (X - self._mu)/self._sigma

    def plot_error(self, error):
        plt.plot(error)
        plt.title("Error VS Iteration")
        plt.show()


if __name__ == '__main__':
    from sklearn.datasets import load_boston

    data = load_boston()
    X, y = data['data'], data['target']
    mylinreg = MyLinearRegressor()
    mylinreg.fit(X, y)
