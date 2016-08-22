#!/usr/bin/env python

import numpy as np
from scipy.optimize import minimize

def logist_nexp(theta, x):

    arg = -np.dot(x, theta)
    arg[arg > 100] = 100   # Handle overflow.
    return np.exp(arg)

def logist_h(theta, x):

    return 1.0 / (1.0 + logist_nexp(theta, x))

def logist_cost(theta, x, y, lamb, nc):

    ht = logist_h(theta, x)
    ht[ht < 1e-15] = 1e-15
    ht[ht > 1.0 - 1e-15] = 1.0 - 1e-15

    grad = np.dot(ht - y, x)

    return [ -np.sum(y * np.log(ht) + (1.0 - y) * np.log(1.0 - ht)), np.dot(ht - y, x) ]

def logist_reg(data, labels, cost_func, t0, lamb=1e-4, nc=2, tol=None, maxiter=100, disp=True):

    res = minimize(cost_func, t0, jac=True, method='L-BFGS-B', args=(data, labels, lamb, nc), tol=tol, options={'maxiter': maxiter, 'disp': disp})
    return res.x

def mult_logist_predict(theta, x, nc):

    norm = logist_nexp(-theta.T, x)
    return norm.T / np.sum(norm, axis=1)

# Assumes class labels are 0, 1, 2, ...
def mult_logist_cost(theta, x, y, lamb, nc):

    theta = theta.reshape(nc, x.shape[1])
    prob = mult_logist_predict(theta, x, nc)

    sm = 0
    for i in range(prob.shape[0]):
        wh = (np.where( y == i ))[0]
        sm -= np.sum(np.log(prob[i,wh]))
    sm += 0.5*lamb*np.sum(np.multiply(theta, theta))   # Weight decay term.

    indic = np.zeros(prob.shape, dtype=int)
    for i in range(prob.shape[0]):
        wh = (np.where( y == i ))[0]
        indic[i,wh] = 1
    grad = np.dot(prob - indic, x)
    grad += lamb*theta   # Weight decay term.
    grad = grad.reshape(grad.size)

    return [ sm, grad ]

if __name__ == '__main__':

    import ml_common

    doMultLogistic = True

    np.random.seed(1)

    # Training data.
    labels_train = ml_common.loadMNISTLabels(r"common/train-labels-idx1-ubyte")
    images_train = ml_common.loadMNISTImages(r"common/train-images-idx3-ubyte")
    images_train = images_train.reshape(images_train.shape[0], images_train.shape[1]*images_train.shape[2])

    # "Real" data.
    labels_real = ml_common.loadMNISTLabels(r"common/t10k-labels-idx1-ubyte")
    images_real = ml_common.loadMNISTImages(r"common/t10k-images-idx3-ubyte")
    images_real = images_real.reshape(images_real.shape[0], images_real.shape[1]*images_real.shape[2])

    if not doMultLogistic:
        # Logistic regression.
        wh_train = (np.where((labels_train == 0) | (labels_train == 1)))[0]
        wh_real  = (np.where((labels_real == 0) | (labels_real == 1)))[0]
        labels_train_01 = labels_train[wh_train]
        images_train_01 = images_train[wh_train,:]
        labels_real_01 = labels_real[wh_real]
        images_real_01 = images_real[wh_real,:]

        t0 = 0.005 * np.random.randn(images_train_01.shape[1])
        t = logist_reg(images_train_01, labels_train_01, logist_cost, t0)
        pred = np.round(logist_h(t, images_real_01)).astype(int)
        diff = labels_real_01 - pred
    else:
        # Multinomial logistic regression.
        for it in np.arange(10):  # Repeat multiple times.
            t0 = 0.005 * np.random.randn(images_train.shape[1] * 10)
            lamb = 1e-4
            nc = int(t0.size/images_train.shape[1])

            if False:
                # Check the gradient calculation.
                t = t0
                cost1, grad1 = mult_logist_cost(t0, images_train, labels_train, lamb, nc)
                for i in np.arange(350,360):
                    t1 = np.copy(t0)
                    t1[i] += 1e-9
                    cost2, grad2 = mult_logist_cost(t1, images_train, labels_train, lamb, nc)
                    delta = (cost2-cost1)/1e-9
                    print(t0[i], cost1, cost2, delta, grad1[i], delta/grad1[i])

            t = logist_reg(images_train, labels_train, mult_logist_cost, t0, lamb=lamb, nc=nc, disp=False)
            prob = mult_logist_predict(t.reshape(nc, images_real.shape[1]), images_real, nc)
            pred = np.argmax(prob, axis=0)
            diff = labels_real - pred

            [ print(diff[i]) for i in range(10) ]
            wh = (diff == 0)
            print("# data: %s, accuracy: %s" % (diff.size, np.mean(wh)))


