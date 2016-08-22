#!/usr/bin/env python

import sys, ml_common, rbm, rbm_stacked, multlogistic
import numpy as np

def viewWeights(rbm, imShape, outfile=""):

    tiledImage = rbm.imageWeights(imShape)

    import pylab as pl

    fig = pl.figure()
    ax = fig.add_subplot(111, aspect=1)
    ax.imshow(tiledImage, interpolation='none', cmap=pl.get_cmap('gray'))
    ax.autoscale(False)
    for i in range(100):
        ax.axhline(imShape[1]*i-0.5, color='black', lw=1)
        ax.axvline(imShape[0]*i-0.5, color='black', lw=1)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if outfile == "":
        pl.show()
    else:
        pl.savefig(outfile, dpi=500)

def viewErrors(diff, prob, images_real, labels_real, imShape):

    wh = (np.where(diff != 0))[0]
    im = images_real[wh,:].reshape(len(wh), imShape[0], imShape[1])
    lb = labels_real[wh]
    pb = prob[:,wh]

    import pylab as pl

    for ind in range(10):
        print("Actual:", lb[ind])
        for i in range(10):
            print(i, 100.*pb[i,ind])

        fig = pl.figure()
        ax = fig.add_subplot(111, aspect=1)
        ax.imshow(im[ind,:,:], interpolation='none', cmap=pl.get_cmap('gray'))
        pl.show()

doMultipleLayers = False

np.random.seed(1)

# Training data.
labels_train = ml_common.loadMNISTLabels(r"common/train-labels-idx1-ubyte")
images_train = ml_common.loadMNISTImages(r"common/train-images-idx3-ubyte")
imShape = (images_train.shape[1], images_train.shape[2])
images_train = images_train.reshape(images_train.shape[0], images_train.shape[1]*images_train.shape[2])

# "Real" data.
labels_real = ml_common.loadMNISTLabels(r"common/t10k-labels-idx1-ubyte")
images_real = ml_common.loadMNISTImages(r"common/t10k-images-idx3-ubyte")
images_real = images_real.reshape(images_real.shape[0], images_real.shape[1]*images_real.shape[2])

if not doMultipleLayers:
    rbm = rbm.RBM(images_train.shape[1], 150, actType=1)
else:
    rbms = multrbm.MultRBM([images_train.shape[1], 300, 10], actType=1)
    rbms.rbms[0].numBatchesPerDataSize = 100
    rbms.rbms[-1].numBatchesPerDataSize = 5

if True:
    if not doMultipleLayers:
        rbm.learn(images_train, maxIter=2000, rate=0.03, wDecay=0.0005)
        rbm.write(r"/tmp/rbm.pickle")
    else:
        rbms.trainLayer(images_train, layerNum=0, maxIter=1000, rate=0.03, wDecay=0.0005)
        rbms.trainLayer(images_train, layerNum=1, maxIter=1000, rate=0.03, wDecay=0.0005)
        rbms.write(r"/tmp/multrbm.pickle")
else:
    if not doMultipleLayers:
        rbm = rbm.read(r"/tmp/rbm.pickle")
        np.random.set_state(rbm.rngState)
    else:
        rbms = rbms.read(r"/tmp/multrbm.pickle")
        np.random.set_state(rbms.rngState)

outfile = "" #r"/tmp/weights.png"
if not doMultipleLayers:
    viewWeights(rbm, imShape, outfile=outfile)
else:
    viewWeights(rbms.rbms[0], imShape, outfile=outfile)

# Use hidden layer as input to multinomial regression.
if not doMultipleLayers:
    hidden_train = np.zeros((images_train.shape[0], rbm.weights.shape[1]-1), dtype=np.int32)
    hidden_real  = np.zeros((images_real.shape[0],  rbm.weights.shape[1]-1), dtype=np.float64)
    for i in range(hidden_train.shape[0]):
        hidden_train[i,:] = rbm.dopass(np.array([images_train[i,:]]), forward=True)
    for i in range(hidden_real.shape[0]):
        hidden_real[i,:] = rbm.dopass(np.array([images_real[i,:]]), forward=True)
    #hist = np.sum(hidden_train, axis=0)
else:
    hidden_train = np.zeros((images_train.shape[0], rbms.numNodes[-1]), dtype=np.int32)
    hidden_real  = np.zeros((images_real.shape[0],  rbms.numNodes[-1]), dtype=np.float64)
    for i in range(hidden_train.shape[0]):
        hidden_train[i,:] = rbms.dopass(np.array([images_train[i,:]]), 0, -1)
    for i in range(hidden_real.shape[0]):
        hidden_real[i,:] = rbms.dopass(np.array([images_real[i,:]]), 0, -1)

print("Training multlogistic regression...")

t0 = 0.005 * np.random.randn(hidden_train.shape[1] * 10)
nc = int(t0.size/hidden_train.shape[1])
t = multlogistic.logist_reg(hidden_train, labels_train, multlogistic.mult_logist_cost, t0, lamb=1e-4, nc=nc, maxiter=300, disp=False)

print("Predicting results...")

prob = multlogistic.mult_logist_predict(t.reshape(nc, hidden_real.shape[1]), hidden_real, nc)
pred = np.argmax(prob, axis=0)
diff = labels_real - pred

[ print(diff[i]) for i in range(10) ]
wh = (diff == 0)
print("# data: %s, accuracy: %s" % (diff.size, np.mean(wh)))

#viewErrors(diff, prob, images_real, labels_real, imShape)

