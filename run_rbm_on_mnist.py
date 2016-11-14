#!/usr/bin/env python

import os, sys, ml_common, rbm, rbm_stacked, multlogistic
import numpy as np

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

def remapOutputNodes(labels_train, labels_real, hidden_train):

    numClasses = np.max(labels_train)+1

    mapping = np.zeros((numClasses,numClasses), dtype=np.int32)
    for i in range(labels_train.size):
        mapping[labels_train[i], np.argmax(hidden_train[i,:])] += 1

    finalmap = np.zeros(mapping.shape[0], dtype=np.uint8)
    for i in range(mapping.shape[0]):
        indi, indj = np.unravel_index(mapping.argmax(), mapping.shape)
        mapping[indi,:] = 0
        mapping[:,indj] = 0
        finalmap[indi] = indj

    #for i in range(mapping.shape[0]):
        #print("label", i, "from node", finalmap[i])

    return finalmap[labels_train], finalmap[labels_real]

doReadRBMFromFile   = False
doTrainRBM          = True
doMultipleLayers    = False
doBackProp          = False
initBackPropWithRBM = False
actType = 'Logistic'
#actType = 'RectiLinear'

tmpdir = r"/tmp"
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

# Train the RBM.
if not doMultipleLayers:
    if doReadRBMFromFile:
        rbm = rbm.RBM(0, 0).read(os.path.join(tmpdir, r"rbm.pickle"))
        np.random.set_state(rbm.rngState)
    else:
        rbm = rbm.RBM(images_train.shape[1], 150, actType=actType, batchSize=500)
    if doTrainRBM:
        rbm.learn(images_train, maxIter=20000, rate=0.01, wDecay=0.01)
        rbm.write(os.path.join(tmpdir, "rbmb.pickle"))

    rbm.viewWeights(imShape, outfile=os.path.join(tmpdir, r"weights.png"))

else:
    if doReadRBMFromFile:
        rbms = rbm_stacked.MultRBM([]).read(os.path.join(tmpdir, r"multrbm.pickle"))
        np.random.set_state(rbms.rngState)
    else:
        rbms = rbm_stacked.MultRBM([images_train.shape[1], 200, 10], actType=actType)

    if doTrainRBM:
        for i in range(len(rbms.rbms)):
            #rbms.rbms[i] = rbms.rbms[i].read(os.path.join(tmpdir, r"rbm_"+str(i)+".pickle"))
            rbms.trainLayer(images_train, layerNum=i, maxIter=20000, rate=0.01, wDecay=0.01)
            #rbms.rbms[i].write(os.path.join(tmpdir, r"rbm_"+str(i)+"b.pickle"))
        rbms.write(os.path.join(tmpdir, r"multrbmb.pickle"))

    rbms.rbms[0].viewWeights(imShape, outfile=os.path.join(tmpdir, r"weights.png"))

    if doBackProp:

        print("Training with backpropagation...")

        if initBackPropWithRBM:
            hidden_train = rbms.dopass(images_train, 0, -1, returnProbs=True)
            labels_train, labels_real = remapOutputNodes(labels_train, labels_real, hidden_train)

        #rbms = rbms.read(os.path.join(tmpdir, r"multrbm_backprop.pickle"))
        rbms.backprop(images_train, labels_train, batchSize=10, rate=0.1, maxIter=100000)
        rbms.write(os.path.join(tmpdir, r"multrbm_backpropb.pickle"))

    rbms.rbms[0].viewWeights(imShape, outfile=os.path.join(tmpdir, r"weightsAfterBP.png"))

    pred = rbms.dopass(images_real, 0, -1, booleanNodes=False, returnProbs=True)
    pred = np.argmax(pred, axis=1)

if not doMultipleLayers or not doBackProp:

    # Use hidden layer as input to multinomial regression.
    if not doMultipleLayers:
        hidden_train = rbm.dopass(images_train, forward=True)
        hidden_real = rbm.dopass(images_real, forward=True)
    else:
        hidden_train = rbms.dopass(images_train, 0, -1)
        hidden_real = rbms.dopass(images_real, 0, -1)

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

