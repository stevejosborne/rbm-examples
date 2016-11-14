#!/usr/bin/env python

import pickle
import numpy as np
import pylab as pl

class RBM:

    def __init__(self, numV, numH, actType='Logistic', batchSize=100):

        self.numV = numV
        self.numH = numH
        if actType == 'Logistic':
            self.actFunc = self.actLogistic
        elif actType == 'RectiLinear':
            self.actFunc = self.actRectLinear
        else:
            raise NotImplementedError("Activation type must be Logistic or RectiLinear")

        self.batchSize = batchSize

        # Initialize weights to have mu=0, sigm=X.
        self.weights = 0.01 * np.random.randn(self.numV, self.numH)
        self.weights = np.insert(self.weights, 0, 0, axis=0)
        self.weights = np.insert(self.weights, 0, 0, axis=1)

    def actLogistic(self, x):

        return 1.0 / (1.0 + np.exp(-x))

    def actRectLinear(self, x):

        y = np.copy(x)
        y[y < 0] = 0

        return y

    def normData(self, data):

        mean = np.mean(data, axis=1)
        std  = np.std(data, axis=1)

        for i in range(data.shape[0]):
            data[i,:] -= mean[i]
            data[i,:] /= std[i]

        return data

    # Recommended weight decay parameter: 0.01 to 0.00001.
    def learn(self, dataIn, rate=0.01, maxIter=3000, wDecay=0.001, verbose=True, useAllData=False):

        data = np.copy(dataIn).astype(np.float64)
        data = np.insert(data, 0, 1, axis=1)
        #data = self.normData(data)
        numData = data.shape[0]

        for it in range(maxIter):

            #np.random.seed(0)

            if useAllData:
                subData = data
            else:
                ind = np.random.uniform(size=self.batchSize, high=data.shape[0]).astype(int)
                subData = data[ind,:]

            forwdHiddenIn = np.dot(subData, self.weights)
            forwdHiddenProb = self.actFunc(forwdHiddenIn)
            forwdHiddenState = (forwdHiddenProb > np.random.rand(self.batchSize, self.numH+1)).astype(int)

            brac0 = np.dot(subData.T, forwdHiddenProb)

            bckwdVisibleIn = np.dot(forwdHiddenState, self.weights.T)
            bckwdVisibleProb = self.actFunc(bckwdVisibleIn)
            bckwdVisibleProb[:,0] = 1
            bckwdHiddenAct = np.dot(bckwdVisibleProb, self.weights)
            bckwdHiddenProb = self.actFunc(bckwdHiddenAct)

            brac1 = np.dot(bckwdVisibleProb.T, bckwdHiddenProb)

            #print("weights:", self.weights)
            #print("delta(weights):", self.rate * ((brac0 - brac1) / self.batchSize))

            self.weights += rate * (brac0 - brac1) / self.batchSize
            if wDecay != 0:
                self.weights -= rate * wDecay * self.weights

            if verbose and it % 50 == 0:
                error = np.sum((subData - bckwdVisibleProb)**2)
                print("Iteration %s: error is %s" % (it, error))

    def dopass(self, data, forward=True, returnProbs=False):

        numData = data.shape[0]
        data = np.insert(data, 0, 1, axis=1)

        weights = self.weights if forward else self.weights.T 
        num = self.numH if forward else self.numV

        act = np.dot(data, weights)
        prob = self.actFunc(act)

        if returnProbs:
            return prob[:,1:]   # Remove the bias.
        else:
            state = (prob > np.random.rand(numData, num+1)).astype(np.int8)
            return state[:,1:]

    def generateSample(self, dataIn, numIter=1000):

        data = np.copy(dataIn)

        for j in range(numIter):
            data = self.dopass(data, forward=True)
            data = self.dopass(data, forward=False, returnProbs=True)

        return data

    def write(self, outfile):

        try:
            pfile = open(outfile, "wb")
        except IOError:
            print("Could not open file: "+outfile)
            return

        self.rngState = np.random.get_state()
        pickle.dump(self, pfile)
        pfile.close()

    def read(self, infile):

        try:
            pfile = open(infile, "rb")
        except IOError:
            print("Could not open file: "+infile)
            return

        pclass = pickle.load(pfile)
        pfile.close()

        return pclass

    def factorize(self, n):

        fact = [ [i, n//i] for i in range(1, int(n**0.5)+1) if n % i == 0 ]
        return sorted(list(set([ item for sublist in fact for item in sublist])))

    def imageWeights(self, imShape):

        if len(imShape) == 2:
            imShape = (imShape[0], imShape[1], 1)

        assert(imShape[0]*imShape[1]*imShape[2] == self.numV)

        fact = self.factorize(self.numH)
        ny = fact[len(fact)//2] if len(fact) > 0 else 1
        nx = self.numH//ny
        if max(nx, ny)/min(nx, ny) > 5:
            ny = int(np.ceil(np.sqrt(self.numH)))
            nx = int(np.ceil(self.numH/ny))

        tiledImage = np.zeros((nx*imShape[0], ny*imShape[1], imShape[2]), dtype=np.float64)
        tempWeights = self.weights[1::,1::].reshape(imShape[0], imShape[1], imShape[2], -1)
        for i in range(nx):
            for j in range(ny):
                tiledImage[i*imShape[0]:(i+1)*imShape[0],j*imShape[1]:(j+1)*imShape[1],:] = tempWeights[:,:,:,i*ny+j]

        return np.squeeze(tiledImage)

    def rgb2gray(self, image):
        return np.dot(image[...,:3], [0.299, 0.587, 0.114])

    def viewWeights(self, imShape, outfile="", grayScale=False, normalize=True, tiledImage=None, title=""):

        if tiledImage is None:
            tiledImage = self.imageWeights(imShape)

        if normalize:
            tiledImage -= np.min(tiledImage)
            tiledImage *= 255.9999/np.max(tiledImage)
            tiledImage = tiledImage.astype(np.uint8)

        if len(tiledImage.shape) == 3 and grayScale:
            tiledImage = self.rgb2gray(tiledImage)

        cmap = pl.get_cmap('gray') if (len(imShape) == 2 or grayScale) else None

        fig = pl.figure()
        ax = fig.add_subplot(111, aspect=1)
        ax.imshow(tiledImage, interpolation='none', cmap=cmap)
        ax.autoscale(False)
        for i in range(tiledImage.shape[1]//imShape[1]+2):
            ax.axvline(imShape[1]*i-0.5, color='black', lw=1)
        for i in range(tiledImage.shape[0]//imShape[0]+2):
            ax.axhline(imShape[0]*i-0.5, color='black', lw=1)

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        if title != "":
            ax.set_title(title)

        if outfile == "":
            pl.show()
        else:
            pl.savefig(outfile, dpi=500)
        pl.close('all')

if __name__ == '__main__':

    # Example from: http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/
    rbm = RBM(6, 2, actType='Logistic')
    data = np.array([[1,1,1,0,0,0], [1,0,1,0,0,0], [1,1,1,0,0,0], [0,0,1,1,1,0], [0,0,1,1,0,0], [0,0,1,1,1,0]])
    rbm.learn(data, maxIter=5000)
    print("Weights:", rbm.weights)
    res = rbm.dopass(np.array([[0,0,0,1,1,0]]), forward=True)
    print("Result:", res)

