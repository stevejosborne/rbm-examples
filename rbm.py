#!/usr/bin/env python

import pickle
import numpy as np

class RBM:

    def __init__(self, numV, numH, actType=1, numBatchesPerDataSize=100):

        self.numV = numV
        self.numH = numH
        self.actFunc = self.actRectLinear if actType == 1 else self.actLogistic
        self.numBatchesPerDataSize = numBatchesPerDataSize

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
    def learn(self, dataIn, rate=0.1, maxIter=1000, wDecay=0, verbose=True):

        data = np.copy(dataIn).astype(np.float64)
        data = np.insert(data, 0, 1, axis=1)
        #data = self.normData(data)
        numData = data.shape[0]
        numSubData = data.shape[0]//self.numBatchesPerDataSize   # Size of mini-batch.

        for it in range(maxIter):

            ind = np.random.uniform(size=numSubData, high=data.shape[0]).astype(int)
            subData = data[ind,:]

            forwdHiddenIn = np.dot(subData, self.weights)
            forwdHiddenProb = self.actFunc(forwdHiddenIn)
            forwdHiddenState = (forwdHiddenProb > np.random.rand(numSubData, self.numH+1)).astype(int)

            brac0 = np.dot(subData.T, forwdHiddenProb)

            bckwdVisibleIn = np.dot(forwdHiddenState, self.weights.T)
            bckwdVisibleProb = self.actFunc(bckwdVisibleIn)
            bckwdVisibleProb[:,0] = 1
            bckwdHiddenAct = np.dot(bckwdVisibleProb, self.weights)
            bckwdHiddenProb = self.actFunc(bckwdHiddenAct)

            brac1 = np.dot(bckwdVisibleProb.T, bckwdHiddenProb)

            #print("weights:", self.weights)
            #print("delta(weights):", self.rate * ((brac0 - brac1) / numSubData))

            self.weights += rate * (brac0 - brac1) / numSubData
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

        assert(imShape[0]*imShape[1] == self.numV)

        fact = self.factorize(self.numH)
        ny = fact[len(fact)//2] if len(fact) > 0 else 1
        nx = self.numH//ny

        tiledImage = np.zeros((nx*imShape[0], ny*imShape[1]), dtype=np.float64)
        for i in range(nx):
            for j in range(ny):
                tiledImage[i*imShape[0]:(i+1)*imShape[0],j*imShape[1]:(j+1)*imShape[1]] = self.weights[1::,1+i*ny+j].reshape(imShape[0], imShape[1])

        return tiledImage

if __name__ == '__main__':

    # Example from: http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/
    rbm = RBM(6, 2, actType=1)
    data = np.array([[1,1,1,0,0,0], [1,0,1,0,0,0], [1,1,1,0,0,0], [0,0,1,1,1,0], [0,0,1,1,0,0], [0,0,1,1,1,0]])
    rbm.learn(data, maxIter=5000)
    print("Weights:", rbm.weights)
    res = rbm.dopass(np.array([[0,0,0,1,1,0]]), forward=True)
    print("Result:", res)

