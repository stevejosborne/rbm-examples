#!/usr/bin/env python

import rbm, pickle
import numpy as np

class MultRBM:

    # numNodes is a list of nodes in each layer starting with the visible nodes.
    def __init__(self, numNodes, batchSize=100, actType='Logistic'):

        self.numNodes = numNodes
        self.actType = actType
        self.batchSize = batchSize
        self.rbms = [ rbm.RBM(numNodes[i], numNodes[i+1], batchSize=self.batchSize, actType=self.actType) for i in range(len(numNodes)-1) ]

    def trainLayer(self, dataIn, layerNum=0, rate=0.1, maxIter=1000, wDecay=0, verbose=True):

        dataProp = np.copy(dataIn)

        for i in range(layerNum):
            dataProp = self.rbms[i].dopass(dataProp, forward=True)

        self.rbms[layerNum].learn(dataProp, rate=rate, maxIter=maxIter, wDecay=wDecay, verbose=verbose)

    # Layer 0 is the visible layer.
    def dopass(self, dataIn, fromLayer, toLayer, booleanNodes=True, returnProbs=False, returnIntermediate=False):

        if fromLayer == -1:
            fromLayer = len(self.numNodes)-1
        if toLayer == -1:
            toLayer = len(self.numNodes)-1
        direc = np.sign(toLayer-fromLayer)
        if direc < 0:
            fromLayer -= 1
            toLayer -= 1

        data = np.copy(dataIn)
        order = np.arange(fromLayer, toLayer, direc)

        if returnIntermediate:
            alldata = [data]

        for i in order:
            data = self.rbms[i].dopass(data, forward=bool(direc+1), returnProbs=returnProbs)
            if returnIntermediate:
                alldata.append(np.copy(data))
            if booleanNodes and (i != order[-1] or not returnProbs):
                data = (data > np.random.rand(data.shape[0], data.shape[1])).astype(np.int8)
        if returnIntermediate:
            return alldata
        else:
            return data

    def calcDeriv(self, act):

        return np.multiply(act, (1.0-act))

    def backprop(self, dataIn, labelsIn, rate=0.1, maxIter=20000, batchSize=10, verbose=True):

        if self.actType != 'Logistic':
            raise NotImplementedError("For backprop function only Logistic activation has been implemented")

        depth = len(self.rbms)
        ratePerSample = rate / batchSize

        labelsIn2D = np.zeros((labelsIn.size, self.rbms[-1].numH), dtype=np.uint8)
        for i in range(labelsIn.size):
            labelsIn2D[i,labelsIn[i]] = 1

        for it in range(maxIter):

            if verbose and it % (maxIter//50) == 0:
                pred = self.dopass(dataIn, 0, -1, booleanNodes=False, returnProbs=True)
                diff = labelsIn - np.argmax(pred, axis=1)
                wh = (diff == 0)
                print("Iteration %s: accuracy is %s" % (it, np.mean(wh)))

            ind = np.random.uniform(size=batchSize, high=dataIn.shape[0]).astype(np.int32)
            labels = labelsIn2D[ind,:]

            forwdHiddenStates = self.dopass(dataIn[ind,:], 0, -1, booleanNodes=False, returnProbs=True, returnIntermediate=True)

            for layer in range(depth-1, -1, -1):
                deriv = self.calcDeriv(forwdHiddenStates[layer+1])
                if layer == depth-1:
                    error = forwdHiddenStates[layer+1] - labels
                else:
                    error = np.dot(delta, self.rbms[layer+1].weights[1::,1::].T)
                delta = np.multiply(error, deriv)

                # Weights shape: (numV,numH). Ignore bias units connected to visible nodes.
                self.rbms[layer].weights[1::,1::] -= ratePerSample * np.dot(forwdHiddenStates[layer].T, delta)
                self.rbms[layer].weights[0,1::] -= ratePerSample * np.sum(delta, axis=0)

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

