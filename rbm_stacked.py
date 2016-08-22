#!/usr/bin/env python

import rbm, pickle
import numpy as np

class MultRBM:

    # numNodes is a list of nodes in each layer starting with the visible nodes.
    def __init__(self, numNodes, actType=1):

        self.numNodes = numNodes
        self.rbms = [ rbm.RBM(numNodes[i], numNodes[i+1], actType=actType) for i in range(len(numNodes)-1) ]

    def trainLayer(self, dataIn, layerNum=0, rate=0.1, maxIter=1000, wDecay=0, verbose=True):

        dataProp = np.copy(dataIn)

        for i in range(layerNum):
            dataProp = self.rbms[i].dopass(dataProp, forward=True)

        self.rbms[layerNum].learn(dataProp, rate=rate, maxIter=maxIter, wDecay=wDecay, verbose=verbose)

    # Layer 0 is the visible layer.
    def dopass(self, dataIn, fromLayer, toLayer, returnProbs=False):

# FIXME. returnProbs.
        if returnProbs:
            raise NotImplementedError("Returning probabilities from MultRBM.dopass() is not implemented.")

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
        for i in order:
            data = self.rbms[i].dopass(data, forward=bool(direc+1))

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

