#!/usr/bin/env python

import struct
import numpy as np

# Returns numpy array of size (numImag, 28, 28).
def loadMNISTImages(infile):

    try:
        file = open(infile, 'rb')
    except IOError:
        print("Could not open file: "+infile)
        return

    magNum = (struct.unpack(">I", file.read(4)))[0]
    assert magNum == 2051, "Bad magic number in "+infile

    numImag = (struct.unpack(">I", file.read(4)))[0]
    numRows = (struct.unpack(">I", file.read(4)))[0]
    numCols = (struct.unpack(">I", file.read(4)))[0]
    num = numImag*numRows*numCols

    #images = np.array(struct.unpack(">"+str(num)+"B", file.read(num)))
    images = np.fromfile(file, dtype=np.dtype('>B'))
    images.shape = (numImag, numRows, numCols)

    file.close()

    return images.astype(np.float64)/255.

# Returns numpy array containing the image labels.
def loadMNISTLabels(infile):

    try:
        file = open(infile, 'rb')
    except IOError:
        print("Could not open file: "+infile)
        return

    magNum = (struct.unpack(">I", file.read(4)))[0]
    assert magNum == 2049, "Bad magic number in "+infile

    numLabels = (struct.unpack(">I", file.read(4)))[0]
    #labels = np.array(struct.unpack(">"+str(numLabels)+"B", file.read(numLabels)))
    labels = np.fromfile(file, dtype=np.dtype('>B'))

    file.close()

    return labels


