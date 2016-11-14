#!/usr/bin/env python

import subprocess, json, os, sys
import rbm as rbmclass
import rbm_stacked as multrbmclass
import tifffile as tiff
import numpy as np
import pylab as pl
from skimage import exposure

def getLandsatData(lat, lon, bands, download=True, tmpdir=""):

    tmpfile = os.path.join(tmpdir, r"landsat_json.txt")

    if download:
        # --cloud specifies maximum cloud percentage.
        subprocess.call(['landsat search --cloud 4 --lat '+str(lat)+' --lon '+str(lon)+' --geojson > '+tmpfile], shell=True)

    data = json.loads(open(tmpfile).read())
    _bandstr = ''.join([str(b) for b in bands])
    sceneIDs = [ dict['properties']['sceneID'] for dict in data['features'] ]

    if download:
        subprocess.call(['landsat download -d '+download_dir+' '+' '.join(sceneIDs)+' --bands '+_bandstr], shell=True)

    files = [ [ os.path.join(download_dir, sceneID, sceneID+"_B"+band+".TIF") for band in _bandstr ] for sceneID in sceneIDs ]

    return data, files

def imageFromFiles(files, bands):

    image = np.array([ tiff.imread(files[i]) for i in range(len(bands)-1, -1, -1) ])  # uint16
    image = np.rollaxis(image, 0, 3)
    image = image.astype(np.float64)
    image -= np.min(image)
    image *= 255.9999/np.max(image)
    image = image.astype(np.uint8)

    return image

def displayImage(image, metadata=None, title="", outfile="", cmap=None, dpi=350):

    if metadata is not None:
        geometry = [ dct['geometry']['coordinates'] for dct in metadata['features'] ]
        x = [ geometry[0][0][i][0] for i in range(4) ]
        y = [ geometry[0][0][i][1] for i in range(4) ]
        extent = (np.min(x), np.max(x), np.min(y), np.max(y))
    else:
        extent = None

    fig, ax = pl.subplots()
    ax.imshow(image, aspect=1, interpolation='none', extent=extent, cmap=cmap)
    if metadata is not None:
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    if title != "":
        ax.set_title(title)

    if outfile != "":
        pl.savefig(outfile, dpi=dpi)
    else:
        pl.show()

# loc[i][1] is the fast varying index.
def patchesFromImage(image, shape):

    patches = image[0:(image.shape[0]//shape[0])*shape[0], 0:(image.shape[1]//shape[1])*shape[1], :]

    locX = np.arange(0, patches.shape[0], shape[0])
    locY = np.arange(0, patches.shape[1], shape[1])

    patches = patches.reshape(patches.shape[0]//shape[0], shape[0], -1, shape[1], patches.shape[2])\
                     .swapaxes(1,2)\
                     .reshape(-1, shape[0], shape[1], patches.shape[2])

    loc = np.array([ (lX, lY) for lX in locX for lY in locY ])

    return patches, loc

def imageFromPatches(patches, loc, mode='full'):

    maxloc0 = max([l[0] for l in loc])
    maxloc1 = max([l[1] for l in loc])

    if mode == 'full':
        image = np.zeros((maxloc0+patches.shape[1], maxloc1+patches.shape[2], patches.shape[3]), dtype=np.uint8)
        for i in range(loc.shape[0]):
            l, p = loc[i], patches[i]
            image[l[0]:l[0]+patches.shape[1], l[1]:l[1]+patches.shape[2], :] = p
    else:
        image = np.zeros((maxloc0//patches.shape[1]+1, maxloc1//patches.shape[2]+1, patches.shape[3]), dtype=np.uint8)
        for i in range(loc.shape[0]):
            l, p = loc[i], patches[i]
            image[l[0]//patches.shape[1], l[1]//patches.shape[2], :] = p[0,0,:]

    return image

def filterEdgePatches(patches, loc):

    badIndices = set()

    maxVal = np.array([ np.max(patch) for patch in patches ])
    wh = (np.where(maxVal == 0))[0]

    numPatchesY = len((np.where(np.array([l[0] for l in loc]) == 0))[0])
    numPatchesX = len(patches)//numPatchesY

    # Add black patches and their neighbors.
    for w in wh:
        badIndices.add(w)
        if loc[w][0] > 0:
            badIndices.add(w-numPatchesY)
        if loc[w][0] < loc[-1][0]:
            badIndices.add(w+numPatchesY)
        if loc[w][1] > 0:
            badIndices.add(w-1)
        if loc[w][1] < loc[-1][1]:
            badIndices.add(w+1)
        if loc[w][0] > 0 and loc[w][1] > 0:
            badIndices.add(w-1-numPatchesY)
        if loc[w][0] > 0 and loc[w][1] < loc[-1][1]:
            badIndices.add(w-numPatchesY+1)
        if loc[w][0] < loc[-1][0] and loc[w][1] > 0:
            badIndices.add(w+numPatchesY-1)
        if loc[w][0] < loc[-1][0] and loc[w][1] < loc[-1][1]:
            badIndices.add(w+1+numPatchesY)

    # Add image border.
    c = (numPatchesX-1)*numPatchesY
    for i in range(numPatchesY):
        badIndices.add(i)
        badIndices.add(c+i)
    c = numPatchesY-1
    for i in range(numPatchesX):
        iY = i*numPatchesY
        badIndices.add(iY)
        badIndices.add(c+iY)

    #for elem in badIndices:
        #patches[elem,:,:,:] = 255

    goodIndices = set()
    for i in range(patches.shape[0]):
        goodIndices.add(i)
    goodIndices -= badIndices

    patches = np.array([ patches[i,:,:,:] for i in goodIndices ])
    loc = np.array([ loc[i,:] for i in goodIndices ])

    return patches, loc

def bitlist2int(bitlist):

    res = 0
    for bit in bitlist:
        res = (res << 1) | bit

    return res

def runRBMs(patches, loc, tmpdir="", showClassMap=True, showPatchClasses=True, matchNumPatches=None):

    data = patches.reshape(patches.shape[0],-1).astype(np.float64)
    for i in range(data.shape[0]):
        data[i,:] -= np.min(data[i,:])
        data[i,:] /= np.max(data[i,:])

    print("Training RBM...")

    rbms = multrbmclass.MultRBM([data.shape[1], 120, 120, 10], actType='Logistic', batchSize=500)

    for i in range(3):
        #rbms.rbms[i] = rbms.rbms[i].read(os.path.join(tmpdir, r"rbm"+str(i)+".pkl"))
        numIter = 10000 if i == 0 else 30000
        rbms.trainLayer(data, layerNum=i, rate=0.01, maxIter=numIter, wDecay=0.01, verbose=True)
        #rbms.rbms[i].write(os.path.join(tmpdir, r"rbm"+str(i)+"b.pkl"))

    # TODO. Read some labels here.
    #labels = np.random.uniform(size=data.shape[0], high=rbms.rbms[-1].weights.shape[1]-1).astype(np.uint8)
    #rbms.backprop(data, labels, rate=0.01, maxIter=3000, batchSize=100)

    rbms.rbms[0].viewWeights((patches.shape[1], patches.shape[2], patches.shape[3]), grayScale=False,\
                             outfile=os.path.join(tmpdir, r"landsat_weights.png"))
    dataFinal = rbms.dopass(data, 0, -1, booleanNodes=True, returnProbs=True)

    print("Outputting results...")

    argmax = np.argmax(dataFinal, axis=1)
    numH = rbms.rbms[-1].weights.shape[1]-1
    #argmax = np.zeros(dataFinal.shape[0])
    #for i in range(argmax.size):
        #argmax[i] = bitlist2int(dataFinal[i,:])
    #numH = 2**(rbms.rbms[-1].weights.shape[1]-1)

    uniq = np.unique(argmax)
    print("num unique:", uniq.size)

    for i in range(numH):
        num = len((np.where(argmax == i))[0])
        print("num "+str(i), num)

    if showPatchClasses:
        imShape = (patches.shape[1], patches.shape[2], patches.shape[3])
        for i in range(numH):
            wh = np.where(argmax == i)
            if len(wh[0]) == 0:
                continue

            #wh = wh[0][np.argsort(np.max(dataFinal[wh], axis=1))]
            wh = wh[0]

            if matchNumPatches is not None:
                wh = wh[0:matchNumPatches]
            ny = int(np.ceil(np.sqrt(len(wh))))
            nx = len(wh)//ny
            tiledImage = np.zeros((nx*imShape[0], ny*imShape[1], imShape[2]), dtype=np.float64)
            tempImage = (data[wh,:]*255.9999).astype(np.uint8).reshape(len(wh), imShape[0], imShape[1], imShape[2])
            for j in range(nx):
                for k in range(ny):
                    tiledImage[j*imShape[0]:(j+1)*imShape[0],k*imShape[1]:(k+1)*imShape[1],:] = tempImage[j*ny+k,:,:,:]

            #displayImage(tiledImage, metadata=None, title="Class "+str(i))
            rbms.rbms[-1].viewWeights(imShape, tiledImage=tiledImage, outfile=os.path.join(tmpdir, r"landsat_class_"+str(i)+".png"))

    if showClassMap:
        tiles = np.zeros((patches.shape[0], patches.shape[1], patches.shape[2], 1), dtype=np.float64)
        for i in range(patches.shape[0]):
            tiles[i,:,:,0] = 1+argmax[i]
        tiles *= 255.9999/np.max(tiles)
        tiles = tiles.astype(np.uint8)
        recon = imageFromPatches(tiles, loc, mode='small')
        displayImage(np.squeeze(recon), metadata=None, cmap=pl.get_cmap('gist_earth'), outfile=os.path.join(tmpdir, r"landsat_classification.png"))


#####################
## Start of script ##
#####################

lat, lon = 37.7749, -122.4194
bands = [2, 3, 4]
images_to_analyse = [6]
do_download = False
download_dir = "../../landsat/downloads"
tmpdir = r"/tmp"
np.random.seed(1)

print("Getting data...")
metadata, files = getLandsatData(lat, lon, bands, download=do_download, tmpdir=tmpdir)

date  = [ dct['properties']['date'] for dct in metadata['features'] ]
cloud = [ dct['properties']['cloud'] for dct in metadata['features'] ]
print("#, Date, Cloud %")
for d, c in zip(date, cloud):
    print(d, c)

for i in images_to_analyse:

    print("Reading image "+str(i))
    try:
        image = imageFromFiles(files[i], bands)
    except FileNotFoundError:
        continue

    patches, loc = patchesFromImage(image, (20, 20, 3))
    patches, loc = filterEdgePatches(patches, loc)

    p1, p2 = np.percentile(image, (0, 99.9))
    image_rescale = exposure.rescale_intensity(image.astype(np.float64), in_range=(p1, p2))
    displayImage(image_rescale, metadata=metadata, outfile=os.path.join(tmpdir, r"image_"+str(i)+".png"), dpi=650)

    print("Data shape:", patches.shape)

    runRBMs(patches, loc, tmpdir=tmpdir, matchNumPatches=100)


