import numpy as np

def quantizationCube(shape, quality=75, mode="y"):
    # takes in the shape of the tensor block and outputs a quantization tensor of same shape
    # shape: frames x rows x cols
    qTensor = np.zeros(shape)
    for frame in range(shape[0]):
        for row in range(shape[1]):
            for col in range(shape[2]):
                qTensor[frame, row, col] = qVal(frame, row, col, mode)

    if quality < 1 or quality > 100:
        raise Exception("Quality factor must be in range [1,100].")

    scalar = 5000 / quality if quality < 50 else 200 - 2 * quality # formula for scaling by quality factor
    qTensor = qTensor * scalar / 100. # scale the quantization matrix
    qTensor[qTensor<1.] = 1. # do not divide by numbers less than 1
    return qTensor.astype(int)

def qVal(frame, row, col, mode="y"):
    # quantization for a single chroma pixel(to be mapped over a frame of pixels)
    # hyperparameters below are to be tuned
    # algorithm credited to https://barbany.github.io/doc/3d-dct.pdf
    if mode=="y":
        Ai = 100
        Ao = 100
        Bi = 0.05
        Bo = 0.1
        C = 16
    else:
        Ai = 100
        Ao = 100
        Bi = 0.05
        Bo = 0.1
        C = 16
    poly = (row + 1) * (col + 1) * (frame + 1)
    q = 0
    if poly <= C:
        q = (Ai*(1 - ((np.exp(-Bi * poly)/(np.exp(-Bi)))))) + 1
    else:
        q = (Ao*(1 - ((np.exp(-Bo * poly))))) + 1
    return q

def difference(frames):
    # frame differencing
    tp = type(frames[0][0][0])
    differenced = np.zeros(np.shape(frames))
    differenced[0] = frames[0]
    for f in range(1, len(frames)):
        differenced[f] = frames[f] - frames[f-1]
    return differenced.astype(tp)

def idifference(differenced):
    # inverse frame differencing
    tp = type(differenced[0][0][0])
    frames = np.zeros(np.shape(differenced))
    frames[0] = differenced[0]
    for f in range(1, len(differenced)):
        frames[f] = frames[f-1] + differenced[f]
    return frames.astype(tp)

def spiralEncode(shape):
    # reorder tensor by increasing index sum
    # returns encoding array and decoding array
    maxIdx = np.prod(shape)
    maxSum = np.sum(shape)+1
    idxMap = np.arange(0,maxIdx).reshape(shape)
    idxs1D = []
    for sum in range(3, maxSum):
        idxs3D = findTrios(shape, sum)
        idxs1D.extend(idxMap[tuple(zip(*idxs3D))])
    spiral = np.array(idxs1D)
    unspiral = np.array([np.where(spiral==i)[0][0] for i in range(maxIdx)])
    return spiral,unspiral

def findTrios(shape, sum):
    # find all index combinations of a given index sum (x+y+z)
    combos = []
    for frame in range(1, shape[0]+1):
        for row in range(1, shape[1]+1):
            for col in range(1, shape[2]+1):
                if frame + row + col == sum:
                    combos.append((frame-1,row-1,col-1))
    return combos
