from compress_functions import *
from differencing import *
from sklearn.decomposition import PCA

def encode_video_block(block, shape, mode="y", quality=75):
    # Input: a 3D array, block, representing an image block over several frames
    #         a string, mode, ("y" for luma, "c" for chroma)
    #         an int, quality, the JPEG quality factor in range [1,100] (defaults to 75)
    # Output: dc_bits, ac_bits, arrays of coefficients for each frame
    #           a bitarray, dc_bits, of Huffman encoded DC coefficients
    #           a bitarray, ac_bits, of Huffman encoded AC coefficients
    #spiral = spiralEncode((shape[0]-1, shape[1], shape[2]))[0]
        
    spiral = spiralEncode(shape)[0]
    qCube = quantizationCube(shape, quality, mode)
    #blockDiff = difference(block)
    #intra = dct2(blockDiff[0])
    #intraQ = quantize(intra, mode, quality)
    #inter = blockDiff[1:]
    #intraZ = zigzag(intraQ)
    #intraZ = np.concatenate((np.array(intraZ),np.array(inter.flatten()[spiral])))
    blockTrans = dct2_3d(block)
    blockQuant = np.round(blockTrans/qCube).astype(int)
    #blockQuant = np.array([quantize(block_c, mode, quality) for block_c in blockTrans])
    #print(blockQuant)
    blockSpiral = blockQuant.flatten()[spiral]
    #blockSpiral = np.array([zigzag(frame) for frame in blockQuant]).flatten()
    blockZRLE = zrle(blockSpiral)
    #blockZRLE = zrle(intraZ.astype(int))
    blockDC = bitarray(encode_huffman(blockZRLE[0], mode)) # DC
    blockAC = ''.join(encode_huffman(v, mode) for v in blockZRLE[1:]) # AC
    return blockDC, blockAC

def encode_video(video, blockDepth=4, quality=75):
    #  mirror pad, chroma downsampling (perform on each frame)
    T_orig, M_orig, N_orig = video.shape[:3]
    T = T_orig + (blockDepth - (T_orig % blockDepth))
    Y_vec, Cb_vec, Cr_vec = [], [], []
    for img in video:
        img = mirror_pad(img[:,:,0:3])
        M, N = img.shape[0:2]
        im_ycbcr = RGB2YCbCr(img)
        Y_vec.append(im_ycbcr[:,:,0])
        Cb_vec.append(im_ycbcr[:,:,1])
        Cr_vec.append(im_ycbcr[:,:,2])

    Y_vec = np.array(Y_vec)
    Cb_vec = np.array(Cb_vec)
    Cr_vec = np.array(Cr_vec)

    

    # Y component
    Y_dc_bits = bitarray()
    Y_ac_bits = bitarray()

    tempRatio = 3
    spatialRatio = 1.1
    
    # Temporal PCA
    temporalY = np.transpose(Y_vec, (2,1,0))
    pca = PCA(n_components=min(np.shape(temporalY)[1:])//tempRatio)
    pcaImg = np.zeros(np.shape(temporalY))
    for f in range(len(temporalY)):
        f_red = pca.fit_transform(temporalY[f])
        pcaImg[f] = pca.inverse_transform(f_red)
    temporalY = np.transpose(pcaImg, (2,1,0))

    # # Spatial PCA
    # Y = np.zeros(np.shape(temporalY))
    # pca = PCA(n_components=int(min(np.shape(temporalY)[1:])/spatialRatio))
    # for f in range(len(temporalY)):
    #     f_red = pca.fit_transform(temporalY[f])
    #     Y[f] = pca.inverse_transform(f_red)
    Y_vec = temporalY
    for i in np.r_[0:M:8]:
        for j in np.r_[0:N:8]:
            for k in np.r_[0:T_orig:blockDepth]:
                block = Y_vec[k:k+blockDepth,i:i+8,j:j+8]
                block = frame_pad(block, blockDepth)
                dc_bits, ac_bits = encode_video_block(block, np.shape(block), "y", quality)
                Y_dc_bits.extend(dc_bits)
                Y_ac_bits.extend(ac_bits)
    # Cb component
    Cb_dc_bits = bitarray()
    Cb_ac_bits = bitarray()

    # Temporal PCA
    temporalCb = np.transpose(Cb_vec, (2,1,0))
    print(np.shape(temporalCb))
    pca = PCA(n_components=min(np.shape(temporalCb)[1:])//tempRatio)
    pcaImg = np.zeros(np.shape(temporalCb))
    for f in range(len(temporalCb)):
        f_red = pca.fit_transform(temporalCb[f])
        pcaImg[f] = pca.inverse_transform(f_red)
    temporalCb = np.transpose(pcaImg, (2,1,0))
    
    # Spatial PCA
    Cb_vec = np.zeros(np.shape(temporalCb))
    pca = PCA(n_components=int(min(np.shape(temporalCb)[1:])/spatialRatio))
    for f in range(len(temporalCb)):
        f_red = pca.fit_transform(temporalCb[f])
        Cb_vec[f] = pca.inverse_transform(f_red)

    Cb = []
    for f in Cb_vec:
        Cb.append(chroma_downsample(temporalCb))
    Cb_vec = np.array(Cb)

    for i in np.r_[0:M//2:8]:
        for j in np.r_[0:N//2:8]:
            for k in np.r_[0:T_orig:blockDepth]:
                block = Cb_vec[k:k+blockDepth,i:i+8,j:j+8]
                block = frame_pad(block, blockDepth)
                dc_bits, ac_bits = encode_video_block(block, np.shape(block), "c", quality)
                Cb_dc_bits.extend(dc_bits)
                Cb_ac_bits.extend(ac_bits)

    # Cr component
    Cr_dc_bits = bitarray()
    Cr_ac_bits = bitarray()

    # Temporal PCA
    temporalCr = np.transpose(Cr_vec, (2,1,0))
    pca = PCA(n_components=min(np.shape(temporalCr)[1:])//tempRatio)
    pcaImg = np.zeros(np.shape(temporalCr))
    for f in range(len(temporalCr)):
        f_red = pca.fit_transform(temporalCr[f])
        pcaImg[f] = pca.inverse_transform(f_red)
    temporalCr = np.transpose(pcaImg, (2,1,0))
    Cr_vec = temporalCr
    # Spatial PCA
    Cr_vec = np.zeros(np.shape(temporalCr))
    pca = PCA(n_components=int(min(np.shape(temporalCr)[1:])/spatialRatio))
    for f in range(len(temporalCr)):
        f_red = pca.fit_transform(temporalCr[f])
        Cr_vec[f] = pca.inverse_transform(f_red)
    
    Cr = []
    for f in Cr_vec:
        Cr.append(chroma_downsample(temporalCr))
    Cr_vec = np.array(Cr)

    for i in np.r_[0:M//2:8]:
        for j in np.r_[0:N//2:8]:
            for k in np.r_[0:T_orig:blockDepth]:
                block = Cr_vec[k:k+blockDepth,i:i+8,j:j+8]
                block = frame_pad(block, blockDepth)
                dc_bits, ac_bits = encode_video_block(block, np.shape(block), "c", quality)
                Cr_dc_bits.extend(dc_bits)
                Cr_ac_bits.extend(ac_bits)

    bits = (Y_dc_bits, Y_ac_bits, Cb_dc_bits, Cb_ac_bits, Cr_dc_bits, Cr_ac_bits)
    return bits

def decode_video_block(dc_gen, ac_gen, blockDepth=4, mode="y", quality=75):
    # Inputs: a generator, dc_gen, that yields decoded Huffman DC coefficients
    #         a generator, ac_gen, that yields decoded Huffman AC coefficients
    #         a string, mode, ("y" for luma, "c" for chroma)
    #         an int, quality, the JPEG quality factor in range [1,100] (defaults to 75)
    block_cqzr = [next(dc_gen)] # initialize list by yielding from DC generator
    while block_cqzr[-1] != (0,0):
        block_cqzr.append(next(ac_gen)) # append to list by yielding from AC generator until (0,0) is encountered
    shape = (blockDepth, 8, 8)
    #interShape = (shape[0]-1, shape[1], shape[2])
    #unspiral = spiralEncode(interShape)[1]
    unspiral = spiralEncode(shape)[1]
    qCube = quantizationCube(shape, quality, mode)

    block_cqz = np.array(unzrle(block_cqzr, blockDepth))
    #intra = block_cqz[:np.prod(shape[1:3])]
    #intraF = unzigzag(intra)
    #inter = block_cqz[np.prod(shape[1:3]):]
    blockUnspiral = block_cqz[unspiral]
    blockUnspiral = blockUnspiral.reshape(shape)
    #blockUnspiral = np.array([unzigzag(block_cqz[i:i+64]) for i in range(0,len(block_cqz),64)]).reshape(shape)
    blockUnQ = np.multiply(blockUnspiral,qCube)
    #blockUnQ = np.array([unquantize(block_cq, mode, quality) for block_cq in blockUnspiral])
    #intraQ = unquantize(intraF, mode, quality)
    #intra = idct2(intraQ)
    #block = np.concatenate((np.array([intra]), blockUnspiral), axis=0)
    block = idct2_3d(blockUnQ)
    #block = idifference(blockUnQ)
    return block

def decode_video(bits, T, M, N, blockDepth=4, quality=75):
    # T:frames, M:rows, N:cols
    Y_dc, Y_ac, Cb_dc, Cb_ac, Cr_dc, Cr_ac = bits

    #     Y_dc_bits = np.array(Y_dc)
    #     Y_ac_bits = np.array(Y_ac)
    #     Cb_dc_bits = np.array(Cb_dc)
    #     Cb_ac_bits = np.array(Cb_ac)
    #     Cr_dc_bits = np.array(Cr_dc)
    #     Cr_ac_bits = np.array(Cr_ac)

    M_orig = M # save original image dimensions
    N_orig = N
    T_orig = T
    M = M_orig + ((16 - (M_orig % 16)) % 16) # dimensions of padded image
    N = N_orig + ((16 - (N_orig % 16)) % 16)
    T = T_orig + (blockDepth - (T_orig % blockDepth))
    num_blocks = (M * N * T) // (64*blockDepth) # number of blocks

    # decode each block at a time
    # Y component
    Y = np.empty((T, M, N))
    Y_dc_gen = decode_huffman(bitarray(Y_dc).to01(), "dc", "y")
    Y_ac_gen = decode_huffman(bitarray(Y_ac).to01(), "ac", "y")
    for m in np.r_[0:M:8]:
        for n in np.r_[0:N:8]:
            for f in np.r_[0:T_orig:blockDepth]:
                block = decode_video_block(Y_dc_gen, Y_ac_gen, blockDepth, "y", quality)
                Y[f:f+blockDepth, m:m+8, n:n+8] = block

    # Cb component
    Cb = np.empty((T, M//2, N//2))
    Cb_dc_gen = decode_huffman(bitarray(Cb_dc).to01(), "dc", "c")
    Cb_ac_gen = decode_huffman(bitarray(Cb_ac).to01(), "ac", "c")
    for m in np.r_[0:M//2:8]:
        for n in np.r_[0:N//2:8]:
            for f in np.r_[0:T_orig:blockDepth]:
                block = decode_video_block(Cb_dc_gen, Cb_ac_gen, blockDepth, "c", quality)
                Cb[f:f+blockDepth, m:m+8, n:n+8] = block

    # Cr component
    Cr = np.empty((T, M//2, N//2))
    Cr_dc_gen = decode_huffman(bitarray(Cr_dc).to01(), "dc", "c")
    Cr_ac_gen = decode_huffman(bitarray(Cr_ac).to01(), "ac", "c")
    for m in np.r_[0:M//2:8]:
        for n in np.r_[0:N//2:8]:
            for f in np.r_[0:T_orig:blockDepth]:
                block = decode_video_block(Cr_dc_gen, Cr_ac_gen, blockDepth, "c", quality)
                Cr[f:f+blockDepth, m:m+8, n:n+8] = block

    video = []
    for i in range(T_orig):
        Cb_f=chroma_upsample(Cb[i])
        Cr_f=chroma_upsample(Cr[i])
        img = YCbCr2RGB(np.stack((Y[i],Cb_f,Cr_f), axis=-1))
        video.append(img[0:M_orig,0:N_orig,:])

    return np.array(video)

"""

3d spiral, 3d quantization

psnr:  27.920130767250413
The compression ratio is: 9.017130398671096
psnr x compression:  251.75945987624584
56 packets

frame-wise zigzag, 3d quantization

psnr:  27.917488729895933
The compression ratio is: 8.040455471209036
psnr x compression:  224.46932500070835
63 packets

frame-wise zigzag, frame-wise quantization

psnr:  28.06686379225472
The compression ratio is: 6.828066037735849
psnr x compression:  191.64239944565244
73 packets

"""


#
