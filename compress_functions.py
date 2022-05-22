import numpy as np
import scipy.fftpack
from scipy.fftpack import dct, idct
from bitarray import bitarray
import queue as Queue
from huffman import *
import os
import math
from os import stat
from PIL import Image
from apng import APNG
import base64

# color scheme conversion functions
def RGB2YCbCr(im_rgb):
    # Input:  a 3D float array, im_rgb, representing an RGB image in range [0.0,255.0]
    # Output: a 3D float array, im_ycbcr, representing a YCbCr image in range [-128.0,127.0]
    b = np.array([-128.0, 0.0, 0.0])
    A = np.array([[0.299, 0.587, 0.114],
                  [-0.168736, -0.331264, 0.5],
                  [0.5, -0.418688, -0.081312]])
    trnsfrm = lambda pixel, A, b: (A @ pixel) + b
    im_ycbcr = np.array([[np.clip(trnsfrm(p, A, b),-128.0, 127.0) for p in row] for row in im_rgb])
    return im_ycbcr

def YCbCr2RGB(im_ycbcr):
    # Input:  a 3D float array, im_ycbcr, representing a YCbCr image in range [-128.0,127.0]
    # Output: a 3D float array, im_rgb, representing an RGB image in range [0.0,255.0]
    A_inv = np.linalg.inv(np.array([[0.299, 0.587, 0.114],
                                    [-0.168736, -0.331264, 0.5],
                                    [0.5, -0.418688, -0.081312]]))
    b = np.array([-128.0, 0.0, 0.0])
    inv_trnsfrm = lambda pixel, A_inv, b: A_inv @ np.subtract(pixel, b)
    im_rgb = np.array([np.array([np.clip(inv_trnsfrm(p, A_inv, b),0.0,255.0) for p in row]) for row in im_ycbcr])
    return im_rgb

# chroma sampling functions
def chroma_downsample(C):
    # Input:  an MxN array, C, of chroma values
    # Output: an (M/2)x(N/2) array, C2, of downsampled chroma values
    C2 = np.array(Image.fromarray(C).resize((np.shape(C)[1]//2, np.shape(C)[0]//2), resample=Image.BILINEAR))
    return C2

def chroma_upsample(C2):
    # Input:  an (M/2)x(N/2) array, C2, of downsampled chroma values
    # Output: an MxN array, C, of chroma values
    C = np.array(Image.fromarray(C2).resize((np.shape(C2)[1]*2, np.shape(C2)[0]*2), resample=Image.BICUBIC))
    return C

# DCT-II functions
def dct2(block):
    # Input:  a 2D array, block, representing an image block
    # Output: a 2D array, block_c, of DCT coefficients
    block_col_trans = scipy.fftpack.dct(block, norm='ortho', axis = 0)
    block_c = scipy.fftpack.dct(block_col_trans, norm='ortho', axis = 1)
    return block_c

def idct2(block_c):
    # Input:  a 2D array, block_c, of DCT coefficients
    # Output: a 2D array, block, representing an image block
    block_col_trans = scipy.fftpack.idct(block_c, norm='ortho', axis = 1)
    block = scipy.fftpack.idct(block_col_trans, norm='ortho', axis = 0)
    return block

def dct2_3d(block):
    # Input:  a 3D array, block, representing an image block
    # Output: a 3D array, block_c, of DCT coefficients
    block_col_trans = scipy.fftpack.dct(block, norm='ortho', axis = 0)
    block_row = scipy.fftpack.dct(block_col_trans, norm='ortho', axis = 1)
    block_c = scipy.fftpack.dct(block_row, norm='ortho', axis = 2)
    return block_c

def idct2_3d(block_c):
    # Input:  a 2D array, block_c, of DCT coefficients
    # Output: a 2D array, block, representing an image block
    block_row = scipy.fftpack.idct(block_c, norm='ortho', axis = 2)
    block_col_trans = scipy.fftpack.idct(block_row, norm='ortho', axis = 1)
    block = scipy.fftpack.idct(block_col_trans, norm='ortho', axis = 0)
    return block

# quantization functions
def quantize(block_c, mode="y", quality=75):
    # Input:  a 2D float array, block_c, of DCT coefficients
    #         a string, mode, ("y" for luma quantization, "c" for chroma quantization)
    #         an int, quality, the JPEG quality factor in range [1,100] (defaults to 75)
    # Output: a 2D int array, block_cq, of quantized DCT coefficients
    if mode=="y":
        Q = np.array([[ 16,  11,  10,  16,  24,  40,  51,  61 ],
                      [ 12,  12,  14,  19,  26,  58,  60,  55 ],
                      [ 14,  13,  16,  24,  40,  57,  69,  56 ],
                      [ 14,  17,  22,  29,  51,  87,  80,  62 ],
                      [ 18,  22,  37,  56,  68,  109, 103, 77 ],
                      [ 24,  36,  55,  64,  81,  104, 113, 92 ],
                      [ 49,  64,  78,  87,  103, 121, 120, 101],
                      [ 72,  92,  95,  98,  112, 100, 103, 99 ]])
    elif mode=="c":
        Q = np.array([[ 17,  18,  24,  47,  99,  99,  99,  99 ],
                      [ 18,  21,  26,  66,  99,  99,  99,  99 ],
                      [ 24,  26,  56,  99,  99,  99,  99,  99 ],
                      [ 47,  66,  99,  99,  99,  99,  99,  99 ],
                      [ 99,  99,  99,  99,  99,  99,  99,  99 ],
                      [ 99,  99,  99,  99,  99,  99,  99,  99 ],
                      [ 99,  99,  99,  99,  99,  99,  99,  99 ],
                      [ 99,  99,  99,  99,  99,  99,  99,  99 ]])
    else:
        raise Exception("String argument must be 'y' or 'c'.")

    if quality < 1 or quality > 100:
        raise Exception("Quality factor must be in range [1,100].")

    scalar = 5000 / quality if quality < 50 else 200 - 2 * quality # formula for scaling by quality factor
    Q = Q * scalar / 100. # scale the quantization matrix
    Q[Q<1.] = 1. # do not divide by numbers less than 1

    # Quantize the 8x8 block
    block_cq = np.round(np.divide(block_c,Q)).astype(int)
    return block_cq

def unquantize(block_cq, mode="y", quality=75):
    # Input:  a 2D int array, block_cq, of quantized DCT coefficients
    #         a string, mode, ("y" for luma quantization, "c" for chroma quantization)
    #         an int, quality, the JPEG quality factor in range [1,100] (defaults to 75)
    # Output: a 2D float array, block_c, of "unquantized" DCT coefficients (they will still be quantized)
    if mode=="y":
        Q = np.array([[ 16,  11,  10,  16,  24,  40,  51,  61 ],
                      [ 12,  12,  14,  19,  26,  58,  60,  55 ],
                      [ 14,  13,  16,  24,  40,  57,  69,  56 ],
                      [ 14,  17,  22,  29,  51,  87,  80,  62 ],
                      [ 18,  22,  37,  56,  68,  109, 103, 77 ],
                      [ 24,  36,  55,  64,  81,  104, 113, 92 ],
                      [ 49,  64,  78,  87,  103, 121, 120, 101],
                      [ 72,  92,  95,  98,  112, 100, 103, 99 ]])
    elif mode=="c":
        Q = np.array([[ 17,  18,  24,  47,  99,  99,  99,  99 ],
                      [ 18,  21,  26,  66,  99,  99,  99,  99 ],
                      [ 24,  26,  56,  99,  99,  99,  99,  99 ],
                      [ 47,  66,  99,  99,  99,  99,  99,  99 ],
                      [ 99,  99,  99,  99,  99,  99,  99,  99 ],
                      [ 99,  99,  99,  99,  99,  99,  99,  99 ],
                      [ 99,  99,  99,  99,  99,  99,  99,  99 ],
                      [ 99,  99,  99,  99,  99,  99,  99,  99 ]])
    else:
        raise Exception("String argument must be 'y' or 'c'.")

    if quality < 1 or quality > 100:
        raise Exception("Quality factor must be in range [1,100].")

    scalar = 5000 / quality if quality < 50 else 200 - 2 * quality # formula for scaling by quality factor
    Q = Q * scalar / 100. # scale the quantization matrix
    Q[Q<1.] = 1. # do not divide by numbers less than 1

    # Un-quantize the 8x8 block
    block_c = np.multiply(block_cq,Q)
    return block_c

# run-length encoding functions
def zigzag(block_cq):
    # Input:  a 2D array, block_cq, of quantized DCT coefficients
    # Output: a list, block_cqz, of zig-zag reordered quantized DCT coefficients
    idx = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41,
           34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23,
           30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]
    unraveled = np.ravel(block_cq)
    block_cqz = [unraveled[idx[i]] for i in range(len(idx))]
    return block_cqz

def unzigzag(block_cqz):
    # Input:  a list, block_cqz, of zig-zag reordered quantized DCT coefficients
    # Output: a 2D array, block_cq, of conventionally ordered quantized DCT coefficients
    idx = [0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41,
           43, 9, 11, 18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38,
           46, 51, 55, 60, 21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63]
    rowLen = 8
    block_cq = np.array([block_cqz[idx[i]] for i in range(len(idx))]).reshape((8,8))
    return block_cq

# Zero run-length encoding
def zrle(block_cqz):
    # Input:  a list, block_cqz, of zig-zag reordered quantized DCT coefficients
    # Output: a list, block_cqzr, of zero-run-length encoded quantized DCT coefficients
    zero_count = 0
    block_cqzr = [block_cqz[0]]
    for i in range(1,len(block_cqz)):
        if max(block_cqz[i:]) == 0:
            block_cqzr.append((0, 0))
            break
        if block_cqz[i] != 0:
            block_cqzr.append((zero_count, block_cqz[i]))
            zero_count = 0
        elif zero_count == 14:
            block_cqzr.append((15,0))
            zero_count = 0
        else:
            zero_count+=1
    if block_cqzr[-1] != (0,0):
        block_cqzr.append((0,0))
    return block_cqzr

def unzrle(block_cqzr, blockDepth=4):
    # Input:  a list, block_cqzr, of zero-run-length encoded quantized DCT coefficients
    # Output: a list, block_cqz, of zig-zag reordered quantized DCT coefficients
    block_cqz = [block_cqzr[0]]
    for i in range(1, len(block_cqzr)):
        if block_cqzr[i] == (0,0):
            num_zeros = 64*blockDepth - len(block_cqz)
            block_cqz.extend([0]*num_zeros)
            break
        elif block_cqzr[i] == (15,0):
            block_cqz.extend([0] * 15)
        else:
            block_cqz.extend([0] * block_cqzr[i][0])
            block_cqz.append(block_cqzr[i][1])
    return block_cqz

# Huffman coding
def encode_block(block, mode="y", quality=75):
    # Input:  a 2D array, block, representing an image component block
    #         a string, mode, ("y" for luma, "c" for chroma)
    #         an int, quality, the JPEG quality factor in range [1,100] (defaults to 75)
    # Output: a bitarray, dc_bits, of Huffman encoded DC coefficients
    #         a bitarray, ac_bits, of Huffman encoded AC coefficients
    block_c = dct2(block)
    block_cq = quantize(block_c, mode, quality)
    block_cqz = zigzag(block_cq)
    block_cqzr = zrle(block_cqz)
    dc_bits = encode_huffman(block_cqzr[0], mode) # DC
    ac_bits = ''.join(encode_huffman(v, mode) for v in block_cqzr[1:]) # AC
    return bitarray(dc_bits), bitarray(ac_bits)

def decode_block(dc_gen, ac_gen, mode="y", quality=75):
    # Inputs: a generator, dc_gen, that yields decoded Huffman DC coefficients
    #         a generator, ac_gen, that yields decoded Huffman AC coefficients
    #         a string, mode, ("y" for luma, "c" for chroma)
    #         an int, quality, the JPEG quality factor in range [1,100] (defaults to 75)
    # Output: a 2D array, block, decoded by and yielded from the two generators
    block_cqzr = [next(dc_gen)] # initialize list by yielding from DC generator
    while block_cqzr[-1] != (0,0):
        block_cqzr.append(next(ac_gen)) # append to list by yielding from AC generator until (0,0) is encountered
    block_cqz = unzrle(block_cqzr)
    block_cq = unzigzag(block_cqz)
    block_c = unquantize(block_cq, mode, quality)
    block = idct2(block_c)
    return block

# mirror pad
def mirror_pad(img):
    # Input:  a 3D float array, img, representing an RGB image in range [0.0,255.0]
    # Output: a 3D float array, img_pad, mirror padded so the number of rows and columns are multiples of 16

    M, N = img.shape[0:2]
    pad_r = ((16 - (M % 16)) % 16) # number of rows to pad
    pad_c = ((16 - (N % 16)) % 16) # number of columns to pad
    img_pad = np.pad(img, ((0,pad_r), (0,pad_c), (0,0)), "symmetric") # symmetric padding
    return img_pad

def frame_pad(block, blockDepth=4):
    print(np.shape(block))
    T, M, N = np.shape(block)[0:3]
    if T == blockDepth:
        return block
    pad_t = (blockDepth - (T % blockDepth)) % blockDepth # number of columns to pad
    print(pad_t, T, blockDepth)
    block_pad = np.pad(block, ((0,pad_t),(0,0),(0,0)), "symmetric")
    return block_pad

# full encode / decode with DCT-II
# TODO: these should be edited for 3D videos and wavelets/temporal locality
def encode_image(img, quality=75):
    # Inputs:  a 3D float array, img, representing an RGB image in range [0.0,255.0]
    #          an int, quality, the JPEG quality factor in range [1,100] (defaults to 75)
    # Outputs: a bitarray, Y_dc_bits, the Y component DC bitstream
    #          a bitarray, Y_ac_bits, the Y component AC bitstream
    #          a bitarray, Cb_dc_bits, the Cb component DC bitstream
    #          a bitarray, Cb_ac_bits, the Cb component AC bitstream
    #          a bitarray, Cr_dc_bits, the Cr component DC bitstream
    #          a bitarray, Cr_ac_bits, the Cr component AC bitstream
    print("encode quality",quality)
    M_orig, N_orig = img.shape[0:2]
    img = mirror_pad(img[:,:,0:3])
    M, N = img.shape[0:2]

    im_ycbcr = RGB2YCbCr(img)
    Y = im_ycbcr[:,:,0]
    Cb = chroma_downsample(im_ycbcr[:,:,1])
    Cr = chroma_downsample(im_ycbcr[:,:,2])

    # Y component
    Y_dc_bits = bitarray()
    Y_ac_bits = bitarray()
    for i in np.r_[0:M:8]:
        for j in np.r_[0:N:8]:
            block = Y[i:i+8,j:j+8]
            dc_bits, ac_bits = encode_block(block, "y", quality)
            Y_dc_bits.extend(dc_bits)
            Y_ac_bits.extend(ac_bits)

    # Cb component
    Cb_dc_bits = bitarray()
    Cb_ac_bits = bitarray()
    for i in np.r_[0:M//2:8]:
        for j in np.r_[0:N//2:8]:
            block = Cb[i:i+8,j:j+8]
            dc_bits, ac_bits = encode_block(block, "c", quality)
            Cb_dc_bits.extend(dc_bits)
            Cb_ac_bits.extend(ac_bits)

    # Cr component
    Cr_dc_bits = bitarray()
    Cr_ac_bits = bitarray()
    for i in np.r_[0:M//2:8]:
        for j in np.r_[0:N//2:8]:
            block = Cr[i:i+8,j:j+8]
            dc_bits, ac_bits = encode_block(block, "c", quality)
            Cr_dc_bits.extend(dc_bits)
            Cr_ac_bits.extend(ac_bits)

    bits = (Y_dc_bits, Y_ac_bits, Cb_dc_bits, Cb_ac_bits, Cr_dc_bits, Cr_ac_bits)

    return bits

def decode_image(bits, M, N, quality=75):
    # Inputs: a tuple, bits, containing the following:
    #              a bitarray, Y_dc_bits, the Y component DC bitstream
    #              a bitarray, Y_ac_bits, the Y component AC bitstream
    #              a bitarray, Cb_dc_bits, the Cb component DC bitstream
    #              a bitarray, Cb_ac_bits, the Cb component AC bitstream
    #              a bitarray, Cr_dc_bits, the Cr component DC bitstream
    #              a bitarray, Cr_ac_bits, the Cr component AC bitstream
    #         ints, M and N, the number of rows and columns in the image
    #         an int, quality, the JPEG quality factor in range [1,100] (defaults to 75)
    # Output: a 3D float array, img, representing an RGB image in range [0.0,255.0]

    Y_dc_bits, Y_ac_bits, Cb_dc_bits, Cb_ac_bits, Cr_dc_bits, Cr_ac_bits = bits # unpack bits tuple

    M_orig = M # save original image dimensions
    N_orig = N
    M = M_orig + ((16 - (M_orig % 16)) % 16) # dimensions of padded image
    N = N_orig + ((16 - (N_orig % 16)) % 16)
    num_blocks = M * N // 64 # number of blocks

    # Y component
    Y_dc_gen = decode_huffman(Y_dc_bits.to01(), "dc", "y")
    Y_ac_gen = decode_huffman(Y_ac_bits.to01(), "ac", "y")
    Y = np.empty((M, N))
    for b in range(num_blocks):
        block = decode_block(Y_dc_gen, Y_ac_gen, "y", quality)
        r = (b*8 // N)*8 # row index (top left corner)
        c = b*8 % N # column index (top left corner)
        Y[r:r+8, c:c+8] = block

    # Cb component
    Cb_dc_gen = decode_huffman(Cb_dc_bits.to01(), "dc", "c")
    Cb_ac_gen = decode_huffman(Cb_ac_bits.to01(), "ac", "c")
    Cb2 = np.empty((M//2, N//2))
    for b in range(num_blocks//4):
        block = decode_block(Cb_dc_gen, Cb_ac_gen, "c", quality)
        r = (b*8 // (N//2))*8 # row index (top left corner)
        c = b*8 % (N//2) # column index (top left corner)
        Cb2[r:r+8, c:c+8] = block

    # Cr component
    Cr_dc_gen = decode_huffman(Cr_dc_bits.to01(), "dc", "c")
    Cr_ac_gen = decode_huffman(Cr_ac_bits.to01(), "ac", "c")
    Cr2 = np.empty((M//2, N//2))
    for b in range(num_blocks//4):
        block = decode_block(Cr_dc_gen, Cr_ac_gen, "c", quality)
        r = (b*8 // (N//2))*8 # row index (top left corner)
        c = b*8 % (N//2) # column index (top left corner)
        Cr2[r:r+8, c:c+8] = block

    Cb = chroma_upsample(Cb2)
    Cr = chroma_upsample(Cr2)

    img = YCbCr2RGB(np.stack((Y,Cb,Cr), axis=-1))

    img = img[0:M_orig,0:N_orig,:] # crop out padded parts

    return img

def jpeg123_encoder(img, outfile, quality=75):
    # Inputs: a 3D uint8 array, img, representing an RGB color image
    #         a string, outfile, of the output binary filename
    #         an int, quality, the JPEG quality factor in range [1,100] (defaults to 75)

    img = img.astype(np.float64)
    M, N = img.shape[0:2]

    bits = encode_image(img, quality=quality)
    #   bits(tuple):
    #     a bitarray, Y_dc_bits, the Y component DC bitstream
    #     a bitarray, Y_ac_bits, the Y component AC bitstream
    #     a bitarray, Cb_dc_bits, the Cb component DC bitstream
    #     a bitarray, Cb_ac_bits, the Cb component AC bitstream
    #     a bitarray, Cr_dc_bits, the Cr component DC bitstream
    #     a bitarray, Cr_ac_bits, the Cr component AC bitstream

    with open(outfile, "wb") as fh:
        # Your code here:
        # Start of Image (SOI) marker - 0xFFD8 (2 bytes)
        fh.write(bytes.fromhex("FFD8"))
        rows, cols = np.shape(img)[:2]
        # Number of rows - unsigned int (2 bytes)
        fh.write(rows.to_bytes(2, byteorder ='big'))
        # Number of columns - unsigned int (2 bytes)
        fh.write(cols.to_bytes(2, byteorder ='big'))
        # Quality factor - unsigned int (2 bytes)
        fh.write(quality.to_bytes(2, byteorder ='big'))
        # Start of Scan (SOS) - 0xFFDA (2 bytes)
        fh.write(bytes.fromhex("FFDA"))
        # Y component DC values - (? bytes)
        yDC = bits[0]
        padLen = (16 - (len(yDC) % 16)) % 16
        yDC.extend([1] * padLen)
        fh.write(yDC)
        # Start of Scan (SOS) - 0xFFDA (2 bytes)
        fh.write(bytes.fromhex("FFDA"))
        # Y component AC values - (? bytes)
        yAC = bits[1]
        padLen = (16 - (len(yAC) % 16)) % 16
        yAC.extend([1] * padLen)
        fh.write(yAC)
        # Start of Scan (SOS) - 0xFFDA (2 bytes)
        fh.write(bytes.fromhex("FFDA"))
        # Cb component DC values - (? bytes)
        cbDC = bits[2]
        padLen = (16 - (len(cbDC) % 16)) % 16
        cbDC.extend([1] * padLen)
        fh.write(cbDC)
        # Start of Scan (SOS) - 0xFFDA (2 bytes)
        fh.write(bytes.fromhex("FFDA"))
        # Cb component AC values - (? bytes)
        cbAC = bits[3]
        padLen = (16 - (len(cbAC) % 16)) % 16
        cbAC.extend([1] * padLen)
        fh.write(cbAC)
        # Start of Scan (SOS) - 0xFFDA (2 bytes)
        fh.write(bytes.fromhex("FFDA"))
        # Cr component DC values - (? bytes)
        crDC = bits[4]
        padLen = (16 - (len(crDC) % 16)) % 16
        crDC.extend([1] * padLen)
        fh.write(crDC)
        # Start of Scan (SOS) - 0xFFDA (2 bytes)
        fh.write(bytes.fromhex("FFDA"))
        # Cr component AC values - (? bytes)
        crAC = bits[5]
        padLen = (16 - (len(crAC) % 16)) % 16
        crAC.extend([1] * padLen)
        fh.write(crAC)
        # End of Image (EOI) marker - 0xFFD9 (2 bytes)
        fh.write(bytes.fromhex("FFD9"))

def jpeg123_decoder(infile):
    # Inputs:  a string, infile, of the input binary filename
    # Outputs: a 3D uint8 array, img_dec, representing a decoded JPEG123 color image

    with open(infile, "rb") as fh:
        SOI = fh.read(2)
        if SOI != bytes.fromhex("FFD8"):
            raise Exception("Start of Image marker not found!")
        M = int.from_bytes(fh.read(2), "big")
        N = int.from_bytes(fh.read(2), "big")
        quality = int.from_bytes(fh.read(2), "big")
        SOS = fh.read(2)
        if SOS != bytes.fromhex("FFDA"):
            raise Exception("Start of Scan marker not found!")
        bits = ()
        for _ in range(5):
            ba = bitarray()
            for b in iter(lambda: fh.read(2), bytes.fromhex("FFDA")): # iterate until next SOS marker
                ba.frombytes(b)
            bits = (*bits, ba)
        ba = bitarray()
        for b in iter(lambda: fh.read(2), bytes.fromhex("FFD9")): # iterate until EOI marker
            ba.frombytes(b)
        bits = (*bits, ba)

    img_dec = decode_image(bits, M, N, quality)

    return img_dec.astype(np.uint8)

# Part 3
# functions to enqueue data
# Convert from bitarray class to b64 bytes and back
def ee123_bitarr_to_base64(bits: bitarray):
    bN = np.uint32(len(bits)).tobytes()
    mybytes = bN + bits.tobytes()
    return base64.b64encode(mybytes)

def ee123_base64_to_bitarr(b64: str):
    mybytes = base64.b64decode(b64)  # Be careful not to overwrite the builtin bytes class!
    N = np.frombuffer(mybytes[:4], dtype='<u4')[0]
    ba = bitarray.bitarray()
    ba.frombytes(mybytes[4:])
    return ba[:N]

def file_to_b64(fname):
    with open(fname, 'rb') as f:
        raw = f.read()
    # Prepend the number of bytes before encoding
    b64 = base64.b64encode(np.uint32(len(raw)).tobytes() + raw)
    return b64

def enqueue_data(callsign, modem, data, address=None, uid=None, fname="myfile.bin",
                 comment="final-project-part4-winningteamnamept4", dest=b"APCAL", bsize=240):
    """
    Inputs:
      callsign: your callsign
      modem: a modem object
      data: data to send as a bytes object
      address: address callsign for the message, defaults to your callsign
      uid: optionally specify a UID, otherwise one will be randomly chosen
      fname: filename to send in start packet
      comment: comment for start packet
      dest: should be APCAL for EE123 data
      bsize: number of bytes to send per packet

    Outputs:
      Qout: a queue containing modulated packets to transmit
      uid: the uid used for this data transfer
    """
    if address is None:
        address = callsign
    if uid is None:
        uid = np.random.randint(0, 10000)
    uid = bytes("{:04d}".format(uid), 'utf-8')
    addrLen = 8
    if len(address) < addrLen:
        pad = addrLen - len(address)
        address = address + " "*pad
    print("Putting packets in Queue for transmission ID=%s" % uid.decode())

    Qout = Queue.Queue()

    digi = b'WIDE1-1,WIDE2-1'

    # Enqueue START packet
    # formt->  :TGTCALL :START####,filename,comment here
    prefix = bytes(":" + address + " :START", 'utf-8')
    suffix = bytes("," + fname + "," + comment, 'utf-8')
    info = prefix + uid + suffix
    start_packet = modem.modulatePacket(callsign=callsign, digi=digi, dest=dest, info=info)
    Qout.put(start_packet)

    # Enqueue data packets
    # format->   :TGTCALL :<data data data>####
    num_packets = len(data) // bsize
    excess_data = len(data) % bsize
    for i in range(num_packets):
        pcktData = data[i*bsize : (i+1)*bsize]
        prefix = bytes(":" + address + " :", 'utf-8')
        packetNum = bytes("{:04d}".format(i), 'utf-8')
        info = prefix + pcktData + packetNum
        body_packet = modem.modulatePacket(callsign=callsign, digi=digi, dest=dest, info=info)
        Qout.put(body_packet)

    #left over packet
    if excess_data > 0:
        excess_pcktData = data[num_packets*bsize :]
        prefix = bytes(":" + address + " :", 'utf-8')
        packetNum = bytes("{:04d}".format(num_packets), 'utf-8')
        info = prefix + excess_pcktData + packetNum
        body_packet = modem.modulatePacket(callsign=callsign, digi=digi, dest=dest, info=info)
        Qout.put(body_packet)

    # Enqueue END packet
    # format->    :YOURCALL :END####
    info = bytes(":" + address + " :END", 'utf-8') + uid
    end_packet = modem.modulatePacket(callsign=callsign, digi=digi, dest=dest, info=info)
    Qout.put(end_packet)

    return Qout, uid

# Compute psnr
# These functions return a runtime warning from overflow
def compute_psnr(I_dec, I_ref):
    # Input:  an array, I_dec, representing a decoded image in range [0.0,255.0]
    #         an array, I_ref, representing a reference image in range [0.0,255.0]
    # Output: a float, PSNR, representing the PSNR of the decoded image w.r.t. the reference image (in dB)
    shape = np.shape(I_ref)
    sum = 0
    for row in range(shape[0]):
        for col in range(shape[1]):
            for p in range(shape[2]):
                sum += np.square(I_ref[row,col,p] - I_dec[row,col,p])
    return 10.0*np.log10(((225.0**2)*(3.0 * float(shape[0]) * float(shape[1]))) / sum)

def compute_psnr_np(I_dec, I_ref):
    # Input:  an array, I_dec, representing a decoded image in range [0.0,255.0]
    #         an array, I_ref, representing a reference image in range [0.0,255.0]
    # Output: a float, PSNR, representing the PSNR of the decoded image w.r.t. the reference image (in dB)
    T, M, N, C = np.shape(I_ref)
    MSE = 0
    coeff = (1/(3*M*N*T))
    for m in range(M):
        for n in range(N):
            for c in range(C):
                for t in range(T):
                    sqr = np.square(np.abs(I_ref[t,m,n,c] - I_dec[t,m,n,c]))
                    MSE += coeff * sqr
    return 10.0*np.log10((225.0**2) / MSE)

def compute_psnr_apng(png_dec, png_ref):
    # Input: an APNG filename, a string filename for the decoded apng file
    #        an APNG filename, a string filename for the reference apng file
    # Output: a float, average PSNR, representing the average PSNR of the decoded image w.r.t. the reference image (in dB)

    # Load apngs into individual png files
    im = APNG.open(png_dec)
    input_files, output_files = [], []
    for i, (png, control) in enumerate(im.frames):
        frame = "output_frames/{i}.png".format(i=i)
        output_files.append(frame)
        png.save(frame)

    im = APNG.open(png_ref)
    for i, (png, control) in enumerate(im.frames):
        frame = "input_frames/{i}.png".format(i=i)
        input_files.append(frame)
        png.save(frame)

    # compute PSNR for each frame and return average
    dec_video, ref_video = [], []
    for i in range(len(output_files)):
        I_dec = Image.open(output_files[i])
        dec_video.append(np.array(I_dec)[:,:,0:3])
        I_ref = Image.open(input_files[i])
        ref_video.append(np.array(I_ref)[:,:,0:3])

    return compute_psnr_np(np.array(dec_video),np.array(ref_video))


def array_from_input():
    # Output: an array, TxMxNx3
    # T = # of frames, (M,N) = shape of image, 3 color channels
    arr = []
    for i in range(len(os.listdir("input_frames"))):
        frame = "input_frames/{i}.png".format(i=i)
        img = Image.open(frame)
        img = np.array(img)[:,:,0:3]
        arr.append(img)
    return np.array(arr)

def make_QR_callback(Qin):
    def queueREPLAY_callback(indata, outdata, frames, time, status):
#         assert frames == 1024
        if status:
            print(status)
        outdata[:] = indata
        Qin.put(indata.copy()[:, 0]) # Global queue
    return queueREPLAY_callback

def decode_ee123_message(msg):
    """
    Input:
      msg: The info field of an APRS message packet

    Output:
      A dict containing the key components the packet may contain

    Message formats:
    :CALLSIGN :START####,filename,comment
    :CALLSIGN :<data data data>####
    :CALLSIGN :END####
    """
    msg = msg.strip()
    addressee = ''
    uid = None
    isstart = False
    isend = False
    seq = None
    data = None  # This should be a bytes object
    filename = ''
    comment = ''

    # Your code here:
    if msg[11:16] == "START":
        # START PACKET
        isstart = True
        addressee = msg[1:9].strip()
        uid = msg[16:20]
        commaSplit = msg[21:].split(",")
        print("split:", commaSplit)
        filename = commaSplit[0]
        comment = commaSplit[1]
    elif msg[11:14] == "END":
        # END PACKET
        isend = True
        uid = msg[14:18]
    else:
        # DATA PACKET
        addressee = msg[1:9].strip()
        data = bytes(msg[11:-4], 'utf-8')
        seq = msg[-4:]
    # End of your code
    return {'addr': addressee, 'uid': uid, 'isstart': isstart,
            'isend': isend, 'seq': seq, 'filename': filename,
            'data': data, 'comment': comment}
