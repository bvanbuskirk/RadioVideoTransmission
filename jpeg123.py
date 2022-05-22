# write/read compressed file encodings
from compress_functions import *
from video_encoding import *
from PIL import Image, ImageFilter

def jpeg123_video_encoder(video, outfile, blockDepth=4,quality=75):
    # Inputs: a 4D uint8 array, img, representing an RGB color image
    #         a string, outfile, of the output binary filename
    #         an int, quality, the JPEG quality factor in range [1,100] (defaults to 75)
    video = video.astype(np.float64)
    T, M, N = video.shape[:3]

    # video processing
    bits = encode_video(video, blockDepth, quality)

    # returns (frames x 6 x arbitrary_bit_len) array
    #    video_bits(tuple):
    #      a bitarray, Y_dc_bits, the Y component DC bitstream
    #      a bitarray, Y_ac_bits, the Y component AC bitstream
    #      a bitarray, Cb_dc_bits, the Cb component DC bitstream
    #      a bitarray, Cb_ac_bits, the Cb component AC bitstream
    #      a bitarray, Cr_dc_bits, the Cr component DC bitstream
    #      a bitarray, Cr_ac_bits, the Cr component AC bitstream

    with open(outfile, "wb") as fh:
        # Start of Image (SOI) marker - 0xFFD8 (2 bytes)
        fh.write(bytes.fromhex("FFD8"))
        frames, rows, cols = np.shape(video)[:3]

        # Number of frames - unsigned int (2 bytes)
        fh.write(frames.to_bytes(2, byteorder ='big'))
        # Number of frames per block - unsigned int(2 bytes)
        fh.write(blockDepth.to_bytes(2, byteorder ='big'))
        # Number of rows - unsigned int (2 bytes)
        fh.write(rows.to_bytes(2, byteorder ='big'))
        # Number of columns - unsigned int (2 bytes)
        fh.write(cols.to_bytes(2, byteorder ='big'))
        # Quality factor - unsigned int (2 bytes)
        fh.write(quality.to_bytes(2, byteorder ='big'))

        # Write all lengths first (3x6 = 18 bytes), then data
        yDC = bits[0]
        padLen = (16 - (len(yDC) % 16)) % 16
        yDC.extend([1] * padLen)
        fh.write(len(yDC).to_bytes(3, byteorder ='big'))

        yAC = bits[1]
        padLen = (16 - (len(yAC) % 16)) % 16
        yAC.extend([1] * padLen)
        fh.write(len(yAC).to_bytes(3, byteorder ='big'))

        cbDC = bits[2]
        padLen = (16 - (len(cbDC) % 16)) % 16
        cbDC.extend([1] * padLen)
        fh.write(len(cbDC).to_bytes(3, byteorder ='big'))

        cbAC = bits[3]
        padLen = (16 - (len(cbAC) % 16)) % 16
        cbAC.extend([1] * padLen)
        fh.write(len(cbAC).to_bytes(3, byteorder ='big'))

        crDC = bits[4]
        padLen = (16 - (len(crDC) % 16)) % 16
        crDC.extend([1] * padLen)
        fh.write(len(crDC).to_bytes(3, byteorder ='big'))

        crAC = bits[5]
        padLen = (16 - (len(crAC) % 16)) % 16
        crAC.extend([1] * padLen)
        fh.write(len(crAC).to_bytes(3, byteorder ='big'))

        # write data bits
        fh.write(yDC)
        fh.write(yAC)
        fh.write(cbDC)
        fh.write(cbAC)
        fh.write(crDC)
        fh.write(crAC)

        # End of Image (EOI) marker - 0xFFD9 (2 bytes)
        fh.write(bytes.fromhex("FFD9"))

def jpeg123_video_decoder(infile, outfile):
    # Inputs:  a string, infile, of the input binary filename
    # Outputs: a 3D uint8 array, img_dec, representing a decoded JPEG123 color image
    print("decoding video!")
    with open(infile, "rb") as fh:
        SOI = fh.read(2)
        while SOI != bytes.fromhex("FFD8"):
            SOI = fh.read(2)
        T = int.from_bytes(fh.read(2), "big")
        blockDepth = int.from_bytes(fh.read(2), "big")
        M = int.from_bytes(fh.read(2), "big")
        N = int.from_bytes(fh.read(2), "big")
        quality = int.from_bytes(fh.read(2), "big")

        bit_lengths = []
        for _ in range(6):
            bit_lengths.append(int.from_bytes(fh.read(3), "big"))

        bits = ()
        for bit_len in bit_lengths:
            ba = bitarray()
            for _ in range(0,bit_len,16):
                b = fh.read(2)
                ba.frombytes(b)
            bits = (*bits, ba)

    video = decode_video(bits, T, M, N, blockDepth, quality)
    """
    filelist = [ f for f in os.listdir("output_frames")]
    for f in filelist:
        os.remove(os.path.join("output_frames", f))
    """
        
    if os.path.exists(outfile):
        os.remove(outfile)
    # video decoding
    output_files = []
    for i in range(T):
        frame = "{i}.png".format(i=i)
        output_files.append(frame)
        img = Image.fromarray(video[i].astype(np.uint8),"RGB")
        # blur does not increase psnr
        #img = img.filter(ImageFilter.GaussianBlur(radius = 0.75))
        img = img.filter(ImageFilter.SHARPEN)
        img = img.save(frame)

    APNG.from_files(output_files, delay=100).save(outfile)
    return outfile












#
