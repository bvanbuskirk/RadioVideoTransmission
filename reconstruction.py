#!/usr/bin/env python3
from bitarray import bitarray
from jpeg123 import *
import numpy as np
import os
import sys


def reconstruct(binfile, outfile):
    # Read in data and settings
    uid = int(binfile.split('.')[0])
    with open(binfile, 'rb') as f:
        B = f.read()


    ## For you to modify
    
    # Check length
    if len(B) == 0:
        print("Empty data!")
        msg = "empty data component"
        ffail = str(uid) + '.fail'
        with open(ffail, 'w') as f:
            f.write(msg)
        return ffail
    if (len(B) - 4) != np.frombuffer(B[:4], dtype='<u4')[0]:
        print("Length of received bytes ({}) != length header ({})".format(
            len(B) - 4, np.frombuffer(B[:4], dtype='<u4')[0]))
        msg = "length mismatch"
        ffail = str(uid) + '.fail'
        with open(ffail, 'w') as f:
            f.write(msg)
        return ffail
    data_out = B[4:]  # Drop uint containing bit length
    
        ## Save as the given filename
    # Write output
    with open("tmp.bin", 'wb') as f:
        # Drop initial uint containing bit length
        f.write(data_out)
    jpeg123_video_decoder("tmp.bin", outfile)
    return outfile
    

def main():
    datadir = sys.argv[1]
    os.chdir(datadir)  # cd into data dir
    binfile = sys.argv[2]
    outfile = sys.argv[3]
    fname = jpeg123_video_decoder(binfile, outfile)
    print("Reconstructed", fname)
    return 0


if __name__ == "__main__":
    main()
