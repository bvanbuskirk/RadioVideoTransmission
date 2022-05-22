import numpy as np
import queue as Queue
import threading
import time
import sys
import multiprocessing
from numpy.fft import fft, fftshift, ifft, ifftshift
from scipy import signal
from scipy import integrate
from scipy.io.wavfile import read as wavread
import ax25
from math import gcd
import sounddevice as sd
from functools import reduce
import base64
from bitarray import bitarray

import numpy.ctypeslib as npct
from ctypes import c_int
from ctypes import c_float

# RPi only command
"""
import RPi.GPIO as GPIO

array_1d_int = npct.ndpointer(dtype=np.int, ndim=1, flags='CONTIGUOUS')

libcd = npct.load_library("./libpll", "pll.c")
libcd.pll.restype = c_int
libcd.pll.argtypes= [array_1d_int, c_int, array_1d_int,array_1d_int,  array_1d_int,array_1d_int, c_int, c_float]
"""

class TNCaprs:
    def __init__(self, fs=48000.0, Abuffer=1024, Nchunks=10):

        #  Implementation of an afsk1200 TNC.
        #
        #  The TNC processes `Abuffer` long buffers until `Nchunks` number of buffers are
        #  collected into a large one.
        #  This is because python is able to more efficiently process larger buffers than smaller ones.
        #  Then, the resulting large buffer is demodulated, sampled, and packets extracted.
        #
        # Inputs:
        #   fs      - sampling rate
        #   TBW     - TBW of the demodulator filters
        #   Abuffer - Input audio buffers from Pyaudio
        #   Nchunks - Number of audio buffers to collect before processing
        #   apll    - nudge factor of the PLL

        ## compute sizes based on inputs
        self.TBW = 2.0   # TBW for the demod filters
        self.N = (int(fs/1200*self.TBW)//2)*2+1   # length of the mark-space filters for demod
        self.fs = fs     # sampling rate
        self.BW = 1200      # BW of filter based on TBW
        self.Abuffer = Abuffer             # size of audio buffer
        self.Nchunks = Nchunks             # number of audio buffers to collect
        self.Nbuffer = Abuffer*Nchunks+(self.N*3-3)         # length of the large buffer for processing
        self.Ns = 1.0*fs/1200.0 # samples per symbol

        ## state variables for the modulator
        self.prev_ph = 0  # previous phase to maintain continuous phase when recalling the function

        ## generate filters for the demodulator
        self.h_lp = signal.firwin(self.N,self.BW/fs*1.0,window='hanning')
        self.h_lpp = signal.firwin(self.N,self.BW*2*1.2/fs,window='hanning')
        self.h_space = self.h_lp*np.exp(1j*2*np.pi*(2200)*(np.r_[0:self.N]-self.N//2)/fs)
        self.h_mark = self.h_lp*np.exp(1j*2*np.pi*(1200)*(np.r_[0:self.N]-self.N//2)/fs)
        self.h_bp = signal.firwin(self.N,self.BW/fs*2.2,window='hanning')*np.exp(1j*2*np.pi*1700*(np.r_[0:self.N]-self.N//2)/fs)

        ## PLL state variables  -- so continuity between buffers is preserved
        self.dpll = np.round(2.0**32 / self.Ns).astype(np.int32)    # PLL step
        self.pll = 0                # PLL counter
        self.ppll = -self.dpll       # PLL counter previous value -- to detect overflow
        self.apll = 0.74             # PLL agressivness (small more agressive)

        ## state variable to NRZI2NRZ
        self.NRZIprevBit = bool(1)

        ## State variables for findPackets
        self.state = 'search'   # state variable: 'search' or 'pkt'
        self.pktcounter = 0   # counts the length of a packet
        self.packet = bitarray([0,1,1,1,1,1,1,0]) # current packet being collected
        self.bitpointer = 0   # pointer to advance the search beyond what was already
                              # searched in the previous buffer

        ## state variables for processBuffer
        self.buff = np.zeros(self.Nbuffer)   # large overlap-save buffer
        self.chunk_count = 0              # chunk counter
        # bits from end of prev buffer to be copied to beginning of new
        self.oldbits = bitarray([0,0,0,0,0,0,0])
        self.Npackets = 0                 # packet counter

    # lcm function
    def lcm(self, numbers):
        return reduce(lambda x, y: (x*y)//gcd(x,y), numbers, 1)

    def NRZ2NRZI(self,NRZ, prevBit=True):
        NRZI = NRZ.copy()
        for n in range(0,len(NRZ)):
            if NRZ[n] :
                NRZI[n] = prevBit
            else:
                NRZI[n] = not(prevBit)
            prevBit = NRZI[n]
        return NRZI


    def NRZI2NRZ(self, NRZI):
        NRZ = NRZI.copy()
        for n in range(0,len(NRZI)):
            NRZ[n] = NRZI[n] == self.NRZIprevBit
            self.NRZIprevBit = NRZI[n]
        return NRZ


    def KISS2bits(self,KISS):
        # function that takes a KISS frame sent via TCP/IP and converts it to an APRSpacket bit stream.
        bits = bitarray(endian="little")
        bits.frombytes(KISS)
        fcs = ax25.FCS()
        for bit in bits:
            fcs.update_bit(bit)
        bits.frombytes(fcs.digest())
        return bitarray('01111110') + ax25.bit_stuff(bits) + bitarray('01111110')


    def bits2KISS(self,bits):
        # function that takes a bitstream of an APRS-packet, removes flags and FCS and unstuffs the bits
        bitsu = ax25.bit_unstuff(bits[8:-8])
        return  bitsu[:-16].tobytes()


    def modulate(self, bits):
        # the function will take a bitarray of bits and will output an AFSK1200 modulated signal of them,
        # sampled at fs Hz
        #  Inputs:
        #         bits  - bitarray of bits
        #         fs    - sampling rate
        # Outputs:
        #         sig    -  returns afsk1200 modulated signal
        # Your code here:
        fs = self.fs
        fss = self.lcm((1200,fs))
        deci = fss//fs

        Nb = fss//1200
        nb = len(bits)
        NRZ = np.ones((nb,Nb))
        for n in range(0,nb):
            if bits[n]:
                NRZ[n,:]=-NRZ[n,:]

        freq = 1700 + 500*NRZ.ravel()
        ph = 2.0*np.pi*integrate.cumtrapz(freq)/fss
        sig = np.cos(ph[::deci])
        # End of your code
        return sig


    def modulatePacket(self, callsign, digi, dest, info, preflags=80, postflags=80):
        # given callsign, digipath, dest, info, number of pre-flags and post-flags, the function contructs
        # an appropriate APRS packet, then converts it to NRZI and calls `modulate`
        # to afsk1200 modulate the packet.
        packet = ax25.UI(destination=dest,source=callsign, info=info, digipeaters=digi.split(b','),)
        prefix = bitarray(np.tile([0,1,1,1,1,1,1,0],(preflags,)).tolist())
        suffix = bitarray(np.tile([0,1,1,1,1,1,1,0],(postflags,)).tolist())
        sig = self.modulate(self.NRZ2NRZI(prefix + packet.unparse()+suffix))

        return sig


    def demod(self, buff):
        # Demodulates a buffer and returns NRZa. Make sure you only return valid samples.
        fs = self.fs
        TBW = self.TBW
        N = (int(fs/1200*TBW)//2)*2+1
        BW = TBW/(1.0*N/fs)
        h_lp = self.h_lp
        h_lpp = self.h_lpp
        h_space = self.h_space
        h_mark = self.h_mark
        h_bp = self.h_bp

        buff = np.convolve(buff.copy(),h_bp,'valid')
        mark = abs(np.convolve(buff,h_mark,mode='valid'))
        space = abs(np.convolve(buff,h_space,mode='valid'))
        NRZ = mark-space
        NRZa = np.convolve(NRZ,h_lpp,mode='valid')
        return NRZa

    """
    def FastPLL(self,NRZa):
        recbits = np.zeros(len(NRZa)//(self.fs//1200)*2,dtype=np.int32)
        pll = np.zeros(1,dtype = np.int32)
        pll[0] = self.pll
        ppll = np.zeros(1,dtype = np.int32)
        ppll[0] = self.ppll

        NRZb = (NRZa > 0).astype(np.int32)
        tot = libcd.pll(NRZb,len(NRZb),recbits,recbits,pll,ppll,self.dpll,self.apll)

        self.ppll = ppll.copy()
        self.pll = pll.copy()

        return bitarray.bitarray(recbits[:tot].tolist())
    """

    def PLL(self, NRZa):
        idx = np.zeros(len(NRZa)//int(self.Ns)*2)   # allocate space to save indices
        c = 0

        for n in range(1,len(NRZa)):
            if (self.pll < 0) and (self.ppll >0):
                idx[c] = n
                c = c+1

            if (NRZa[n] >= 0) !=  (NRZa[n-1] >=0):
                self.pll = np.int32(self.pll*self.apll)

            self.ppll = self.pll
            self.pll = np.int32(self.pll+ self.dpll)

        return idx[:c].astype(np.int32)


    def findPackets(self,bits):
        # function take a bitarray and looks for AX.25 packets in it.
        # It implements a 2-state machine of searching for flag or collecting packets
        flg = bitarray([0,1,1,1,1,1,1,0])
        packets = []
        n = self.bitpointer

        # Loop over bits
        while (n < len(bits)-7) :
            # default state is searching for packets
            if self.state == 'search':
                # look for 1111110, because can't be sure if the first zero is decoded
                # well if the packet is not padded.
                if bits[n:n+7] == flg[1:]:
                    # flag detected, so switch state to collecting bits in a packet
                    # start by copying the flag to the packet
                    # start counter to count the number of bits in the packet
                    self.state = 'pkt'
                    self.packet=flg.copy()
                    self.pktcounter = 8
                    # Advance to the end of the flag
                    n = n + 7
                else:
                    # flag was not found, advance by 1
                    n = n + 1
            # state is to collect packet data.
            elif self.state == 'pkt':
                # Check if we reached a flag by comparing with 0111111
                # 6 times ones is not allowed in a packet, hence it must be a flag (if there's no error)
                if bits[n:n+7] == flg[:7]:
                    # Flag detected, check if packet is longer than some minimum
                    if self.pktcounter > 200:
                        #print('packet found!')
                        # End of packet reached! append packet to list and switch to searching state
                        # We don't advance pointer since this our packet might have been
                        # flase detection and this flag could be the beginning of a real packet
                        self.state = 'search'
                        self.packet.extend(flg)
                        packets.append(self.packet.copy())
                    else:
                        # packet is too short! false alarm. Keep searching
                        # We don't advance pointer since this this flag could be the beginning of a real packet
                        self.state = 'search'
                # No flag, so collect the bit and add to the packet
                else:
                    # check if packet is too long... if so, must be false alarm
                    if self.pktcounter < 2680:
                        # Not a false alarm, collect the bit and advance pointer
                        self.packet.append(bits[n])
                        self.pktcounter = self.pktcounter + 1
                        n = n + 1
                    else:  #runaway packet
                        #runaway packet, switch state to searching, and advance pointer
                        self.state = 'search'
                        n = n + 1

        self.bitpointer = n-(len(bits)-7)
        return packets

    # function to generate a checksum for validating packets
    def genfcs(self,bits):
        # Generates a checksum from packet bits
        fcs = ax25.FCS()
        for bit in bits:
            fcs.update_bit(bit)

        digest = bitarray(endian="little")
        digest.frombytes(fcs.digest())

        return digest


    # function to parse packet bits to information
    def decodeAX25(self,bits, deepsearch=False):
        ax = ax25.AX25()
        ax.info = "bad packet"
        ax.parse(bits)
        return ax


    def processBuffer(self, buff_in):
        # function processes an audio buffer. It collect several small into a large one
        # Then it demodulates and finds packets.
        #
        # The function uses overlap and save convolution
        # The function returns packets when they become available. Otherwise, returns empty list

        N = self.N
        NN = N*3 - 3

        Nchunks = self.Nchunks
        Abuffer = self.Abuffer
        fs = self.fs
        Ns = self.Ns

        validPackets=[]
        packets=[]
        NRZI=[]
        idx = []
        bits = []

        # Fill in buffer at the right place
        self.buff[NN+self.chunk_count*Abuffer:NN+(self.chunk_count+1)*Abuffer] = buff_in.copy()
        self.chunk_count = self.chunk_count + 1

        # number of chunk reached -- process large buffer
        if self.chunk_count == Nchunks:
            # Demodulate to get NRZI
            NRZI = self.demod(self.buff)

            # compute sampling points, using PLL
            bits = self.PLL(NRZI)
            # In case that buffer is too small, raise an error -- must have at least 7 bits worth
            if len(bits) < 7:
                raise ValueError('number of bits too small for buffer')

            # concatenate end of previous buffer to current one
            bits = self.oldbits + self.NRZI2NRZ(bits)

            # store end of bit buffer to next buffer
            self.oldbits = bits[-7:].copy()

            # look for packets
            packets = self.findPackets(bits)

            # Copy end of sample buffer to the beginning of the next (overlap and save)
            self.buff[:NN] = self.buff[-NN:].copy()

            # reset chunk counter
            self.chunk_count = 0

            # checksum test for all detected packets
            for n in range(0,len(packets)):
                if len(packets[n]) > 200:
                    try:
                        ax = self.decodeAX25(packets[n])
                    except:
                        ax = ax25.AX25()
                        ax.info = "bad packet"
                    if ax.info != 'bad packet' and ax.info != 'no decode':
                        validPackets.append(packets[n])

        return validPackets





# Define the callback factory so we can specify which non-global queue we want
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
