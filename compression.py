 # EE123 Spring 2022 Final Project
# Authors: Brandon VanBuskirk, Steven Foryoung, Matthew Haddad
import numpy as np
import sys
from compress_functions import *
from TNCaprs import *
from jpeg123 import *
# png file selection
cal = "Rolling_cal"
andy = "Andy_Video"
dog = "dog"
simpson = "simpson"
simple_shape = "simple_shape"
river = "river_flowing"
ice = "ice_growing"
milky_way = "milkyway"
comp_img = "competition_img"

all_pngs = [cal, andy, dog, simpson, simple_shape, river, ice, milky_way, comp_img]


file = comp_img

input_png = file + ".png"
jpeg123 = file + ".jpeg123"
output_png = "rec_" + file + ".png"

# clear input files
filelist = [ f for f in os.listdir("input_frames")]
for f in filelist:
    os.remove(os.path.join("input_frames", f))

# Load apng into individual png files
im = APNG.open(input_png)
input_files, rgb_array = [], []
for i, (png, control) in enumerate(im.frames):
    frame = "input_frames/{i}.png".format(i=i)
    input_files.append(frame)
    png.save(frame)

# load pngs into a numpy array
rgb_array = []
for i in range(len(input_files)):
    img = Image.open(input_files[i])
    rgb_array.append(np.array(img)[:,:,0:3])

print("video dims: ",np.shape(rgb_array))
"""
rgb_array = np.array(rgb_array)
dim1 = np.min([np.shape(rgb_array[i])[0] for i in range(np.shape(rgb_array)[0])])
dim2 = np.min([np.shape(rgb_array[i])[1] for i in range(np.shape(rgb_array)[0])])
print("dim",dim1)
print(dim2)
rgb_array = rgb_array[:dim1,:dim2]
"""

rgb_array = np.array(rgb_array)
video = rgb_array.astype(np.float64)
T, M, N = rgb_array.shape[:3]

jpeg123_video_encoder(np.array(rgb_array), jpeg123, blockDepth=9, quality=20)

# test decoder here
img_dec = jpeg123_video_decoder(jpeg123, output_png)

psnr = compute_psnr_apng(input_png, output_png)
print(input_png)
print("quality: ", 35)
print("psnr: ",psnr)
#comp_ratio = stat(input_png).st_size / stat(jpeg123).st_size
#print("The compression ratio is:", comp_ratio)

#print("psnr x compression: ", psnr * comp_ratio)

callsign = "EE1233-8"
fs = 48000
modem = TNCaprs(fs=fs, Abuffer=1024, Nchunks=10)

data_base64 = file_to_b64(jpeg123)
Qout, uid = enqueue_data(callsign=callsign, modem=modem, data=data_base64, uid=1010, fname=output_png)
packets = Qout.qsize()
print(packets, "packets")

#
