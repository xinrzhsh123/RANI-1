"""
CNN layers - Classes for layers for convolutional neural networks
Builds upon the Keras layer
"""

"""
Import python packages
"""
import pdb # QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQq
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import shelve
from contextlib import closing

import os
import glob
import re
import numpy as np
import tifffile as tiff
from numpy.fft import fft2, ifft2, fftshift
from skimage.io import imread
from scipy import ndimage
import threading
import scipy.ndimage as ndi
from scipy import linalg
import re
import random
import itertools
import h5py
import datetime
import shutil
import matlab.engine
eng = matlab.engine.start_matlab()

from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from scipy.ndimage.morphology import binary_fill_holes
from skimage import morphology as morph
from pywt import WaveletPacket2D
from skimage.transform import resize
from numpy.fft import fft2, ifft2, fftshift
from skimage.io import imread
from skimage.filters import threshold_otsu
import skimage as sk
from sklearn.utils.linear_assignment_ import linear_assignment

from theano.tensor.nnet import conv
from theano.tensor.signal.pool import pool_2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.ifelse import ifelse

from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Activation, merge, Dense, Flatten
from keras.preprocessing.image import random_rotation, random_shift, random_shear, random_zoom, random_channel_shift
from keras.preprocessing.image import transform_matrix_offset_center, apply_transform, flip_axis, array_to_img, img_to_array, load_img, list_pictures
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.engine import Layer, InputSpec
from keras.utils import np_utils
from keras import activations as activations

try:
	from keras import initializations as initializations
except ImportError:
	from keras import initializers as initializations
from keras import regularizers as regularizers
from keras import constraints as constraints




def process_image(channel_img, win_x, win_y, std = False, remove_zeros = False):

	if std:
		avg_kernel = np.ones((2*win_x + 1, 2*win_y + 1))
		channel_img -= ndimage.convolve(channel_img, avg_kernel)/avg_kernel.size
		std = np.std(channel_img)
		channel_img /= std
		return channel_img

	if remove_zeros:
		channel_img /= 255
		avg_kernel = np.ones((2*win_x + 1, 2*win_y + 1))
		channel_img -= ndimage.convolve(channel_img, avg_kernel)/avg_kernel.size
		return channel_img

	else:  
               if np.max(channel_img)>1:
                  #channel_img=channel_img/np.max(channel_img)
                  pTri= np.mean(channel_img)
                  #pTri= np.percentile(channel_img,50)
               else: pTri= 0.5
	       channel_img /= pTri
	       avg_kernel = np.ones((2*win_x + 1, 2*win_y + 1))
	       channel_img -= ndimage.convolve(channel_img, avg_kernel)/avg_kernel.size
                #p99 = np.percentile(channel_img, 99.99)
                #p01 = np.percentile(channel_img, 0.01)
                #channel_img = ((channel_img- p01) / (p99-p01))
                #channel_img[channel_img>1]=1
                #channel_img[channel_img<0]=0
                
	       return channel_img


def nikon_getfiles(direc_name,channel_name):
	imglist = os.listdir(direc_name)
	imgfiles = [i for i in imglist if channel_name in i]

	def sorted_nicely(l):
		convert = lambda text: int(text) if text.isdigit() else text
		alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
		return sorted(l, key = alphanum_key)

	imgfiles = sorted_nicely(imgfiles)
	return imgfiles

def get_image(file_name):
	if '.tif' in file_name:
		im = np.float32(tiff.TIFFfile(file_name).asarray())
	else:
		im = np.float32(imread(file_name))
	return im
def format_coord(x,y,sample_image):
	numrows, numcols = sample_image.shape
	col = int(x+0.5)
	row = int(y+0.5)
	if col>= 0 and col<numcols and row>=0 and row<numrows:
		z = sample_image[row,col]
		return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x,y,z)
	else:
		return 'x=%1.4f, y=1.4%f'%(x,y)

def get_image_sizes(data_location, channel_names):
	img_list_channels = []
	for channel in channel_names:
		img_list_channels += [nikon_getfiles(data_location, channel)]
	img_temp = get_image(os.path.join(data_location, img_list_channels[0][0]))

	return img_temp.shape
	
def get_images_from_directory(data_location, channel_names):
	img_list_channels = []
	for channel in channel_names:
		img_list_channels += [nikon_getfiles(data_location, channel)]

	img_temp = get_image(os.path.join(data_location, img_list_channels[0][0]))
        img_x_even=img_temp.shape[0]/2*2
        img_y_even=img_temp.shape[1]/2*2
	n_channels = len(channel_names)
	all_images = []
	for stack_iteration in xrange(len(img_list_channels[0])):
		all_channels = np.zeros((1, n_channels, img_x_even,img_y_even), dtype = 'float32')
		for j in xrange(n_channels):
			channel_img = get_image(os.path.join(data_location, img_list_channels[j][stack_iteration]))
			all_channels[0,j,:,:] = channel_img[0:img_x_even,0:img_y_even]
		all_images += [all_channels]
	
	return all_images


