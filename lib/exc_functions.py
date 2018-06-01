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
from img_functions import nikon_getfiles, get_images_from_directory,process_image
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


"""
Helper functions
"""

def set_weights(model, weights_path):
	f = h5py.File(weights_path ,'r')

	# for key in f.keys():
	# 	g = f[key]
	# 	weights = [g[k] for k in g.keys()]
	
	# print f['model_weights'].keys()	

	for layer in model.layers:
		if 'tensorprod2d' in layer.name:
			idsplit = layer.name.split('_')[-1]
			layer.name = 'dense_' + idsplit

		if 'sparse_convolution2d' in layer.name:
			idsplit = layer.name.split('_')[-1]
			layer.name = 'convolution2d_' + idsplit

	for layer in model.layers:
		if 'model_weights' in f.keys():
			if layer.name in f['model_weights'].keys():
				if 'bn' in layer.name:
					g = f['model_weights'][layer.name]
					keys = ['{}_gamma'.format(layer.name), '{}_beta'.format(layer.name), '{}_running_mean'.format(layer.name), '{}_running_std'.format(layer.name)]
					weights = [g[key] for key in keys]
					layer.set_weights(weights)

				if 'batch' in layer.name:
					g = f['model_weights'][layer.name]
					keys = ['{}_gamma'.format(layer.name), '{}_beta'.format(layer.name), '{}_running_mean'.format(layer.name), '{}_running_std'.format(layer.name)]
					weights = [g[key] for key in keys]
					layer.set_weights(weights)

				else:
					g = f['model_weights'][layer.name]
					weights = [g[key] for key in g.keys()]
					layer.set_weights(weights)
		else:
			# In case old keras saving convention is used
			if layer.name in f.keys():
				if 'bn' in layer.name:
					g = f[layer.name]
					keys = ['{}_gamma'.format(layer.name), '{}_beta'.format(layer.name), '{}_running_mean'.format(layer.name), '{}_running_std'.format(layer.name)]
					weights = [g[key] for key in keys]
					layer.set_weights(weights)

				if 'batch' in layer.name:
					g = f[layer.name]
					keys = ['{}_gamma'.format(layer.name), '{}_beta'.format(layer.name), '{}_running_mean'.format(layer.name), '{}_running_std'.format(layer.name)]
					weights = [g[key] for key in keys]
					layer.set_weights(weights)

				else:
					g = f[layer.name]
					weights = [g[key] for key in g.keys()]
					layer.set_weights(weights)

	return model
def sparse_pool_output_length(input_length, pool_size, stride):
	return input_length - stride*(pool_size-1)
def sparse_pool(input_image, stride = 2, pool_size = (2,2), mode = 'max'):
	pooled_array = []
	counter = 0
	for offset_x in xrange(stride):
		for offset_y in xrange(stride):
			pooled_array +=[pool_2d(input_image[:, :, offset_x::stride, offset_y::stride], pool_size, st = (1,1), mode = mode, padding = (0,0), ignore_border = True)]
			counter += 1

	# Concatenate pooled image to create one big image
	running_concat = []
	for it in xrange(stride):
		running_concat += [T.concatenate(pooled_array[stride*it:stride*(it+1)], axis = 3)]
	concatenated_image = T.concatenate(running_concat,axis = 2)

	pooled_output_array = []

	for it in xrange(counter+1):
		pooled_output_array += [T.tensor4()]

	pooled_output_array[0] = concatenated_image

	counter = 0
	for offset_x in xrange(stride):
		for offset_y in xrange(stride):
			pooled_output_array[counter+1] = T.set_subtensor(pooled_output_array[counter][:, :, offset_x::stride, offset_y::stride], pooled_array[counter])
			counter += 1
	return pooled_output_array[counter]
def tensorprod_softmax(x):
	e_output = T.exp(x - x.max(axis = 1, keepdims=True))
	softmax = e_output/e_output.sum(axis = 1, keepdims = True)
	return softmax

def sparse_W(W_input, stride = 2, filter_shape = (0,0,0,0)):
	W_new = theano.shared(value = np.zeros((filter_shape[0], filter_shape[1], stride*(filter_shape[2]-1)+1, stride*(filter_shape[3]-1)+1),dtype = theano.config.floatX), borrow = True)
	W_new_1 = T.set_subtensor(W_new[:,:,0::stride,0::stride],W_input)
	new_filter_shape = (filter_shape[0], filter_shape[1], stride*(filter_shape[2]-1)+1, stride*(filter_shape[3]-1)+1)
	return W_new_1, new_filter_shape
def conv_output_length(input_length, filter_size, border_mode, stride):
	if input_length is None:
		return None
	assert border_mode in {'same', 'valid'}
	if border_mode == 'same':
		output_length = input_length
	elif border_mode == 'valid':
		output_length = input_length - filter_size + 1
	return (output_length + stride - 1) // stride

"""
Keras layers
"""

class sparse_Convolution2D(Layer):
	'''Convolution operator for filtering windows of two-dimensional inputs.
	When using this layer as the first layer in a model,
	provide the keyword argument `input_shape`
	(tuple of integers, does not include the sample axis),
	e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.
	# Examples
	```python
		# apply a 3x3 convolution with 64 output filters on a 256x256 image:
		model = Sequential()
		model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 256, 256)))
		# now model.output_shape == (None, 64, 256, 256)
		# add a 3x3 convolution on top, with 32 output filters:
		model.add(Convolution2D(32, 3, 3, border_mode='same'))
		# now model.output_shape == (None, 32, 256, 256)
	```
	# Arguments
		nb_filter: Number of convolution filters to use.
		nb_row: Number of rows in the convolution kernel.
		nb_col: Number of columns in the convolution kernel.
		init: name of initialization function for the weights of the layer
			(see [initializations](../initializations.md)), or alternatively,
			Theano function to use for weights initialization.
			This parameter is only relevant if you don't pass
			a `weights` argument.
		activation: name of activation function to use
			(see [activations](../activations.md)),
			or alternatively, elementwise Theano function.
			If you don't specify anything, no activation is applied
			(ie. "linear" activation: a(x) = x).
		weights: list of numpy arrays to set as initial weights.
		border_mode: 'valid' or 'same'.
		subsample: tuple of length 2. Factor by which to subsample output.
			Also called strides elsewhere.
		W_regularizer: instance of [WeightRegularizer](../regularizers.md)
			(eg. L1 or L2 regularization), applied to the main weights matrix.
		b_regularizer: instance of [WeightRegularizer](../regularizers.md),
			applied to the bias.
		activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
			applied to the network output.
		W_constraint: instance of the [constraints](../constraints.md) module
			(eg. maxnorm, nonneg), applied to the main weights matrix.
		b_constraint: instance of the [constraints](../constraints.md) module,
			applied to the bias.
		dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
			(the depth) is at index 1, in 'tf' mode is it at index 3.
			It defaults to the `image_dim_ordering` value found in your
			Keras config file at `~/.keras/keras.json`.
			If you never set it, then it will be "th".
		bias: whether to include a bias (i.e. make the layer affine rather than linear).
	# Input shape
		4D tensor with shape:
		`(samples, channels, rows, cols)` if dim_ordering='th'
		or 4D tensor with shape:
		`(samples, rows, cols, channels)` if dim_ordering='tf'.
	# Output shape
		4D tensor with shape:
		`(samples, nb_filter, new_rows, new_cols)` if dim_ordering='th'
		or 4D tensor with shape:
		`(samples, new_rows, new_cols, nb_filter)` if dim_ordering='tf'.
		`rows` and `cols` values might have changed due to padding.
	'''
	def __init__(self, nb_filter, nb_row, nb_col,
				 d = 1, init='glorot_uniform', activation='linear', weights=None,
				 border_mode='valid', subsample=(1, 1), dim_ordering=K.image_dim_ordering(),
				 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
				 W_constraint=None, b_constraint=None,
				 bias=True, **kwargs):

		if border_mode not in {'valid', 'same'}:
			raise Exception('Invalid border mode for Convolution2D:', border_mode)
		self.nb_filter = nb_filter
		self.nb_row = nb_row
		self.nb_col = nb_col
		self.init = initializations.get(init)
		self.activation = activations.get(activation)
		self.d = d
		assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
		self.border_mode = border_mode
		self.subsample = tuple(subsample)
		assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
		self.dim_ordering = dim_ordering

		self.W_regularizer = regularizers.get(W_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)
		self.activity_regularizer = regularizers.get(activity_regularizer)

		self.W_constraint = constraints.get(W_constraint)
		self.b_constraint = constraints.get(b_constraint)

		self.bias = bias
		self.input_spec = [InputSpec(ndim=4)]
		self.initial_weights = weights
		super(sparse_Convolution2D, self).__init__(**kwargs)

	def build(self, input_shape):
		if self.dim_ordering == 'th':
			stack_size = input_shape[1]
			self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
		elif self.dim_ordering == 'tf':
			stack_size = input_shape[3]
			self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
		else:
			raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

		self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
		self.sparse_W, self.sparse_W_shape = sparse_W(self.W, stride = self.d, filter_shape = self.W_shape)
		self.nb_row_sparse = self.sparse_W_shape[2]
		self.nb_col_sparse = self.sparse_W_shape[3]

		if self.bias:
			self.b = K.zeros((self.nb_filter,), name='{}_b'.format(self.name))
			self.trainable_weights = [self.W, self.b]
		else:
			self.trainable_weights = [self.W]
		self.regularizers = []

		if self.W_regularizer:
			self.W_regularizer.set_param(self.W)
			self.regularizers.append(self.W_regularizer)

		if self.bias and self.b_regularizer:
			self.b_regularizer.set_param(self.b)
			self.regularizers.append(self.b_regularizer)

		if self.activity_regularizer:
			self.activity_regularizer.set_layer(self)
			self.regularizers.append(self.activity_regularizer)

		self.constraints = {}
		if self.W_constraint:
			self.constraints[self.W] = self.W_constraint
		if self.bias and self.b_constraint:
			self.constraints[self.b] = self.b_constraint

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def get_output_shape_for(self, input_shape):
		if self.dim_ordering == 'th':
			rows = input_shape[2]
			cols = input_shape[3]
		elif self.dim_ordering == 'tf':
			rows = input_shape[1]
			cols = input_shape[2]
		else:
			raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

		rows = conv_output_length(rows, self.nb_row_sparse,
								  self.border_mode, self.subsample[0])
		cols = conv_output_length(cols, self.nb_col_sparse,
								  self.border_mode, self.subsample[1])

		if self.dim_ordering == 'th':
			return (input_shape[0], self.nb_filter, rows, cols)
		elif self.dim_ordering == 'tf':
			return (input_shape[0], rows, cols, self.nb_filter)
		else:
			raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

	def call(self, x, mask=None):
		output = K.conv2d(x, self.sparse_W, strides=self.subsample,
						  border_mode=self.border_mode,
						  dim_ordering=self.dim_ordering,
						  filter_shape=self.sparse_W_shape)
		if self.bias:
			if self.dim_ordering == 'th':
				output += K.reshape(self.b, (1, self.nb_filter, 1, 1))
			else:
				raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
		output = self.activation(output)
		return output

	def get_config(self):
		config = {'nb_filter': self.nb_filter,
				  'nb_row': self.nb_row,
				  'nb_col': self.nb_col,
				  'nb_row_sparse': self.nb_row_sparse,
				  'nb_col_sparse': self.nb_col_sparse,
				  'init': self.init.__name__,
				  'activation': self.activation.__name__,
				  'border_mode': self.border_mode,
				  'subsample': self.subsample,
				  'dim_ordering': self.dim_ordering,
				  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
				  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
				  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
				  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
				  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
				  'bias': self.bias}
		base_config = super(sparse_Convolution2D, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class sparse_MaxPooling2D(Layer):
	def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
				 dim_ordering=K.image_dim_ordering(), **kwargs):
		super(sparse_MaxPooling2D, self).__init__(**kwargs)
		self.pool_size = tuple(pool_size)
		if strides is None:
			strides = self.pool_size
		self.strides = tuple(strides)
		assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
		self.border_mode = border_mode
		assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
		self.dim_ordering = dim_ordering
		self.input_spec = [InputSpec(ndim=4)]

	def get_output_shape_for(self, input_shape):
		if self.dim_ordering == 'th':
			rows = input_shape[2]
			cols = input_shape[3]
		else:
			raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

		rows = sparse_pool_output_length(rows, self.pool_size[0], self.strides[0])
		cols = sparse_pool_output_length(cols, self.pool_size[1], self.strides[1])

		if self.dim_ordering == 'th':
			return (input_shape[0], input_shape[1], rows, cols)
		else:
			raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

	def _pooling_function(self, inputs, pool_size, strides,
						  border_mode, dim_ordering):
		output = sparse_pool(inputs, pool_size = pool_size, stride = strides[0])
		return output

	def call(self, x, mask=None):
		output = self._pooling_function(inputs=x, pool_size=self.pool_size,
										strides=self.strides,
										border_mode=self.border_mode,
										dim_ordering=self.dim_ordering)
		return output

	def get_config(self):
		config = {'pool_size': self.pool_size,
				  'border_mode': self.border_mode,
				  'strides': self.strides,
				  'dim_ordering': self.dim_ordering}
		base_config = super(sparse_MaxPooling2D, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class TensorProd2D(Layer):
	def __init__(self, input_dim, output_dim,
				 init='glorot_uniform', activation='linear', weights=None,
				 border_mode='valid', subsample=(1, 1), dim_ordering=K.image_dim_ordering(),
				 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
				 W_constraint=None, b_constraint=None,
				 bias=True, **kwargs):

		if border_mode not in {'valid', 'same'}:
			raise Exception('Invalid border mode for Convolution2D:', border_mode)
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.init = initializations.get(init)
		self.activation = activations.get(activation)
		assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
		self.border_mode = border_mode
		self.subsample = tuple(subsample)
		assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
		self.dim_ordering = dim_ordering

		self.W_regularizer = regularizers.get(W_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)
		self.activity_regularizer = regularizers.get(activity_regularizer)

		self.W_constraint = constraints.get(W_constraint)
		self.b_constraint = constraints.get(b_constraint)

		self.bias = bias
		self.input_spec = [InputSpec(ndim=4)]
		self.initial_weights = weights
		super(TensorProd2D, self).__init__(**kwargs)

	def build(self, input_shape):
		if self.dim_ordering == 'th':
			self.W_shape = (self.input_dim, self.output_dim)
		else:
			raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
		self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
		if self.bias:
			self.b = K.zeros((self.output_dim,), name='{}_b'.format(self.name))
			self.trainable_weights = [self.W, self.b]
		else:
			self.trainable_weights = [self.W]
		self.regularizers = []

		if self.W_regularizer:
			self.W_regularizer.set_param(self.W)
			self.regularizers.append(self.W_regularizer)

		if self.bias and self.b_regularizer:
			self.b_regularizer.set_param(self.b)
			self.regularizers.append(self.b_regularizer)

		if self.activity_regularizer:
			self.activity_regularizer.set_layer(self)
			self.regularizers.append(self.activity_regularizer)

		self.constraints = {}
		if self.W_constraint:
			self.constraints[self.W] = self.W_constraint
		if self.bias and self.b_constraint:
			self.constraints[self.b] = self.b_constraint

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def get_output_shape_for(self, input_shape):
		if self.dim_ordering == 'th':
			rows = input_shape[2]
			cols = input_shape[3]
			return(input_shape[0], self.output_dim, rows, cols)
		else:
			raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

	def call(self, x, mask=None):

		output = T.tensordot(x, self.W, axes = [1,0]).dimshuffle(0,3,1,2) 

		if self.bias:
			if self.dim_ordering == 'th':
				output += self.b.dimshuffle('x',0,'x','x') 
			else:
				raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
		output = self.activation(output)
		return output

	def get_config(self):
		config = {'input_dim': self.input_dim,
				  'output_dim': self.output_dim,
				  'init': self.init.__name__,
				  'activation': self.activation.__name__,
				  'border_mode': self.border_mode,
				  'subsample': self.subsample,
				  'dim_ordering': self.dim_ordering,
				  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
				  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
				  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
				  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
				  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
				  'bias': self.bias}
		base_config = super(TensorProd2D, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))











"""
Executing convnets
"""

def run_model(image, model, win_x = 30, win_y = 30, process = False):

        image_size_x = image.shape[2]
	image_size_y = image.shape[3]

	evaluate_model = K.function(
		[model.layers[0].input, K.learning_phase()],
		[model.layers[-1].output]
		) 

	n_features = model.layers[-1].output_shape[1]

        model_output = evaluate_model([image,0])[0]
	model_output = model_output[0,:,:,:]
		
	model_output = np.pad(model_output, pad_width = [(0,0), (win_x, win_x),(win_y,win_y)], mode = 'constant', constant_values = [(0,0), (0,0), (0,0)])
	return model_output


def run_models_list(data_location, channel_names,  output_location, model_name, list_model_weights, n_features = 3, image_size_x = 1080, image_size_y = 1280, win_x = 30, win_y = 30, process = True, save = True, nuclear_prefix='', NumReferImg=0, Nextcascade= False,images=None):
        #
	batch_input_shape = (1,len(channel_names),image_size_x+win_x, image_size_y+win_y)
	model = model_name(batch_input_shape = batch_input_shape, n_features = n_features, weights_path = list_model_weights[0])
	n_features = model.layers[-1].output_shape[1]#lots of warning
        
        if images is None:
                image_list = get_images_from_directory(data_location, channel_names)
                   
        else:  image_list=images;
        
        for image in image_list:
                if process:
	            for j in xrange(image.shape[1]):
		           image[0,j,:,:] = process_image(image[0,j,:,:], win_x, win_y)   
	
	        model_outputs = []
	        for weights_path in list_model_weights:
                    #print "Now model:" + weights_path[-4]
		    model = set_weights(model, weights_path = weights_path)
		    processed_image= run_model(image, model, win_x = win_x, win_y = win_y)
                    processed_image_list= [processed_image]
                    model_outputs += [np.stack(processed_image_list, axis = 0)]

	        # Average all images
	        model_output = np.stack(model_outputs, axis = 0)
	        model_output = np.mean(model_output, axis = 0)

	img_files =[]
	for channel in channel_names:
		img_files += [nikon_getfiles(data_location, channel)]
        
	# Save images
	
        if save:
		for img in xrange(model_output.shape[0]):
			for feat in xrange(n_features-1):
				feature = model_output[img,feat,:,:]
				cnnout_name = os.path.join(output_location, 'f' + str(feat) + '_' + img_files[0+NumReferImg][img][0:-4] +'_'+ nuclear_prefix + r'.png')
				tiff.imsave(cnnout_name,feature[0:image_size_x*2,0:image_size_y*2])
        if Nextcascade:
                shutil.copyfile(data_location+img_files[0+NumReferImg][0],output_location+'image_'+img_files[0+NumReferImg][0])
 
	from keras.backend.common import _UID_PREFIXES
	for key in _UID_PREFIXES:
		_UID_PREFIXES[key] = 0

	return model_output


