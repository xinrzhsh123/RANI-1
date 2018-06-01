

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils

from exc_functions import tensorprod_softmax, sparse_Convolution2D, sparse_MaxPooling2D, TensorProd2D, set_weights
from keras.models import Model
from keras.layers import Input, Activation, merge, Dense, Flatten, Lambda, merge
import os
import h5py

"""
Vanilla convnets
"""



def net_normalization_51(n_features = 3, n_channels = 2, reg = 1e-5, init = 'he_normal'):  #QQ
	print "Using feature net 51x51 with batch normalization"
	model = Sequential()
	model.add(Convolution2D(32, 5, 5, init = init, border_mode='valid', input_shape=(n_channels, 51, 51), W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))	
	model.add(Convolution2D(32, 5, 5, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))	
	model.add(Convolution2D(64, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))	
	model.add(Convolution2D(64, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(128, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(Convolution2D(128, 3, 3, init = init, border_mode ='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(Convolution2D(128, 3, 3, init = init, border_mode ='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))


	model.add(Flatten())

	model.add(Dense(200, init = init, W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Dense(n_features, init = init, W_regularizer = l2(reg)))
	model.add(Activation('softmax'))

	return model

def sparse_net_normalization_51(batch_input_shape = (1,2,1080,1280), n_features = 3, reg = 1e-5, init = 'he_normal', weights_path = None):

	model = Sequential()
	d = 1
	model.add(sparse_Convolution2D(32, 5, 5, d = d, init = init, batch_input_shape = batch_input_shape, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(sparse_Convolution2D(32, 5, 5, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(sparse_Convolution2D(32, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
	d *= 2

	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))	
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode ='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode ='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
	d *= 2

	model.add(sparse_Convolution2D(128, 3, 3, d = d, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))	
	model.add(sparse_Convolution2D(128, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(sparse_Convolution2D(128, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))	

	model.add(TensorProd2D(128, 200, init = init, W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, init = init, W_regularizer = l2(reg)))
	model.add(Activation(tensorprod_softmax))

	model = set_weights(model, weights_path)

	return model

