'''Train a simple deep CNN on a HeLa dataset.
GPU run command:
	THEANO_FLAGS='mode=FAST_RUN,device=gpu,floatX=float32' python training_template.py

'''

from __future__ import print_function
from keras.optimizers import SGD, RMSprop
import sys
sys.path.insert(0, '../lib/')
from tra_functions import rate_scheduler, train_model_sample
from modelsetting import net_normalization_51 as the_model##########################WQ

import pdb

import os
import datetime
import numpy as np
import time
start = time.clock()
 

batch_size = 256 # why 256
n_epoch = 30


#dataset = "CCNN_RevisedLabelBasedCNN3Dlightsheet_51-51_11P11I_20180131"  # training data
dataset = "CNN_FTB2_0d5_51_1P_180531"  # training data
expt = "51" # model name

direc_save = "../trained_clearNuclear_model"
direc_data = "training_patch"


optimizer = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)

lr_sched = rate_scheduler(lr = 0.001, decay = 0.95)
class_weight = {0:1, 1:1, 2:1, 3:1} #????
todays_date = datetime.datetime.now().strftime("%Y%m%d")
todays_date=todays_date[2:]
for iterate in xrange(5):

	model = the_model(n_channels = 1, n_features = 3, reg = 1e-4)       # For cascaded CNN, n_channels=2

  
	train_model_sample(model = model, dataset = dataset, optimizer = optimizer, 
		expt = expt, it = iterate, batch_size = batch_size, n_epoch = n_epoch,
		direc_save = direc_save, direc_data = direc_data, 
		lr_sched = lr_sched, rotate = True, flip = True, shear = False, todays_date=todays_date)


	del model
	from keras.backend.common import _UID_PREFIXES
	for key in _UID_PREFIXES.keys():
		_UID_PREFIXES[key] = 0

end = time.clock()
print('total time of training cnn is:%s Seconds'%(end-start))
 
		
