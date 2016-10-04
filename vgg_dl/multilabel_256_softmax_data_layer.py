# imports
import caffe

import numpy as np
from random import shuffle
import cPickle as cp
import scipy.io as sio

import sys

class MultilabelSoftmaxDataLayer(caffe.Layer):

	"""
	This is a simple syncronous datalayer for training a multilabel model on
	CIFAR.
	"""

	def setup(self, bottom, top):

		self.top_names = ['data', 'label_hash', 'label_data']

		# === Read input parameters ===

		# params is a python dictionary with layer parameters.
		params = eval(self.param_str)

		# Check the paramameters for validity.

		# store input as class variables
		self.phase = params['phase']
		self.batch_size = params['batch_size']

		# Create a batch loader to load the images.
		self.batch_loader = BatchLoader(params, None)

		# === reshape tops ===
		# since we use a fixed input image size, we can shape the data layer
		# once. Else, we'd have to do it in the reshape call.
		top[0].reshape(
			self.batch_size, 3, params['height'], params['width'])
		# Note the 20 channels (because PASCAL has 20 classes.)
		top[1].reshape(self.batch_size, params['n_labels'])
		top[2].reshape(self.batch_size)
		print "PythonDataLayer init success", params

		# print_info("PythonDataLayer", params)

	def forward(self, bottom, top):
		"""
		Load data.
		"""
		# imgs, labels, label_cls = self.batch_loader.load_next_batch()
		for i in xrange(self.batch_size):
			img, label, label_cls = self.batch_loader.load_next_batch()
			top[0].data[i,...] = img
			top[1].data[i,...] = label
			top[2].data[i,...] = label_cls


	def reshape(self, bottom, top):
		"""
		There is no need to reshape the data, since the input is of fixed size
		(rows and columns)
		"""
		pass

	def backward(self, top, propagate_down, bottom):
		"""
		These layers does not back propagate
		"""
		pass


class BatchLoader(object):

	"""
	This class abstracts away the loading of images.
	Images can either be loaded singly, or in a batch. The latter is used for
	the asyncronous data layer to preload batches while other processing is
	performed.
	"""

	def __init__(self, params, result):
		self.result = result
		self.batch_size = params['batch_size']
		self.height = params['height']
		self.width = params['width']
		self.is_train = (params['phase']=='TRAIN')
		self.n_labels = params['n_labels']

		# get data
		self.data = None 
		self.multilabels = None 
		self.labels = None 
		
		# get list of image indexes.
		self._cur = 0  
		self.n_data = 5000 
		self.indexlist = np.arange(self.n_data, dtype=np.int32) 
		
		# preprocess: compute img mean
		self.img_mean = np.load('./cifar10_data/imagenet_mean.npy').reshape((3,256,256))

	def update_data(self, traindata, H=None, trainlabel=None):
		self.data = traindata
		if H is not None:
			print "updating H..."
			assert H.shape[1] == self.n_labels
			self.multilabels = np.where(H>=0, 1, -1)
		else:
			self.multilabels = np.zeros((len(traindata), self.n_labels))
		if trainlabel is not None:
			print "updating training label..."
			self.labels = trainlabel
		else:
			self.labels = np.zeros(len(traindata))
		self.n_data = len(self.data)
		self.indexlist = np.arange(self.n_data, dtype=np.int32)
		self._cur = 0

	def load_next_batch(self):
		"""
		Load the next image in a batch.
		"""
		if self._cur < len(self.indexlist):
			index = self.indexlist[self._cur]
			self._cur += 1

		else:
			if self.is_train:
				shuffle(self.indexlist)
			index = self.indexlist[0]
			self._cur = 1	
		
		imgs = self.data[index].astype(np.float32)
		imgs[:,:,:] = imgs[::-1,:,:]
		imgs -= self.img_mean
		rnd_height = imgs.shape[1] - self.height + 1
		rnd_width = imgs.shape[2] - self.width + 1
		imgs_crop = np.zeros((3, self.height, self.width))

		# data argumentation
		if self.is_train:
			x = np.random.randint(rnd_height)
			y = np.random.randint(rnd_width)
			if np.random.randint(2) == 0:
				imgs_crop[...] = imgs[:,x:x+self.height,y:y+self.width]
			else:
				imgs_crop[...] = imgs[:,x:x+self.height,-y-1:-y-self.width-1:-1]
		else:
			imgs_crop[...] = imgs[:,rnd_height/2:rnd_height/2+self.height,rnd_width/2:rnd_width/2+self.width]
		labels = self.multilabels[index].astype(np.float32)
		label_cls = self.labels[index].astype(np.float32)

		# disturb labels (optional)
		# if self.is_train:
		#	label_disturb = np.where(np.random.rand(self.n_labels)<0.3, -1, 1)
		#	labels *= label_disturb

		
		return imgs_crop, labels, label_cls
