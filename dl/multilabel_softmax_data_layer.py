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
		imgs, labels, label_cls = self.batch_loader.load_next_batch()
		top[0].data[...] = imgs
		top[1].data[...] = labels
		top[2].data[...] = label_cls


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
		# self.data = np.load('./cifar10_data/cifar10_data.npy')
		# self.multilabels = np.where(np.load('./cifar10_data/cifar10_hash.npy')>0, 1, -1)
		# self.labels = np.load('./cifar10_data/cifar10_label.npy')
		
		# get list of image indexes.
		# self._cur = 0 
		# self.n_data = 5000 
		# self.indexlist = np.arange(self.n_data, dtype=np.int32) 
		
		# preprocess: compute img mean
		self.img_mean = np.load('./cifar10_data/cifar10_mean.npy').reshape((1, 3, self.height, self.width))

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
		if self._cur + self.batch_size <= len(self.indexlist):
			index = self.indexlist[self._cur:self._cur+self.batch_size]
			self._cur += self.batch_size

		else:
			index = np.zeros(self.batch_size, dtype=np.int32)
			index[:len(self.indexlist)-self._cur] = self.indexlist[self._cur:]
			if self.is_train:
				shuffle(self.indexlist)
			index[len(self.indexlist)-self._cur:] = self.indexlist[:self.batch_size-len(self.indexlist)+self._cur]
			self._cur += self.batch_size - len(self.indexlist)
		
		# data argumentation
		imgs = self.data[index].astype(np.float32)
		imgs[:,:,:,:] = imgs[:,::-1,:,:]
		imgs -= self.img_mean
		if self.is_train:
			flip_ind = np.argwhere(np.random.rand(self.batch_size)>0.5)[:,0]
			imgs[flip_ind,:,:,:] = imgs[flip_ind,:,:,::-1]
		labels = self.multilabels[index].astype(np.float32)
		label_cls = self.labels[index].astype(np.float32)

		# disturb labels (optional)
		# label_disturb = np.where(np.random.rand(self.batch_size, self.n_labels)<0.3, -1, 1)
		# labels *= label_disturb
		
		return imgs, labels, label_cls
