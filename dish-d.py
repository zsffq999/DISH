import numpy as np
from data import hash_value, hash_evaluation
from scipy.linalg import eigh
import time
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from bqp import *

import sys

# whether use VGG net for evaluation
use_vgg = False

if use_vgg:
	sys.path.extend(['./vgg_dl'])
	from vgg_dl import cnn_hash
else:
	sys.path.extend(['./dl'])
	from dl import cnn_hash

class DISH_D(object):
	def __init__(self, r, numlabel):
		self.r = r # num of hash bits
		self.numlabel = numlabel

		# Hash code and out-of-sample labels
		self.H = None
		self.trainlabel = None

		# tuning parameters
		self.mu = 1e-4
		self.lmda = 1e-2

		# classifiers in W-step
		self.classifier = 'CNN'

	def train(self, traindata, trainlabel):
		n = len(traindata)
		mu = self.mu * n
		# shuffle data
		indexes = np.arange(n, dtype=np.int32)
		np.random.shuffle(indexes)
		traindata = traindata[indexes]
		trainlabel = trainlabel[indexes]
		
		# pairwise label matrix, S = 2*P*P.T-1_{n*n}
		if len(trainlabel.shape) >= 2:
			assert trainlabel.shape[1] == self.numlabel
			P = csr_matrix(trainlabel, dtype=np.float32)
			P = P.T
		else:
			P = csr_matrix((np.ones(n),[np.arange(n, dtype=np.int32), trainlabel]), shape=(n,self.numlabel), dtype=np.float32)
			P = P.T
		H = np.zeros((n,self.r))


		self.cls = cnn_hash.CNN_hash(self.r)
		self.cls.train(traindata, np.where(np.random.rand(n,self.r)>=0.5, 1, -1), trainlabel)
		Y = self.cls.predict(traindata)

		# step 2: discrete optimization
		print '\nSTEP 2: Discrete Optimization...'
		H = np.where(Y>=0, 1, -1).astype(np.float32)
		
		h = np.zeros(n)
		h1 = np.zeros(n)

		bqp = AMF_BQP(P.T, 2*self.r, -self.r, H)
		# bqp = AMF_deg3_BQP(P.T, 1.0/3*self.r, -2*self.r, 11.0/3*self.r, -self.r, H)
		TT = 3
		for t in range(TT):
			print '\nIter No: %d' % t

			# step 2.2: fix W, optimize H
			KK_W = Y
			for rr in range(self.r):
				if (rr+1) % 10 == 0:
					print 'rr:', rr
				h[:] = H[:,rr]
				H[:,rr] = 0
				q = -0.5 * mu * (np.log(1.0+np.exp(-KK_W[:,rr])) - np.log(1.0+np.exp(KK_W[:,rr])))
				
				bqp.H = H
				bqp.q = q
				h1[:] = bqp_cluster(bqp, h)
				if bqp.neg_obj(h1) <= bqp.neg_obj(h):
					H[:,rr] = h1
				else:
					H[:,rr] = h

			self.cls.train(traindata, H, trainlabel)
			
			Y = self.cls.predict(traindata)
						
			print np.sum(H==np.where(Y>=0, 1, -1)) / float(n*self.r)
			if t-1 < TT:
				H = np.where(Y>=0, 1, -1).astype(np.float32)

		
		self.trainlabel = trainlabel
		self.H = np.copy(H)

	def queryhash(self, qdata):
		Y = self.cls.predict(qdata)
		Y = np.where(Y>=0, 1, 0)
		return hash_value(Y)

	def basehash(self, data):
		return self.queryhash(data)


def test_cnn(n_bit):
	np.random.seed(17)
	if use_vgg:
		X = np.load('cifar10_data/cifar10_256_data.npy')
	else:
		X = np.load('cifar10_data/cifar10_data.npy')
	Y = np.load('cifar10_data/cifar10_label.npy')
	
	is_train = np.load('cifar10_data/cifar10_istrain.npy')
	
	train_idx = np.argwhere(is_train==1)[:,0]
	nottrain_idx = np.argwhere(is_train==0)[:,0]
	np.random.shuffle(nottrain_idx)
	test_idx = nottrain_idx[-1000:]
	base_idx = np.setdiff1d(np.arange(60000, dtype=np.int32), test_idx)
	
	traindata = X[train_idx]
	trainlabel = Y[train_idx]
	testdata = X[test_idx]
	testlabel = Y[test_idx]
	
	basedata = X[base_idx]
	baselabel = Y[base_idx]
	
	del X
	
	# train model
	dksh = DISH_D(n_bit, 10)
	tic = time.time()
	dksh.train(traindata, trainlabel)
	toc = time.time()
	print 'time:', toc-tic

	H_test = dksh.queryhash(testdata)
	H_base = dksh.queryhash(basedata)

	# make labels
	gnd_truth = np.array([y == baselabel for y in testlabel]).astype(np.int8)

	print 'testing...'

	res = hash_evaluation(H_test, H_base, gnd_truth, len(H_base), topN=len(H_base), trn_time=toc-tic)
	print 'MAP:', res['map'], 'Pre2:', res['pre2']


if __name__ == "__main__":
	test_cnn(32)
