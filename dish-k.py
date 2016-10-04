import numpy as np
from data import hash_value, hash_evaluation
from scipy.linalg import eigh
import time
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from bqp import *

import threading

class DISH_K(object):
	def __init__(self, r, m, numlabel, kernel):
		self.r = r # num of hash bits
		self.m = m # num of anchors
		self.kernel = kernel # kernel function
		self.anchors = None # anchor points
		self.W = None # parameter to optimize
		self.numlabel = numlabel
		self.mvec = None # mean vector

		# Hash code and out-of-sample labels
		self.H = None
		self.trainlabel = None

		# tuning parameters
		self.mu = 1e-4
		self.lmda = 0

		# classifiers in W-step
		self.classifier = 'LineR'

	def train(self, traindata, trainlabel):
		n = len(traindata)
		mu = self.mu * n
		
		# shuffle data
		indexes = np.arange(n, dtype=np.int32)
		np.random.shuffle(indexes)
		traindata = traindata[indexes]
		trainlabel = trainlabel[indexes]

		print 'determine anchors...'

		# determine anchors
		anchoridx = np.copy(indexes)
		np.random.shuffle(anchoridx)
		anchoridx = anchoridx[:self.m]
		self.anchors = traindata[anchoridx]

		# kernel matrix and mean
		KK = self.kernel(traindata, self.anchors)
		self.mvec = np.mean(KK, axis=0).reshape((1, self.m))
		KK = KK - self.mvec

		# pairwise label matrix, S = 2*P*P.T-1_{n*n}
		if len(trainlabel.shape) >= 2:
			assert trainlabel.shape[1] == self.numlabel
			P = csr_matrix(trainlabel, dtype=np.float32)
			P = P.T
		else:
			P = csr_matrix((np.ones(n),[np.arange(n, dtype=np.int32), trainlabel]), shape=(n,self.numlabel), dtype=np.float32)
			P = P.T
		H = np.zeros((n,self.r))

		# projection optimization
		RM = np.dot(KK.T, KK)
		W = np.zeros((self.m, self.r), dtype=np.float32) # parameter W
		b = np.zeros(self.r) # parameter b
		LM = self.r*(2*np.dot(P.dot(KK).T, P.dot(KK)) - np.dot(np.sum(KK.T, axis=1, keepdims=True), np.sum(KK, axis=0, keepdims=True)))


		# step 1: initialize with spectral relaxation
		# step 1.1: batch coordinate optimization
		h0 = np.zeros(n)
		print '\nSTEP 1: Initialize with spectral relaxation...'
		print 'step 1.1...'
		
		for rr in range(self.r):
			if rr > 0:
				tmp = np.dot(KK.T, h0.reshape((n,1)))
				LM -= np.dot(tmp, tmp.T)
			(V, U) = eigh(LM, RM, eigvals_only=False)
			W[:,rr] = U[:,self.m-1]
			tmp = np.dot(np.dot(W[:,rr].T, RM), W[:,rr])
			W[:,rr] *= np.sqrt(n/tmp)

			h0 = np.where(np.dot(KK, W[:,rr]) >= 0, 1, -1)
			H[:,rr] = h0
		

		# step 2: discrete optimization
		print '\nSTEP 2: Discrete Optimization...'
		RM += self.lmda * np.eye(self.m)
		h = np.zeros(n)
		h1 = np.zeros(n)
		if self.classifier == 'LogR':
			cls = []
			for i in xrange(self.r):
				cls.append(LogisticRegression(C=1.0/self.lmda))
		elif self.classifier == 'SVM':
			cls = []
			for i in xrange(self.r):
				cls.append(LinearSVC(C=1.0/self.lmda))
		else:
			invRM = np.linalg.inv(RM)

		if self.classifier == 'LogR' or self.classifier == 'SVM':
			def multi_run(n_threads, num, cls, KK, H):
				for i in xrange(num, self.r, n_threads):
					cls[i].fit(KK, H[:,i])

		bqp = AMF_BQP(P.T, 2*self.r, -self.r, H)
		# bqp = AMF_deg3_BQP(P.T, 1.0/3*self.r, -2*self.r, 11.0/3*self.r, -self.r, H)

		for t in range(3):
			print '\nIter No: %d' % t

			# step 2.1: fix W, optimize H
			KK_W = np.dot(KK, W)
			for rr in range(self.r):
				if (rr+1) % 10 == 0:
					print 'rr:', rr
				h[:] = H[:,rr]
				H[:,rr] = 0
				if self.classifier == 'SVM':
					q = -0.5 * mu / self.lmda * (np.where(KK_W[:,rr]>1, 0, 1-KK_W[:,rr]) - np.where(KK_W[:,rr]<-1, 0, 1+KK_W[:,rr]))
				elif self.classifier == 'LogR':
					q = -0.5 * mu / self.lmda * (np.log(1.0+np.exp(-KK_W[:,rr])) - np.log(1.0+np.exp(KK_W[:,rr])))
				else:
					q = KK_W[:,rr]
				# bqp = AMF_BQP(P.T, 2*self.r, -self.r, H, q)
				bqp.H = H
				bqp.q = q
				h1[:] = bqp_cluster(bqp, h)
				if bqp.neg_obj(h1) <= bqp.neg_obj(h):
					H[:,rr] = h1
				else:
					H[:,rr] = h


			# step 2.2: fix H, optimize W
			# For SVM or LR
			if self.classifier == 'SVM' or self.classifier == 'LogR':
				threads = []
				n_threads = 16
				for i in xrange(n_threads):
					thr = threading.Thread(target=multi_run, args=(n_threads, i, cls, KK, H))
					threads.append(thr)

				for i in xrange(n_threads):
					threads[i].start()

				for i in xrange(n_threads):
					threads[i].join()

				for rr in xrange(self.r):
					W[:,rr] = cls[rr].coef_[0]
					b[rr] = cls[rr].intercept_[0]
			else:
				W = np.dot(invRM, np.dot(KK.T, H))


		self.W = W
		self.trainlabel = trainlabel
		self.H = np.copy(H)
		self.b = b


	def queryhash(self, qdata):
		Kdata = self.kernel(qdata, self.anchors)
		Kdata -= self.mvec
		Y = np.dot(Kdata, self.W) + self.b
		Y = np.where(Y>=0, 1, 0)
		return hash_value(Y)

	def basehash(self, data):
		# for symmetric hashing
		return self.queryhash(data)

		# for asymmetric hashing
		# return hash_value(np.where(self.H>0, 1, 0))


def RBF(X, Y):
	lenX = X.shape[0]
	lenY = Y.shape[0]
	X2 = np.dot(np.sum(X * X, axis=1).reshape((lenX, 1)), np.ones((1, lenY), dtype=np.float32))
	Y2 = np.dot(np.ones((lenX, 1), dtype=np.float32), np.sum(Y * Y, axis=1).reshape((1, lenY)))
	return np.exp(2*np.dot(X,Y.T) - X2 - Y2)

def test(n_bit):
	np.random.seed(17)
	X = np.load('cifar10_data/cifar10_gist.npy')
	Y = np.load('cifar10_data/cifar10_label.npy')

	traindata = X[:59000]
	trainlabel = Y[:59000]
	basedata = X[:59000]
	baselabel = Y[:59000]
	testdata = X[59000:]
	testlabel = Y[59000:]

	# train model
	dish = DISH_K(n_bit, 1000, 10, RBF)
	tic = time.time()
	dish.train(traindata, trainlabel)
	toc = time.time()
	print 'time:', toc-tic

	H_test = dish.queryhash(testdata)
	H_base = dish.basehash(basedata)

	# make labels
	gnd_truth = np.array([y == baselabel for y in testlabel]).astype(np.int8)

	print 'testing...'

	res = hash_evaluation(H_test, H_base, gnd_truth, 59000)
	print 'MAP:', res['map']

if __name__ == "__main__":
	test(64)
