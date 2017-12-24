#!/usr/bin/env python2

import numpy as np
from sklearn import hmm

if __name__ == "__main__":
	#startprob = np.array([0.6, 0.3, 0.1])
	#transmat = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.3, 0.3, 0.4]])
	#means = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
	#covars = np.tile(np.identity(2), (3, 1, 1))
	#model = hmm.GaussianHMM(3, "full", startprob, transmat)
	#model.means_ = means
	#model.covars_ = covars
	#X, Z = model.sample(100)
	#print(X)
	#print(Z)

	model = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=1000)
	#data = np.array([[0.7, 0.3], [0.3, 0.7], [0.6, 0.4], [0.9, 0.1]])
	a = [0.1, 0.2, 0.3, 0.4]
	b = [0.9, 0.8, 0.7, 0.6]
	data = np.column_stack([a, b])

	print(data)

	model.fit([data])

	np.set_printoptions(precision=2)
	np.set_printoptions(suppress=True)

	print(model.transmat_)
	#print(model.statprob_)

	test = np.array([[0.3, 0.7], [0.6, 0.4], [0.6, 0.4], [0.6, 0.4]])

	print(model.predict(data))
	print(model.score(data))
	print(model.score(test))
	print(model.sample(3))
