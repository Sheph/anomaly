#!/usr/bin/env python3

import cv2
import numpy as np
import pandas as pd
import seaborn as sns

#from sklearn import hmm
from sklearn import mixture
from sklearn.decomposition import PCA
import time
import sys
import scipy.ndimage
import MOG3
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LogNorm
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

def classify(features):
	lowest_bic = np.infty
	#bic = []
	n_components_range = range(1, 5)
	#cv_types = ['spherical', 'tied', 'diag', 'full']
	cv_types = ['full']
	best_gmm = None
	for cv_type in cv_types:
		for n_components in n_components_range:
			gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type, random_state=0)
			gmm.fit(features)
			cbic = gmm.bic(features)
			print(cbic)
			if cbic < lowest_bic:
				lowest_bic = cbic
				best_gmm = gmm
	print(best_gmm.n_components)
	return best_gmm

def classify2(features):
	gmm = mixture.BayesianGaussianMixture(n_components=10, random_state=0, covariance_type='full')
	gmm.fit(features)
	return gmm

def classify3(features):
	newdata = features
	#newdata = scipy.ndimage.filters.gaussian_filter(features, (1.5, 1.5))
	newdata = scipy.ndimage.filters.gaussian_filter(features, (1.5, 1.5))
	n_components = np.arange(1, 15)
	BIC = np.zeros(n_components.shape)
	lowest_bic = np.infty
	best_gmm = None

	for i, n in enumerate(n_components):
		gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=0)
		gmm.fit(newdata)
		BIC[i] = gmm.bic(newdata)
		print(BIC[i])
		if BIC[i] < lowest_bic:
			lowest_bic = BIC[i]
			best_gmm = gmm

	print(best_gmm.n_components)
	plt.plot(BIC)
	plt.show()
	return best_gmm

if __name__ == "__main__":
	df = pd.read_csv("data.csv")
	df.drop(df.columns[0], inplace=True, axis=1)
	df.drop(["w", "h", "mx", "my", "Rf"], inplace=True, axis=1)

	scaler = StandardScaler()





	#df = df.head(500)

	df2 = df.drop(df.columns[0], axis=1)



	#df2 = pd.DataFrame(scaler.fit_transform(df2), columns=df.columns[1:])

	print(df2.describe())

	#x1, y1 = np.random.multivariate_normal([3, 3], [[0.1, 0.1], [0.1, 0.2]], int(len(df2) / 2) + 1).T
	#x2, y2 = np.random.multivariate_normal([3, 3], [[-1.1, -1.1], [-1.1, 1.2]], int(len(df2) / 2)).T
	#df2["x"] = np.concatenate((x1, x2))
	#df2["y"] = np.concatenate((y1, y2))

	pca = PCA(n_components=2)
	X1 = pca.fit_transform(df2)

	#bgmm = classify(df2)
	#bgmm = classify2(df2)
	bgmm = classify3(df2)

	labels = bgmm.predict(df2)

	#print(len(set(labels)))

	#plt.scatter(X1[:, 0], X1[:, 1], c=labels)
	plt.scatter(df2["x"], df2["y"], c=labels, s=40)
	plt.scatter(bgmm.means_[:,1], bgmm.means_[:,0], s=120)


	sns.jointplot(x=df2["x"], y=df2["y"], kind="hex", color="k");

	#plt.hist2d(df2["x"], df2["y"], bins=[16, 9], norm=LogNorm())

	plt.show()

	#fig = plt.figure(1, figsize=(4, 3))
	#plt.clf()

	#ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

	#ax.scatter(df2["x"], df2["y"], np.histogram2d(df2["x"], df2["y"], bins=(17, 10)))

	#plt.show()

	#hist, xedges, yedges = np.histogram2d(df2["x"], df2["y"], bins=(17, 10))

	#xpos, ypos = np.meshgrid(xedges[:-1] + 0.1, yedges[:-1] + 0.1)
	#xpos = xpos.flatten('F')
	#ypos = ypos.flatten('F')
	#zpos = np.zeros_like(xpos)

	#dx = 0.01 * np.ones_like(zpos)
	#dy = dx.copy()
	#dz = hist.flatten()

	#fig = plt.figure()
	#ax = fig.add_subplot(111, projection='3d')

	#ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

	#plt.show()


	mlabel = np.max(labels)

	cap = cv2.VideoCapture('z3.avi')


	for i in range(mlabel + 1):
		print("Now playing cluster %d!" % i)
		frms = df["frame"][labels == i]
		print(len(frms)/len(df["frame"]))
		frms = np.unique(frms)
		for frm in frms:
			cap.set(cv2.CAP_PROP_POS_FRAMES, frm)
			ok, frame = cap.read()
			aspect = float(frame.shape[1]) / frame.shape[0]
			frame = cv2.resize(frame, (int(240 * aspect), 240), interpolation = cv2.INTER_AREA)
			cv2.imshow('frame', frame)
			k = cv2.waitKey(1) & 0xff
			if k == 32:
				k = cv2.waitKey() & 0xff
			if k == 27:
				break
		print("Done!")
		k = cv2.waitKey()

	cap.release()


	#pca = PCA(n_components=3)
	#X2 = pca.fit_transform(df2)

	#fig = plt.figure(1, figsize=(4, 3))
	#plt.clf()

	#ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

	#ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2])

	#plt.show()


