#!/usr/bin/env python2

import cv2
import cv
import numpy as np
from sklearn import hmm
from sklearn import mixture
import time
import sys
import scipy.ndimage

pch1_s = 12.0
pch1_t = 10.0
th = 180
tm = 10
frame_skip = 0
num_learn_frames = 10000

#pch1_s = 1.0
#pch1_t = 1.0

def find_blobs(img):
	img_32 = np.int32(img)
	img_32[img_32 > 0] = 1

	label_count = 2

	blobs = []

	for y in xrange(0, img_32.shape[0]):
		row = img_32[y]
		if np.count_nonzero(row) == 0:
			continue;
		for x in xrange(0, img_32.shape[1]):
			if row[x] != 1:
				continue
			rect = cv2.floodFill(img_32, None, (x, y), label_count, 0, 0, 4)[1]
			minx = 10000;
			miny = 10000;
			maxx = 0;
			maxy = 0;
			for i in xrange(rect[1], rect[1] + rect[3]):
				row2 = img_32[i]
				for j in xrange(rect[0], rect[0] + rect[2]):
					if row2[j] != label_count:
						continue;
					if j < minx:
						minx = j;
					if i < miny:
						miny = i;
					if j > maxx:
						maxx = j;
					if i > maxy:
						maxy = i;
			label_count += 1
			blobs.append(((minx, miny), (maxx, maxy)))

	#print(len(blobs))
	return blobs

def calc_pch1(pch, fgmask):
	pch[fgmask > 0] += 255.0 / pch1_s
	pch[pch > 255] = 255
	pch[fgmask <= 0] -= 255.0 / pch1_t
	pch[pch < 0] = 0

def classify(features):
	lowest_bic = np.infty
	bic = []
	n_components_range = range(5, 15)
	#cv_types = ['spherical', 'tied', 'diag', 'full']
	cv_types = ['full']
	best_gmm = None
	print("training...")
	for cv_type in cv_types:
		for n_components in n_components_range:
			gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type)
			gmm.fit(features)
			bic.append(gmm.bic(features))
			if bic[-1] < lowest_bic:
				lowest_bic = bic[-1]
				best_gmm = gmm
	print(best_gmm.n_components)

if __name__ == "__main__":
	#print(cv2.__version__)

	#cv2.

	pch1 = None

	cap = cv2.VideoCapture('2.avi')
	bg_subtractor = cv2.BackgroundSubtractorMOG2(history=500, varThreshold=16.0, bShadowDetection=False);
	bg_subtractor.setInt("nmixtures", 6)
	bg_subtractor.setDouble("backgroundRatio", 0.7)

	#bg_subtractor = cv2.BackgroundSubtractorMOG();

	params = cv2.SimpleBlobDetector_Params()

	# Change thresholds
	params.minThreshold = 0
	params.maxThreshold = 255


	# Filter by Area.
	#params.filterByArea = True
	#params.minArea = 1500

	# Filter by Circularity
	#params.filterByCircularity = True
	#params.minCircularity = 0.1

	# Filter by Convexity
	#params.filterByConvexity = True
	#params.minConvexity = 0.87

	# Filter by Inertia
	#params.filterByInertia = True
	#params.minInertiaRatio = 0.01

	blob_detector = cv2.SimpleBlobDetector(params)

	prev_gray = None

	frame_num = 0

	features = []

	while cap.isOpened():
		num_learn_frames -= 1

		if num_learn_frames < 0:
			break;

		ret, frame = cap.read()

		if not ret:
			break

		frame_num -= 1

		if (frame_num > 0):
			continue;

		frame_num = frame_skip

		aspect = float(frame.shape[1]) / frame.shape[0]

		frame = cv2.resize(frame, (int(480 * aspect), 480), interpolation = cv.CV_INTER_AREA)

		gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

		gray = cv2.GaussianBlur(gray, (21, 21), 0)

		fgmask = bg_subtractor.apply(gray, learningRate=0.002)

		#fgmask[fgmask < 255] = 0

		#frame[fgmask <= 0] = 0

		cv2.namedWindow("fgmask", cv.CV_WINDOW_NORMAL);
		cv2.imshow("fgmask", fgmask)

		if pch1 is None:
			pch1 = np.zeros(gray.shape, np.float);

		calc_pch1(pch1, fgmask);

		pixel_level_events = pch1.copy();
		pixel_level_events[pch1 < th] = 0;

		pixel_level_events = np.uint8(pixel_level_events)

		cv2.namedWindow("ple", cv.CV_WINDOW_NORMAL);
		cv2.imshow("ple", pixel_level_events)

		#kernel = np.ones((2, 2), np.uint8)
		#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
		#pixel_level_events = cv2.erode(pixel_level_events, kernel, iterations=1)
		#pixel_level_events = cv2.morphologyEx(pixel_level_events, cv2.MORPH_OPEN, kernel)

		#blobs = find_blobs(pixel_level_events)

		frame_cp = frame.copy()

		labeled_array, num_features = scipy.ndimage.measurements.label(pixel_level_events)

		blobs = scipy.ndimage.measurements.find_objects(labeled_array, num_features)
		for b in blobs:
			sx = b[1].stop - b[1].start;
			sy = b[0].stop - b[0].start;
			if sx > 5 or sy > 5:
				b_data = pixel_level_events[b]
				Rf = float(np.count_nonzero(b_data)) / b_data.size
				diff = (np.abs(gray[b] - prev_gray[b])) > tm
				Rm = float(np.count_nonzero(diff)) / diff.size
				x = (b[1].start + b[1].stop) / 2
				y = (b[0].start + b[0].stop) / 2
				cv2.rectangle(frame_cp, (b[1].start, b[0].start), (b[1].stop, b[0].stop), (0,0,255), 2)
				features.append([x, y, sx, sy, Rf, Rm])

		#for b in blobs:
		#	cv2.rectangle(frame, b[0], b[1], (0,0,255), 2)

		#keypoints = blob_detector.detect(pixel_level_events)

		#if len(keypoints) > 0:
		#	keypoints[0].size *= 5;

		#frame = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

		#cv2.namedWindow("ple_opened", cv.CV_WINDOW_NORMAL);
		#cv2.imshow("ple_opened", pixel_level_events)

		cv2.imshow('frame',frame_cp)
		cv2.waitKey(1)

		prev_gray = gray

		#if len(keypoints) > 0:
		#	pass
			#print(keypoints[0].pt)
			#print(keypoints[0].size)
			#time.sleep(1)

	#print(pch1_masked[0, 0]);

	classify(np.array(features))

	time.sleep(100);

	cap.release()
	cv2.destroyAllWindows()
