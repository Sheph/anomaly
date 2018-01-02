#!/usr/bin/env python3

import cv2
import numpy as np
import pandas as pd
#from sklearn import hmm
#from sklearn import mixture
#import time
#import sys
#import scipy.ndimage
import scipy.ndimage as ndimage

scale_height = 240
target_fps = 8
bg_learning_rate_sec = 8
obj_min_w_prc=6
obj_min_h_prc=6
obj_max_w_prc=60
obj_max_h_prc=60
pch1_s_sec = 2.0
pch1_t_sec = 5.0 / 3.0
ple_th = 180
ple_tm = 10
ple_tb = 100
bin_size_prc=10

aspect = 1
bg_sub = None
frame_h = 0
frame_w = 0
fps = 0
bg_learning_rate = 0
bg_kernel = None
frame_blur = 0
obj_min_w = 0
obj_min_h = 0
obj_max_w = 0
obj_max_h = 0
bin_size_w = 0
bin_size_h = 0

pch1_s = 0.0
pch1_t = 0.0

def calc_pch1(pch, fgmask):
	global pch1_s
	global pch1_t
	pch[fgmask > 0] += 255.0 / pch1_s
	pch[pch > 255] = 255
	pch[fgmask <= 0] -= 255.0 / pch1_t
	pch[pch < 0] = 0

def apply_bg_sub(frame):
	mask = bg_sub.apply(frame, bg_learning_rate)

	bg = bg_sub.getBackgroundImage()
	bg_gray = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	frame_gray_b = cv2.GaussianBlur(frame_gray, (frame_blur, frame_blur), 0)
	bg_gray_b = cv2.GaussianBlur(bg_gray, (frame_blur, frame_blur), 0)
	_, thr = cv2.threshold(cv2.absdiff(bg_gray_b, frame_gray_b), 20, 255, cv2.THRESH_BINARY);
	mask = cv2.bitwise_and(mask, thr)

	mask_127 = (mask == 127).astype('uint8') * 255
	mask_255 = (mask == 255).astype('uint8') * 255
	mask_255_d = cv2.dilate(mask_255, bg_kernel, iterations = 1)
	mask_127_d = cv2.dilate(mask_127, bg_kernel, iterations = 1)
	mask = (mask_255_d > 0) & ( (mask_127_d > 0) | (mask_255 > 0) )
	return ((mask > 0).astype('uint8') * 255), frame_gray_b

def process(cap):
	global scale_height
	global target_fps
	global bg_learning_rate_sec

	global aspect
	global bg_sub
	global frame_h
	global frame_w
	global fps
	global bg_learning_rate
	global bg_kernel
	global frame_blur
	global obj_min_w
	global obj_min_h
	global obj_max_w
	global obj_max_h
	global pch1_s
	global pch1_t
	global bin_size_w
	global bin_size_h

	fps = cap.get(cv2.CAP_PROP_FPS)

	bg_learning_rate = 1.0 / (bg_learning_rate_sec * target_fps)

	ok, frame = cap.read()

	aspect = float(frame.shape[1]) / frame.shape[0]

	bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=17, detectShadows=False)
	bg_sub.setNMixtures(5)
	bg_sub.setBackgroundRatio(0.5)

	frame = cv2.resize(frame, (int(scale_height * aspect), scale_height), interpolation = cv2.INTER_AREA)

	frame_h, frame_w = frame.shape[:2]

	bg_ksize = int(frame_h / 105);
	bg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(bg_ksize, bg_ksize))

	erode_ksize = int(frame_h / 45);
	dilate_ksize = int(frame_h / 35);

	erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_ksize, erode_ksize))
	dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))

	frame_blur = int(frame_h / 45)

	obj_min_w=obj_min_w_prc * scale_height / 100
	obj_min_h=obj_min_h_prc * scale_height / 100
	obj_max_w=obj_max_w_prc * scale_height / 100
	obj_max_h=obj_max_h_prc * scale_height / 100

	pch1_s = pch1_s_sec * target_fps
	pch1_t = pch1_t_sec * target_fps

	bin_size_h = bin_size_prc * frame_h / 100
	bin_size_w = frame_w / int(frame_w / bin_size_h)

	num_bins_h = int(frame_h / bin_size_h)
	num_bins_w = int(frame_w / bin_size_w)

	assert(float(num_bins_w) * bin_size_w == frame_w)
	assert(float(num_bins_h) * bin_size_h == frame_h)

	######

	bg_learning_frames = bg_learning_rate_sec * target_fps

	pch1 = np.zeros((frame_h, frame_w), np.float);

	prev_gray = None

	features = []

	i = 0
	j = 0
	while ok:
		ok, frame = cap.read()
		if not ok:
			break
		j += 1
		if (j < fps / target_fps):
			continue
		j = 0
		i += 1

		frame = cv2.resize(frame, (frame_w, frame_h), interpolation = cv2.INTER_AREA)

		cv2.imshow('frame', frame)

		if bg_learning_frames > 0:
			bg_learning_frames -= 1
			bg_sub.apply(frame, bg_learning_rate)
			continue

		fgmask, gray = apply_bg_sub(frame)

		mask2 = cv2.erode(fgmask, erode_kernel, iterations=1)
		mask2 = cv2.dilate(mask2, dilate_kernel, iterations=2)

		cv2.imshow('mask2', mask2)

		calc_pch1(pch1, mask2);

		pixel_level_events = pch1.copy();
		pixel_level_events[pch1 < ple_th] = 0;

		pixel_level_events = np.uint8(pixel_level_events)

		pch1b = pch1.copy().astype('uint8');

		cv2.imshow("ple", pixel_level_events)
		cv2.imshow("pch1", pch1b)

		frame2 = frame.copy()

		labeled_array, num_features = ndimage.measurements.label(mask2)
		for f in range(1, num_features + 1):
			b = ndimage.find_objects(labeled_array == f)[0]
			x = b[1].start
			y = b[0].start
			w = b[1].stop - b[1].start;
			h = b[0].stop - b[0].start;

			b_data_pch = pch1b[b]
			nz_pch = float(np.count_nonzero(b_data_pch))

			pch1_m = pch1b[b].mean()

			#if (w >= obj_min_w) & (h >= obj_min_h) & (w <= obj_max_w) & (h <= obj_max_h) & (prev_gray is not None) & (pch1[b].mean() > ple_tb):
			if (w >= obj_min_w) & (h >= obj_min_h) & (w <= obj_max_w) & (h <= obj_max_h) & (prev_gray is not None) & (nz_pch > 0):
				b_data = pixel_level_events[b]
				nz = float(np.count_nonzero(b_data))
				Rf = nz / b_data.size
				diff = (np.abs(gray[b] - prev_gray[b])) > ple_tm
				if nz == 0:
					Rm = 0.0
				else:
					Rm = float(np.count_nonzero(b_data[diff])) / nz
				#Rm = float(np.count_nonzero(diff)) / diff.size
				cx = (b[1].start + b[1].stop) / 2
				cy = (b[0].start + b[0].stop) / 2

				if nz_pch != 0:
					cv2.rectangle(frame2, (x, y), (x + w, y + h), (255, 255, 255), 2)
					#Rf = nz_pch / b_data_pch.size
					Rf = np.sum(b_data_pch) / (255 * b_data_pch.size)
					com = ndimage.measurements.center_of_mass(b_data_pch)
					cv2.circle(frame2, (x + int(com[1]), y + int(com[0])), 2, (255, 255, 255));
					cv2.circle(frame2, (int(cx), int(cy)), 2, (0, 0, 255));
					#features.append([cap.get(cv2.CAP_PROP_POS_FRAMES), cx, cy, w, h, (x + com[1] - cx), (y + com[0] - cy)])
					#features.append([cap.get(cv2.CAP_PROP_POS_FRAMES), cx, cy, (x + com[1] - cx), (y + com[0] - cy)])
					features.append([cap.get(cv2.CAP_PROP_POS_FRAMES), cx, cy, w, h, Rf, (x + com[1] - cx), (y + com[0] - cy)])
				#features.append([cap.get(cv2.CAP_PROP_POS_FRAMES), cx / frame_w, cy / frame_h, w / frame_w, h / frame_h, Rf])

		cv2.imshow('frame2', frame2)

		if prev_gray is not None:
			diff_mask = np.abs(gray - prev_gray) > ple_tm
			cv2.imshow('diff_mask', diff_mask.astype('uint8') * 255)

		prev_gray = gray

		k = cv2.waitKey(1) & 0xff
		if k == 32:
			k = cv2.waitKey() & 0xff
		if k == 27:
			break

	df = pd.DataFrame(features, columns = ["frame", "x", "y", "w", "h", "Rf", "mx", "my"])
	#df = pd.DataFrame(features, columns = ["frame", "x", "y", "mx", "my"])
	#df = pd.DataFrame(features, columns = ["frame", "x", "y", "w", "h", "Rf"])
	df.to_csv("data.csv", encoding='utf-8')

if __name__ == "__main__":
	cap = cv2.VideoCapture('test3.avi')
	process(cap)
	cap.release()
	cv2.destroyAllWindows()
