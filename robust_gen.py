#!/usr/bin/env python3

import cv2
import numpy as np
import pandas as pd
import math, pickle

def aspect_fit(where, what):
	aspect = float(what[0]) / what[1]
	if aspect >= 1.0:
		rect = [0.0, 0.0, float(where[0]), float(where[0]) / aspect]
		if rect[3] <= where[1]:
			rect[1] = float(where[1] - rect[3]) / 2.0
		else:
			rect[3] = float(where[1])
			rect[2] = float(where[1]) * aspect
			rect[0] = float(where[0] - rect[2]) / 2.0
	else:
		rect = [0.0, 0.0, float(where[1]) * aspect, float(where[1])]
		if rect[2] <= where[0]:
			rect[0] = float(where[0] - rect[2]) / 2.0
		else:
			rect[2] = float(where[0])
			rect[3] = float(where[0]) / aspect
			rect[1] = float(where[1] - rect[3]) / 2.0

	return (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))

def resize_fit(frame, desired_size):
	r = aspect_fit(desired_size, (frame.shape[1], frame.shape[0]))

	frame = cv2.resize(frame, (r[2], r[3]), interpolation = cv2.INTER_AREA)

	color = [0, 0, 0]
	return cv2.copyMakeBorder(frame, r[1], desired_size[1] - r[1] - r[3], r[0], desired_size[0] - r[0] - r[2], cv2.BORDER_CONSTANT, value=color)

def calc_bcd_space(flow, cell_size, Ndirs, mag_thres, Tb):
	h, w = flow.shape[1:3]

	Nx = int(w / cell_size)
	Ny = int(h / cell_size)

	bcd = np.zeros((Ny, Nx, Ndirs + 1), np.uint8)

	hofbins = np.arange(-math.pi, math.pi + 1e-6, 2 * math.pi / Ndirs)

	fx, fy = flow[:,:,:,0], flow[:,:,:,1]
	orientation = np.arctan2(fy, fx)
	magnitude = np.sqrt(fx*fx+fy*fy)
	for i in range(Nx):
		for j in range(Ny):
			or1 = orientation[:, j * cell_size:(j + 1) * cell_size, i * cell_size:(i + 1) * cell_size]
			mag1 = magnitude[:, j * cell_size:(j + 1) * cell_size, i * cell_size:(i + 1) * cell_size]

			mag_gz = mag1 > mag_thres
			pruned_or1 = or1[mag_gz]

			hist, _ = np.histogram(pruned_or1.flatten(), bins=hofbins)
			hist = np.insert(hist, 0, np.count_nonzero(mag_gz == 0))
			hist = hist.astype(np.float32) / (cell_size * cell_size)
			hist = (hist >= Tb).astype(np.uint8)

			bcd[j, i, :] = hist

	return bcd

def calc_bcd_time(flow, cell_size, Ndirs, mag_thres, Tb):
	h, w = flow.shape[:2]

	Nx = int(w / cell_size)
	Ny = int(h / cell_size)

	bcd = np.zeros((Ny, Nx, Ndirs + 1), np.uint8)

	hofbins = np.arange(-math.pi, math.pi + 1e-6, 2 * math.pi / Ndirs)

	fx, fy = flow[:,:,0], flow[:,:,1]
	orientation = np.arctan2(fy, fx)
	magnitude = np.sqrt(fx*fx+fy*fy)
	for i in range(Nx):
		for j in range(Ny):
			or1 = orientation[j * cell_size:(j + 1) * cell_size, i * cell_size:(i + 1) * cell_size]
			mag1 = magnitude[j * cell_size:(j + 1) * cell_size, i * cell_size:(i + 1) * cell_size]

			mag_gz = mag1 > mag_thres
			pruned_or1 = or1[mag_gz]

			hist, _ = np.histogram(pruned_or1.flatten(), bins=hofbins)
			hist = np.insert(hist, 0, np.count_nonzero(mag_gz == 0))
			hist = hist.astype(np.float32) / (cell_size * cell_size)
			hist = (hist >= Tb).astype(np.uint8)

			bcd[j, i, :] = hist

	return bcd

def process(cap):
	scale_width = 400
	scale_height = 256
	target_fps = 10
	cell_size = 16
	Ndirs = 8
	mag_thres = 0.05
	Tb = 0.2
	Tsec = 1

	T = Tsec * target_fps

	#fps = cap.get(cv2.CAP_PROP_FPS)
	fps = 25

	print(fps)

	ok, frame = cap.read()

	frame = resize_fit(frame, (scale_width, scale_height))

	frame_h, frame_w = frame.shape[:2]

	print(frame_w, frame_h)

	assert(frame_w % cell_size == 0)
	assert(frame_h % cell_size == 0)

	Nx = int(frame_w / cell_size)
	Ny = int(frame_h / cell_size)

	print(Nx, Ny, T)

	prevgray = None

	features_space = []
	features_time = []

	flow_accum = []

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

		frame = resize_fit(frame, (scale_width, scale_height))

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		if prevgray is not None:
			flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
			bcd = calc_bcd_time(flow, cell_size, Ndirs, mag_thres, Tb)
			features_time.append([cap.get(cv2.CAP_PROP_POS_FRAMES) / fps, cap.get(cv2.CAP_PROP_POS_FRAMES), bcd])
			#flow_accum.append(flow)
			#if len(flow_accum) >= T:
			#	flow = np.array(flow_accum)
			#	flow_accum = []
			#	bcd = calc_bcd_space(flow, cell_size, Ndirs, mag_thres, Tb)
			#	features_space.append([cap.get(cv2.CAP_PROP_POS_FRAMES) / fps, cap.get(cv2.CAP_PROP_POS_FRAMES), bcd])
			cv2.imshow('frame', frame)
		prevgray = gray

		k = cv2.waitKey(1) & 0xff
		if k == 32:
			k = cv2.waitKey() & 0xff
		if k == 27:
			break

	features = { "frame_h" : frame_h, "frame_w" : frame_w, "T" : T, "Tsec" : Tsec, "features" : features_time }

	print("Done!")
	with open("robust_3.bin", "wb") as f:
		pickle.dump(features, f)

if __name__ == "__main__":
	#cap = cv2.VideoCapture('reception_test_weirdo.avi')
	cap = cv2.VideoCapture('Datasets/Pedestrian/test.avi')
	#cap = cv2.VideoCapture('kitchen1.avi')



	process(cap)

	cap.release()
	cv2.destroyAllWindows()
