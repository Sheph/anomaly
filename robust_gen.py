#!/usr/bin/env python3

import cv2
import numpy as np
import pandas as pd
import math, pickle

def calc_bcd(flow, cell_size, Ndirs, mag_thres, Tb):
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
	scale_height = 240
	target_fps = 10
	cell_size = 16
	Ndirs = 8
	mag_thres = 0.05
	Tb = 0.2

	#fps = cap.get(cv2.CAP_PROP_FPS)
	fps = 25

	print(fps)

	ok, frame = cap.read()

	aspect = float(frame.shape[1]) / frame.shape[0]

	frame = cv2.resize(frame, (int(scale_height * aspect), scale_height), interpolation = cv2.INTER_AREA)

	frame_h, frame_w = frame.shape[:2]

	print(frame_w, frame_h)

	assert(frame_w % cell_size == 0)
	assert(frame_h % cell_size == 0)

	Nx = int(frame_w / cell_size)
	Ny = int(frame_h / cell_size)

	print(Nx, Ny)

	prevgray = None

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

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		if prevgray is not None:
			flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
			bcd = calc_bcd(flow, cell_size, Ndirs, mag_thres, Tb)

			features.append([cap.get(cv2.CAP_PROP_POS_FRAMES) / fps, cap.get(cv2.CAP_PROP_POS_FRAMES), bcd])

			cv2.imshow('frame', frame)
		prevgray = gray

		k = cv2.waitKey(1) & 0xff
		if k == 32:
			k = cv2.waitKey() & 0xff
		if k == 27:
			break

	print("Done!")
	with open("robust_1.bin", "wb") as f:
		pickle.dump(features, f)

if __name__ == "__main__":
	cap = cv2.VideoCapture('reception_test_weirdo.avi')
	process(cap)

	cap.release()
	cv2.destroyAllWindows()
