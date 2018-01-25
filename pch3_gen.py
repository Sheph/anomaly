#!/usr/bin/env python3

import cv2
import numpy as np
import pandas as pd
import json

def cart2pol(x, y):
	rho = np.sqrt(x**2 + y**2)
	phi = np.arctan2(y, x)
	return(rho, phi)

def process(cap, preds):
	scale_height = 240
	target_fps = 10
	obj_min_h_prc=1
	obj_min_w_prc=1
	obj_max_h_prc=60
	obj_max_w_prc=60

	#fps = cap.get(cv2.CAP_PROP_FPS)
	fps = 25

	print(fps)

	ok, frame = cap.read()

	aspect = float(frame.shape[1]) / frame.shape[0]

	frame = cv2.resize(frame, (int(scale_height * aspect), scale_height), interpolation = cv2.INTER_AREA)

	frame_h, frame_w = frame.shape[:2]

	obj_min_w=obj_min_w_prc * scale_height / 100
	obj_min_h=obj_min_h_prc * scale_height / 100
	obj_max_w=obj_max_w_prc * scale_height / 100
	obj_max_h=obj_max_h_prc * scale_height / 100

	pred_i = 0

	trackers = []

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

		cur_pred = preds[pred_i]

		frame = cv2.resize(frame, (frame_w, frame_h), interpolation = cv2.INTER_AREA)

		orig_frame = frame.copy()

		frm = cap.get(cv2.CAP_PROP_POS_FRAMES)

		for tr in trackers:
			tr_obj = tr[0]
			bbox = tr[1]
			#c_speed_x = tr[2]
			#c_speed_y = tr[3]
			#c_cnt = tr[4]
			ok2, ubbox = tr_obj.update(orig_frame)
			ubbox = (int(ubbox[0]), int(ubbox[1]), int(ubbox[2]), int(ubbox[3]))
			if ok2:
				tr[2] = ubbox
				cx = bbox[0] + bbox[2] / 2.0
				cy = bbox[1] + bbox[3] / 2.0

				mx = ubbox[0] + ubbox[2] / 2.0
				my = ubbox[1] + ubbox[3] / 2.0

				tr[3] += mx - cx
				tr[4] += my - cy
				tr[5] += 1
			else:
				if tr[5] > 0:
					tr[2] = None
				tr[5] = 0

		for tr in trackers:
			tr_obj = tr[0]
			bbox = tr[1]
			ubbox = tr[2]
			if ubbox is None:
				continue
			cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
			cv2.rectangle(frame, (ubbox[0], ubbox[1]), (ubbox[0] + ubbox[2], ubbox[1] + ubbox[3]), (0, 0, 255), 2)

			cx = bbox[0] + bbox[2] / 2.0
			cy = bbox[1] + bbox[3] / 2.0
			w = float(bbox[2])
			h = float(bbox[3])

			mx = ubbox[0] + ubbox[2] / 2.0
			my = ubbox[1] + ubbox[3] / 2.0

			Rf = tr[6]
			sx = mx - cx
			sy = my - cy

			#sx, sy = cart2pol(sx, sy)

			#w = w * h
			#h = 0

			features.append([cap.get(cv2.CAP_PROP_POS_FRAMES) / fps, cap.get(cv2.CAP_PROP_POS_FRAMES), cx, cy, w, h, Rf, sx, sy])

			tr[1] = tr[2]

			if tr[5] == 0:
				tr[2] = None

		if trackers:
			cv2.imshow('frame', frame)

		if frm >= cur_pred[0]:
			boxes = cur_pred[1]
			trackers = []
			for b in boxes:
				cls = b[0]
				clsid = b[1]
				prob = b[2]
				x = int(b[3] * frame_w)
				y = int(b[4] * frame_h)
				w = int(b[5] * frame_w)
				h = int(b[6] * frame_h)

				if cls != "person":
					continue

				if (w >= obj_min_w) & (h >= obj_min_h) & (w <= obj_max_w) & (h <= obj_max_h):
					cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
					# should use 0.1 as detect_thresh, but no python api :(
					# change in c++ source
					tr = cv2.TrackerKCF_create()
					#tr = cv2.Tracker_create('KCF')
					tr.init(orig_frame, (x, y, w, h))
					tr.update(orig_frame)
					trackers.append([tr, (x, y, w, h), (x, y, w, h), 0, 0, 0, prob])

			pred_i += 1
			if pred_i >= len(preds):
				break

		k = cv2.waitKey(1) & 0xff
		if k == 32:
			k = cv2.waitKey() & 0xff
		if k == 27:
			break

	print("Done!")
	df = pd.DataFrame(features, columns = ["time", "frame", "x", "y", "w", "h", "Rf", "mx", "my"])
	df.to_csv("data_kitchen3.csv", encoding='utf-8')

if __name__ == "__main__":
	#cap = cv2.VideoCapture('Datasets/UCSDPed1/combined/test.avi')
	#process(cap, json.load(open('Datasets/UCSDPed1/combined/test_boxes.json')))
	#cap = cv2.VideoCapture('Datasets/Pedestrian/train.avi')
	#process(cap, json.load(open('ped_train_boxes.json')))
	#cap = cv2.VideoCapture('Datasets/Crossroads1/test.avi')
	#process(cap, json.load(open('cross1_test_boxes.json')))
	#cap = cv2.VideoCapture('z3.avi')
	#process(cap, json.load(open('z3_boxes.json')))

	#cap = cv2.VideoCapture('reception_short_train.avi')
	#process(cap, json.load(open('reception_short_train_boxes.json')))

	cap = cv2.VideoCapture('kitchen3.avi')
	process(cap, json.load(open('kitchen3.json')))

	cap.release()
	cv2.destroyAllWindows()
