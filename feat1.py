#!/usr/bin/env python2

import psycopg2
import base64
import struct
import numpy as np
import hexdump

def decode_optical_flow(buf):
	ver, = struct.unpack('I', buf[0:4])
	if ver != 101:
		raise Exception('Bad format version: {0}'.format(ver))
	step, size = struct.unpack('II', buf[4:12])
	flow = struct.unpack('ff'*size, buf[12:-4])
	print size
	print hexdump.dump(buf[-48:], sep=":")
	aspect_ratio, = struct.unpack('f', buf[-4:])
	np_flow = np.array(flow).reshape((int(size/step), step, 2))
	return dict(flow = np_flow, aspect_ratio = aspect_ratio)

def decode_feature(feature_type, buf):
	if feature_type == 'OpticalFlow':
		return decode_optical_flow(buf)
	raise Exception('Cannot decode feature: {0}'.format(feature_type))

if __name__ == "__main__":
	conn = psycopg2.connect("host='127.0.0.1' dbname='trassir3' user='postgres' password='postgres'");
	c = conn.cursor()
	c.execute("SELECT ts, channel, type, data FROM features WHERE type = %s LIMIT 1", ("OpticalFlow",))
	rows = c.fetchall()
	for ts, channel, tp, data in rows:
		feat = decode_feature(tp, base64.b64decode(data))
		#print("%d %s %s %s" % (ts, channel, tp, str(feat)))
		#print len(feat['flow'][0])
		if feat['aspect_ratio'] != 0.0:
			print feat['aspect_ratio']
