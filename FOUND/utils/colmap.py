import numpy as np
import json
import os

def _remove_ext(f):
	return os.path.splitext(f)[0]

def load_colmap_data(colmap_json: str, image_list: list = None):
	"""Load R, t values from COLMAP for images,
	image_list = optional list N
	returns dict of R [ N x 3 x 3 ], t [ N x 3 ], params (camera params).

	Note that coordinates have already been converted into PyTorch3D's coord system in the data generation script."""

	with open(colmap_json) as infile:
		data = json.load(infile)

	if image_list is None:
		image_list = [x['pth'] for x in data['images']]

	N = len(image_list)
	R = {}
	T = {}

	for n, i in enumerate(image_list):
		idx = [x for x in data['images'] if x['pth'] == i]
		if len(idx) == 0:
			raise ValueError(f"No COLMAP data found for {i}")
		elif len(idx) > 1:
			raise ValueError(f"{len(idx)} COLMAP data points found for {i}")

		image_data = idx[0]
		pth = _remove_ext(image_data['pth'])

		R[pth] = np.array(image_data['R']).T
		T[pth] = np.array(image_data['T'])

	return dict(image_list=image_list, R=R, T=T, params=data['camera'])

def quat_to_3x3(quat):
	q0 = quat[0]
	q1 = quat[1]
	q2 = quat[2]
	q3 = quat[3]
	r00 = 2 * (q0 * q0 + q1 * q1) - 1
	r01 = 2 * (q1 * q2 - q0 * q3)
	r02 = 2 * (q1 * q3 + q0 * q2)
	r10 = 2 * (q1 * q2 + q0 * q3)
	r11 = 2 * (q0 * q0 + q2 * q2) - 1
	r12 = 2 * (q2 * q3 - q0 * q1)
	r20 = 2 * (q1 * q3 - q0 * q2)
	r21 = 2 * (q2 * q3 + q0 * q1)
	r22 = 2 * (q0 * q0 + q3 * q3) - 1
	return np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])

"""
Load raw colmap data from automatic reconstruction
"""
def raw_load_colmap_data(colmap_dir: str, image_list: list = None):
	COLMAP_CAMERAS = "/cameras.txt"
	COLMAP_IMAGES = "/images.txt"
	EXTENSION = ".jpg"
	
	rot = {}
	tr = {}
	params = {}
	
	with open(colmap_dir + COLMAP_IMAGES, "r") as imgfile:
		data = imgfile.read().replace('\n', ' ').split(' ')
		
		for i, token in enumerate(data):
			if token.endswith(EXTENSION):
				# We've reached the last image description token - filename
				# Now, read the previous 8 tokens.
				img_params = data[i-8:i+1]
				q3 = img_params[0]
				q0 = img_params[1]
				q1 = img_params[2]
				q2 = img_params[3]
				tx = img_params[4]
				ty = img_params[5]
				tz = img_params[6]
				img_name = _remove_ext(img_params[8])
				
				rot[img_name] = quat_to_3x3([float(q0), float(q1), float(q2), float(q3)])
				tr[img_name] = np.array([tx, ty, tz]).astype(float)
	
	with open(colmap_dir + COLMAP_CAMERAS, "r") as camfile:
		data = camfile.readlines()[3].replace('\n', ' ').split(' ')
		params["f"] = float(data[4])
		params["cx"] = float(data[5])
		params["cy"] = float(data[6])
	
	return dict(image_list=image_list, R=rot, T=tr, params=params)