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

def _rot_q_to_3x3(qvec):
	return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])

"""
Load raw colmap data from automatic reconstruction
"""

def _helper_read_colmap_images_txt(colmap_dir: str, image_list: list = None):
	COLMAP_IMAGES = "/images.txt"
	SUPPORTED_EXTENSIONS = tuple(".jpg", ".png")
	
	# Token-to-index lookup table for easier parsing.
	token_to_index = {
		'image_id': 0,
		'rot_qw': 1,
		'rot_qx': 2,
		'rot_qy': 3,
		'rot_qz': 4,
		'trans_x': 5,
		'trans_y': 6,
		'trans_z': 7,
		'camera_id': 8,
		'name': 9,
	}

	rot = {}
	tr = {}

	with open(colmap_dir + COLMAP_IMAGES, "r") as f:
		# Read the file contents into an array of lines where each line is a set of tokens that form it.
		data = f.readlines()
		data = [list(filter(None, line.replace('\n', '').split(' '))) for line in data]

		# Find each line containing a token that ends with .jpg
		for line in data:
			for token in line:
				if token.endswith(SUPPORTED_EXTENSIONS):
					if len(line) != 10:
						raise RuntimeError('malformed colmap images.txt file')
					
					if image_list != None and token not in image_list:
						continue
					
					# Parse the image info
					file_name_no_ext = _remove_ext(line[token_to_index['name']])
					rot_quat = np.array([float(v) for v in line[token_to_index['rot_qw']-1:token_to_index['rot_qz']]])
					tr_vec = np.array([float(v) for v in line[token_to_index['trans_x']-1:token_to_index['trans_z']]])
					# np.roll(rot_quat, -1)

					rot[file_name_no_ext] = _rot_q_to_3x3(rot_quat).T
					tr[file_name_no_ext] = tr_vec
		
	return dict(R=rot, T=tr)

def _helper_read_colmap_cameras_txt(colmap_dir: str):
	COLMAP_CAMERAS = "/cameras.txt"

	params = {}
	with open(colmap_dir + COLMAP_CAMERAS, 'r') as f:
		data = f.readlines()[3].replace('\n', ' ').split(' ')
		params['width'] = float(data[2])
		params['height'] = float(data[3])
		params['f'] = float(data[4])
		params['cx'] = float(data[5])
		params['cy'] = float(data[6])
		params['k'] = float(data[7])
	
	return params

def raw_load_colmap_data(colmap_dir: str, image_list: list = None):
	image_data = _helper_read_colmap_images_txt(colmap_dir, image_list)
	camera_data = _helper_read_colmap_cameras_txt(colmap_dir)
	return dict(image_list=image_list, R=image_data['R'], T=image_data['T'], params=camera_data)