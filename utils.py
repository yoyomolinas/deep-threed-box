
import numpy as np
from PIL import ImageOps, Image

def project_2d(K, coords_3d):
    """
    Project 3d coordinates onto the image plane by
    :param K: instrinsic camera matrix (3, 4)
    :param coords_3d: 3d coordinates of vertices in world space
    :return coords_2d: array of image coordinates ofv given vertices i.e. [(x, y, 1), ...]
    """
    coords_3d_homo = np.vstack([coords_3d, np.ones(coords_3d.shape[-1])])
    coords_2d = K.dot(coords_3d_homo)
    coords_2d = coords_2d / coords_2d[2]
    coords_2d = coords_2d[:2]
    return coords_2d

def compute_orientation(K, angle, bbox_2d):
	"""
	Compute global orientation of angle
	:param K: instrinsic camera matrix (3, 4)
	:param angle: angle of rotation araound y axis in radians in range 0, 2*pi
	:param bbox_2d: 2d bounding box coordinates in the image (xmin, ymin, xmax, ymax)
	:return: global angle of rotation around y axis in radians in any range
	"""
	xmin, ymin, xmax, ymax = bbox_2d
	x = (xmin + xmax) / 2
	focal_length = K[0, 0]
	u_distance = x - K[0, 2]
	rot_cam = np.arctan(u_distance/focal_length)
	rot_global = rot_cam + angle
	return rot_global

def compute_3d_coordinates(K, T, angle, dimensions, bbox_2d):
	"""
	Compute coordinates of object in world space given the following params
	:param K: instrinsic camera matrix (3, 4)
	:param T: object translation vector x, y, z in meters
	:param angle: angle of rotation araound y axis in radians in range 0, 2*pi
	:param dimensions: object dimensions (height, width, length)
	:param coords_3d: 3d coordinates in object space
	:param bbox_2d: 2d bounding box coordinates in the image (xmin, ymin, xmax, ymax)
	:return corners_3d: 3d coordinates of vertices in world space
	"""
	rot_global = compute_orientation(K, angle, bbox_2d)
	# rot_global = np.clip(rot_global - np.pi, -np.pi, np.pi) # shift to range -pi, pi
	print("rot global ", 180 * rot_global / np.pi)
	R = np.array([[np.cos(rot_global), 0, np.sin(rot_global)], 
					[0, 1, 0],
					[-np.sin(rot_global), 0, np.cos(rot_global)]])
	T = T.reshape(3, 1)
	height, width, length = dimensions
	x_corners = [-length/2, length/2, length/2, length/2, length/2, -length/2, -length/2, -length/2]
	y_corners = [-height, -height, 0, 0, -height, -height, 0, 0]
	z_corners = [-width/2, -width/2, -width/2, width/2, width/2, width/2, width/2, -width/2]
	corners_3d = np.array([x_corners, y_corners, z_corners])
	corners_3d = R.dot(corners_3d) 
	corners_3d += T	
	return corners_3d

def solve_for_translations(K, dimensions, rot_local, rot_global, bbox_2d):
	"""
	Compute translations in test time by solving a linear systems of equations
	:param K: instrinsic camera matrix (3, 4)
	:param rot_local: angle of rotation around y axis in radians in object space 
	:param rot_global: angle of rotation around y axis in radians in world space 
	:param bbox_2d: 2d bounding box coordinates in the image (xmin, ymin, xmax, ymax)
	:return: translation vector
	"""
	bbox = bbox_2d
	print("bbox : ", bbox)
	# rotation matrix
	R = np.array([[ np.cos(rot_global), 0,  np.sin(rot_global)],
					[          0,             1,             0          ],
					[-np.sin(rot_global), 0,  np.cos(rot_global)]])
	A = np.zeros((4, 3))
	b = np.zeros((4, 1))
	I = np.identity(3)

	xmin_corr, xmax_corr, ymin_corr, ymax_corr = corresponding_vertices(dimensions, rot_local, soft_range=8)

	X  = np.stack([xmin_corr, xmin_corr,
				   ymin_corr, ymax_corr])
	# X: [x, y, z] in object coordinate
	X = X.T
	print("shape: ", X.shape)

	# construct equation (4, 3 )
	for i in range(4):
		matrix = np.vstack([I, np.matmul(R, X[:,i])]).T
		last_row = np.zeros(4, )
		last_row[-1] = 1
		matrix = np.vstack([matrix, last_row])
		M = np.matmul(K, matrix)
		# x
		if i % 2 == 0:
			print("bbox[%i] "%i, bbox[i])
			A[i, :] = M[0, 0:3] - bbox[i] * M[2, 0:3]
			b[i, :] = M[2, 3] * bbox[i] - M[0, 3]
		# y
		else:
			A[i, :] = M[1, 0:3] - bbox[i] * M[2, 0:3]
			b[i, :] = M[2, 3] * bbox[i] - M[1, 3]
	# solve x, y, z, using method of least square
	translation_vector = np.matmul(np.linalg.pinv(A), b)
	

	tx, ty, tz = [float(np.around(tran, 2)) for tran in translation_vector]
	ret = np.array([tx, ty, tz])
	return ret

def corresponding_vertices(dimensions, rot_local, soft_range = 8):
	"""
	Find corners that correspond to xmin, ymin, xmax and ymax on the bounding box
	:param dimensions: height width length of 3d box
	:param rot_local: angle of rotation around y axis in radians in object space 
	:return : matrix of xmin, ymin, xmax, ymax in 3x4 shape
	"""
	height, width, length = dimensions
	# x_corners = [length, length, length, length, 0, 0, 0, 0]
	# y_corners = [height, height, 0, 0, height, height, 0, 0]
	# z_corners = [0, 0, 0, width, width, width, width, 0]

	# x_corners = [i - length / 2 for i in x_corners]
	# y_corners = [i - height for i in y_corners]
	# z_corners = [i - width / 2 for i in z_corners]
	# x_corners = [-length/2, length/2, length/2, length/2, length/2, -length/2, -length/2, -length/2]
	# y_corners = [-height, -height, 0, 0, -height, -height, 0, 0]
	# z_corners = [-width/2, -width/2, -width/2, width/2, width/2, width/2, width/2, -width/2]

	# corners_3d = np.transpose(np.array([x_corners, y_corners, z_corners]))
	# point0 = corners_3d[0, :]
	# point1 = corners_3d[1, :]
	# point2 = corners_3d[2, :]
	# point3 = corners_3d[3, :]
	# point4 = corners_3d[4, :]
	# point5 = corners_3d[5, :]
	# point6 = corners_3d[6, :]
	# point7 = corners_3d[7, :]
	x_corners = [length, length, length, length, 0, 0, 0, 0]
	y_corners = [height, 0, height, 0, height, 0, height, 0]
	z_corners = [0, 0, width, width, width, width, 0, 0]

	x_corners = [i - length / 2 for i in x_corners]
	y_corners = [i - height for i in y_corners]
	z_corners = [i - width / 2 for i in z_corners]

	corners_3d = np.transpose(np.array([x_corners, y_corners, z_corners]))
	point1 = corners_3d[0, :]
	point2 = corners_3d[1, :]
	point3 = corners_3d[2, :]
	point4 = corners_3d[3, :]
	point5 = corners_3d[6, :]
	point6 = corners_3d[7, :]
	point7 = corners_3d[4, :]
	point8 = corners_3d[5, :]

	# set up projection relation based on local orientation
	xmin_corr = xmax_corr = ymin_corr = ymax_corr = np.array([0, 0, 0])
	# print("rot local", 180 * rot_local / np.pi)
	rot_local = rot_local % (2 * np.pi)
	if 0 < rot_local < np.pi / 2:
		print('rot local < pi/2')
		xmin_corr = point8
		xmax_corr = point2
		ymin_corr = point2
		ymax_corr = point5

	if np.pi / 2 <= rot_local <= np.pi:
		print(' pi/2 < rot local < pi')
		xmin_corr = point6
		xmax_corr = point4
		ymin_corr = point5
		ymax_corr = point8

	if np.pi < rot_local <= 3 / 2 * np.pi:
		print(' np.pi < rot_local <= 3 / 2 * np.pi')
		# xmin_corr = point2
		# xmax_corr = point88
		# ymin_corr = point8
		# ymax_corr = point1
		xmin_corr = point4
		xmax_corr = point8
		ymin_corr = point6
		ymax_corr = point5
		print(xmin_corr, xmax_corr, ymin_corr, ymax_corr)

	if 3 * np.pi / 2 <= rot_local <= 2 * np.pi:
		print("3 * np.pi / 2 <= rot_local <= 2 * np.pi")
		xmin_corr = point4
		xmax_corr = point6
		ymin_corr = point6
		ymax_corr = point5
		


	# soft constraint
	# div = soft_range * np.pi / 180
	# if 0 < rot_local < div or 2*np.pi-div < rot_local < 2*np.pi:
	# 	xmin_corr = point8
	# 	xmax_corr = point6
	# 	ymin_corr = point6
	# 	ymax_corr = point5

	# if np.pi - div < rot_local < np.pi + div:
	# 	xmin_corr = point2
	# 	xmax_corr = point4
	# 	ymin_corr = point8
	# 	ymax_corr = point1

	return xmin_corr, xmax_corr, ymin_corr, ymax_corr

def recover_angle(anchors, confidences, num_bins):
	"""
	Recover angle from anchor and confidence bins
	:param anchors: anchor bins generated by utils.compute_anchors function
	:param confidences: confidence scores for each anchor
	:param num_bins: specifies number of anchor bins
	:return: angle in range 0, 2*pi
	"""
	# select anchor from bins
	max_anchor_idx = np.argmax(confidences)
	anchor = anchors[max_anchor_idx]
	# compute the angle offset
	if anchor[1] > 0:
		angle_offset = np.arccos(anchor[0])
	else:
		angle_offset = -np.arccos(anchor[0])

	# add the angle offset to the center ray of each bin to obtain the local orientation
	wedge = 2 * np.pi / num_bins
	angle = angle_offset + max_anchor_idx * wedge

	# angle modulo 2pi, if exceed 2pi
	# angle = angle % (2 * np.pi)

	# change to ray back to [-pi, pi]
	# angle = angle_l - np.pi / 2
	# if angle > np.pi:
	# 	angle -= 2 * np.pi
	angle = round(angle, 2)

	return angle


def resize(img, new_size, keep_aspect_ratio = False, sample=Image.NEAREST):
	"""
	Resize image in specified form 
	:param img: PIL.Image 
	:param new_size: tuple indicating (width, height)
	:param keep_aspect_ratio: specifies whether to keep aspect ratio when resize, i.e pads if true
	:return: resized image
	"""
	if keep_aspect_ratio:
		return resampling_with_original_ratio(img, new_size, sample)
	else:
		return img.resize(new_size, sample)


def resampling_with_original_ratio(img, required_size, sample=Image.NEAREST):
	"""Resizes the image to maintain the original aspect ratio by adding pixel padding where needed.
	For example, if your model's input tensor requires a square image but your image is landscape (and
	you don't want to reshape the image to fit), pass this function your image and the required square
	dimensions, and it returns a square version by adding the necessary amount of black pixels on the
	bottom-side only. If the original image is portrait, it adds black pixels on the right-side
	only.
	Args:
	img (:obj:`PIL.Image`): The image to resize.
	required_size (list): The pixel width and height [x, y] that your model requires for input.
	sample (int): A resampling filter for image resizing.
		This can be one of :attr:`PIL.Image.NEAREST` (recommended), :attr:`PIL.Image.BOX`,
		:attr:`PIL.Image.BILINEAR`, :attr:`PIL.Image.HAMMING`, :attr:`PIL.Image.BICUBIC`,
		or :attr:`PIL.Image.LANCZOS`. See `Pillow filters
		<https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters>`_.
	Returns:
	A 2-tuple with a :obj:`PIL.Image` object for the resized image, and a tuple of floats
	representing the aspect ratio difference between the original image and the returned image
	(x delta-ratio, y delta-ratio).
	"""
	old_size = img.size
	# Resizing image with original ratio.
	resampling_ratio = min(
		required_size[0] / old_size[0],
		required_size[1] / old_size[1]
	)
	new_size = (
		int(old_size[0] * resampling_ratio),
		int(old_size[1] * resampling_ratio)
	)
	new_img = img.resize(new_size, sample)
	# Expand it to required size.
	delta_w = required_size[0] - new_size[0]
	delta_h = required_size[1] - new_size[1]
	padding = (0, 0, delta_w, delta_h)
	ratio = (new_size[0] / required_size[0], new_size[1] / required_size[1])
	return ImageOps.expand(new_img, padding)

def compute_anchors(angle, num_bins, overlap_ratio):
	"""
	compute angle offset and which bin the angle lies in
	:param angle: fixed local orientation [0, 2pi]
	:param num_bins: number of bins specified in config.py
	:param overlap_ratio: defined in config.py
	:return: [bin number, angle offset]

	For two bins:

	if angle < pi, l = 0, r = 1
		if    angle < 1.65, return [0, angle]
		elif  pi - angle < 1.65, return [1, angle - pi]

	if angle > pi, l = 1, r = 2
		if    angle - pi < 1.65, return [1, angle - pi]
		elif     2pi - angle < 1.65, return [0, angle - 2pi]
	"""
	anchors = []

	wedge = 2. * np.pi / num_bins  # 2pi / bin = pi
	l_index = int(angle / wedge)  # angle/pi
	r_index = l_index + 1

	# (angle - l_index*pi) < pi/2 * 1.05 = 1.65
	if (angle - l_index * wedge) < wedge / 2 * (1 + overlap_ratio / 2):
		anchors.append([l_index, angle - l_index * wedge])

	# (r*pi + pi - angle) < pi/2 * 1.05 = 1.65
	if (r_index * wedge - angle) < wedge / 2 * (1 + overlap_ratio / 2):
		anchors.append([r_index % num_bins, angle - r_index * wedge])

	return anchors

