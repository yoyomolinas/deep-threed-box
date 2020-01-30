
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


def compute_3d_coordinates(K, T, alpha, dimensions, bbox_2d):
    """
    Compute coordinates of object in world space given the following params
    :param K: instrinsic camera matrix (3, 4)
    :param T: object translation vector x, y, x in meters
    :param alpha: alpha rotation araound y axis in radians in range 0, 2*pi
    :param dimensions: object dimensions (height, width, length)
    :param coords_3d: 3d coordinates in object space
    :param bbox_2d: 2d bounding box coordinates in the image (ymin, xmin, ymax, xmax)
    :return corners_3d: 3d coordinates of vertices in world space
    """
    ymin, xmin, ymax, xmax = bbox_2d
    x = (xmin + xmax) / 2
    focal_length = K[0, 0]
    u_distance = x - K[0, 2]
    rot_cam = np.arctan(u_distance/focal_length)
    rot_local = np.clip(alpha - np.pi, -np.pi, np.pi) # shift and clip range
    rot_global = np.round(rot_cam + rot_local, 2)
    R = np.array([[np.cos(rot_global), 0, np.sin(rot_global)], 
                  [0, 1, 0],
                  [-np.sin(rot_global), 0, np.cos(rot_global)]])
    T = T.reshape(3, 1)
    height, width, length = dimensions
    x_corners = [-length/2, length/2, length/2, length/2, length/2, -length/2, -length/2, -length/2]
    y_corners = [-height, -height, 0, 0, -height, -height, 0, 0]
    z_corners = [-width/2, -width/2, -width/2, width/2, width/2, width/2, width/2, -width/2]
    corners_3d = np.array([x_corners, y_corners, z_corners])
    corners_3d = R.dot(corners_3d) + T
    return corners_3d


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
