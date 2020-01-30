
import numpy as np

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
