import numpy as np
import cv2

def draw_3d_box(img, coords, color = (0, 255, 0), line_width = 2):
    """
    Draw 3d bounding box given coords of every point
    """
    assert coords.shape == (2, 8)
    coords = coords.astype(int)
    p0 = (coords[0][0], coords[1][0])
    p1 = (coords[0][1], coords[1][1])
    p2 = (coords[0][2], coords[1][2])
    p3 = (coords[0][3], coords[1][3])
    p4 = (coords[0][4], coords[1][4])
    p5 = (coords[0][5], coords[1][5])
    p6 = (coords[0][6], coords[1][6])
    p7 = (coords[0][7], coords[1][7])
    
    lines = [
         (p0, p1),
         (p0, p5),
         (p0, p7),
         (p1, p2),
         (p1, p4),
         (p2, p3),
         (p2, p7), 
         (p3, p6), 
         (p3, p4), 
         (p4, p5), 
         (p5, p6), 
         (p6, p7)
    ]

    # draw lines
    for p1, p2 in lines:
        cv2.line(img, p1, p2, color, line_width)
    # draw points
    for i in range(8):
        x, y = coords[0][i], coords[1][i]
        cv2.circle(img, (x, y), 1, color, 2)    

"""

# Example visualization of 3d bounding boxes 

import reader
import utils
import visualization as vis
from PIL import Image
import numpy as np
import cv2

# Read data
data = reader.KittiReader("data/")

# Visualize
j = -200
image_path = data.image_paths[j]
calib_path = data.calibration_paths[j]
K = reader.KittiReader.read_intrinsic_matrix(calib_path)
image_data = list(filter(lambda rec: rec['image'] == image_path, data.image_data))
img = cv2.imread(image_path)
for obj in image_data:
    cv2.rectangle(img, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), (255, 0, 0), 4)
    coords_3d = utils.compute_3d_coordinates(K,
                                        obj['trans'], 
                                        obj['alpha'], 
                                        obj['dims'], 
                                        (obj['ymin'], obj['xmin'], obj['ymax'], obj['xmax']))
    coords_2d = utils.project_2d(K, coords_3d)
    vis.draw_3d_box(img, coords_2d)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
Image.fromarray(img)
"""