import os
import csv
import numpy as np
from tqdm import tqdm
from config import Config

config = Config()

class KittiReader():
    # Kitti README :  https://github.com/pratikac/kitti/blob/master/readme.raw.txt
    def __init__(self):

        self.data_dir = config.data_dir
        self.categories = config.categories

        label_dir = os.path.join(self.data_dir, 'labels')
        image_dir = os.path.join(self.data_dir, 'images')
        calib_dir = os.path.join(self.data_dir, 'calib')

        # List of annotations 
        # Example of annotation : 
        # {'name': 'Car',
        # 'image': '../data/training/images/000135.png',
        # 'xmin': 367,
        # 'ymin': 194,
        # 'xmax': 448,
        # 'ymax': 247,
        # 'dims': array([1.42, 1.54, 3.5 ]),
        # 'trans': array([-6.21,  2.14, 22.33]),
        # 'alpha': 4.971592653589793}
        # Use `list(filter(lambda e: e['image'] == self.image_paths[0], self.image_data))` to get annotatios for 0th image
        self.image_data = [] 

        # List of image paths
        self.image_paths = []

        # List of calibration paths - same order as self.image_paths
        self.calibration_paths = []
        
        # Read data into above lists
        for i, fn in enumerate(os.listdir(label_dir)):
            label_full_path = os.path.join(label_dir, fn)
            image_full_path = os.path.join(image_dir, fn.replace('.txt', '.png'))
            calib_full_path = os.path.join(calib_dir, fn)
            
            self.calibration_paths.append(calib_full_path)
            self.image_paths.append(image_full_path)
            fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw', 'dl',
                          'lx', 'ly', 'lz', 'ry']
            

            with open(label_full_path, 'r') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
                for line, row in enumerate(reader):
                    if row['type'] in self.categories:
                        new_alpha = self.shift_alpha(row['alpha'])
                        translations = np.array([float(row['lx']), float(row['ly']), float(row['lz'])])
                        dimensions = np.array([float(row['dh']), float(row['dw']), float(row['dl'])])
                        annotation = {'name': row['type'], 'image': image_full_path, 'calib' : calib_full_path,
                                        'xmin': int(float(row['xmin'])), 'ymin': int(float(row['ymin'])),
                                        'xmax': int(float(row['xmax'])), 'ymax': int(float(row['ymax'])),
                                        'dims': dimensions, 'trans' : translations, 'alpha': new_alpha}

                        self.image_data.append(annotation)

    def get_average_dimension(self):
        """
        Average dimensions of objects in the dataset
        :return dims_avg: dimensions average in height, width length format, di
        """
        dims_avg = {key: np.array([0, 0, 0]) for key in self.KITTI_cat}
        dims_cnt = {key: 0 for key in self.KITTI_cat}

        for i in range(len(self.image_data)):
            current_data = self.image_data[i]
            if current_data['name'] in self.KITTI_cat:
                dims_avg[current_data['name']] = dims_cnt[current_data['name']] * dims_avg[current_data['name']] + \
                                                 current_data['dims']
                dims_cnt[current_data['name']] += 1
                dims_avg[current_data['name']] /= dims_cnt[current_data['name']]
        return dims_avg

    @staticmethod
    def read_intrinsic_matrix(path):
        """
        Read calibration file for intrinsic camera matrix.  Use P2 (left color camera) in the file
        :param path: path to calibration file : typically smt like training/calib/00000.txt
        :return K: intrinsic camera matrix
        """
        with open(path, 'r') as calib_file:
            for line in calib_file:
                if 'P2' in line:
                    K = line.split(' ')
                    K = np.asarray([float(i) for i in K[1:]])
                    K = np.reshape(K, (3,4))
        return K

    @staticmethod
    def shift_alpha(alpha):
        """
        Change the range of orientation from [-pi, pi] to [0, 2pi]
        :param alpha: original orientation in KITTI
        :return: new alpha
        """
        new_alpha = float(alpha)
        # new_alpha = np.clip((float(alpha) + np.pi) % (2* np.pi), 0, 2 * np.pi) 
        # new_alpha = float(alpha) + np.pi / 2.
        # if new_alpha < 0:
        #     new_alpha = new_alpha + 2. * np.pi
        #     # make sure angle lies in [0, 2pi]
        # new_alpha = new_alpha - int(new_alpha / (2. * np.pi)) * (2. * np.pi)
        return new_alpha
