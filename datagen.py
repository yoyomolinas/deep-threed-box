import copy
import random
import numpy as np
from itertools import permutations
from tensorflow import keras
import time
from PIL import Image
import cv2
from imgaug import augmenters as iaa
import utils
from config import Config

config = Config()

class BatchGenerator(keras.utils.Sequence):
    """
    This batch generator generates batches of augmented images, labels, and attributes.
    """
    def __init__(self,
                kitti_reader,
                random_seed = 10, 
                jitter = False,
                mode = 'train'):
        """
        :param kitti_reader: kitti data reader from reader.py
        :param batch_size:
        :param shuffle: true if shuffle dataset
        :param jitter: true if augment images
        :param mode: train or test
        """
        random.seed(random_seed)
        assert mode in ['train', 'val']
        self.keep_aspect_ratio = config.keep_aspect_ratio
        self.kitti_reader = kitti_reader
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.aug_pipe = self.get_aug_pipeline(p = 0.5)
        self.index = list(range(len(self.kitti_reader.image_data)))
        random.shuffle(self.index)
        self.mode = mode
        if self.mode == 'train' :
            self.index = self.index[:int(len(self.index) * config.split)]
        else:
            self.index = self.index[int(len(self.index) * config.split):]
        self.images = {} # {image_path : RGB image}
        self.jitter = jitter
        
    def __len__(self):
        """
        :return :Number of batches in this generator
        """
        return int(len(self.index) / self.batch_size)

    def on_epoch_end(self):
        """
        Function called in the end of every epoch.
        """
        np.random.shuffle(self.index)

    def __get_bounds__(self, idx):
        """
        Retrieve bounds for specified index
        :param idx: index 
        :return left bound, right bound:
        """
        #Define bounds of the image range in current batch
        l_bound = idx*self.batch_size #left bound
        r_bound = (idx+1)*self.batch_size #right bound

        if r_bound > len(self.index):
            r_bound = len(self.index)
            # Keep batch size stable when length of images is not a multiple of batch size.
            l_bound = r_bound - self.batch_size
        return l_bound, r_bound

    def preprocess(self, image, annot):
        """
        Crop and augment patch 
        :param image: PIL image
        :param annot: annotation dictionary, see example in kitti reader
        :return image, annot: numpy array, annotations dict with updated angle if image is augmented
        """
        new_annot = copy.deepcopy(annot)
        width, height = image.size
        xmin, ymin, xmax, ymax = annot['xmin'], annot['ymin'], annot['xmax'], annot['ymax']  
        # jitter with bbox coords
        if self.jitter:      
            xmin = np.clip(xmin  + np.random.randint(-config.jit, config.jit+1), 0, width - 1)
            ymin = np.clip(ymin  + np.random.randint(-config.jit, config.jit+1), 0, height - 1) 
            xmax = np.clip(xmax  + np.random.randint(-config.jit, config.jit+1), 0, width - 1)
            ymax = np.clip(ymax  + np.random.randint(-config.jit, config.jit+1), 0, height - 1)

        bbox = (xmin, ymin, xmax, ymax)
        crop = image.crop(bbox)
        crop = utils.resize(crop, self.image_size, keep_aspect_ratio = self.keep_aspect_ratio)
        crop = np.array(crop)
        
        # TODO flip crop and coordinates
        if self.jitter:
            # if np.randint(0, 10) < 3:
            #     crop = cv2.flip(crop, 1)
            #     new_annot['alpha'] = (2. * np.pi - annot['alpha']) 
            pass
        
        # jitter with image res and pixels
        if self.jitter:
            crop = self.aug_pipe.augment_image(crop)
        
        return crop, new_annot

    def get_aug_pipeline(self, p = 0.2):
        # Helper Lambda

        sometimes = lambda aug: iaa.Sometimes(p, aug)

        aug_pipe = iaa.Sequential(
            [
                iaa.OneOf(
                    [
                        sometimes(iaa.Multiply((0.5, 0.5), per_channel=0.5)),
                        sometimes(iaa.GaussianBlur((0, 3.0))),  # blur images with a sigma between 0 and 3.0
                        sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.04 * 255), per_channel=0.5)),
                        
                    ]
                )
            ],
            random_order=True
        )

        return aug_pipe  

    def __getitem__(self, i):
        """
        Abstract function from Sequence class - called every iteration in model.fit_generator function.
        :param i: batch id
        :return X, Y
        """
        X = []
        Y = []
        l_bound, r_bound = self.__get_bounds__(i)
        for j in range(l_bound, r_bound):
            idx = self.index[j]
            annot = self.kitti_reader.image_data[idx]
            image_path = annot['image']
            if image_path in self.images.keys():
                image = self.images[image_path]
            else:                
                image = cv2.imread(annot['image'])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                self.images[image_path] = image
            crop, annot = self.preprocess(image, annot)

            # prepare target values
            dimensions = annot['dims']
            orientation = np.zeros((config.bin, 2))
            confidence = np.zeros(config.bin)

            # compute anchors and respective orientation and confidence value
            anchors = utils.compute_anchors(annot['alpha'], config.bin, config.overlap)
            for anchor in anchors:
                # each angle is represented in sin and cos
                orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
                confidence[anchor[0]] = 1
            confidence = confidence / np.sum(confidence) # normalize confidence ?
            X.append(crop)
            Y.append([dimensions, orientation, confidence])
            # print(orientation)


        X = np.array(X)
        Y_dims = np.array(list(map(lambda y: y[0], Y)))
        Y_ort = np.array(list(map(lambda y: y[1], Y)))
        Y_conf = np.array(list(map(lambda y: y[2], Y)))
        return X, [Y_dims, Y_ort, Y_conf]