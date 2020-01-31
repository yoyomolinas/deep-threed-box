import os
class Config():
    def __init__(self):
        # points to data path
        self.data_dir = os.path.abspath("data/training")

        # set the base network: vgg16, vgg16_conv or mobilenet_v2
        self.network = 'mobilenet_v2'

        # set the bin size
        self.bin = 2

        # set the train/val split
        self.split = 0.8

        # set overlapping
        self.overlap = 0.1

        # set jittered
        self.jit = 3

        # set the image size
        self.image_size = 224, 224

        # set whether to keep aspect ratio when cropping
        self.keep_aspect_ratio = False

        # set the batch size
        self.batch_size = 1

        # set the categories
        self.categories = ['Car', 'Cyclist', 'Pedestrian']
