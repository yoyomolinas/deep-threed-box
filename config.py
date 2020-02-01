import os
import json

# points to data path
data_dir = os.path.abspath("data/training")

# set the base network: must be one of the keys in models.MODELS
model = 'mobilenet_v2'

# set the bin size
bin = 2

# set the train/val split
split = 0.8

# set overlapping
overlap = 0.1

# set jittered
jit = True

# set the image size width, height
input_size = 224, 224

# set whether to keep aspect ratio when cropping
keep_aspect_ratio = False

# set the batch size
batch_size = 1

# set the categories
categories = ['Car', 'Cyclist', 'Pedestrian']

def save(dir=None, path=None):
    if dir is None:
        assert path is not None
    else:
        path = os.path.join(dir, 'config.json')
    data = {
        'data_dir' = data_dir
        'network' = network
        'bin' = bin
        'split' = split
        'overlap' = overlap
        'jit' = jit
        'image_size' = image_size
        'keep_aspect_ratio' = keep_aspect_ratio
        'batch_size' = batch_size
        'categories' = categories
    }

    with open(path, 'w') as f:
        json.dump(data, f)

def read(dir = None, path= None):
    if dir is None:
        assert path is not None
    else:
        path = os.path.join(dir, 'config.json')

    with open(path, 'r') as f:
        data = json.load(f)
    return data

