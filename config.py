import os
import json
import models
import tensorflow as tf

# Data path
DATA_DIR = os.path.abspath("data/training")

# set the bin size
NUM_BINS_DEFAULT = 2

# set overlapping
OVERLAP_DEFAULT = 0.1

# crop range -n, n+1
CROP_RANGE_DEFAULT = 3

# set the categories
CATEGORIES_DEFAULT = ['Car', 'Cyclist', 'Pedestrian']

# training ratio
TRAIN_RATIO_DEFAULT = 0.8

# random seed used when splitting training and validation
RANDOM_SEED_DEFAULT = 10

# path to save model checkpoints and tensorboard logs 
SAVE_TO_DEFAULT = "progress/test/"

# batch size
BATCH_SIZE_DEFAULT = 32

# number of epochs
NUM_EPOCHS_DEFAULT = 500

# image size (width, height)
INPUT_SIZE_DEFAULT = [224, 224] 

# loss weigths for dimension, orientation, confidence outputs respectively
LOSS_WEIGHTS_DEFAULT = [1., 10., 5.]

# model name that matches a key models.MODELS
MODEL_DEFAULT = 'mobilenet_v2'

# apply image augmentations to images if true
JITTER_DEFAULT = True

# keep aspect ratio of cropped patches when resizing them to input size 
KEEP_ASPECT_RATIO_DEFAULT = False

# MODELS dictionary that maps model names to construction functions 
MODELS = {
    'mobilenet_v2': models.mobilenet_v2.construct,
}

def save(
    save_to,
    model =None,
    input_size = None,
    keep_aspect_ratio = None,
    jitter = None,
    batch_size = None,
    num_epochs = None,
    loss_weights = LOSS_WEIGHTS_DEFAULT,
    train_ratio = TRAIN_RATIO_DEFAULT,
    random_seed = RANDOM_SEED_DEFAULT,
    num_bins = NUM_BINS_DEFAULT,
    overlap_ratio = OVERLAP_DEFAULT,
    categories = CATEGORIES_DEFAULT,
    **kwargs):
    """
    Save configuration paramteres into file
    """
    path = os.path.join(save_to, 'config.json')
    assert model in MODELS.keys()
    data = {
        'save_to' : save_to,
        'model' : model,
        'num_bins' : num_bins,
        'train_ratio' : train_ratio,
        'random_seed' : random_seed,
        'overlap_ratio' : overlap_ratio,
        'jitter' : jitter,
        'input_size' : input_size,
        'keep_aspect_ratio' : keep_aspect_ratio,
        'batch_size' : batch_size,
        'categories' : categories,
        'tf_version' : tf.__version__
    }
    data.update(kwargs)

    with open(path, 'w') as f:
        json.dump(data, f)
    
    return data

def read(read_from):
    """
    Read configuration file
    :param read_from: directory where config.json file is
    :return: dictionary with config params
    """
    path = os.path.join(read_from, 'config.json')
    with open(path, 'r') as f:
        data = json.load(f)
    assert data['model'] in MODELS.keys()
    return data

