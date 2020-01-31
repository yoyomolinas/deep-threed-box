from os.path import join
import pickle
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from absl import app, flags, logging
from absl.flags import FLAGS

import utils
import models 
from batchgen import BatchGenerator
import callbacks

"""
This script trains a model on triplets.
Example usage: 
    python train.py --save_path progress/mobil-pre- --epochs 100 --batch_size 32 --model_type 1 --input_size 256,256 --augment --loss_weights 1,100 --crop
"""

DEFAULT_SAVE_PATH = "progress/test/"
# DEFAULT_OUTPUT_SIZE = None
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 30
DEFAULT_IMAGE_SIZE = [224, 224] # width, height
DEFAULT_MODEL_TYPE = 0

flags.DEFINE_string('save_path', DEFAULT_SAVE_PATH, 'path to save checkpoints and logs')
flags.DEFINE_boolean('overwrite', False, 'Overwrite given save path')
flags.DEFINE_boolean('augment', False, 'Apply image augmentation')
flags.DEFINE_boolean('crop', False, 'Train on crops of clothes')
flags.DEFINE_string('from_ckpt', None, 'path to continue training on checkpoint')
flags.DEFINE_integer('batch_size', DEFAULT_BATCH_SIZE, 'batch size')
flags.DEFINE_list('input_size', DEFAULT_IMAGE_SIZE, 'input size in (width, height) format')
flags.DEFINE_integer('epochs', DEFAULT_NUM_EPOCHS, 'number of epochs')
flags.DEFINE_integer('model_type', DEFAULT_MODEL_TYPE, 'integer model type - %s'%str(models.ENUM_MODELS_DICT))
