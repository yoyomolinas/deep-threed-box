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
This script deploys a trained model
Example usage: 
    python deploy.py --ckpt progress/bigx/ckpt/ --save_to deploy/bigx.hdf5 --model_type 0 --input_size 256,256
"""

DEFAULT_IMAGE_SIZE = [256, 256] # width, height
DEFAULT_MODEL_TYPE = 0

flags.DEFINE_string('ckpt', None, 'path to hdf5 checkpoint file - should contain weights only')
flags.DEFINE_string('save_to', None, 'path to save keras hdf5 model file')
flags.DEFINE_list('input_size', DEFAULT_IMAGE_SIZE, 'input size in (width, height) format')
flags.DEFINE_integer('model_type', DEFAULT_MODEL_TYPE, 'integer model type - %s'%str(models.ENUM_MODELS_DICT))

def main(_argv):
    assert FLAGS.ckpt is not None, 'Provide ckpt argument'
    assert FLAGS.save_to is not None, 'Provide save_to argument'
    assert FLAGS.model_type in models.ENUM_MODELS_DICT.keys()
    input_size = (int(FLAGS.input_size[0]) , int(FLAGS.input_size[1])) # (width, height)
    input_shape = (int(FLAGS.input_size[1]), int(FLAGS.input_size[0]), 3)
    
    # Prepare network
    model = models.ENUM_MODELS_DICT[FLAGS.model_type](input_shape=input_shape)
    logging.info("Loading weights from %s"%FLAGS.ckpt)
    model.load_weights(FLAGS.ckpt)
    
    # Concatenate layers 
    attrs_layer = model.get_layer("attributes").output
    cats_layer = model.get_layer("categories").output
    outputs = [cats_layer, attrs_layer]
    model = keras.models.Model(inputs = model.inputs, outputs = outputs)
    model.compile(optimizer = 'adam', loss = 'mse', metrics = [keras.metrics.Recall(), keras.metrics.Precision()])
    
    model.summary()
    logging.info("Saved model to %s"%FLAGS.save_to)
    model.save(FLAGS.save_to)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass