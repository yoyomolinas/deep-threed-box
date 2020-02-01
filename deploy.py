import os
import pickle
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

from absl import app, flags, logging
from absl.flags import FLAGS

import config
import utils
from models improt MODELS
from batchgen import BatchGenerator
import callbacks

"""
This script deploys a trained model
Example usage: 
    python deploy.py --ckpt progress/mobilenet_v2/ckpt/1.12.hdf5 --out deploy/mobilenet_v2 --model_type 0 --input_size 224,224 
"""

DEFAULT_IMAGE_SIZE = [224, 224] # width, height
DEFAULT_MODEL_TYPE = 0

flags.DEFINE_string('in', None, 'path to model - typically smt. like progress/mobilenet_v2-1')
flags.DEFINE_string('out', None, 'directory to save keras, tflite model files')

def main(_):
    assert FLAGS.in is not None, 'Provide in argument'
    assert FLAGS.out is not None, 'Provide out argument'
    cfg = config.read(dir = FLAGS.in)
    input_size = (cfg['input_size'][0]) , int(cfg['input_size'][1])) # (width, height)
    input_shape = (int(cfg['input_size'][1]), int(cfg['input_size'][0]), 3)
    
    # Prepare network
    model = models.MODELS[cfg['model']](input_shape=input_shape, num_bins = cfg['bin'])

    # Find ckpt with lowest loss
    min_loss = np.inf
    min_loss_weight_file = ""
    for filename in os.listdir(os.path.join(FLAGS.in, 'ckpt')):
        w_str = filename[(len('weights') + 1):filename.find('hdf5') - 1]
        if filename[(len('weights') + 1):cfilename.find('hdf5') - 1] == 'nan':
            continue
        
        w_float = float(w_str)
        if w_float < min_loss:
            min_loss = w_float
            min_loss_weight_file = filename
    
    weights_path = os.path.join(FLAGS.in, 'ckpt', min_loss_weight_file)
    logging.info("Loading weights from %s"%weights_path)
    model.load_weights(weights_path)
    
    # Add custom output
    # model = keras.models.Model(inputs = model.inputs, outputs = outputs)
    model.compile(optimizer = 'adam', loss = 'mse'])
    
    model.summary()
    keras_save_path = os.path.join(FLAGS.out, 'model.hdf5')
    model.save(keras_save_path)
    logging.info("Saved keras model to %s"%keras_save_path)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass