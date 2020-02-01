import numpy as np
from tensorflow import keras

from absl import app, flags, logging
from absl.flags import FLAGS

import utils
from models import MODELS
from datagen import BatchGenerator
import loss
import callbacks
import reader
import config
import datagen


"""
This script trains a model on triplets.
Example usage: 
    python train.py --save_path progress/test --epochs 500 --batch_size 32 --model_type 0 --input_size 224,224 --augment 
"""

CONFIG = config.Config()
DEFAULT_SAVE_PATH = "progress/test/"
# DEFAULT_OUTPUT_SIZE = None
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 500
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
flags.DEFINE_integer('model_type', DEFAULT_MODEL_TYPE, 'integer model type - %s'%str(MODELS))

def main(_argv):
    assert FLAGS.model_type in MODELS.keys()
    assert not ((FLAGS.overwrite) and (FLAGS.from_ckpt is not None))
    input_size = (int(FLAGS.input_size[0]) , int(FLAGS.input_size[1])) # (width, height)
    input_shape = (int(FLAGS.input_size[1]), int(FLAGS.input_size[0]), 3)
    logging.info("Loading data")
    # Load data
    kitti_reader = reader.KittiReader()

    # Define batch generators
    logging.info("Creating batch generators")
    traingen = datagen.BatchGenerator(kitti_reader, jitter = FLAGS.augment, mode = 'train')
    valgen = datagen.BatchGenerator(kitti_reader, jitter = False, mode = 'val')

    # Prepare network
    model = MODELS[FLAGS.model_type](input_shape=input_shape, num_bins = CONFIG.bin)
    
    # Setup and compile model
    model.compile(optimizer = 'adam', 
                loss={'dimensions': 'mean_squared_error', 'orientation': loss.orientation_loss, 'confidence': 'binary_crossentropy'},
                loss_weights={'dimensions': 1., 'orientation': 10., 'confidence': 5.})
    
    # logging.info("Compiled model with loss weights:%s"%str(loss_weights))
    model.summary()

    if FLAGS.from_ckpt is not None:
        logging.info("Loading weights from %s"%FLAGS.from_ckpt)
        model.load_weights(FLAGS.from_ckpt)

    logging.info("Starting training")
    model.fit(traingen,
                steps_per_epoch=8000,
                epochs=500,
                verbose=1,
                validation_data=valgen,
                validation_steps=1000,
                shuffle=True,
                callbacks=callbacks.generate_keras_callbacks(FLAGS.save_path, overwrite = FLAGS.overwrite),
                max_queue_size=3)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass