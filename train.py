import os
import numpy as np
from tensorflow import keras

from absl import app, flags, logging
from absl.flags import FLAGS

import utils
from datagen import BatchGenerator
import loss
import callbacks
import reader
import config
import datagen


"""
This script trains a model on triplets.
Example usage: 
    python train.py --save_to progress/test --num_epochs 10 --batch_size 8 --model mobilenet_v2 --input_size 224,224 --jitter --overwrite
"""

flags.DEFINE_string('save_to', config.SAVE_TO_DEFAULT, 'directory to save checkpoints and logs')
flags.DEFINE_boolean('overwrite', False, 'Overwrite given save path')
flags.DEFINE_string('from_ckpt', None, 'path to continue training on checkpoint')
flags.DEFINE_boolean('jitter', config.JITTER_DEFAULT, 'Apply image augmentation')
flags.DEFINE_integer('batch_size', config.BATCH_SIZE_DEFAULT, 'batch size')
flags.DEFINE_list('input_size', config.INPUT_SIZE_DEFAULT, 'input size in (width, height) format')
flags.DEFINE_boolean('keep_aspect_ratio', config.KEEP_ASPECT_RATIO_DEFAULT, 'keep aspect ratio when resizing patches')
flags.DEFINE_list('loss_weights',config.LOSS_WEIGHTS_DEFAULT, 'loss weights size in (w_dimension, w_orientation, w_confidence) format')
flags.DEFINE_integer('num_bins', config.NUM_BINS_DEFAULT, 'numebr of bins used in orientation regression')
flags.DEFINE_integer('num_epochs', config.NUM_EPOCHS_DEFAULT, 'number of epochs')
flags.DEFINE_string('model', config.MODEL_DEFAULT, 'integer model type - %s'%str(config.MODELS.keys()))

def main(_argv):
    assert not ((FLAGS.overwrite) and (FLAGS.from_ckpt is not None))
    input_size = list(map(int, FLAGS.input_size)) # (width, height)
    input_shape = (input_size[1], input_size[0], 3)
    loss_weights = list(map(float, FLAGS.loss_weights))

    # Load data
    logging.info("Loading data")
    kitti_reader = reader.KittiReader()

    # Define batch generators
    logging.info("Creating batch generators")
    traingen = datagen.BatchGenerator(
        kitti_reader, 
        batch_size=FLAGS.batch_size,
        keep_aspect_ratio=FLAGS.keep_aspect_ratio,
        input_size = input_size,
        num_bins = FLAGS.num_bins,
        jitter = FLAGS.jitter,
        mode = 'train')
    valgen = datagen.BatchGenerator(
        kitti_reader, 
        batch_size=FLAGS.batch_size,
        keep_aspect_ratio=FLAGS.keep_aspect_ratio,
        input_size = input_size,
        num_bins = FLAGS.num_bins,
        jitter = False,
        mode = 'val')

    # Prepare network
    logging.info("Constructing model")
    model = config.MODELS[FLAGS.model](input_shape=input_shape, num_bins = FLAGS.num_bins)
    
    # Setup and compile model
    model.compile(optimizer = 'adam', 
                loss={'dimensions': 'mean_squared_error', 'orientation': loss.orientation_loss, 'confidence': 'binary_crossentropy'},
                loss_weights={'dimensions': loss_weights[0], 'orientation': loss_weights[1], 'confidence': loss_weights[2]})
    
    logging.info("Compiled model with loss weights:%s"%str(loss_weights))
    model.summary()

    if FLAGS.from_ckpt is not None:
        logging.info("Loading weights from %s"%FLAGS.from_ckpt)
        model.load_weights(FLAGS.from_ckpt)

    logging.info("Genrating callbacks")
    train_callbacks = callbacks.get(directory = FLAGS.save_to, overwrite = FLAGS.overwrite)
    
    cfg = config.save(
        FLAGS.save_to, 
        model = FLAGS.model, 
        input_size = input_size, 
        keep_aspect_ratio=FLAGS.keep_aspect_ratio,
        jitter = FLAGS.jitter,
        batch_size = FLAGS.batch_size,
        num_epochs = FLAGS.num_epochs,
        num_bins=FLAGS.num_bins,
        loss_weights = loss_weights
        )
    logging.info("Saving config : %s"%str(cfg))
    logging.info("Starting training")
    model.fit(traingen,
                steps_per_epoch=1000,
                epochs=FLAGS.num_epochs,
                verbose=1,
                validation_data=valgen,
                validation_steps=100,
                callbacks=train_callbacks,
                workers = 8,
                max_queue_size=3)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass