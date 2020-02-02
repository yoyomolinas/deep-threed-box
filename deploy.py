import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from absl import app, flags, logging
from absl.flags import FLAGS

import config
import utils
import datagen 
import callbacks
import reader

"""
This script deploys a trained model for keras, tflite, edgetpu
Example usage: 
    python deploy.py --read_from progress/test --save_to deploy/test 
"""

flags.DEFINE_string('read_from', None, 'model directory - typically smt. like progress/mobilenet_v2-1')
flags.DEFINE_string('save_to', None, 'directory to save keras, tflite model files')
flags.DEFINE_boolean('overwrite', False, 'Overwrite given save path')
flags.mark_flag_as_required('read_from')
flags.mark_flag_as_required('save_to')


class RepresentativeDataset:
    """
    Used when converting to tflite
    Modify this function to generate different sets of images for quantization
    """
    def __init__(self, cfg, limit = 1000):
        self.kitti_reader = reader.KittiReader()
        self.limit = limit
        self.generator = datagen.BatchGenerator(
            self.kitti_reader, 
            batch_size=1,
            keep_aspect_ratio=cfg['keep_aspect_ratio'],
            input_size = cfg['input_size'],
            num_bins = cfg['num_bins'],
            jitter = False,
            mode = 'val')

    def gen(self):
        for i in range(self.limit):
            img_batch, _ = self.generator.__getitem__(i)
            yield [img_batch.astype(np.float32)]


def main(_argv):
    assert FLAGS.read_from is not None, 'Provide read_from argument'
    assert FLAGS.save_to is not None, 'Provide save_to argument'
    cfg = config.read(FLAGS.read_from) # read configuration file
    input_size = cfg['input_size'] # (width, height)
    input_shape = (cfg['input_size'][1], cfg['input_size'][0], 3)
    
    # Prepare network
    model = config.MODELS[cfg['model']](input_shape=input_shape, num_bins = cfg['num_bins'])

    # Find ckpt with lowest loss by parsing model files
    min_loss = np.inf
    min_loss_weight_file = None
    for filename in os.listdir(os.path.join(FLAGS.read_from, 'ckpt')):
        cur_loss = callbacks.parse_weight_path(filename)
        if cur_loss < min_loss:
            min_loss = cur_loss
            min_loss_weight_file = filename
    
    weights_path = os.path.join(FLAGS.read_from, 'ckpt', min_loss_weight_file)
    
    # Load weights
    logging.info("Loading weights from %s"%weights_path)
    model.load_weights(weights_path)
    
    # Add custom output
    # model = keras.models.Model(inputs = model.inputs, outputs = outputs)
    model.compile(optimizer = 'adam', loss = 'mse')
    # model.summary()

    # Create directory for deployment
    dt = utils.DirectoryTree(FLAGS.save_to, overwrite = True)

    # Define paths for saving models
    keras_save_path = os.path.join(dt.path, 'model.hdf5')
    tflite_save_path = os.path.join(dt.path, 'model.tflite')
    
    # Save keras model
    model.save(keras_save_path)
    logging.info("Saved keras model to %s"%keras_save_path)

    # Convert and save tflite model
    logging.info("Forming representative dataset")
    representative_dataset = RepresentativeDataset(cfg, limit = 10)
    # Currentry, it seems that tf.lite.TFLiteConverter does not suppert inference_input/output_type yet.
    # So we have to use tf.compat.v1.lite.TFLiteConverter.
    logging.info("Converting keras model : %s to tflite"%keras_save_path)
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(keras_save_path,
                                                                        input_shapes={'image' : [1, *input_shape]})
    converter.representative_dataset = representative_dataset.gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_save_path, 'wb') as o_:
        o_.write(tflite_model)
    interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
    interpreter.allocate_tensors()
    logging.info("Successfull tflite conversion. Saved file to %s"%tflite_save_path)
    
    # Compile for edgetpu
    os.system("edgetpu_compiler %s --out_dir %s"%(tflite_save_path, FLAGS.save_to))
    logging.info("Successfull compiled for EdgeTPU. Saved model to %s"%os.path.join(FLAGS.save_to, 'model_edgetpu.tflite'))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass