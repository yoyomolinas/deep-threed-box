# Must use tf-nightly 1.15

import tensorflow as tf
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt
from os.path import join
from tqdm import tqdm
import random
from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
keras = tf.keras

flags.DEFINE_string('out', './', 'output directory for the .tflite file')
flags.DEFINE_string('keras_model', None, 'path to keras model .hdf5 file')
flags.DEFINE_integer('input_size', 128, 'input image size') 

class RepresentativeDataset:
    def __init__(self, path = ""):
        self.train_images = np.random.random((2, 224, 224, 3)).astype(np.float32)
    def gen(self):
        for img in self.train_images:
            yield [np.array([img])]

def main(_argv):
    TFLITE_MODEL_PATH = FLAGS.out
    KERAS_MODEL_PATH = FLAGS.keras_model
    SIZE = (FLAGS.input_size, FLAGS.input_size)
    BATCH_SIZE = 1 # Recommended higher batch size has bad quantized performance for some reason
    LIMIT_ = 100
    logging.info("Forming representative dataset")
    representative_dataset = RepresentativeDataset()
    # Currentry, it seems that tf.lite.TFLiteConverter does not suppert inference_input/output_type yet.
    # So we have to use tf.compat.v1.lite.TFLiteConverter.
    logging.info("Converting keras model : %s to tflite."%KERAS_MODEL_PATH)
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(KERAS_MODEL_PATH, 
                                                                        input_shapes={'input_1' : [BATCH_SIZE, SIZE[0], SIZE[0], 3]})
    converter.representative_dataset = representative_dataset.gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(TFLITE_MODEL_PATH, 'wb') as o_:
        o_.write(tflite_model)
    logging.info("Successfull tflite conversion. Saved file to %s"%TFLITE_MODEL_PATH)
    logging.info("Trying tflite interpreter..")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    logging.info("Successfull allocation of tensors.")

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass