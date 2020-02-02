from tensorflow.keras import callbacks as cb
from utils import DirectoryTree
from absl import logging
import os
import parse

LOSS_VARIABLE_TO_MONITOR = 'val_loss'

def LimitCheckpoints(ckpt_path, n=16):
    '''
    Callback to maintain only the checkpoints that have minimal validation loss
    Usage:
        callback = LimitCheckpoints('progress/test/ckpt', n=16)
    '''
    def f(epoch):
        if len(os.listdir(ckpt_path)) <= n:
            return

        max_loss = 0
        file_to_remove = None
        for filename in os.listdir(ckpt_path):
            cur_loss = parse_weight_path(filename)
            if cur_loss > max_loss:
                max_loss = cur_loss
                file_to_remove = filename

        if filename is not None:
            os.remove(os.path.join(ckpt_path, filename))

    return cb.LambdaCallback(on_epoch_end=lambda epoch, logs: f(epoch))

def format_weight_path():
    """
    Function to standardize weights path generation
    Val loss is used as the differentiation param in weights path.
    If you change this function you must adapt parse_weights_path, its inverse brother
    :return : formatable string with val_loss to be inserted in runtime
    """
    return "weights.{val_loss:.5f}.hdf5"

def parse_weight_path(weight_path):
    """
    Parse weight filename and extract the LOSS_VARIABLE_TO_MONITOR
    :return : floating point representation of LOSS_VARIABLE_TO_MONITOR
    """
    variables = parse.parse(format_weight_path(), weight_path)
    return variables[LOSS_VARIABLE_TO_MONITOR]


# Generates callbacks
def get(
    directory = None,
    checkpoint = True,
    tensorboard = True,
    limit_checkpoints = True,
    early_stopping = True,
    overwrite = False):
    

    dt = DirectoryTree(directory)
    if checkpoint:
        dt.add('ckpt', overwrite = overwrite)
    if tensorboard:
        dt.add('logs', overwrite = overwrite)

    callbacks = []
    if checkpoint:
        checkpoint = cb.ModelCheckpoint(filepath=os.path.join(dt.ckpt.path, format_weight_path()),
                                                monitor=LOSS_VARIABLE_TO_MONITOR,
                                                verbose=1,
                                                mode="auto",
                                                save_best_only = False,
                                                save_weights_only = True)
        callbacks.append(checkpoint)

    if tensorboard:
        tensorboard = cb.TensorBoard(log_dir=dt.logs.path)
        callbacks.append(tensorboard)
    
    if limit_checkpoints:
        keep_limited = LimitCheckpoints(dt.ckpt.path)
        callbacks.append(keep_limited)
    
    if early_stopping:
        early_stop = cb.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, mode='min', verbose=1)
        callbacks.append(early_stop)

    return callbacks