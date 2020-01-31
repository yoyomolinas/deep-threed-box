from tensorflow.keras import callbacks as cb
from utils import DirectoryTree
from absl import logging

def keep_limited_checkpoints(base, n=16):
    '''
    Callback to maintain only the checkpoints that have minimal validation loss
    Usage:
        keeplimited = keep_limited_checkpoints('temp/checkpoints/modelname', n=3)
    '''
    def f(epoch):
        import os
        if len(os.listdir(base)) <= n:
            return

        max_loss = 0
        rm_cp = None
        for cp in os.listdir(base):
            if cp[(len('weights') + 1):cp.find('hdf5') - 1] == 'nan':
                rm_cp = cp
                break
            
            if float(cp[(len('weights') + 1):cp.find('hdf5') - 1]) > max_loss:
                max_loss = float(cp[len('weights') + 1:cp.find('hdf5') - 1])
                rm_cp = cp

        if rm_cp is not None:
            #            logging.debug('Removing %s'%rm_cp)
            os.remove(os.path.join(base, rm_cp))

    return cb.LambdaCallback(on_epoch_end=lambda epoch, logs: f(epoch))

    # Generates callbacks
def generate_keras_callbacks(
    path,
    use_save_checkpoint = True,
    use_tensorboard = True,
    use_keep_limited_checkpoints = True,
    use_early_stopping = True,
    overwrite = False):
    
    from os.path import join
    
    dt = DirectoryTree(path)
    if use_save_checkpoint:
        dt.add('ckpt', overwrite = overwrite)
    if use_tensorboard:
        dt.add('logs', overwrite = overwrite)

    # print(dt.path, dt.ckpt.path, dt.logs.path)
    # logging.info('Initializing callbacks...')

    callbacks = []
    if use_save_checkpoint:
        checkpoint_manager = cb.ModelCheckpoint(filepath=join(dt.ckpt.path, "weights.{val_loss:.5f}.hdf5"),
                                                monitor="val_loss",
                                                verbose=1,
                                                mode="auto",
                                                save_best_only = False,
                                                save_weights_only = True)
        callbacks.append(checkpoint_manager)

    if use_tensorboard:
        tensorboard = cb.TensorBoard(log_dir=dt.logs.path)
        callbacks.append(tensorboard)
    
    if use_keep_limited_checkpoints:
        keep_limited = keep_limited_checkpoints(dt.ckpt.path)
        callbacks.append(keep_limited)
    
    if use_early_stopping:
        early_stop = cb.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, mode='min', verbose=1)
        callbacks.append(early_stop)

    return callbacks