from tensorflow import keras
layers = keras.layers

def l2_normalize(x):
    import tensorflow as tf
    return tf.nn.l2_normalize(x, axis=2)

def network(input_shape, num_bins):
    mobil = keras.applications.mobilenet_v2.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet', pooling='avg')
    x = mobil.outputs[0]
    # x = layers.Reshape((1, 1, 1280))(x)
    
    # Dimensions branch
    dimensions = layers.Dense(3, name = 'dimensions')(x)
    # dimensions = layers.Conv2D(3, (1,1), name='d_conv')(x)
    # dimensions = layers.Reshape((3,), name='dimensions')(dimensions)

    # Orientation branch
    orientation = layers.Dense(4)(x)
    orientation = layers.Reshape((num_bins, -1))(orientation)
    orientation = layers.Lambda(l2_normalize, name='orientation')(orientation)

    # Confidence branch
    confidence = layers.Dense(num_bins, name='confidence', activation = 'softmax')(x)

    # Build model
    model = keras.Model(mobil.inputs, [dimensions, orientation, confidence])

    return model