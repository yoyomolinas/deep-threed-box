from tensorflow import keras
layers = keras.layers

def l2_normalize(x):
    import tensorflow as tf
    return tf.nn.l2_normalize(x, axis=2)

def construct(input_shape, num_bins):
    input_tensor = layers.Input(shape = input_shape, name = 'image')
    net = keras.applications.mobilenet.MobileNet(input_tensor=input_tensor, include_top=False, weights='imagenet', pooling='avg')
    x = net.outputs[0]
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
    model = keras.Model(net.inputs, [dimensions, orientation, confidence])

    return model

# Simple test function
if __name__ == '__main__':
    construct((224, 224, 3), 2).summary()