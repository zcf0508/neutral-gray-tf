import tensorflow as tf
from tensorflow import keras


def batchNormRelu(input):
    x = keras.layers.BatchNormalization()(input)
    x = keras.layers.Activation("relu")(x)
    return x


class ResidualBlock(keras.layers.Layer):
    initializer = tf.keras.initializers.HeNormal()
    kernel_regularizer = None  # tf.keras.regularizers.l2(0.01)

    def __init__(self, filters, kernelSize, downSample=False):
        self.filters = filters
        self.kernelSize = kernelSize
        self.downSample = downSample

    def conv2d(self, strides):
        return keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernelSize,
            strides=strides,
            padding="same",
            kernel_initializer=self.initializer,
            kernel_regularizer=self.kernel_regularizer,
        )

    def __call__(self, input):
        identity = input
        x = self.conv2d(2 if self.downSample else 1)(input)
        x = batchNormRelu(x)
        x = self.conv2d(1)(x)
        x = keras.layers.BatchNormalization()(x)

        if self.downSample:
            identity = self.conv2d(2)(identity)

        x = keras.layers.Add()([identity, x])
        x = keras.layers.Activation("relu")(x)

        return x
