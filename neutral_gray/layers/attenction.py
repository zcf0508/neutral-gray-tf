import tensorflow as tf
from tensorflow import keras


class AttenctionBlock(keras.layers.Layer):
    def __init__(self, filters=3):
        super(AttenctionBlock, self).__init__()
        self.filters = filters

        self.conv2d_1 = keras.layers.Conv2D(
            filters=filters, kernel_size=(3, 3), padding="same"
        )
        self.prelu = keras.layers.PReLU(shared_axes=[1, 2])
        self.conv2d_2 = keras.layers.Conv2D(
            filters=filters, kernel_size=(1, 1), padding="same"
        )

    def _postion(self, input):
        positon_avg = tf.reduce_mean(input, axis=3)
        positon_max = tf.reduce_max(input, axis=3)
        x = tf.stack([positon_avg, positon_max], axis=-1)
        x = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=(7, 7),
            padding="same",
            activation="sigmoid",
        )(x)
        x = keras.layers.Multiply()([x, input])
        return x

    def _channel(self, input):
        channel_avg = keras.layers.GlobalAvgPool2D(keepdims=True)(input)
        x = keras.layers.Conv2D(
            filters=self.filters, kernel_size=(1, 1), padding="same"
        )(channel_avg)
        x = keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=(1, 1),
            padding="same",
            activation="sigmoid",
        )(x)
        x = keras.layers.Multiply()([x, input])
        return x

    def __call__(self, input):
        x = self.conv2d_1(input)
        x = self.prelu(x)
        x = self.conv2d_1(x)
        x = keras.layers.Concatenate()([self._postion(x), self._channel(x)])
        x = self.conv2d_2(x)
        x = keras.layers.Add()([input, x])
        return x
