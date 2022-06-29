import tensorflow as tf
from tensorflow import keras


class MFAMBlock(keras.layers.Layer):
    def __init__(self, filters=3):
        super(MFAMBlock, self).__init__()
        self.filters = filters
        self.conv2d = keras.layers.Conv2D(
            filters=self.filters, kernel_size=(1, 1), padding="same"
        )

    def _up_sample(self, input):
        x = keras.layers.Conv2DTranspose(
            filters=self.filters, kernel_size=(1, 1), strides=2, padding="same"
        )(input)
        x = keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1, padding="same")(x)
        return x

    def _down_sample(self, input):
        x = keras.layers.Conv2D(
            filters=self.filters, kernel_size=(1, 1), strides=2, padding="same"
        )(input)
        x = keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1, padding="same")(x)
        return x

    def __call__(self, input1, input2, input3, scale=2):
        if scale == 1:
            input2 = keras.layers._up_sample(input2)
            input3 = keras.layers._up_sample(input3)
        if scale == 2:
            input1 = self._down_sample(input1)
            input3 = self._up_sample(input3)
        if scale == 3:
            input1 = keras.layers._down_sample(input1)
            input2 = keras.layers._down_sample(input2)

        x = keras.layers.Add()([input1, input2, input3])
        x = keras.layers.GlobalAvgPool2D(keepdims=True)(x)
        x = self.conv2d(x)
        x = keras.layers.PReLU(shared_axes=[1, 2])(x)

        c1 = self.conv2d(x)
        c2 = self.conv2d(x)
        c3 = self.conv2d(x)
        y1 = keras.layers.Softmax()(c1)
        y2 = keras.layers.Softmax()(c2)
        y3 = keras.layers.Softmax()(c3)

        x1 = keras.layers.Multiply()([y1, input1])
        x2 = keras.layers.Multiply()([y2, input2])
        x3 = keras.layers.Multiply()([y3, input3])

        x = keras.layers.Add()([x1, x2, x3])

        return x
