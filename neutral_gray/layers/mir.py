import tensorflow as tf
from tensorflow import keras


def upSample(input, filters):
    x = keras.layers.Conv2DTranspose(
        filters=filters, kernel_size=(1, 1), strides=2, padding="same"
    )(input)
    x = keras.layers.PReLU(shared_axes=[1, 2])(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1, padding="same")(x)
    return x


def downSample(input, filters):
    x = keras.layers.Conv2D(
        filters=filters, kernel_size=(1, 1), strides=2, padding="same"
    )(input)
    x = keras.layers.PReLU(shared_axes=[1, 2])(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1, padding="same")(x)
    return x


class MFAMBlock(keras.layers.Layer):
    def __init__(self, filters=3):
        super(MFAMBlock, self).__init__()
        self.filters = filters
        self.conv2d = keras.layers.Conv2D(
            filters=self.filters, kernel_size=(1, 1), padding="same"
        )

    def __call__(self, input1, input2, input3, scale=2):
        if scale == 1:
            input2 = upSample(input2, self.filters)
            input3 = upSample(input3, self.filters)
            input3 = upSample(input3, self.filters)
        if scale == 2:
            input1 = downSample(input1, self.filters)
            input3 = upSample(input3, self.filters)
        if scale == 3:
            input1 = downSample(input1, self.filters)
            input1 = downSample(input1, self.filters)
            input2 = downSample(input2, self.filters)

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
