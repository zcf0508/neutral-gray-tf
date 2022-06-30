import tensorflow as tf
from tensorflow import keras

from .layers.attenction import AttenctionBlock
from .layers.residual import ResidualBlock
from .layers.mir import MFAMBlock, downSample, upSample
IMG_WIDTH = None
IMG_HEIGHT = None


class MyLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return (
            0.01 * keras.losses.BinaryCrossentropy()(y_true, y_pred)
            + 0.5 * keras.losses.MeanSquaredError()(y_true, y_pred)
            + 0.5 * keras.losses.MeanAbsoluteError()(y_true, y_pred)
        )


class GRAY:
    def __init__(self):
        self.residualBlock = ResidualBlock(128, 3)

        self.attenction_16 = AttenctionBlock(16)
        self.attenction_32 = AttenctionBlock(32)
        self.attenction_64 = AttenctionBlock(64)

        self.mir_16 = MFAMBlock(16)
        self.mir_32 = MFAMBlock(32)
        self.mir_64 = MFAMBlock(64)

        self.conv2d_16 = keras.layers.Conv2D(
            filters=16, kernel_size=(3, 3), padding="same"
        )
        self.conv2d_32 = keras.layers.Conv2D(
            filters=32, kernel_size=(3, 3), padding="same"
        )
        self.conv2d_64 = keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), padding="same"
        )

    def getModel(self):
        input = keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))

        down1 = downSample(input, 16)
        down2 = downSample(down1, 32)
        down3 = downSample(down2, 64)
        down4 = downSample(down3, 128)

        res = self.residualBlock(down4)
        res = self.residualBlock(res)
        res = self.residualBlock(res)
        res = self.residualBlock(res)
        res = self.residualBlock(res)

        # 16
        down1_up_atten = self.attenction_16(
            keras.layers.UpSampling2D(size=(2, 2))(down1)
        )
        down1_direct_atten = self.attenction_16(down1)
        down1_down_atten = self.attenction_16(
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(down1)
        )

        down1_mir1 = self.mir_16(
            down1_up_atten, down1_direct_atten, down1_down_atten, 1
        )
        down1_mir2 = self.mir_16(
            down1_up_atten, down1_direct_atten, down1_down_atten, 2
        )
        down1_mir3 = self.mir_16(
            down1_up_atten, down1_direct_atten, down1_down_atten, 3
        )

        down1_mir1_atten = self.attenction_16(down1_mir1)
        down1_mir2_atten = self.attenction_16(down1_mir2)
        down1_mir3_atten = self.attenction_16(down1_mir3)

        down1_mir = self.mir_16(down1_mir1_atten, down1_mir2_atten, down1_mir3_atten, 2)
        down1_mir = self.conv2d_16(down1_mir)

        # 32
        down2_up_atten = self.attenction_32(
            keras.layers.UpSampling2D(size=(2, 2))(down2)
        )
        down2_direct_atten = self.attenction_32(down2)
        down2_down_atten = self.attenction_32(
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(down2)
        )

        down2_mir1 = self.mir_32(
            down2_up_atten, down2_direct_atten, down2_down_atten, 1
        )
        down2_mir2 = self.mir_32(
            down2_up_atten, down2_direct_atten, down2_down_atten, 2
        )
        down2_mir3 = self.mir_32(
            down2_up_atten, down2_direct_atten, down2_down_atten, 3
        )

        down2_mir1_atten = self.attenction_32(down2_mir1)
        down2_mir2_atten = self.attenction_32(down2_mir2)
        down2_mir3_atten = self.attenction_32(down2_mir3)

        down2_mir = self.mir_32(down2_mir1_atten, down2_mir2_atten, down2_mir3_atten, 2)
        down2_mir = self.conv2d_32(down2_mir)

        # 64
        down3_up_atten = self.attenction_64(
            keras.layers.UpSampling2D(size=(2, 2))(down3)
        )
        down3_direct_atten = self.attenction_64(down3)
        down3_down_atten = self.attenction_64(
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(down3)
        )

        down3_mir1 = self.mir_64(
            down3_up_atten, down3_direct_atten, down3_down_atten, 1
        )
        down3_mir2 = self.mir_64(
            down3_up_atten, down3_direct_atten, down3_down_atten, 2
        )
        down3_mir3 = self.mir_64(
            down3_up_atten, down3_direct_atten, down3_down_atten, 3
        )

        down3_mir1_atten = self.attenction_64(down3_mir1)
        down3_mir2_atten = self.attenction_64(down3_mir2)
        down3_mir3_atten = self.attenction_64(down3_mir3)

        down3_mir = self.mir_64(down3_mir1_atten, down3_mir2_atten, down3_mir3_atten, 2)
        down3_mir = self.conv2d_64(down3_mir)

        # 合并
        up3 = upSample(res, 128)
        up3 = keras.layers.Concatenate()([down3_mir, up3])

        up2 = upSample(up3, 64)
        up2 = keras.layers.Concatenate()([down2_mir, up2])

        up1 = upSample(up2, 32)
        up1 = keras.layers.Concatenate()([down1_mir, up1])

        output = keras.layers.Conv2DTranspose(3, 3, strides=2, padding="same")(up1)

        model = keras.Model(input, output)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(3e-4),
            # loss = keras.losses.BinaryCrossentropy(),
            # loss = keras.losses.MeanAbsoluteError(),
            # loss = keras.losses.MeanSquaredError(), # l2 loss
            loss=MyLoss(),
            metrics=["accuracy"],
        )

        return model
