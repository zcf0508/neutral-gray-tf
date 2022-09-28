import tensorflow as tf
keras = tf.keras

from .layers.attenction import AttenctionBlock
from .layers.residual import ResidualBlock
from .layers.mir import MFAMBlock, downSample, upSample
from .config import FILTER, LOSS_WEIGHT, RES

IMG_WIDTH = None
IMG_HEIGHT = None


class MyLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        a = keras.losses.BinaryCrossentropy()(y_true, y_pred)
        b = keras.losses.MeanSquaredError()(y_true, y_pred)
        c = keras.losses.MeanAbsoluteError()(y_true, y_pred)
        # print(f'二值交叉熵-{tf.print(a)} 均方误差-{tf.print(b)} 平均绝对误差-{tf.print(c)}')
        # 像素准确度
        # tf.print(tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32)))
        return (
            LOSS_WEIGHT[0] * a
            + LOSS_WEIGHT[1] * b
            + LOSS_WEIGHT[2] * c
        )


class GRAY:
    def __init__(self):
        self.residualBlock = ResidualBlock(FILTER[3], 3)

        self.attenction_0 = AttenctionBlock(FILTER[0])
        self.attenction_1 = AttenctionBlock(FILTER[1])
        self.attenction_2 = AttenctionBlock(FILTER[2])

        self.mir_0 = MFAMBlock(FILTER[0])
        self.mir_1 = MFAMBlock(FILTER[1])
        self.mir_2 = MFAMBlock(FILTER[2])

        self.conv2d_0 = keras.layers.Conv2D(
            filters=FILTER[0], kernel_size=(3, 3), padding="same"
        )
        self.conv2d_1 = keras.layers.Conv2D(
            filters=FILTER[1], kernel_size=(3, 3), padding="same"
        )
        self.conv2d_2 = keras.layers.Conv2D(
            filters=FILTER[2], kernel_size=(3, 3), padding="same"
        )

    def getModel(self):
        input = keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))

        down1 = downSample(input, FILTER[0])
        down2 = downSample(down1, FILTER[1])
        down3 = downSample(down2, FILTER[2])
        down4 = downSample(down3, FILTER[3])

        res = self.residualBlock(down4)
        for _ in range(RES):
            res = self.residualBlock(res)

        # 1
        down1_up_atten = self.attenction_0(
            keras.layers.UpSampling2D(size=(2, 2))(down1)
        )
        down1_direct_atten = self.attenction_0(down1)
        down1_down_atten = self.attenction_0(
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(down1)
        )

        down1_mir1 = self.mir_0(
            down1_up_atten, down1_direct_atten, down1_down_atten, 1
        )
        down1_mir2 = self.mir_0(
            down1_up_atten, down1_direct_atten, down1_down_atten, 2
        )
        down1_mir3 = self.mir_0(
            down1_up_atten, down1_direct_atten, down1_down_atten, 3
        )

        down1_mir1_atten = self.attenction_0(down1_mir1)
        down1_mir2_atten = self.attenction_0(down1_mir2)
        down1_mir3_atten = self.attenction_0(down1_mir3)

        down1_mir = self.mir_0(down1_mir1_atten, down1_mir2_atten, down1_mir3_atten, 2)
        down1_mir = self.conv2d_0(down1_mir)

        # 2
        down2_up_atten = self.attenction_1(
            keras.layers.UpSampling2D(size=(2, 2))(down2)
        )
        down2_direct_atten = self.attenction_1(down2)
        down2_down_atten = self.attenction_1(
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(down2)
        )

        down2_mir1 = self.mir_1(
            down2_up_atten, down2_direct_atten, down2_down_atten, 1
        )
        down2_mir2 = self.mir_1(
            down2_up_atten, down2_direct_atten, down2_down_atten, 2
        )
        down2_mir3 = self.mir_1(
            down2_up_atten, down2_direct_atten, down2_down_atten, 3
        )

        down2_mir1_atten = self.attenction_1(down2_mir1)
        down2_mir2_atten = self.attenction_1(down2_mir2)
        down2_mir3_atten = self.attenction_1(down2_mir3)

        down2_mir = self.mir_1(down2_mir1_atten, down2_mir2_atten, down2_mir3_atten, 2)
        down2_mir = self.conv2d_1(down2_mir)

        # 3
        down3_up_atten = self.attenction_2(
            keras.layers.UpSampling2D(size=(2, 2))(down3)
        )
        down3_direct_atten = self.attenction_2(down3)
        down3_down_atten = self.attenction_2(
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(down3)
        )

        down3_mir1 = self.mir_2(
            down3_up_atten, down3_direct_atten, down3_down_atten, 1
        )
        down3_mir2 = self.mir_2(
            down3_up_atten, down3_direct_atten, down3_down_atten, 2
        )
        down3_mir3 = self.mir_2(
            down3_up_atten, down3_direct_atten, down3_down_atten, 3
        )

        down3_mir1_atten = self.attenction_2(down3_mir1)
        down3_mir2_atten = self.attenction_2(down3_mir2)
        down3_mir3_atten = self.attenction_2(down3_mir3)

        down3_mir = self.mir_2(down3_mir1_atten, down3_mir2_atten, down3_mir3_atten, 2)
        down3_mir = self.conv2d_2(down3_mir)

        # 合并
        up3 = upSample(res, FILTER[3])
        up3 = keras.layers.Concatenate()([down3_mir, up3])

        up2 = upSample(up3, FILTER[2])
        up2 = keras.layers.Concatenate()([down2_mir, up2])

        up1 = upSample(up2, FILTER[1])
        up1 = keras.layers.Concatenate()([down1_mir, up1])

        output = keras.layers.Conv2DTranspose(3, 3, strides=2, padding="same")(up1)

        model = keras.Model(input, output)

        return model

    def load_model(model_path: str):
        return tf.keras.models.load_model(
            model_path, custom_objects={"MyLoss": MyLoss()}
        )
