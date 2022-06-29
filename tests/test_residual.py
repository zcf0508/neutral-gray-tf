import unittest

import tensorflow as tf
from neutral_gray.layers.residual import ResidualBlock


class TestResidualBlock(unittest.TestCase):
    def test_1(self):
        residual_block = ResidualBlock(32, 3)

        x = tf.keras.Input(shape=(14, 14, 32))
        y = residual_block(x)
        self.assertEqual(y.shape, (None, 14, 14, 32))
