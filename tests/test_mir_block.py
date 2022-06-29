import unittest

import tensorflow as tf
from neutral_gray.layers.mir import MFAMBlock


class TestMFAMBlock(unittest.TestCase):
    def test_1(self):
        mir_block = MFAMBlock(4)

        x1 = tf.keras.Input(shape=(224, 224, 4))
        x2 = tf.keras.Input(shape=(112, 112, 4))
        x3 = tf.keras.Input(shape=(56, 56, 4))
        y = mir_block(x1, x2, x3)
        self.assertEqual(y.shape, (None, 112, 112, 4))
