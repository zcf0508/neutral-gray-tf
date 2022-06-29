import unittest

import tensorflow as tf
from neutral_gray.layers.attenction import AttenctionBlock


class TestAttentionBlock(unittest.TestCase):
    def test_1(self):
        attenction_block = AttenctionBlock(4)

        x = tf.keras.Input(shape=(112, 112, 4))
        y = attenction_block(x)
        self.assertEqual(y.shape, (None, 112, 112, 4))
