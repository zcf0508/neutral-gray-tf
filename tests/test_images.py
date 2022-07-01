import unittest
from matplotlib import pyplot as plt
import numpy as np

import tensorflow as tf
from neutral_gray.images import ImageLoder


class TestImages(unittest.TestCase):
    def test_1(self):
        (train_images, train_results), (test_images, test_results) = ImageLoder(
            "./data/0", "./data/1"
        ).load_data(1)

        self.assertEqual(train_images.shape[1:4], (336, 224, 3))
