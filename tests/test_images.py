import unittest
from matplotlib import pyplot as plt
import numpy as np

import tensorflow as tf
from neutral_gray.images import ImageLoderV1, ImageLoderV2

from neutral_gray.config import IMG_WIDTH


class TestImages(unittest.TestCase):
    def test_1(self):
        (train_images, train_results), (test_images, test_results) = ImageLoderV1(
            "./data/0", "./data/1"
        ).load_data(1)

        self.assertEqual(train_images.shape[1:4], (IMG_WIDTH, IMG_WIDTH, 3))

    def test_2(self):
        images = ImageLoderV2("./data/0", "./data/1").load_data()
        train_images, train_results = next(iter(images))

        test_images_data = ImageLoderV2("./data_test/0", "./data_test/1").load_data()
        test_images, test_results = next(iter(test_images_data))

        self.assertEqual(train_images.shape[1:4], (IMG_WIDTH, IMG_WIDTH, 3))
