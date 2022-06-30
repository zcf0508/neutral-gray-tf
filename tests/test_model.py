import unittest

import tensorflow as tf
from neutral_gray.model import GRAY


class TestGRAY(unittest.TestCase):
    def test_1(self):
        model = GRAY().getModel()

        model.summary()
