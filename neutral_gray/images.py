#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from .config import IMG_WIDTH, BATCH_SIZE

import os


def _is_image_file(filename: str):
    return any(
        filename.lower().endswith(extension)
        for extension in [
            ".jpg",
        ]
    )


def encode_image(image_url: str):
    img_raw = tf.io.read_file(image_url)
    img_tensor = tf.image.decode_jpeg(img_raw)

    img_tensor = tf.image.resize(img_tensor, [int(IMG_WIDTH * 3 / 2), IMG_WIDTH])
    img_final = img_tensor / 255.0  # 归一化

    # 预览
    # plt.imshow((np.squeeze(img_final) * 255).astype(np.uint8))
    # plt.show()

    return img_final


def decode_image(img_array: npt.ArrayLike, save_path: str = ""):
    """
    decode_image(
        model.predict(...)[0], 'path/to/save'
    )
    """
    result = tf.clip_by_value(np.asarray(img_array) * 255, 0, 255)
    result = tf.cast(result, dtype=tf.uint8)

    # 预览
    # plt.imshow(result)
    # plt.show()

    if save_path:
        tf.keras.utils.save_img(save_path, result)

    return result


class ImageLoderV1:
    def __init__(self, source_dir: str, result_dir: str):
        self.sources = []
        self.results = []
        self.__get_source__(source_dir)
        self.__get_result__(result_dir)

    def __get_source__(self, source_dir: str):
        sources = []
        images = [
            os.path.join(source_dir, x)
            for x in os.listdir(source_dir)
            if _is_image_file(x)
        ]
        for image_url in images:
            sources.append(encode_image(image_url))

        self.sources = np.array(sources)

    def __get_result__(self, result_dir: str):
        results = []
        images = [
            os.path.join(result_dir, x)
            for x in os.listdir(result_dir)
            if _is_image_file(x)
        ]
        for image_url in images:
            results.append(encode_image(image_url))

        self.results = np.array(results)

    def load_data(self, rate=1):
        img_len = np.size(self.sources, 0)

        # 随机打乱图片顺序
        state = np.random.get_state()
        np.random.shuffle(self.sources)
        np.random.set_state(state)
        np.random.shuffle(self.results)

        if np.size(self.sources, 0) == np.size(self.results, 0):
            train_len = int(img_len * 0.85 * rate)
            test_len = int((img_len - int(img_len * 0.85)) * rate)
            return (self.sources[0:train_len], self.results[0:train_len]), (
                self.sources[train_len : (train_len + test_len)],
                self.results[train_len : (train_len + test_len)],
            )


AUTOTUNE = tf.data.experimental.AUTOTUNE


class ImageLoderV2:
    def __init__(self, source_dir: str, result_dir: str, cached: bool = True):
        self.source_dir = source_dir
        self.result_dir = result_dir
        self.cached = cached
        self.image_count = 0
        self.images_ds = None
        self.create_data_set()

    def create_data_set(self):
        source_images_urls = [
            os.path.join(self.source_dir, x)
            for x in os.listdir(self.source_dir)
            if _is_image_file(x)
        ]
        self.image_count = len(source_images_urls)
        result_images_urls = [
            os.path.join(self.result_dir, x)
            for x in os.listdir(self.result_dir)
            if _is_image_file(x)
        ]
        if len(source_images_urls) != len(result_images_urls):
            raise RuntimeError(f"源数据和结果数据数量不相等")

        ds = tf.data.Dataset.from_tensor_slices(
            (source_images_urls, result_images_urls)
        )

        self.images_ds = ds.map(
            lambda source, result: (encode_image(source), encode_image(result))
        )

    def load_data(self):

        ds = (
            self.images_ds.cache(filename="./cache.tf-data")
            if self.cached
            else self.images_ds
        ).apply(tf.data.experimental.shuffle_and_repeat(buffer_size=self.image_count))
        ds = ds.repeat()
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds
