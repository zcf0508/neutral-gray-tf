#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageEnhance

from .config import IMG_WIDTH

import os


def is_image_file(filename):
    return any(
        filename.lower().endswith(extension)
        for extension in [
            ".jpg",
        ]
    )


class ImageLoder:
    def __init__(self, source_dir, result_dir):
        self.sources = []
        self.results = []
        self.__get_source__(source_dir)
        self.__get_result__(result_dir)

    def __get_source__(self, source_dir):
        sources = []
        images = [
            os.path.join(source_dir, x)
            for x in os.listdir(source_dir)
            if is_image_file(x)
        ]
        for image_url in images:
            img = Image.open(image_url)
            aspect_ratio = img.height / img.width

            img = img.resize(
                (IMG_WIDTH, int(IMG_WIDTH * aspect_ratio)), Image.Resampling.LANCZOS
            )

            # img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)  # 水平翻转
            # enh_bri = ImageEnhance.Brightness(img)
            # img = enh_bri.enhance(factor=0.9)  # 亮度
            # img = img.rotate(10)  # 旋转

            img_array = (np.asarray(img) - 127.5) / 127.5  # 归一化
            # img_gray  = np.dot(img_array,[0.299,0.587,0.114]) / 255.0 # 转为黑白

            # 预览
            # plt.imshow((np.squeeze(img_array) * 127.5 + 127.5).astype(np.uint8))
            # plt.show()

            sources.append(img_array)

        self.sources = np.array(sources)

    def __get_result__(self, result_dir):
        results = []
        images = [
            os.path.join(result_dir, x)
            for x in os.listdir(result_dir)
            if is_image_file(x)
        ]
        for image_url in images:
            img = Image.open(image_url)
            aspect_ratio = img.height / img.width

            img = img.resize(
                (IMG_WIDTH, int(IMG_WIDTH * aspect_ratio)), Image.Resampling.LANCZOS
            )

            # img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)  # 水平翻转
            # enh_bri = ImageEnhance.Brightness(img)
            # img = enh_bri.enhance(factor=0.9)  # 亮度
            # img = img.rotate(10)  # 旋转

            img_array = (np.asarray(img) - 127.5) / 127.5  # 归一化
            # img_gray  = np.dot(img_array,[0.299,0.587,0.114]) / 255.0 # 转为黑白

            # 预览
            # plt.imshow((np.squeeze(img_array) * 127.5 + 127.5).astype(np.uint8))
            # plt.show()

            results.append(img_array)

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
