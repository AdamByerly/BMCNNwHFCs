# Copyright 2021 Adam Byerly. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Portions Copyright 2016 The TensorFlow Authors
# (Also licensed under Apache 2.0)
# https://github.com/tensorflow/models/blob/master/research/slim/
#   preprocessing/inception_preprocessing.py

import abc
import cv2
import numpy as np
from python.constructs.loggable import Loggable
import tensorflow as tf


class InputPipelineBase(Loggable, metaclass=abc.ABCMeta):
    PARALLEL_INPUT_CALLS = 16
    TRAINING_BUFFER_SIZE = 1024

    def __init__(self, classes, image_size, image_channels):
        Loggable.__init__(self)
        self._classes                 = classes
        self._image_size              = image_size
        self._image_channels          = image_channels

    def get_class_count(self):
        return self._classes

    def get_image_size(self):
        return self._image_size

    def get_image_channels(self):
        return self._image_channels

    @abc.abstractmethod
    def get_training_dataset(self, epoch, shuffle=True):
        """
        This method should return a tf.data.Dataset for the training data.
        This dataset should be ready for consumption and thus be batched and
        have all processing applied.
        """
        pass

    @abc.abstractmethod
    def get_validation_dataset(self, epoch, shuffle=False):
        """
        This method should return a tf.data.Dataset for the validation data.
        This dataset should be ready for consumption and thus be batched and
        have all processing applied.
        """
        pass

    @staticmethod
    def _image_center(image, label):
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image, label

    def _image_squish_random(self, image, label):
        img_size = self.get_image_size()
        rand_amts = tf.minimum(tf.abs(tf.random.normal([2], 0, .33)), .9999)
        width_mod = tf.cast(tf.floor(
            (rand_amts[0] * (img_size / 4)) + 1), tf.int32)
        offset_mod = tf.cast(tf.floor(rand_amts[1] * 2.0), tf.int32)
        offset = (width_mod // 2) + offset_mod
        image = tf.image.resize(image,
            [img_size, img_size - width_mod],
            method=tf.image.ResizeMethod.LANCZOS3,
            preserve_aspect_ratio=False,
            antialias=True)
        image = tf.image.pad_to_bounding_box(
            image, 0, offset, img_size, img_size + offset_mod)
        image = tf.image.crop_to_bounding_box(
            image, 0, 0, img_size, img_size)
        return image, label

    def _image_rotate_random(self, image, label):
        img_size = self.get_image_size()
        channels = self.get_image_channels()
        rand_amts = tf.maximum(tf.minimum(
            tf.random.normal([2], 0, .33), .9999), -.9999)
        angle = rand_amts[0] * 30  # degrees
        new_image = tf.py_function(
            self._image_rotate_random_py_func,
            (image, angle), tf.float32)
        new_image = tf.cond(rand_amts[1] > 0, lambda: image, lambda: new_image)
        new_image = tf.reshape(new_image, [img_size, img_size, channels])
        return new_image, label

    def _image_rotate_random_py_func(self, image, angle):
        img_size = self.get_image_size()
        rot_mat = cv2.getRotationMatrix2D(
            (img_size/2, img_size/2), float(angle), 1.0)
        rotated = cv2.warpAffine(image.numpy(), rot_mat,
            (img_size, img_size))
        return rotated

    def _image_erase_random(self, image, label):
        img_size = self.get_image_size()
        sess = tf.compat.v1.Session()
        with sess.as_default():
            rand_amts = tf.random.uniform([2])
            x = tf.cast(tf.floor(rand_amts[0]*19)+4, tf.int32)
            y = tf.cast(tf.floor(rand_amts[1]*19)+4, tf.int32)
            patch = tf.zeros([4, 4])
            mask = tf.pad(patch, [[x, img_size-x-4],
                [y, img_size-y-4]],
                mode='CONSTANT', constant_values=1)
            image = tf.multiply(image, tf.expand_dims(mask, -1))
            return image, label

    def _distort_image(self, image, label, bbox=None):
        if bbox is None:
            bbox = np.empty([1, 0, 4])
        bbox_begin, bbox_size, distort_bbox =\
            tf.image.sample_distorted_bounding_box(
                tf.shape(image), bounding_boxes=bbox,
                min_object_covered=0.1, aspect_ratio_range=[0.75, 1.33],
                area_range=[0.05, 1.0], max_attempts=100,
                use_image_if_no_bounding_boxes=True)

        # Crop the image to the specified bounding box.
        image = tf.slice(image, bbox_begin, bbox_size)

        # This resizing operation may distort the images because the aspect
        # ratio is not respected.
        image = tf.image.resize(image, [self._image_size, self._image_size],
                    method=tf.image.ResizeMethod.LANCZOS3)

        image.set_shape([self._image_size, self._image_size, 3])

        image = tf.image.random_flip_left_right(image)

        distortion_order = tf.cast(tf.math.floor(
            tf.random.uniform([1])*2), tf.int32)[0]

        def distort_order1(img):
            img = tf.image.random_brightness(img, max_delta=32. / 255.)
            img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
            img = tf.image.random_hue(img, max_delta=0.2)
            img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
            return img

        def distort_order2(img):
            img = tf.image.random_brightness(img, max_delta=32. / 255.)
            img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
            img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
            img = tf.image.random_hue(img, max_delta=0.2)
            return img

        image = tf.cond(tf.math.equal(distortion_order, tf.constant(0)),
                    lambda: distort_order1(image),
                    lambda: distort_order2(image))

        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label

    # noinspection PyUnusedLocal
    def _massage_image(self, image, label, bbox=None):
        # image = tf.image.central_crop(image, central_fraction=0.875)
        image = tf.image.resize(image, [self._image_size, self._image_size],
                    method=tf.image.ResizeMethod.LANCZOS3)
        image.set_shape([self._image_size, self._image_size, 3])
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label
