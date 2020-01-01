# Copyright 2020 Adam Byerly. All Rights Reserved.
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

import os
import cv2
import tensorflow as tf
tf2 = tf.compat.v2

MNIST_IMG_SIZE = 28
MNIST_CLASSES = 10
MNIST_TRAIN_IMAGE_COUNT = 60000
MNIST_EVAL_IMAGE_COUNT = 10000
PARALLEL_INPUT_CALLS = 16


class InputPipeline(object):
    def __init__(self, data_base_dir):
        self._training_images_file = os.path.join(data_base_dir,
                                        "train-images-idx3-ubyte")
        self._training_labels_file = os.path.join(data_base_dir,
                                        "train-labels-idx1-ubyte")
        self._validation_images_file = os.path.join(data_base_dir,
                                        "t10k-images-idx3-ubyte")
        self._validation_labels_file = os.path.join(data_base_dir,
                                        "t10k-labels-idx1-ubyte")

    @staticmethod
    def get_class_count():
        return MNIST_CLASSES

    @staticmethod
    def get_train_image_count():
        return MNIST_TRAIN_IMAGE_COUNT

    @staticmethod
    def get_eval_image_count():
        return MNIST_EVAL_IMAGE_COUNT

    @staticmethod
    def read_label(tf_bytestring):
        label = tf.io.decode_raw(tf_bytestring, tf.uint8)
        label = tf.reshape(label, [])
        label = tf.one_hot(label, MNIST_CLASSES)
        return label

    @staticmethod
    def read_image(tf_bytestring):
        image = tf.io.decode_raw(tf_bytestring, tf.uint8)
        image = tf.cast(image, tf.float32) / 256.0
        image = tf.reshape(image, [MNIST_IMG_SIZE, MNIST_IMG_SIZE, 1])
        return image

    def load_dataset(self, image_file, label_file):
        imagedataset = tf.data.FixedLengthRecordDataset(
            image_file, MNIST_IMG_SIZE * MNIST_IMG_SIZE, header_bytes=16)
        imagedataset = imagedataset.map(self.read_image,
            num_parallel_calls=PARALLEL_INPUT_CALLS)
        labelsdataset = tf.data.FixedLengthRecordDataset(
            label_file, 1, header_bytes=8)
        labelsdataset = labelsdataset.map(self.read_label,
            num_parallel_calls=PARALLEL_INPUT_CALLS)
        dataset = tf.data.Dataset.zip((imagedataset, labelsdataset))
        return dataset

    @staticmethod
    def image_center(image, label):
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image, label

    @staticmethod
    def image_shift_rand(image, label):
        image = tf.reshape(image, [MNIST_IMG_SIZE, MNIST_IMG_SIZE])
        nonzero_x_cols = tf.cast(tf.where(tf.greater(
            tf.reduce_sum(image, axis=0), 0)), tf.int32)
        nonzero_y_rows = tf.cast(tf.where(tf.greater(
            tf.reduce_sum(image, axis=1), 0)), tf.int32)
        left_margin = tf.reduce_min(nonzero_x_cols)
        right_margin = MNIST_IMG_SIZE - tf.reduce_max(nonzero_x_cols) - 1
        top_margin = tf.reduce_min(nonzero_y_rows)
        bot_margin = MNIST_IMG_SIZE - tf.reduce_max(nonzero_y_rows) - 1
        rand_dirs = tf.random.uniform([2])
        dir_idxs = tf.cast(tf.floor(rand_dirs * 2), tf.int32)
        rand_amts = tf.minimum(tf.abs(tf.random.normal([2], 0, .33)), .9999)
        x_amts = [tf.floor(-1.0 * rand_amts[0] *
                  tf.cast(left_margin, tf.float32)), tf.floor(rand_amts[0] *
                  tf.cast(1 + right_margin, tf.float32))]
        y_amts = [tf.floor(-1.0 * rand_amts[1] *
                  tf.cast(top_margin, tf.float32)), tf.floor(rand_amts[1] *
                  tf.cast(1 + bot_margin, tf.float32))]
        x_amt = tf.cast(tf.gather(x_amts, dir_idxs[1], axis=0), tf.int32)
        y_amt = tf.cast(tf.gather(y_amts, dir_idxs[0], axis=0), tf.int32)
        image = tf.reshape(image, [MNIST_IMG_SIZE * MNIST_IMG_SIZE])
        image = tf.roll(image, y_amt * MNIST_IMG_SIZE, axis=0)
        image = tf.reshape(image, [MNIST_IMG_SIZE, MNIST_IMG_SIZE])
        image = tf.transpose(image)
        image = tf.reshape(image, [MNIST_IMG_SIZE * MNIST_IMG_SIZE])
        image = tf.roll(image, x_amt * MNIST_IMG_SIZE, axis=0)
        image = tf.reshape(image, [MNIST_IMG_SIZE, MNIST_IMG_SIZE])
        image = tf.transpose(image)
        image = tf.reshape(image, [MNIST_IMG_SIZE, MNIST_IMG_SIZE, 1])
        return image, label

    @staticmethod
    def image_squish_random(image, label):
        rand_amts = tf.minimum(tf.abs(tf.random.normal([2], 0, .33)), .9999)
        width_mod = tf.cast(tf.floor(
            (rand_amts[0] * (MNIST_IMG_SIZE / 4)) + 1), tf.int32)
        offset_mod = tf.cast(tf.floor(rand_amts[1] * 2.0), tf.int32)
        offset = (width_mod // 2) + offset_mod
        image = tf.image.resize(image,
            [MNIST_IMG_SIZE, MNIST_IMG_SIZE - width_mod],
            method=tf2.image.ResizeMethod.LANCZOS3,
            preserve_aspect_ratio=False,
            antialias=True)
        image = tf.image.pad_to_bounding_box(
            image, 0, offset, MNIST_IMG_SIZE, MNIST_IMG_SIZE + offset_mod)
        image = tf.image.crop_to_bounding_box(
            image, 0, 0, MNIST_IMG_SIZE, MNIST_IMG_SIZE)
        return image, label

    @staticmethod
    def image_rotate_random_py_func(image, angle):
        rot_mat = cv2.getRotationMatrix2D(
            (MNIST_IMG_SIZE/2, MNIST_IMG_SIZE/2), angle, 1.0)
        rotated = cv2.warpAffine(image.numpy(), rot_mat,
            (MNIST_IMG_SIZE, MNIST_IMG_SIZE))
        return rotated

    @staticmethod
    def image_rotate_random(image, label):
        rand_amts = tf.maximum(tf.minimum(
            tf.random.normal([2], 0, .33), .9999), -.9999)
        angle = rand_amts[0] * 30  # degrees
        new_image = tf.py_function(InputPipeline.image_rotate_random_py_func,
            (image, angle), tf.float32)
        new_image = tf.cond(rand_amts[1] > 0, lambda: image, lambda: new_image)
        return new_image, label

    @staticmethod
    def image_erase_random(image, label):
        sess = tf.compat.v1.Session()
        with sess.as_default():
            rand_amts = tf.random.uniform([2])
            x = tf.cast(tf.floor(rand_amts[0]*19)+4, tf.int32)
            y = tf.cast(tf.floor(rand_amts[1]*19)+4, tf.int32)
            patch = tf.zeros([4, 4])
            mask = tf.pad(patch, [[x, MNIST_IMG_SIZE-x-4],
                [y, MNIST_IMG_SIZE-y-4]],
                mode='CONSTANT', constant_values=1)
            image = tf.multiply(image, tf.expand_dims(mask, -1))
            return image, label

    def get_training_dataset(self, batch_size):
        with tf.name_scope("train_input"):
            dataset = self.load_dataset(self._training_images_file,
                                        self._training_labels_file)
            dataset = dataset.shuffle(buffer_size=MNIST_TRAIN_IMAGE_COUNT)
            dataset = dataset.map(self.image_rotate_random,
                num_parallel_calls=PARALLEL_INPUT_CALLS)
            dataset = dataset.map(self.image_shift_rand,
                num_parallel_calls=PARALLEL_INPUT_CALLS)
            dataset = dataset.map(self.image_squish_random,
                num_parallel_calls=PARALLEL_INPUT_CALLS)
            dataset = dataset.map(self.image_erase_random,
               num_parallel_calls=PARALLEL_INPUT_CALLS)
            dataset = dataset.map(self.image_center,
                num_parallel_calls=PARALLEL_INPUT_CALLS)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(-1)
            return dataset

    def get_validation_dataset(self, batch_size):
        with tf.name_scope("eval_input"):
            dataset = self.load_dataset(self._validation_images_file,
                                        self._validation_labels_file)
            dataset = dataset.cache()
            dataset = dataset.map(self.image_center,
                num_parallel_calls=PARALLEL_INPUT_CALLS)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(-1)
            return dataset
