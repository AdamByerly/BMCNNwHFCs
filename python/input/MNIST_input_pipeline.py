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

import os
from python.input.input_pipeline_base import InputPipelineBase
import tensorflow as tf


class MNIST(InputPipelineBase):
    IMG_SIZE = 28
    IMG_CHANNELS = 1
    CLASSES = 10

    def __init__(self, data_base_dir,
            augment_training_data=True, augmentation_type=1):
        InputPipelineBase.__init__(self,
            self.CLASSES, self.IMG_SIZE, self.IMG_CHANNELS)
        self._augment_training_data  = augment_training_data
        self._augmentation_type      = augmentation_type

        self._training_images_file   = os.path.join(data_base_dir,
                                        "train-images-idx3-ubyte")
        self._training_labels_file   = os.path.join(data_base_dir,
                                        "train-labels-idx1-ubyte")
        self._validation_images_file = os.path.join(data_base_dir,
                                        "t10k-images-idx3-ubyte")
        self._validation_labels_file = os.path.join(data_base_dir,
                                        "t10k-labels-idx1-ubyte")

    def get_image_size(self):
        return self.IMG_SIZE

    def get_training_dataset(self, epoch, shuffle=True):
        dataset = self._load_dataset(
            self._training_images_file, self._training_labels_file)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.TRAINING_BUFFER_SIZE)
        dataset = self._augment(dataset)
        dataset = self._get_pipeline_end(dataset)
        return dataset

    def get_validation_dataset(self, epoch, shuffle=False):
        dataset = self._load_dataset(
            self._validation_images_file, self._validation_labels_file)
        dataset = self._get_pipeline_end(dataset)
        return dataset

    def get_n_training_samples(self, n, augment=False, shuffle=False):
        dataset = self._load_dataset(
            self._training_images_file, self._training_labels_file)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.TRAINING_BUFFER_SIZE)
        if augment:
            dataset = self._augment(dataset)
        dataset = self._get_pipeline_end(dataset)
        dataset = dataset.batch(n)
        return dataset

    def _get_pipeline_end(self, dataset):
        dataset = dataset.map(self._prepare_sample,
            num_parallel_calls=self.PARALLEL_INPUT_CALLS)
        dataset = dataset.map(self._image_center,
            num_parallel_calls=self.PARALLEL_INPUT_CALLS)
        return dataset.prefetch(-1)

    def _load_dataset(self, image_file, label_file):
        imagedataset = tf.data.FixedLengthRecordDataset(image_file,
            self.IMG_SIZE * self.IMG_SIZE, header_bytes=16)
        imagedataset = imagedataset.map(self._read_image,
            num_parallel_calls=self.PARALLEL_INPUT_CALLS)
        labelsdataset = tf.data.FixedLengthRecordDataset(
            label_file, 1, header_bytes=8)
        labelsdataset = labelsdataset.map(self._read_label,
            num_parallel_calls=self.PARALLEL_INPUT_CALLS)
        dataset = tf.data.Dataset.zip((imagedataset, labelsdataset))
        return dataset

    def _read_image(self, tf_bytestring):
        image = tf.io.decode_raw(tf_bytestring, tf.uint8)
        image = tf.cast(image, tf.float32) / 256.0
        image = tf.reshape(image,
            [self.IMG_SIZE, self.IMG_SIZE, self.IMG_CHANNELS])
        return image

    @staticmethod
    def _read_label(tf_bytestring):
        label = tf.io.decode_raw(tf_bytestring, tf.uint8)
        label = tf.reshape(label, [])
        return label

    def _augment(self, dataset):
        if self._augment_training_data:
            if self._augmentation_type == 1:
                dataset = dataset.map(self._image_rotate_random,
                    num_parallel_calls=self.PARALLEL_INPUT_CALLS)
                dataset = dataset.map(self._image_shift_rand,
                    num_parallel_calls=self.PARALLEL_INPUT_CALLS)
                dataset = dataset.map(self._image_squish_random,
                    num_parallel_calls=self.PARALLEL_INPUT_CALLS)
                dataset = dataset.map(self._image_erase_random,
                    num_parallel_calls=self.PARALLEL_INPUT_CALLS)
            elif self._augmentation_type == 2:
                dataset = dataset.map(self._image_shift_rand,
                    num_parallel_calls=self.PARALLEL_INPUT_CALLS)
            elif self._augmentation_type == 3:
                dataset = dataset.map(self._image_shift_rand_max2,
                    num_parallel_calls=self.PARALLEL_INPUT_CALLS)
            else:
                dataset = dataset.map(self._distort_image,
                    num_parallel_calls=self.PARALLEL_INPUT_CALLS)
        return dataset

    def _prepare_sample(self, image, label):
        return image, tf.one_hot(label, self.get_class_count())

    def _image_shift_rand_max2(self, image, label):
        return self._image_shift_common(image,
            label, self.get_image_size(), 2, 2, 2, 2)

    def _image_shift_rand(self, image, label):
        return self._image_shift_rand_static(
            self.get_image_size(), image, label)

    @staticmethod
    def _image_shift_rand_static(img_size, image, label):
        image = tf.reshape(image, [img_size, img_size])
        nonzero_x_cols = tf.cast(tf.where(tf.greater(
            tf.reduce_sum(image, axis=0), 0)), tf.int32)
        nonzero_y_rows = tf.cast(tf.where(tf.greater(
            tf.reduce_sum(image, axis=1), 0)), tf.int32)
        left_margin = tf.reduce_min(nonzero_x_cols)
        right_margin = img_size - tf.reduce_max(nonzero_x_cols) - 1
        top_margin = tf.reduce_min(nonzero_y_rows)
        bot_margin = img_size - tf.reduce_max(nonzero_y_rows) - 1
        return MNIST._image_shift_common(image, label, img_size,
            left_margin, right_margin, top_margin, bot_margin)

    @staticmethod
    def _image_shift_common(image, label, img_size,
            left_margin, right_margin, top_margin, bot_margin):
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
        image = tf.reshape(image, [img_size * img_size])
        image = tf.roll(image, y_amt * img_size, axis=0)
        image = tf.reshape(image, [img_size, img_size])
        image = tf.transpose(image)
        image = tf.reshape(image, [img_size * img_size])
        image = tf.roll(image, x_amt * img_size, axis=0)
        image = tf.reshape(image, [img_size, img_size])
        image = tf.transpose(image)
        image = tf.reshape(image, [img_size, img_size, 1])
        return image, label
