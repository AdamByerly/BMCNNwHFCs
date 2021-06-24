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

# Portions Copyright 2021 The TensorFlow Datasets Authors
# (Also licensed under Apache 2.0)
# https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/
#   image_classification/smallnorb.py

import os
import numpy as np
from python.input.input_pipeline_base import InputPipelineBase
import tensorflow as tf


# noinspection PyPep8Naming
class smallNORB(InputPipelineBase):
    IMG_SIZE = 96
    IMG_CHANNELS = 1
    CLASSES = 5

    def __init__(self, data_base_dir, augment_training_data=True,
            resize_images_to=48, crop_size=32):
        InputPipelineBase.__init__(self,
            self.CLASSES, crop_size, self.IMG_CHANNELS)
        self._augment_training_data = augment_training_data
        self._resize_images_to      = resize_images_to
        self._crop_size             = crop_size

        self._files = ["smallnorb-5x46789x9x18x6x2x96x96-training-info.mat",
                       "smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat",
                       "smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat",
                       "smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat",
                       "smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat",
                       "smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat"]

        self._files = [os.path.join(data_base_dir, f) for f in self._files]

        self._file_data = dict()

    def get_training_dataset(self, epoch, shuffle=True):
        dataset = self._load_dataset(self._files[0],
                    self._files[1], self._files[2])
        dataset = dataset.map(self._choose_image,
            num_parallel_calls=self.PARALLEL_INPUT_CALLS)
        dataset = dataset.map(self._random_crop,
            num_parallel_calls=self.PARALLEL_INPUT_CALLS)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.TRAINING_BUFFER_SIZE)
        if self._augment_training_data:
            dataset = dataset.map(self._augment,
                        num_parallel_calls=self.PARALLEL_INPUT_CALLS)
        dataset = self._get_pipeline_end(dataset)
        return dataset

    def get_validation_dataset(self, epoch, shuffle=False):
        dataset = self._load_dataset(self._files[3],
                    self._files[4], self._files[5])
        dataset = dataset.map(self._choose_image,
            num_parallel_calls=self.PARALLEL_INPUT_CALLS)
        dataset = dataset.map(self._central_crop,
            num_parallel_calls=self.PARALLEL_INPUT_CALLS)
        dataset = self._get_pipeline_end(dataset)
        return dataset

    # noinspection PyUnusedLocal
    def _choose_image(self, image1, image2,
            label, instance, elevation, azimuth, lighting):
        # # Randomly choose 1 of the 2 images from the pair
        # rand_choice = tf.random.uniform([1], maxval=2, dtype=tf.int32)
        # image = tf.gather([image1, image2], rand_choice, axis=0)
        # Choose the first image of the pair
        image = image1
        image = tf.reshape(image, [self.IMG_SIZE,
                    self.IMG_SIZE, self.IMG_CHANNELS])
        image = tf.image.resize(image,
                    [self._resize_images_to, self._resize_images_to],
                    tf.image.ResizeMethod.BILINEAR, False, False)
        return image, label

    def _random_crop(self, image, label):
        image = tf.image.random_crop(image, tf.stack([self._crop_size,
                    self._crop_size, self.IMG_CHANNELS]))
        return image, label

    def _central_crop(self, image, label):
        offset = ( self._resize_images_to - self._crop_size ) // 2
        image = tf.image.crop_to_bounding_box(image,
                    offset, offset, self._crop_size, self._crop_size)
        return image, label

    @staticmethod
    def _augment(image, label):
        image = tf.image.random_brightness(image, max_delta=0.25)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        return image, label

    def _get_pipeline_end(self, dataset):
        dataset = dataset.map(self._prepare_sample,
            num_parallel_calls=self.PARALLEL_INPUT_CALLS)
        dataset = dataset.map(self._image_center,
            num_parallel_calls=self.PARALLEL_INPUT_CALLS)
        return dataset.prefetch(-1)

    def _prepare_sample(self, image, label):
        return image, tf.one_hot(label, self.get_class_count())

    def _load_dataset(self, info_file, cat_file, dat_file):
        info_array, cat_array, dat_array = \
            self._read_files(info_file, cat_file, dat_file)

        # Each item is a 7-tuple where:
        #   0: image 1 (of the paired images)
        #   1: image 2 (of the paired images)
        #   2: class label
        #   3: instance
        #   4: elevation label
        #   5: azimuth label
        #   6: lighting label
        def gen():
            for d, c, i in zip(info_array, cat_array, dat_array):
                yield 1-i[0]/256., 1-i[1]/256., c, d[0], d[1], d[2], d[3]

        return tf.data.Dataset.from_generator(gen,
            output_types=(tf.float32, tf.float32, tf.uint8,
                          tf.uint8, tf.uint8, tf.uint8, tf.uint8))

    def _read_files(self, info_path, cat_path, dat_path):
        # There's a memory leak somewhere in _read_binary_matrix (I suspect
        #  tf.io.gfile.GFile), so we cache our data and only read it once.
        # This makes sense anyway.
        def read_file(file_path):
            if file_path not in self._file_data.keys():
                ary = self._read_binary_matrix(file_path)
                self._file_data[file_path] = ary
            else:
                ary = self._file_data[file_path]
            return ary

        info_array = read_file(info_path)
        cat_array  = read_file(cat_path)
        dat_array  = read_file(dat_path)

        dat_array  = np.expand_dims(dat_array, axis=4)

        # Azimuth values are 0, 2, 4, .., 34.  Divide by 2 to get proper labels.
        info_array = np.copy(info_array)  # Make read-only buffer writable.
        info_array[:, 2] = info_array[:, 2] / 2

        return info_array, cat_array, dat_array

    @staticmethod
    def _read_binary_matrix(filename):
        with tf.io.gfile.GFile(filename, "rb") as f:
            s = f.read()

            # Data is stored in little-endian byte order.
            int32_dtype = np.dtype("int32").newbyteorder("<")

            # The first 4 bytes contain a magic # that specifies the data type.
            magic = int(np.frombuffer(s, dtype=int32_dtype, count=1))
            if magic == 507333717:
                data_dtype = np.dtype("uint8")
            elif magic == 507333716:
                data_dtype = np.dtype("int32").newbyteorder("<")
            else:
                raise ValueError("Invalid magic value for data type!")

            # The second 4 bytes contain an int32 with the number of
            #  dimensions of the stored array.
            ndim = int(np.frombuffer(s, dtype=int32_dtype, count=1, offset=4))

            # The next ndim x 4 bytes contain the shape of the array in int32.
            dims = np.frombuffer(s, dtype=int32_dtype, count=ndim, offset=8)

            # If the array has less than three dimensions, three int32 are
            #  still used to save the shape info (remaining int32 are simply
            #  set to 1). The shape info hence uses max(3, ndim) bytes.
            bytes_used_for_shape_info = max(3, ndim) * 4

            # The remaining bytes are the array.
            data = np.frombuffer(s, dtype=data_dtype,
                    offset=8+bytes_used_for_shape_info)

            return data.reshape(tuple(dims))
