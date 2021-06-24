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

import abc
from python.input.input_pipeline_base import InputPipelineBase
import tensorflow as tf


class CifarInputPipeline(InputPipelineBase, metaclass=abc.ABCMeta):
    IMG_SIZE = 32
    IMG_CHANNELS = 3

    def __init__(self, classes,
            augment_training_data=True, augmentation_type=1):
        InputPipelineBase.__init__(self,
            classes, self.IMG_SIZE, self.IMG_CHANNELS)

        self._augment_training_data = augment_training_data
        self._augmentation_type     = augmentation_type

    def get_training_dataset(self, epoch, shuffle=True):
        dataset = self._load_dataset(self._get_training_files())
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.TRAINING_BUFFER_SIZE)
        dataset = self._augment(dataset)
        dataset = self._get_pipeline_end(dataset)
        return dataset

    def get_validation_dataset(self, epoch, shuffle=False):
        dataset = self._load_dataset(self._get_validation_files())
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.TRAINING_BUFFER_SIZE)
        dataset = self._get_pipeline_end(dataset)
        return dataset

    @abc.abstractmethod
    def _get_training_files(self):
        pass

    @abc.abstractmethod
    def _get_validation_files(self):
        pass

    @abc.abstractmethod
    def _get_class_byte_offset(self):
        pass

    @abc.abstractmethod
    def _get_image_byte_offset(self):
        pass

    def _load_dataset(self, image_files):
        imagedataset = tf.data.FixedLengthRecordDataset(image_files,
            self.IMG_SIZE * self.IMG_SIZE * self.IMG_CHANNELS +
            self._get_class_byte_offset() + 1, header_bytes=0)
        imagedataset = imagedataset.map(self._read_image,
            num_parallel_calls=self.PARALLEL_INPUT_CALLS)
        return imagedataset

    def _read_image(self, tf_bytestring):
        nums = tf.io.decode_raw(tf_bytestring, tf.uint8)
        label = tf.reshape(nums[self._get_class_byte_offset():
                                self._get_class_byte_offset()+1], [])
        image = tf.cast(nums[self._get_image_byte_offset():],
                    tf.float32) / 256.0
        image = tf.reshape(image,
                    [self.IMG_CHANNELS, self.IMG_SIZE, self.IMG_SIZE])
        image = tf.transpose(image, (1, 2, 0))
        return image, label

    def _get_pipeline_end(self, dataset):
        dataset = dataset.map(self._prepare_sample,
            num_parallel_calls=self.PARALLEL_INPUT_CALLS)
        dataset = dataset.map(self._image_center,
            num_parallel_calls=self.PARALLEL_INPUT_CALLS)
        return dataset.prefetch(-1)

    def _augment(self, dataset):
        if self._augment_training_data:
            dataset = dataset.map(self._distort_image,
                num_parallel_calls=self.PARALLEL_INPUT_CALLS)
        return dataset

    def _prepare_sample(self, image, label):
        return image, tf.one_hot(label, self.get_class_count())
