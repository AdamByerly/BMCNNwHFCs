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
from python.input.cifar_input_pipeline import CifarInputPipeline
import tensorflow as tf


class Cifar10(CifarInputPipeline):
    CLASSES = 10

    def __init__(self, data_base_dir,
            augment_training_data=True, augmentation_type=1):
        CifarInputPipeline.__init__(self, self.CLASSES,
            augment_training_data, augmentation_type)

        self._data_base_dir = data_base_dir

    def _get_training_files(self):
        return tf.data.Dataset.from_tensor_slices(
            [os.path.join(self._data_base_dir, "data_batch_1.bin"),
             os.path.join(self._data_base_dir, "data_batch_2.bin"),
             os.path.join(self._data_base_dir, "data_batch_3.bin"),
             os.path.join(self._data_base_dir, "data_batch_4.bin"),
             os.path.join(self._data_base_dir, "data_batch_5.bin")])

    def _get_validation_files(self):
        return os.path.join(self._data_base_dir, "test_batch.bin")

    def _get_class_byte_offset(self):
        return 0

    def _get_image_byte_offset(self):
        return 1
