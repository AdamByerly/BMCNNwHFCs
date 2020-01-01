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

import tensorflow as tf
tf1 = tf.compat.v1


def get_accuracies(logits, labels):
    with tf.device("/device:GPU:0"), tf.name_scope("metrics"):
        labels = tf.argmax(labels, 1)
        in_top_1 = tf1.nn.in_top_k(logits, labels, 1)
        acc_top_1 = tf.reduce_mean(tf.cast(in_top_1, tf.float32))
    return acc_top_1

