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

from python.constructs.loggable import Loggable
import tensorflow as tf


class MarginPlusMeanSquaredError(Loggable):
    def __init__(self):
        Loggable.__init__(self)

    def __call__(self, features, labels, model_output, batch_size):
        margin = 0.4
        downweight = 0.5
        logits = model_output[0] - 0.5
        positive_cost = labels * tf.cast(tf.less(logits, margin),
            tf.float32) * tf.pow(logits - margin, 2)
        negative_cost = (1 - labels) * tf.cast(tf.greater(logits, -margin),
            tf.float32) * tf.pow(logits + margin, 2)
        margin_loss = 0.5 * positive_cost + downweight * 0.5 * negative_cost
        recon_loss = tf.square(tf.subtract(features, model_output[1]))
        return (tf.reduce_sum(margin_loss)
            + tf.reduce_sum(recon_loss*0.0005)) / batch_size


class CategoricalCrossEntropyPlusMeanSquaredError(Loggable):
    def __init__(self, from_logits=True):
        Loggable.__init__(self)
        self._from_logits = from_logits

    def __call__(self, features, labels, model_output, batch_size):
        label_loss = tf.keras.losses.categorical_crossentropy(
            labels, model_output[0], from_logits=self._from_logits)
        recon_loss = tf.square(tf.subtract(features, model_output[1]))
        return (tf.reduce_sum(label_loss)
            + tf.reduce_sum(recon_loss*0.0005)) / batch_size


class CategoricalCrossEntropy(Loggable):
    def __init__(self, from_logits=True):
        Loggable.__init__(self)
        self._from_logits = from_logits

    def __call__(self, features, labels, model_output, batch_size):
        loss = tf.keras.losses.categorical_crossentropy(
            labels, model_output[0], from_logits=self._from_logits)
        return tf.reduce_sum(loss) / batch_size


class MeanSquaredError(Loggable):
    def __init__(self, from_logits=True):
        Loggable.__init__(self)
        self._from_logits = from_logits

    def __call__(self, features, labels, model_output, batch_size):
        loss = tf.square(tf.subtract(features, model_output[1]))
        return tf.reduce_sum(loss) / batch_size
