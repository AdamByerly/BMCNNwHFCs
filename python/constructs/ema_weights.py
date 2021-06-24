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


class EMAWeights(Loggable):
    def __init__(self, validation_weights_ema_rate, trainable_variables):
        Loggable.__init__(self)
        self.ema_object                   = None
        self._validation_weights_ema_rate = validation_weights_ema_rate
        self._trainable_variables         = trainable_variables

        self._backup_vars = [tf.Variable(var.read_value(),
            dtype=var.value().dtype, trainable=False)
            for var in self._trainable_variables]

    def get_members(self):
        return {"self._validation_weights_ema_rate":
                str(self._validation_weights_ema_rate)}

    def update(self):
        with tf.name_scope("ema_weights"):
            if self.ema_object is None:
                self.ema_object = tf.train.ExponentialMovingAverage(
                                    decay=self._validation_weights_ema_rate,
                                    zero_debias=True)
            self.ema_object.apply(self._trainable_variables)

    def load_training_weights(self):
        for var, bck in zip(self._trainable_variables, self._backup_vars):
            var.assign(bck.read_value())

    def load_validation_weights(self):
        for var, bck in zip(self._trainable_variables, self._backup_vars):
            bck.assign(var.read_value())
            var.assign(self.ema_object.average(var))
