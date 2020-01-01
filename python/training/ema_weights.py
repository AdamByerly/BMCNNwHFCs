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


class EMAWeights(object):
    def __init__(self, validation_weights_ema_rate, global_step):
        self.ema = None
        self.validation_weights_ema_rate = validation_weights_ema_rate
        self.global_step = global_step
        with tf.device("/device:CPU:0"),\
                tf1.variable_scope("vars/non_ema_weight_copies"):
            self.ema_vars = tf1.get_collection(
                tf1.GraphKeys.TRAINABLE_VARIABLES)
            self.backup_vars = [tf1.get_variable(var.op.name,
                dtype=var.value().dtype, trainable=False,
                initializer=var.initialized_value()) for var in self.ema_vars]

    def get_update_op(self, train_op):
        with tf.name_scope("ema_weights"):
            self.ema = tf.train.ExponentialMovingAverage(
                        decay=self.validation_weights_ema_rate,
                        num_updates=self.global_step,
                        zero_debias=True)
            with tf.control_dependencies([train_op]):
                return self.ema.apply(self.ema_vars)

    def load_training_weights(self, session):
        session.run([tf.group(*(tf1.assign(var, bck.read_value())
            for var, bck in zip(self.ema_vars, self.backup_vars)))])

    def load_validation_weights(self, session):
        backup_weights = tf.group(*(tf1.assign(bck, var.read_value())
            for var, bck in zip(self.ema_vars, self.backup_vars)))
        with tf.control_dependencies([backup_weights]):
            session.run([tf.group(*(tf1.assign(var,
                self.ema.average(var).read_value())
                for var in self.ema_vars if var is not None))])
