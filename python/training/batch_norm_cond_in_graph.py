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
from tensorflow.python.training import moving_averages
tf1 = tf.compat.v1


def batch_norm(op_name, inputs, is_training,
        decay=0.9997, epsilon=0.001, variable_getter=None):
    moving_collections = [tf1.GraphKeys.GLOBAL_VARIABLES,
                          tf1.GraphKeys.MOVING_AVERAGE_VARIABLES]
    inputs_shape       = inputs.get_shape()
    params_shape       = inputs_shape[-1:]

    with tf.device("/device:CPU:0"),\
            tf1.variable_scope("vars/bns", None, [inputs],
            reuse=tf1.AUTO_REUSE):
        beta            = tf1.get_variable(
                            "beta_"+op_name, shape=params_shape,
                            initializer=tf.zeros_initializer(),
                            custom_getter=variable_getter)
        moving_mean     = tf1.get_variable(
                            "moving_mean_"+op_name, params_shape,
                            initializer=tf.zeros_initializer(),
                            trainable=False, collections=moving_collections)
        moving_variance = tf1.get_variable(
                            "moving_variance_"+op_name, params_shape,
                            initializer=tf.ones_initializer(),
                            trainable=False, collections=moving_collections)

    def training_func():
        # Calculate the moments based on the individual batch.
        axis                = list(range(len(inputs_shape) - 1))
        mean, variance      = tf.nn.moments(inputs, axis)
        upd_moving_mean     = moving_averages.assign_moving_average(
                                    moving_mean, mean, decay)
        upd_moving_variance = moving_averages.assign_moving_average(
                                    moving_variance, variance, decay)
        tf1.add_to_collection(tf1.GraphKeys.UPDATE_OPS, upd_moving_mean)
        tf1.add_to_collection(tf1.GraphKeys.UPDATE_OPS, upd_moving_variance)
        return mean, variance

    def inferring_func():
        # Just use the moving_mean and moving_variance.
        return moving_mean, moving_variance

    with tf.name_scope(op_name):
        mean_to_use, variance_to_use = tf.cond(is_training,
                                        lambda: training_func(),
                                        lambda: inferring_func())

        # Normalize the activations.
        outputs = tf.nn.batch_normalization(inputs, mean_to_use,
                    variance_to_use, beta, None, epsilon, name=op_name)
        outputs.set_shape(inputs.get_shape())
        return outputs
