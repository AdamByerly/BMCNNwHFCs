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

import tensorflow as tf


def reshape(op_name, in_tensor, new_shape):
    with tf.name_scope(op_name):
        return tf.reshape(in_tensor, new_shape, name=op_name)


def flatten(op_name, in_tensor):
    with tf.name_scope(op_name):
        input_size = in_tensor.get_shape().as_list()
        size = input_size[1]*input_size[2]*input_size[3]
        return tf.reshape(in_tensor, [-1, size], name=op_name)


def fc(op_name, in_tensor, variable, activation=tf.nn.relu):
    with tf.name_scope(op_name):
        x = tf.matmul(in_tensor, variable.variable, name="matmul_"+op_name)
        if activation is None:
            return x
        return activation(x, name="activation_" + op_name)


def fc_w_batch_norm(op_name,
        is_training, in_tensor, variable, activation=tf.nn.relu):
    with tf.name_scope(op_name):
        x = tf.matmul(in_tensor, variable.variable, name="matmul_"+op_name)
        x = batch_norm("bn_" + op_name, is_training, x, variable)
        if activation is None:
            return x
        return activation(x, name="activation_" + op_name)


def fc_w_bias(op_name, in_tensor, variable, activation=tf.nn.relu):
    with tf.name_scope(op_name):
        x = tf.matmul(in_tensor, variable.variable, name="matmul_"+op_name)
        x = tf.add(x, variable.bias, name="add_" + op_name)
        if activation is None:
            return x
        return activation(x, name="activation_" + op_name)


def avg_pool(op_name, in_tensor,
        padding="VALID", window_shape=None, strides=None):
    return pool(op_name, in_tensor, "AVG", padding, window_shape, strides)


def max_pool(op_name, in_tensor,
        padding="VALID", window_shape=None, strides=None):
    return pool(op_name, in_tensor, "MAX", padding, window_shape, strides)


def pool(op_name, in_tensor, pooling_type,
        padding="VALID", window_shape=None, strides=None):
    if strides is None:
        strides = [2, 2]
    if window_shape is None:
        window_shape = [2, 2]
    with tf.name_scope(op_name):
        return tf.nn.pool(input=in_tensor, window_shape=window_shape,
            strides=strides, pooling_type=pooling_type, padding=padding,
            name=op_name)


def conv(op_name, is_training, in_tensor, variable,
        strides=None, padding="VALID", activation_fn=tf.nn.relu):
    if strides is None:
        strides = [1, 1, 1, 1]
    with tf.name_scope(op_name):
        x = tf.nn.conv2d(in_tensor, variable.variable,
            strides=strides, padding=padding, name=op_name)
        x = batch_norm("bn_" + op_name, is_training, x, variable)
        return activation_fn(x, name="activate_" + op_name)


def conv_transpose(op_name, is_training, in_tensor, variable,
        output_shape, strides=None, padding="SAME", activation_fn=tf.nn.relu):
    if strides is None:
        strides = [1, 1, 1, 1]
    with tf.name_scope(op_name):
        x = tf.nn.conv2d_transpose(in_tensor, variable.variable,
            output_shape, strides=strides, padding=padding, name=op_name)
        x = batch_norm("bn_" + op_name, is_training, x, variable)
        return activation_fn(x, name="activate_" + op_name)


def caps_from_conv_zxy(op_name, in_tensor, cap_count, cap_dims):
    with tf.name_scope(op_name):
        x = tf.transpose(in_tensor, [0, 3, 1, 2], name="transpose_" + op_name)
        return tf.reshape(x, [-1, 1, cap_count, cap_dims],
            name="caps_shape_" + op_name)


def caps_from_conv_xyz(op_name, in_tensor, cap_count, cap_dims):
    with tf.name_scope(op_name):
        return tf.reshape(in_tensor, [-1, 1, cap_count, cap_dims],
            name="caps_shape_" + op_name)


def hvc_from_xyz(op_name, is_training, in_tensor, variable):
    with tf.name_scope(op_name):
        x = tf.reduce_sum(tf.multiply(in_tensor, variable.variable), 2)
        x = batch_norm("bn_" + op_name, is_training, x, variable)
        return tf.nn.relu(x, "relu_" + op_name)


def hvc_from_zxy(op_name, is_training, in_tensor, variable):
    with tf.name_scope(op_name):
        x = tf.reduce_sum(tf.multiply(in_tensor, variable.variable), 2)
        x = batch_norm("bn_" + op_name, is_training, x, variable)
        return tf.nn.relu(x, "relu_" + op_name)


def batch_norm(op_name, is_training,
        in_tensor, variable, decay=0.9997, epsilon=0.001):
    with tf.name_scope(op_name):
        if is_training:
            axis           = list(range(len(in_tensor.shape)-1))
            mean, variance = tf.nn.moments(in_tensor, axis)
            variable.bn_mean.assign(
                (variable.bn_mean*decay)+(mean*(1-decay)))
            variable.bn_variance.assign(
                (variable.bn_variance*decay)+(variance*(1-decay)))
        else:
            mean     = variable.bn_mean
            variance = variable.bn_variance

        with tf.name_scope(op_name):
            return tf.nn.batch_normalization(in_tensor, mean,
                variance, variable.bn_beta, None, epsilon, name=op_name)
