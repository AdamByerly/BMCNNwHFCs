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


class Variable(object):
    def __init__(self, initial_value, name):
        with tf.device("/device:CPU:0"):
            self.variable = tf.Variable(initial_value,
                aggregation=tf.VariableAggregation.MEAN, name=name)

    def get_trainable_variables(self):
        return [self.variable]

    def get_savable_variables(self):
        return [self.variable]

    def __str__(self):
        return "Variable({})".format(self.variable)


class VariableWBias(object):
    def __init__(self, initial_value, name, bias_shape_idx=-1):
        with tf.device("/device:CPU:0"):
            self.variable = tf.Variable(initial_value,
                aggregation=tf.VariableAggregation.MEAN, name=name)
            self.bias = tf.Variable(tf.zeros(
                shape=[initial_value.shape[bias_shape_idx], ]))

    def get_trainable_variables(self):
        return [self.variable, self.bias]

    def get_savable_variables(self):
        return [self.variable, self.bias]

    def __str__(self):
        return "VariableWBias({};{})".format(self.variable, self.bias)


class VariableWBatchNorm(object):
    def __init__(self, initial_value, name, trainable=True, bn_shape_idx=-1):
        with tf.device("/device:CPU:0"):
            self.variable = tf.Variable(initial_value,
                aggregation=tf.VariableAggregation.MEAN, name=name,
                trainable=trainable)
            self.bn_beta = tf.Variable(tf.zeros(
                shape=[initial_value.shape[bn_shape_idx], ]),
                aggregation=tf.VariableAggregation.MEAN,
                name=name+"_bn_beta")
            self.bn_mean = tf.Variable(tf.zeros(
                shape=[initial_value.shape[bn_shape_idx], ]),
                aggregation=tf.VariableAggregation.MEAN,
                name=name+"_bn_mean", trainable=False)
            self.bn_variance = tf.Variable(tf.ones(
                shape=[initial_value.shape[bn_shape_idx], ]),
                aggregation=tf.VariableAggregation.MEAN,
                name=name+"_bn_variance", trainable=False)

    def get_trainable_variables(self):
        return [self.variable, self.bn_beta]

    def get_savable_variables(self):
        return [self.variable, self.bn_beta, self.bn_mean, self.bn_variance]

    def __str__(self):
        return "VariableWBatchNorm({};{};{};{})".format(
            self.variable, self.bn_beta, self.bn_mean, self.bn_variance)


def get_variable(shape, name):
    return Variable(tf.random_normal_initializer()(shape=shape), name=name)


def get_bias_variable(shape, name, bias_shape_idx=-1):
    return VariableWBias(tf.random_normal_initializer()(shape=shape),
        name=name, bias_shape_idx=bias_shape_idx)


def get_batch_norm_variable(
        shape, name, initializer=tf.random_normal_initializer(),
        trainable=True, bn_shape_idx=-1):
    return VariableWBatchNorm(initializer(shape=shape),
        name=name, trainable=trainable, bn_shape_idx=bn_shape_idx)


def get_conv_batch_norm_variable(name,
        input_size, filter_size_h, filter_size_w, filters):
    return get_batch_norm_variable(
        [filter_size_h, filter_size_w, input_size, filters], name)


def get_conv_1x1_batch_norm_variable(name, input_size, filters):
    return get_conv_batch_norm_variable(name, input_size, 1, 1, filters)


def get_conv_3x3_batch_norm_variable(name, input_size, filters):
    return get_conv_batch_norm_variable(name, input_size, 3, 3, filters)


def get_conv_transpose_batch_norm_variable(name,
        input_size, filter_size_h, filter_size_w, filters):
    return get_batch_norm_variable(
        [filter_size_h, filter_size_w, filters, input_size], name,
        bn_shape_idx=-2)


def get_conv_transpose_3x3_batch_norm_variable(name, input_size, filters):
    return get_conv_transpose_batch_norm_variable(name, input_size, 3, 3,
        filters)


def get_conv_transpose_4x4_batch_norm_variable(name, input_size, filters):
    return get_conv_transpose_batch_norm_variable(name, input_size, 4, 4,
        filters)


def get_hvc_from_xyz_batch_norm_variable(name,
        input_size, output_size, capsule_dimensions):
    return get_batch_norm_variable(
        [output_size, input_size, capsule_dimensions], name, bn_shape_idx=-1)


def get_hvc_from_zxy_batch_norm_variable(name,
        input_size, output_size, capsule_dimensions):
    return get_batch_norm_variable(
        [output_size, input_size, capsule_dimensions], name, bn_shape_idx=-1)
