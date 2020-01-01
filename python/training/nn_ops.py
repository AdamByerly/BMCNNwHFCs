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

from python.training.batch_norm_cond_in_graph import batch_norm
import tensorflow as tf
tf1 = tf.compat.v1


def l2_regularizer(weight):
    def regularizer(t):
        return weight * tf.nn.l2_loss(t)
    return regularizer


def conv_no_bias(op_name, in_tensor, filter_size_h, filter_size_w, filters,
        strides=(1, 1, 1, 1), padding="VALID",
        weight_decay=0.0005, stddev=0.1):
    with tf.device("/device:CPU:0"), \
         tf1.variable_scope("vars/convs", reuse=tf1.AUTO_REUSE):
        input_size = in_tensor.get_shape().as_list()[3]
        shape = (filter_size_h, filter_size_w, input_size, filters)
        w = tf1.get_variable("W_" + op_name, shape=shape,
            regularizer=l2_regularizer(weight_decay),
            initializer=tf1.truncated_normal_initializer(stddev=stddev))
    with tf.name_scope(op_name):
        return tf.nn.conv2d(in_tensor, w,
            strides=strides, padding=padding, name=op_name)


def make_conv_3x3(op_name, is_training,
        in_tensor, filters, weight_decay=0.0005, use_bn=True):
    with tf.name_scope(op_name):
        conv = conv_no_bias("conv_" + op_name, in_tensor, 3, 3, filters,
            weight_decay=weight_decay)
        if use_bn:
            conv = batch_norm("bn_" + op_name, conv, is_training)
        return tf.nn.relu(conv, "relu_" + op_name)


def caps_from_conv(op_name, in_tensor, cap_dims):
    with tf.name_scope(op_name):
        shape = in_tensor.get_shape().as_list()
        cap_count = shape[1] * shape[2] * shape[3] // cap_dims
        return tf.reshape(in_tensor,
            [-1, 1, cap_count, cap_dims], name="caps_shape_" + op_name)


def caps_from_conv2(op_name, in_tensor, cap_dims):
    with tf.name_scope(op_name):
        shape = in_tensor.get_shape().as_list()
        cap_count = shape[1] * shape[2] * shape[3] // cap_dims
        transposed = tf.transpose(in_tensor, [0, 3, 1, 2],
            name="transpose_op_name")
        return tf.reshape(transposed,
            [-1, 1, cap_count, cap_dims], name="caps_shape_" + op_name)


def make_hvc(op_name, is_training, in_tensor,
        out_caps, cap_dims, weight_decay=0.0005, use_bn=True):
    with tf.device("/device:CPU:0"), \
         tf1.variable_scope("vars/caps", reuse=tf1.AUTO_REUSE):
        in_caps_sz = in_tensor.get_shape().as_list()[2]
        w_out_cap = tf1.get_variable("w_" + op_name,
            shape=[out_caps, in_caps_sz, cap_dims],
            regularizer=l2_regularizer(weight_decay),
            initializer=tf1.glorot_uniform_initializer())
    with tf.name_scope(op_name):
        ocap = tf.reduce_sum(tf.multiply(in_tensor, w_out_cap), 3)
        if use_bn:
            ocap = batch_norm("bn_" + op_name, ocap, is_training)
        return tf.nn.relu(ocap, "relu_" + op_name)


def make_hvc2(op_name, is_training, in_tensor,
        out_caps, cap_dims, weight_decay=0.0005, use_bn=True):
    with tf.device("/device:CPU:0"), \
         tf1.variable_scope("vars/caps", reuse=tf1.AUTO_REUSE):
        in_caps_sz = in_tensor.get_shape().as_list()[2]
        w_out_cap = tf1.get_variable("w_" + op_name,
            shape=[out_caps, in_caps_sz, cap_dims],
            regularizer=l2_regularizer(weight_decay),
            initializer=tf1.glorot_uniform_initializer())
    with tf.name_scope(op_name):
        ocap = tf.reduce_sum(tf.multiply(in_tensor, w_out_cap), 2)
        if use_bn:
            ocap = batch_norm("bn_" + op_name, ocap, is_training)
        return tf.nn.relu(ocap, "relu_" + op_name)
