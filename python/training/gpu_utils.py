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


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            if g is not None:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def optimize(optimizer, global_step, grads):
    with tf.control_dependencies(tf1.get_collection(tf1.GraphKeys.UPDATE_OPS)):
        with tf.device("/device:CPU:0"), tf.name_scope("apply_grads"):
            return optimizer.apply_gradients(grads, global_step)


def make_towers(optimizer, features, labels, num_gpus, tower_fn, **kwargs):
    with tf.name_scope("batch_split"):
        feature_splits = tf.split(features, num_or_size_splits=num_gpus)
        label_splits   = tf.split(labels, num_or_size_splits=num_gpus)

    labels_list, preds_list, logits_list, loss_list, grads =\
        ([], [], [], [], [])
    for device_idx in range(num_gpus):
        with tf.device("/device:GPU:%d" % device_idx),\
             tf.name_scope("tower%d" % device_idx):
            preds, logits, loss = tower_fn(feature_splits[device_idx],
                                    label_splits[device_idx], **kwargs)
            labels_list.append(label_splits[device_idx])
            preds_list .append(preds)
            logits_list.append(logits)
            loss_list  .append(loss)
            if optimizer is not None:
                grads.append(optimizer.compute_gradients(loss))
    with tf.device("/device:CPU:0"), tf.name_scope("merge_towers"):
        labels = tf.concat(labels_list, 0)
        preds  = tf.concat(preds_list, 0)
        logits = tf.concat(logits_list, 0)
        loss   = tf.reduce_mean(loss_list, 0, name="total_loss")
        if optimizer is not None:
            grads = average_gradients(grads)
    return preds, logits, loss, labels, grads
