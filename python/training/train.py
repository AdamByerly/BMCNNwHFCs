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

import argparse
from datetime import datetime
from python.training.output import Output
from python.training.metrics import get_accuracies
from python.training.ema_weights import EMAWeights
from python.training.input_pipeline import InputPipeline
from python.training.gpu_utils import make_towers, optimize
from python.training.batch_norm_cond_in_graph import batch_norm
from python.training.nn_ops import make_conv_3x3, caps_from_conv2, make_hvc2
from python.training.train_and_eval_loops import train, validate
import tensorflow as tf
tf1 = tf.compat.v1


def make_tower(features, labels, is_training, count_classes, merge_strategy=0):
    t = is_training
    c = count_classes
    images = tf.reshape(features, [-1, 28, 28, 1])

    conv1   = make_conv_3x3("conv_1", t, images, 32, weight_decay=0.)
    conv2   = make_conv_3x3("conv_2", t, conv1, 48, weight_decay=0.)
    conv3   = make_conv_3x3("conv_3", t, conv2, 64, weight_decay=0.)
    pcap3   = caps_from_conv2("pcap3", conv3, 484)
    ocap3   = make_hvc2("ocap3", t, pcap3, c, 484, weight_decay=0.)
    logits3 = tf.reduce_sum(ocap3, axis=2, name="logits3")

    conv4   = make_conv_3x3("conv_4", t, conv3, 80, weight_decay=0.)
    conv5   = make_conv_3x3("conv_5", t, conv4, 96, weight_decay=0.)
    conv6   = make_conv_3x3("conv_6", t, conv5, 112, weight_decay=0.)
    pcap6   = caps_from_conv2("pcap6", conv6, 256)
    ocap6   = make_hvc2("ocap6", t, pcap6, c, 256, weight_decay=0.)
    logits6 = tf.reduce_sum(ocap6, axis=2, name="logits6")

    conv7   = make_conv_3x3("conv_7", t, conv6, 128, weight_decay=0.)
    conv8   = make_conv_3x3("conv_8", t, conv7, 144, weight_decay=0.)
    conv9   = make_conv_3x3("conv_9", t, conv8, 160, weight_decay=0.)
    pcap9   = caps_from_conv2("pcap9", conv9, 100)
    ocap9   = make_hvc2("ocap9", t, pcap9, c, 100, weight_decay=0.)
    logits9 = tf.reduce_sum(ocap9, axis=2, name="logits9")

    with tf.name_scope("logits"):
        logits = tf.stack([logits3, logits6, logits9], axis=2)
        initer = tf1.ones_initializer()
        trainable = False
        if merge_strategy == 1:
            initer = tf1.glorot_uniform_initializer()
            trainable = True
        elif merge_strategy == 2:
            trainable = True
        with tf.device("/device:CPU:0"), \
             tf1.variable_scope("vars/branch_weights", reuse=tf1.AUTO_REUSE):
            branch_weights = tf1.get_variable("branch_weights",
                shape=[3], initializer=initer, trainable=trainable)
            logits = tf.multiply(logits, branch_weights)
        logits = batch_norm("bn_logits", logits, is_training)
        logits = tf.reduce_sum(logits, axis=2, name="logits")

    with tf.name_scope("loss"):
        preds = tf.nn.softmax(logits=logits)
        tf.stop_gradient(labels)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))\
               + tf1.losses.get_regularization_loss()

    return preds, logits, loss


def go(start_epoch, end_epoch, run_name, weights_file,
        profile_compute_time_steps, save_summary_info_steps,
        batch_size, gpus, log_dir, data_dir, ema_decay_rate,
        merge_strategy):
    tf1.disable_eager_execution()
    tf1.disable_control_flow_v2()
    tf1.logging.set_verbosity(tf1.logging.ERROR)
    tf1.reset_default_graph()

    out = Output(log_dir, run_name, weights_file,
        profile_compute_time_steps, save_summary_info_steps)

    ############################################################################
    # Data feeds
    ############################################################################
    out.log_msg("Setting up data feeds...")
    input_pipe  = InputPipeline(data_dir)
    classes     = input_pipe.get_class_count()
    train_steps = input_pipe.get_train_image_count()//batch_size
    eval_steps  = input_pipe.get_eval_image_count()//batch_size
    train_data  = input_pipe.get_training_dataset(batch_size)
    eval_data   = input_pipe.get_validation_dataset(batch_size)

    ############################################################################
    # Tensorflow Graph
    ############################################################################
    out.log_msg("Building Graph Operations...")
    with tf.device("/device:CPU:0"):  # set the default device to the CPU
        global_step = tf1.train.get_or_create_global_step()

        with tf.name_scope("input/placeholders"):
            is_training = tf1.placeholder(dtype=bool, name="is_training")

        with tf.name_scope("input/data_itetator"):
            data_iterator = tf1.data.Iterator.from_structure(
                tf1.data.get_output_types(train_data),
                tf1.data.get_output_shapes(eval_data))
            features, labels = data_iterator.get_next()

        with tf.name_scope("learning_rate"):
            decay_steps = int(train_steps * 1.0)
            learning_rate = tf1.train.exponential_decay(0.001,
                global_step, decay_steps, 0.98, staircase=True)
            learning_rate = tf.maximum(learning_rate, 1e-6)
        optimizer = tf1.train.AdamOptimizer(learning_rate)

        _, logits, loss, labels, grads = make_towers(
            optimizer, features, labels, gpus, make_tower,
            **{"is_training": is_training, "count_classes": classes,
               "merge_strategy": merge_strategy})

        acc_top_1 = get_accuracies(logits, labels)
        train_op = optimize(optimizer, global_step, grads)
        ema_weights = EMAWeights(ema_decay_rate, global_step)
        ema_weights_update_op = ema_weights.get_update_op(train_op)

    ############################################################################
    # Tensorflow session
    ############################################################################
    out.log_msg("Starting Session...")
    with tf1.Session(config=tf1.ConfigProto(allow_soft_placement=True)) as sess:
        out.set_session_graph(sess.graph)

        if weights_file is not None:
            out.log_msg("Restoring weights file: {}".format(weights_file))
            tf1.train.Saver().restore(sess, weights_file)
        else:
            tf1.global_variables_initializer().run()

        for e in range(start_epoch, end_epoch + 1):
            train(out, sess, e, train_steps, ema_weights_update_op, loss,
                global_step, learning_rate, is_training, train_data,
                data_iterator)

            out.log_msg("Loading EMA of past weights...")
            ema_weights.load_validation_weights(sess)

            validate(out, sess, e, eval_steps, loss, acc_top_1,
                global_step, learning_rate, is_training, eval_data,
                data_iterator)

            out.log_msg("Loading current weights...")
            ema_weights.load_training_weights(sess)

        out.log_msg("Finished.")
        sess.close()

    out.close_files()


################################################################################
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--merge_strategy", default=2, type=float)
    p.add_argument("--ema_decay_rate", default=0.999, type=float)
    p.add_argument("--start_epoch", default=1, type=int)
    p.add_argument("--end_epoch", default=300, type=int)
    p.add_argument("--run_name", default=None)
    p.add_argument("--weights_file", default=None)
    p.add_argument("--profile_compute_time_steps", default=None, type=int)
    p.add_argument("--save_summary_info_steps", default=None, type=int)
    p.add_argument("--batch_size", default=120, type=int)
    p.add_argument("--gpus", default=1, type=int)
    p.add_argument("--trials", default=32, type=int)
    p.add_argument("--log_dir",
        default="../../data/weights/learnable_ones_init")
    p.add_argument("--data_dir", default=r"../../data/mnist_data")
    a = p.parse_args()

    for i in range(a.trials):
        rn = datetime.now().strftime("%Y%m%d%H%M%S")\
            if a.run_name is None else a.run_name + "_" + str(i)
        go(start_epoch=a.start_epoch, end_epoch=a.end_epoch,
            run_name=rn, weights_file=a.weights_file,
            profile_compute_time_steps=a.profile_compute_time_steps,
            save_summary_info_steps=a.save_summary_info_steps,
            batch_size=a.batch_size, gpus=a.gpus, log_dir=a.log_dir,
            data_dir=a.data_dir, ema_decay_rate=a.ema_decay_rate,
            merge_strategy=a.merge_strategy)
