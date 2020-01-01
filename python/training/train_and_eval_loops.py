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


def train(out, sess, epoch, training_steps, train_op, loss_op,
          global_step, learning_rate, is_training_ph,
          training_data, data_iterator, after_train_step=None):
    train_step, g_step = (0, 0)
    sess.run(data_iterator.make_initializer(training_data))
    try:
        while True:
            out.train_step_begin(train_step)

            _, l, g_step, lr = sess.run(
                [train_op, loss_op, global_step, learning_rate],
                feed_dict={is_training_ph: True},
                options=out.get_run_options(),
                run_metadata=out.get_run_metadata())

            out.train_step_end(
                sess, epoch, g_step, train_step, l, lr, training_steps,
                feed_dict={is_training_ph: True})
            train_step += 1
    except tf.errors.OutOfRangeError:
        pass
    out.train_end(sess, epoch, g_step)


def validate(out, sess, epoch, validation_steps, loss_op,
             acc_top_1_op, global_step, learning_rate,
             is_training_ph, validation_data, data_iterator):
    val_step, g_step, acc_top1, test_loss, lr = (0, 0, 0, 0, 0)
    sess.run(data_iterator.make_initializer(validation_data))
    try:
        while True:
            g_step, l, acc1, lr = sess.run(
                [global_step, loss_op, acc_top_1_op,
                 learning_rate], feed_dict={is_training_ph: False})
            acc_top1 = (acc1 + (val_step * acc_top1)) / (val_step + 1)
            test_loss = (l + (val_step * test_loss)) / (val_step + 1)

            out.validation_step_end(val_step, validation_steps)
            val_step += 1
    except tf.errors.OutOfRangeError:
        pass
    out.validation_end(sess, epoch, g_step, test_loss, lr, acc_top1)
