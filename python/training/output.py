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

import os
from datetime import datetime
import tensorflow as tf
tf1 = tf.compat.v1


class Output:
    def __init__(self, log_dir, run_name, prior_weights_file=None,
            profile_compute_time_every_n_steps=None,
            save_summary_info_every_n_steps=None,
            console_log_interval=10):
        self.run_name             = run_name
        self.latest_weights_file  = prior_weights_file
        self.pctens               = profile_compute_time_every_n_steps
        self.ssiens               = save_summary_info_every_n_steps
        self.log_dir              = log_dir
        model_file_base           = os.path.join(log_dir, run_name)
        self.model_file_base      = os.path.join(model_file_base, "weights")
        self.console_log_interval = console_log_interval

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.tb_writer            = None
        self.tf_saver_best_top1   = None
        self.tf_saver_best_loss   = None
        self.tf_saver_latest      = None
        self.run_options          = None
        self.run_metadata         = None
        self.best_top1_accuracy   = 0
        self.best_loss            = 9999

    def train_step_begin(self, step):
        if self.pctens is not None and step % self.pctens == 0:
            self.run_options = tf.RunOptions(
                report_tensor_allocations_upon_oom=True,
                trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()

    def train_step_end(self, session, epoch, global_step,
            step, loss, lr, number_of_training_steps, feed_dict):
        self.log_metrics(session=session, epoch=epoch, loss=loss, lr=lr,
            step=step, steps_per_epoch=number_of_training_steps, is_test=False)
        self.log_run_metadata(global_step, step)
        self.log_summaries(session, global_step, step, feed_dict)
        self.tb_writer.flush()

    def train_end(self, session, epoch, global_step):
        self.save_model_latest(session, epoch, global_step)

    def validation_step_end(self, step, number_of_validation_steps):
        if step % self.console_log_interval == 0:
            self.log_msg("Validating (step {}/{})...".format(
                step + 1, number_of_validation_steps), True)

    def validation_end(self, session, epoch,
            global_step, test_loss, lr, top1_accuracy):
        self.log_metrics(session=session, epoch=epoch, loss=test_loss, lr=lr,
            top1_accuracy=top1_accuracy, is_test=True)
        self.tb_writer.flush()
        if top1_accuracy >= self.best_top1_accuracy:
            self.best_top1_accuracy = top1_accuracy
            self.save_model_best_top1(session, epoch, global_step)
        if test_loss < self.best_loss:
            self.best_loss = test_loss
            self.save_model_best_loss(session, epoch, global_step)

    def log_metrics(self, session, epoch, loss, lr, step=0,
            steps_per_epoch=0, is_test=False, top1_accuracy=None):
        prefix = "Test" if is_test else "Train"
        summary = tf1.Summary()
        s_loss = summary.value.add()
        s_loss.tag = "{}/Loss".format(prefix)
        s_loss.simple_value = loss
        if not is_test:
            s_lr = summary.value.add()
            s_lr.tag = "{}/LR".format(prefix)
            s_lr.simple_value = lr
        if is_test:
            s_accuracy1 = summary.value.add()
            s_accuracy1.tag = "{}/Top-1 Accuracy".format(prefix)
            s_accuracy1.simple_value = top1_accuracy

        step_number = (epoch - 1) * steps_per_epoch + step

        if step_number % self.console_log_interval == 0:
            if is_test:
                self.tb_writer.add_summary(summary, epoch)
                self.log_msg("[TEST] - Epoch {}".format(epoch))
            else:
                self.tb_writer.add_summary(summary, step_number)
                self.log_msg("[TRAIN] - Epoch {}, Step {}"
                    .format(epoch, step+1))
            self.log_msg("loss: {}".format(loss), False)

        if top1_accuracy is not None:
            self.log_msg("top1: {}".format(top1_accuracy), False)

        if not is_test:
            with tf1.variable_scope("vars/branch_weights", reuse=True):
                branch_weights = tf1.get_variable("branch_weights")
                weights = session.run([branch_weights])
                summary = tf1.Summary()
                s_loss = summary.value.add()
                s_loss.tag = "{}/branch_1_weight".format(prefix)
                s_loss.simple_value = weights[0][0]
                self.tb_writer.add_summary(summary, step_number)
                summary = tf1.Summary()
                s_loss = summary.value.add()
                s_loss.tag = "{}/branch_2_weight".format(prefix)
                s_loss.simple_value = weights[0][1]
                self.tb_writer.add_summary(summary, step_number)
                summary = tf1.Summary()
                s_loss = summary.value.add()
                s_loss.tag = "{}/branch_3_weight".format(prefix)
                s_loss.simple_value = weights[0][2]
                self.tb_writer.add_summary(summary, step_number)

    def log_run_metadata(self, global_step, step):
        if self.run_metadata is not None and step % self.pctens == 0:
            self.tb_writer.add_run_metadata(
                self.run_metadata, "step{}".format(global_step))

    def log_summaries(self, session, global_step, step, feed_dict):
        if self.ssiens is not None and step % self.ssiens == 0:
            summary_op = tf1.summary.merge_all()
            if summary_op is not None:
                summary_str = session.run(summary_op, feed_dict=feed_dict)
                self.tb_writer.add_summary(summary_str, global_step)

    def save_model_best_top1(self, session, epoch, g_step):
        self.tf_saver_best_top1.save(session, "{}-{}-best_top1"
            .format(self.model_file_base, epoch), global_step=g_step)

    def save_model_best_loss(self, session, epoch, g_step):
        self.tf_saver_best_loss.save(session, "{}-{}-best_loss"
            .format(self.model_file_base, epoch), global_step=g_step)

    def save_model_latest(self, session, epoch, g_step):
        self.latest_weights_file = self.tf_saver_latest.save(
            session, "{}-{}-latest".format(self.model_file_base, epoch),
            global_step=g_step)

    def set_session_graph(self, session_graph):
        self.tb_writer = tf1.summary.FileWriter(
            os.path.join(self.log_dir, self.run_name), session_graph)
        self.tf_saver_best_top1 = tf1.train.Saver(max_to_keep=2)
        self.tf_saver_best_loss = tf1.train.Saver(max_to_keep=2)
        self.tf_saver_latest    = tf1.train.Saver(max_to_keep=2)

    def close_files(self):
        self.tb_writer.close()

    def get_run_options(self):
        return self.run_options

    def get_run_metadata(self):
        return self.run_metadata

    @staticmethod
    def log_msg(msg, put_time=True):
        t_str = "                     "
        if put_time:
            t_str = datetime.now().strftime("%Y%m%d %H:%M:%S.%f")[:-3]
        print("{} - {}".format(t_str, msg))
