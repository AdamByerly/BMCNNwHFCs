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

import os
from datetime import datetime
from python.constructs.loggable import Loggable
import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2


class Output(Loggable):
    def __init__(self, log_dir, run_name, profile_batch_start=None,
            profile_batch_end=None, console_log_interval=1, include_top1=True,
            include_top5=False):
        Loggable.__init__(self)
        self._log_dir                     = os.path.join(log_dir, run_name)
        self._run_name                    = run_name
        self._pbs                         = profile_batch_start
        self._pbe                         = profile_batch_start \
                                                if profile_batch_end is None \
                                                else profile_batch_end
        self._console_log_interval        = console_log_interval
        self._summary_writer              = tf.summary.create_file_writer(
                                                self._log_dir)
        self._profile_started             = False
        self._profile_finished            = False
        self._ckpt_latest                 = None
        self._ckpt_latest_mgr             = None
        self._ckpt_best_top1              = None
        self._ckpt_best_top1_mgr          = None
        self._best_top1_accuracy          = 0
        self._include_top1                = include_top1
        self._include_top5                = include_top5
        self._ckpt_best_top5              = None
        self._ckpt_best_top5_mgr          = None
        self._best_top5_accuracy          = 0
        self._trained_4_full_epoch_once   = False
        self._max_train_step              = 0
        self._validated_4_full_epoch_once = False
        self._max_validation_step         = 0

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)

    def get_log_dir(self):
        return self._log_dir

    def get_summary_writer(self):
        return self._summary_writer

    def train_step_begin(self, step):
        if self._pbs is not None and step + 1 >= self._pbs \
                and self._profile_started is not True \
                and self._profile_finished is not True:
            tf.profiler.experimental.start(self._log_dir)
            self.print_msg("Profiling started...")
            self._profile_started = True

    def train_step_end(self, epoch, step, loss, lr, total_steps):
        self._log_train_metrics(epoch, step, total_steps, loss, lr)

        if self._profile_started and step + 1 >= self._pbe \
                and not self._profile_finished:
            tf.profiler.experimental.stop()
            self.print_msg("Profiling finished...")
            self._profile_finished = True

    # noinspection PyUnusedLocal
    def train_end(self, epoch):
        self._trained_4_full_epoch_once = True

    def validation_step_end(self, step):
        if step > self._max_validation_step:
            self._max_validation_step = step

        if step % self._console_log_interval == 0:
            total_steps = self._max_validation_step \
                if self._validated_4_full_epoch_once is True else "?"
            self.print_msg("Validating (step {}/{})..."
                .format(step, total_steps), True)

    # noinspection PyUnusedLocal
    def validation_end(self, model, optimizer, ema_weights,
            epoch, loss, top1_accuracy=None, top5_accuracy=None):
        self._validated_4_full_epoch_once = True
        self._log_test_metrics(epoch, loss, top1_accuracy, top5_accuracy)

        # TODO: Add ema_weights.ema_object to checkpoints.
        #  Currently, tf doesn't seem to support this.
        #  See https://github.com/tensorflow/tensorflow/issues/38452
        #  Until such time as that is addressed or a workaround is devised,
        #   it's good to know that the variables that art loaded at the time the
        #   checkpoint is saved are the ones that are saved.
        #  So, if the last interaction with the ema_weights object was a call
        #   to load_validation_weights(), then the averaged weights are the
        #   ones that are saved.  Likewise, if the last interaction with the
        #   ema_weights object was a call to load_training_weights(), then it
        #   is the unaveraged weights that are saved.
        #  Since all checkpoints are currently being saved in this method, all
        #   saved weights are the averaged weights.

        if self._ckpt_latest is None:
            self._ckpt_latest = tf.train.Checkpoint(
                vars=model.get_all_savable_variables(), optimizer=optimizer)
            self._ckpt_latest_mgr = tf.train.CheckpointManager(
                self._ckpt_latest, self._log_dir,
                max_to_keep=2, checkpoint_name='latest')

        self._ckpt_latest_mgr.save(epoch)
        if self._include_top1:
            if self._ckpt_best_top1 is None:
                self._ckpt_best_top1 = tf.train.Checkpoint(
                    vars=model.get_all_savable_variables(), optimizer=optimizer)
                self._ckpt_best_top1_mgr = tf.train.CheckpointManager(
                    self._ckpt_best_top1, self._log_dir,
                    max_to_keep=2, checkpoint_name='best_top1')

            if top1_accuracy >= self._best_top1_accuracy:
                self._best_top1_accuracy = top1_accuracy
                self._ckpt_best_top1_mgr.save(epoch)

        if self._include_top5:
            if self._ckpt_best_top5 is None:
                self._ckpt_best_top5 = tf.train.Checkpoint(
                    vars=model.get_all_savable_variables(), optimizer=optimizer)
                self._ckpt_best_top5_mgr = tf.train.CheckpointManager(
                    self._ckpt_best_top5, self._log_dir,
                    max_to_keep=2, checkpoint_name='best_top5')

            if top5_accuracy >= self._best_top5_accuracy:
                self._best_top1_accuracy = top5_accuracy
                self._ckpt_best_top5_mgr.save(epoch)

    def _log_train_metrics(self, epoch, epoch_steps, total_steps, loss, lr):
        with self._summary_writer.as_default():
            tf.summary.scalar("Train/Loss", loss, total_steps)
            tf.summary.scalar("Train/LR", lr, total_steps)

            if epoch_steps > self._max_train_step:
                self._max_train_step = epoch_steps

            total_steps = self._max_train_step \
                if self._trained_4_full_epoch_once is True else "?"

            self.print_msg("[TRAIN] - Epoch {}, Step {}/{}"
                .format(epoch, epoch_steps, total_steps))
            self._log_common_metrics(loss)

    def _log_test_metrics(self, epoch, loss, top1_accuracy, top5_accuracy):
        with self._summary_writer.as_default():
            tf.summary.scalar("Test/Loss", loss, epoch)
            self.print_msg("[TEST] - Epoch {}".format(epoch))
            if self._include_top1:
                tf.summary.scalar("Test/Top-1 Accuracy",
                    top1_accuracy, epoch)
                self.print_msg("top1: {}".format(top1_accuracy), False)
            if self._include_top5:
                tf.summary.scalar("Test/Top-5 Accuracy",
                    top5_accuracy, epoch)
                self.print_msg("top5: {}".format(top5_accuracy), False)
            self._log_common_metrics(loss)

    def _log_common_metrics(self, loss):
        self.print_msg("loss: {}".format(loss), False)

    def log_method_info(self, method_info):
        with self._summary_writer.as_default():
            tf.summary.text("Logging/" + method_info[0],
                tf.convert_to_tensor(method_info[1]), 0)

    def log_graph_of_func(self, func, args):
        with self._summary_writer.as_default():
            func_graph = func.get_concrete_function(*args).graph
            summary_ops_v2.graph(func_graph)

    def log_loggables(self, loggables):
        with self._summary_writer.as_default():
            for loggable in loggables:
                file, data = loggable.get_log_data()
                tf.summary.text("Logging/"+file, tf.convert_to_tensor(data), 0)

    @staticmethod
    def print_msg(msg, put_time=True):
        t_str = "                     "
        if put_time:
            t_str = datetime.now().strftime("%Y%m%d %H:%M:%S.%f")[:-3]
        print("{} - {}".format(t_str, msg))
