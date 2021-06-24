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

import inspect
import numpy as np
from python.constructs.loggable import Loggable
import tensorflow as tf


class Loops(Loggable):
    def __init__(self, in_pipe, out, strategy, model, optimizer,
            learning_rate, loss_object, metrics, ema_weights, batch_size,
            weights_file=None):
        Loggable.__init__(self)
        self._in_pipe        = in_pipe
        self._out            = out
        self._strategy       = strategy
        self._model          = model
        self._optimizer      = optimizer
        self._learning_rate  = learning_rate
        self._loss_object    = loss_object
        self._metrics        = metrics
        self._ema_weights    = ema_weights
        self._batch_size     = batch_size
        self._verify_restore = False
        self._graph_logged   = False

        if weights_file is not None:
            self._out.print_msg("Restoring weights file: {}"
                .format(weights_file))
            ckpt = tf.train.Checkpoint(
                    vars=self._model.get_all_savable_variables(),
                    optimizer=self._optimizer)
            self._ckpt_status    = ckpt.restore(weights_file)
            self._verify_restore = True

    def do_recon_epoch(self, epoch):
        self._train(epoch)

        self._learning_rate.increase_step()
        self._verify_restore_func()

    def do_epoch(self, epoch):
        self._train(epoch)

        self._out.print_msg("Loading EMA of past weights...")
        self._ema_weights.load_validation_weights()

        self._validate(epoch)

        self._out.print_msg("Loading current weights...")
        self._ema_weights.load_training_weights()

        self._learning_rate.increase_step()
        self._verify_restore_func()

    def _verify_restore_func(self):
        if self._verify_restore:
            self._out.print_msg("Verifying that all variables were restored...")
            self._ckpt_status.assert_consumed()
            self._verify_restore = False

    def _train(self, epoch):
        self._metrics.reset()

        dataset = self._strategy.experimental_distribute_datasets_from_function(
            lambda input_context: self._dataset_distribute(input_context,
                self._in_pipe.get_training_dataset(epoch)))

        step = 0
        for features, labels in dataset:
            self._out.train_step_begin(step)
            lr = self._learning_rate(self._optimizer.iterations)

            loss, outputs = self._distributed_train_step(features, labels)

            self._log_graph_if_possible(
                self._distributed_train_step, (features, labels))

            if self._ema_weights is not None:
                self._ema_weights.update()

            self._out.train_step_end(epoch, step,
                loss, lr, self._optimizer.iterations)

            step += 1

        self._out.train_end(epoch)

    def _validate(self, epoch):
        self._metrics.reset()

        dataset = self._strategy.experimental_distribute_datasets_from_function(
            lambda input_context: self._dataset_distribute(input_context,
                self._in_pipe.get_validation_dataset(epoch)))

        step = 0
        for features, labels in dataset:
            loss, top1, top5, outputs = self._distributed_eval_step(
                features, labels)

            self._metrics.update_loss(loss)
            self._metrics.update_accuracy(top1, top5)

            self._out.validation_step_end(step)

            step += 1

        self._out.validation_end(self._model, self._optimizer,
            self._ema_weights, epoch, self._metrics.get_loss(),
            self._metrics.get_top1(), self._metrics.get_top5())

    def _log_graph_if_possible(self, graph_func, args):
        if not self._graph_logged and inspect.getsource(
                graph_func).find("@tf.function") >= 0:
            self._graph_logged = True
            self._out.print_msg(
                "Logging graph of model train function...")
            self._out.log_graph_of_func(graph_func, args)

    def _combine_replica_outputs(self, outputs):
        if self._strategy.num_replicas_in_sync > 1:
            outputs = np.reshape(np.array(
                [v for v in outputs.values if v.shape[0] > 0]),
                [-1, self._in_pipe.get_image_size(),
                 self._in_pipe.get_image_size(), 1])
        return outputs

    def _dataset_distribute(self, input_context, dataset):
        batch_size = input_context.get_per_replica_batch_size(
                        self._batch_size)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset.shard(input_context.num_input_pipelines,
            input_context.input_pipeline_id)

    @tf.function
    def _distributed_train_step(self, features, labels):
        def train_step(features_, labels_, loss_object):
            with tf.name_scope("train"), tf.GradientTape() as tape:
                model_out = self._model.forward(features_, labels_, True)
                with tf.name_scope("loss"):
                    loss = loss_object(features_,
                            labels_, model_out, self._batch_size)

            with tf.name_scope("apply_gradients"):
                gradients = tape.gradient(loss,
                    self._model.get_all_trainable_variables())
                self._optimizer.apply_gradients(zip(gradients,
                    self._model.get_all_trainable_variables()))

            return loss, model_out

        losses, outputs = self._strategy.run(
            train_step, args=(features, labels, self._loss_object))

        with tf.name_scope("reduce_train_replicas"):
            return self._strategy.reduce(
                tf.distribute.ReduceOp.SUM, losses, axis=None), outputs

    @tf.function
    def _distributed_eval_step(self, features, labels):
        def validation_step(features_, labels_,
                loss_object, computing_top1, computing_top5):
            with tf.name_scope("validation"):
                model_out = self._model.forward(features_, labels_, False)
                with tf.name_scope("loss"):
                    loss = loss_object(features_,
                            labels_, model_out, self._batch_size)

                top1, top5 = 0, 0
                if computing_top1:
                    top1   = tf.reduce_mean(
                                tf.keras.metrics.top_k_categorical_accuracy(
                                    labels_, model_out[0], k=1))
                if computing_top5:
                    top5   = tf.reduce_mean(
                                tf.keras.metrics.top_k_categorical_accuracy(
                                    labels_, model_out[0], k=5))

                return loss, top1, top5, model_out

        losses, top1s, top5s, outputs = self._strategy.run(
            validation_step, args=(features, labels, self._loss_object,
                self._metrics.computing_top1(), self._metrics.computing_top5()))

        with tf.name_scope("reduce_validation_replicas"):
            l = self._strategy.reduce(
                    tf.distribute.ReduceOp.SUM, losses, axis=None)

            t1, t5 = 0, 0
            if self._metrics.computing_top1():
                t1 = self._strategy.reduce(
                        tf.distribute.ReduceOp.MEAN, top1s, axis=None)
            if self._metrics.computing_top5():
                t5 = self._strategy.reduce(
                        tf.distribute.ReduceOp.MEAN, top5s, axis=None)

            return l, t1, t5, outputs
