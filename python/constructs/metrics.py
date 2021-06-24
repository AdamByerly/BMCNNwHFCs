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

from python.constructs.loggable import Loggable
import tensorflow as tf


class Metrics(Loggable):
    def __init__(self, compute_top1=True, compute_top5=False):
        Loggable.__init__(self)

        self._compute_top1     = compute_top1
        self._compute_top5     = compute_top5
        self._loss_counter     = 0
        self._loss_value       = 0
        self._acc_counter      = 0
        self._top1_value       = 0
        self._top5_value       = 0

    def computing_top1(self):
        return self._compute_top1

    def computing_top5(self):
        return self._compute_top5

    def get_loss(self):
        return self._loss_value

    def get_top1(self):
        return self._top1_value

    def get_top5(self):
        return self._top5_value

    def reset(self):
        self._loss_counter     = 0
        self._loss_value       = 0
        self._acc_counter      = 0
        self._top1_value       = 0
        self._top5_value       = 0

    def update_loss(self, loss):
        with tf.name_scope("metrics"):
            self._loss_value = ((loss +
                (self._loss_counter * self._loss_value))
                    / (self._loss_counter + 1))
            self._loss_counter += 1

    def update_accuracy(self, top1=None, top5=None):
        with tf.name_scope("metrics"):
            if self._compute_top1:
                self._top1_value = ((top1 +
                    (self._acc_counter * self._top1_value))
                    / (self._acc_counter + 1))

            if self._compute_top5:
                self._top5_value = ((top5 +
                    (self._acc_counter * self._top5_value))
                    / (self._acc_counter + 1))

            self._acc_counter += 1
