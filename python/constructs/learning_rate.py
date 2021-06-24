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


class ExponentialDecay(
        tf.keras.optimizers.schedules.ExponentialDecay, Loggable):
    pass


class ConstantLearningRate(Loggable):
    def __init__(self, learning_rate):
        Loggable.__init__(self)
        self.learning_rate = learning_rate

    def increase_step(self):
        pass

    def reset_step(self):
        pass

    def __call__(self, step=None):
        return self.learning_rate

    def get_config(self):
        return {
            "learning_rate": self.learning_rate,
        }


class CappedExponentialDecay(
        tf.keras.optimizers.schedules.ExponentialDecay, Loggable):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate,
            staircase=False, minimum_lr=0.0, name=None):
        tf.keras.optimizers.schedules.ExponentialDecay.__init__(self,
            initial_learning_rate, decay_steps, decay_rate, staircase, name)
        Loggable.__init__(self)
        self.minimum_lr = minimum_lr

    def increase_step(self):
        pass

    def reset_step(self):
        pass

    def __call__(self, step=None):
        lr = super(CappedExponentialDecay, self).__call__(step)
        return tf.maximum(lr, self.minimum_lr)

    def get_config(self):
        config = super(CappedExponentialDecay, self).get_config()
        config["minimum_lr"] = self.minimum_lr
        return config


class ManualExponentialDecay(Loggable):
    def __init__(self, initial_learning_rate, decay_rate, minimum_lr=0.0):
        Loggable.__init__(self)
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.minimum_lr = minimum_lr
        self.decay_step = 0

    def increase_step(self):
        self.decay_step += 1

    def reset_step(self):
        self.decay_step = 0

    def __call__(self, step=None):
        lr = self.initial_learning_rate*(self.decay_rate**self.decay_step)
        return tf.maximum(lr, self.minimum_lr)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_rate": self.decay_rate,
            "minimum_lr": self.minimum_lr
        }
