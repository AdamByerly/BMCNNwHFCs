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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import argparse
from datetime import datetime
from python.constructs.loops import Loops
from python.constructs.output import Output
from python.constructs.optimizer import Adam
from python.constructs.metrics import Metrics
from python.constructs.loggable import Loggable
from python.constructs.ema_weights import EMAWeights
from python.constructs.loss import MeanSquaredError
from python.constructs.loss import CategoricalCrossEntropy
from python.constructs.loss import MarginPlusMeanSquaredError
from python.constructs.loss import CategoricalCrossEntropyPlusMeanSquaredError
from python.constructs.learning_rate import ManualExponentialDecay
from python.input.MNIST_input_pipeline import MNIST
from python.input.cifar10_input_pipeline import Cifar10
from python.input.cifar100_input_pipeline import Cifar100
from python.input.smallNORB_input_pipeline import smallNORB
from python.models.SmallImageBranchingMerging import SmallImageBranchingMerging
import tensorflow as tf


def go(run_name, end_epoch, data_dir, input_pipeline, log_dir,
       batch_size, merge_strategy, loss_type, use_hvcs=True, hvc_type=1,
       initial_filters=32, filter_growth=16, hvc_dims=None,
       use_augmentation=True, augmentation_type=1, total_convolutions=None,
       branches_after=None, reconstruct_from_hvcs=False,
       profile_batch_start=None, profile_batch_end=None):

    out = Output(log_dir, run_name, profile_batch_start, profile_batch_end)

    if input_pipeline == 3:
        in_pipe = Cifar10(data_dir, use_augmentation, augmentation_type)
    elif input_pipeline == 4:
        in_pipe = Cifar100(data_dir, use_augmentation, augmentation_type)
    elif input_pipeline == 5:
        in_pipe = smallNORB(data_dir, use_augmentation, 48, 32)
    else:
        in_pipe = MNIST(data_dir, use_augmentation, augmentation_type)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        out.print_msg("Building model...")

        model = SmallImageBranchingMerging(in_pipe.get_class_count(),
                    in_pipe.get_image_size(), in_pipe.get_image_channels(),
                    merge_strategy, use_hvcs, hvc_type, initial_filters,
                    filter_growth, hvc_dims, total_convolutions, branches_after,
                    reconstruct_from_hvcs)

        if loss_type == 2:
            loss = MeanSquaredError()
        elif loss_type == 3:
            loss = MarginPlusMeanSquaredError()
        elif loss_type == 4:
            loss = CategoricalCrossEntropyPlusMeanSquaredError()
        else:
            loss = CategoricalCrossEntropy()

        lr          = ManualExponentialDecay(0.001, 0.98, 1e-7)
        optimizer   = Adam(lr)
        metrics     = Metrics(True, False)
        ema_weights = EMAWeights(0.999, model.get_all_trainable_variables())
        loops       = Loops(in_pipe, out, strategy, model, optimizer,
                        lr, loss, metrics, ema_weights, batch_size)

        out.log_method_info(Loggable.get_this_method_info())
        out.log_loggables([out, in_pipe, model,
            lr, optimizer, loss, metrics, ema_weights, loops])

        for epoch in range(1, end_epoch+1):
            out.print_msg("Starting epoch {}...".format(epoch))
            loops.do_epoch(epoch)


################################################################################
# Entry point
################################################################################
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default=None)
    p.add_argument("--end_epoch", default=300, type=int)
    p.add_argument("--data_dir", default=r"../../../Datasets/smallNORB_data")
    p.add_argument("--input_pipeline", default=5, type=int)
    p.add_argument("--log_dir", default="../logs")
    p.add_argument("--batch_size", default=120, type=int)
    p.add_argument("--merge_strategy", default=2, type=float)
    p.add_argument("--loss_type", default=1, type=int)
    p.add_argument("--use_hvcs", default=True, type=bool)
    p.add_argument("--hvc_type", default=2, type=int)
    p.add_argument("--initial_filters", default=32, type=int)
    p.add_argument("--filter_growth", default=16, type=int)
    p.add_argument("--hvc_dims", default=[96, 144, 192], type=int)
    p.add_argument("--use_augmentation", default=True, type=bool)
    p.add_argument("--augmentation_type", default=1, type=int)
    p.add_argument("--total_convolutions", default=11, type=int)
    p.add_argument("--branches_after", default=[4, 7, 10])
    p.add_argument("--reconstruct_from_hvcs", default=False, type=bool)
    p.add_argument("--profile_batch_start", default=None, type=int)
    p.add_argument("--profile_batch_end", default=None, type=int)
    p.add_argument("--trials", default=1, type=int)
    a = p.parse_args()

    for i in range(a.trials):
        if a.run_name is None:
            rn = datetime.now().strftime("%Y%m%d%H%M%S")
        else:
            rn = a.run_name + ("" if a.trials <= 1 else "_" + str(i))

        go(run_name=rn, end_epoch=a.end_epoch, data_dir=a.data_dir,
           input_pipeline=a.input_pipeline, log_dir=a.log_dir,
           batch_size=a.batch_size, merge_strategy=a.merge_strategy,
           loss_type=a.loss_type, use_hvcs=a.use_hvcs, hvc_type=a.hvc_type,
           initial_filters=a.initial_filters, filter_growth=a.filter_growth,
           hvc_dims=a.hvc_dims, use_augmentation=a.use_augmentation,
           augmentation_type=a.augmentation_type,
           total_convolutions=a.total_convolutions,
           branches_after=a.branches_after,
           reconstruct_from_hvcs=a.reconstruct_from_hvcs,
           profile_batch_start=a.profile_batch_start,
           profile_batch_end=a.profile_batch_end)
