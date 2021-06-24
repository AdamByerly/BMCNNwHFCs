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
import matplotlib.pyplot as plt
from python.constructs.loops import Loops
from python.constructs.output import Output
from python.constructs.optimizer import Adam
from python.constructs.metrics import Metrics
from python.constructs.loggable import Loggable
from python.constructs.loss import MeanSquaredError
from python.constructs.learning_rate import ManualExponentialDecay
from python.input.MNIST_input_pipeline import MNIST
from python.models.BranchingMerging import SmallImageBranchingMerging
import tensorflow as tf


def save_images(log_dir, generated_data, epoch):
    generated_data = tf.squeeze(generated_data)

    grid_size = 10

    fig = plt.figure(figsize=(grid_size, grid_size))

    range_top = generated_data.shape[0]
    if grid_size ** 2 < range_top:
        range_top = grid_size ** 2

    for i_ in range(range_top):
        plt.subplot(grid_size, grid_size, i_ + 1)
        plt.imshow(generated_data[i_, :, :], cmap='Greys')
        plt.axis('off')

    filename = os.path.join(log_dir,
        "images_at_epoch_{:04d}.png".format(epoch))
    plt.savefig(filename)
    plt.close(fig)


def go(run_name, end_epoch, data_dir, log_dir, merge_strategy, batch_size,
       hvc_type=1, hvc_dims=None, use_augmentation=True, augmentation_type=1,
       total_convolutions=None, branches_after=None):

    out = Output(log_dir, run_name, None, None)

    in_pipe = MNIST(data_dir, use_augmentation, augmentation_type)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        out.print_msg("Building model...")

        model = SmallImageBranchingMerging(in_pipe.get_class_count(),
                    in_pipe.get_image_size(), in_pipe.get_image_channels(),
                    merge_strategy, True, hvc_type, hvc_dims,
                    total_convolutions, branches_after, True)

        lr          = ManualExponentialDecay(0.001, 0.98, 1e-7)
        optimizer   = Adam(lr)
        loss        = MeanSquaredError()
        metrics     = Metrics(True, False)
        loops       = Loops(in_pipe, out, strategy, model, optimizer,
                        lr, loss, metrics, None, batch_size)

        out.log_method_info(Loggable.get_this_method_info())
        out.log_loggables([out, in_pipe, model,
            lr, optimizer, loss, metrics, loops])

        for epoch in range(1, end_epoch+1):
            out.print_msg("Starting epoch {}...".format(epoch))
            loops.do_recon_epoch(epoch)
            for features, labels in in_pipe.get_n_training_samples(100):
                model_out = model.forward(features, labels, False)
                save_images(out.get_log_dir(), model_out[1], epoch)
                break


################################################################################
# Entry point
################################################################################
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default=None)
    p.add_argument("--end_epoch", default=300, type=int)
    p.add_argument("--data_dir", default=r"../../../Datasets/mnist_data")
    p.add_argument("--log_dir", default="../logs")
    p.add_argument("--batch_size", default=120, type=int)
    p.add_argument("--merge_strategy", default=2, type=float)
    p.add_argument("--hvc_type", default=2, type=int)
    p.add_argument("--hvc_dims", default=[8], type=int)
    p.add_argument("--use_augmentation", default=False, type=bool)
    p.add_argument("--augmentation_type", default=1, type=int)
    p.add_argument("--total_convolutions", default=9, type=int)
    p.add_argument("--branches_after", default=[8])
    a = p.parse_args()

    rn = datetime.now().strftime("%Y%m%d%H%M%S")
    go(run_name=rn, end_epoch=a.end_epoch, data_dir=a.data_dir,
       log_dir=a.log_dir, merge_strategy=a.merge_strategy,
       batch_size=a.batch_size, hvc_type=a.hvc_type,
       hvc_dims=a.hvc_dims, use_augmentation=a.use_augmentation,
       augmentation_type=a.augmentation_type,
       total_convolutions=a.total_convolutions,
       branches_after=a.branches_after)
