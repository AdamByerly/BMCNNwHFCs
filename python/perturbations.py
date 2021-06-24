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
import numpy as np
import matplotlib.pyplot as plt
from python.input.MNIST_input_pipeline import MNIST
from python.models.BranchingMerging import SmallImageBranchingMerging
import tensorflow as tf


def start_image():
    return plt.figure(figsize=(10, 10))


def write_image(generated_data, image_idx, fig_pos):
    generated_data = tf.squeeze(generated_data)

    plt.subplot(10, 10, 1 + fig_pos)
    plt.imshow(generated_data[image_idx, :, :], cmap='Greys')
    plt.axis('off')


def finish_image(fig, filename):
    plt.savefig(filename)
    plt.close(fig)


def go(data_dir, output_dir, weights_file, merge_strategy, hvc_type, hvc_dims,
       perturbation_width, total_convolutions=None, branches_after=None):

    in_pipe = MNIST(data_dir, False, 0)

    model = SmallImageBranchingMerging(in_pipe.get_class_count(),
                in_pipe.get_image_size(), in_pipe.get_image_channels(),
                merge_strategy, True, hvc_type, hvc_dims, total_convolutions,
                branches_after, True)

    print("Restoring weights file: {}".format(weights_file))
    ckpt = tf.train.Checkpoint(vars=model.get_all_savable_variables())
    ckpt_status = ckpt.restore(weights_file)
    ckpt_status.expect_partial()

    features_to_use, labels_to_use = [], []
    for features, labels in in_pipe.get_n_training_samples(237):
        features_to_use = tf.gather(features,
                            [1, 3, 5, 7, 92, 236, 62, 52, 55, 4])
        labels_to_use   = tf.gather(labels,
                            [1, 3, 5, 7, 92, 236, 62, 52, 55, 4])
        break

    for digit_idx in range(10):
        hvc_dimensions = 100 if hvc_type == 1 else hvc_dims[0]
        for i in range(hvc_dimensions):
            fig = start_image()
            for j in range(100):
                perturbation = np.dstack((
                    np.zeros((10, 10, i)),
                    np.ones((10, 10, 1)),
                    np.zeros((10, 10, hvc_dimensions-1-i))))

                perturbation *= (j-50)*perturbation_width

                model_out = model.forward(features_to_use,
                                labels_to_use, False, perturbation)
                write_image(model_out[1], digit_idx, j)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            finish_image(fig, os.path.join(output_dir, "images_{}_{}.png")
                .format(digit_idx, i))


################################################################################
# Entry point
################################################################################
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=r"../../../../Datasets/mnist_data")
    p.add_argument("--output_dir", default=r"../../perturbations_paper")
    p.add_argument("--weights_file",
        default=r"../../logs_etc/20210525105356/latest-300")
    p.add_argument("--merge_strategy", default=2, type=float)
    p.add_argument("--hvc_type", default=2)
    p.add_argument("--hvc_dims", default=[8], type=int)
    p.add_argument("--perturbation_width", default=.01)
    p.add_argument("--total_convolutions", default=9, type=int)
    p.add_argument("--branches_after", default=[8])
    a = p.parse_args()

    go(data_dir=a.data_dir, output_dir=a.output_dir,
       weights_file=a.weights_file, merge_strategy=a.merge_strategy,
       hvc_type=a.hvc_type, hvc_dims=a.hvc_dims,
       perturbation_width=a.perturbation_width,
       total_convolutions=a.total_convolutions,
       branches_after=a.branches_after)
