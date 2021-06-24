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
from python.input.MNIST_input_pipeline import MNIST
from python.input.cifar10_input_pipeline import Cifar10
from python.input.cifar100_input_pipeline import Cifar100
from python.input.smallNORB_input_pipeline import smallNORB
from python.models.BranchingMerging import SmallImageBranchingMerging
import tensorflow as tf


def go(data_dir, log_dir, output_file, input_pipeline, merge_strategy,
       use_hvcs=True, hvc_type=1, hvc_dims=None, total_convolutions=None,
       branches_after=None):

    files = []
    for dirname, _, filenames in os.walk(log_dir):
        file = list(set([os.path.join(dirname,
            os.path.splitext(fn)[0]) for fn in filenames]))
        if len(file) > 0:
            files.append(file[0])

    if input_pipeline == 3:
        in_pipe = Cifar10(data_dir, False, 0)
    elif input_pipeline == 4:
        in_pipe = Cifar100(data_dir, False, 0)
    elif input_pipeline == 5:
        in_pipe = smallNORB(data_dir, False, 48, 32)
    else:
        in_pipe = MNIST(data_dir, False, 1)

    branch_weights = []

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        print("Building model...")

        model = SmallImageBranchingMerging(in_pipe.get_class_count(),
                    in_pipe.get_image_size(), in_pipe.get_image_channels(),
                    merge_strategy, use_hvcs, hvc_type, hvc_dims,
                    total_convolutions, branches_after, False)

        for weights_file in files:
            print("Restoring weights file: {}".format(weights_file))
            ckpt = tf.train.Checkpoint(
                    vars=model.get_all_savable_variables())
            ckpt.restore(weights_file).expect_partial()

            branch_weights.append(model.branch_weights.variable.numpy())

    print("Saving final branch weights...")
    # (False Positive)
    # noinspection PyTypeChecker
    np.savetxt(output_file, np.array(branch_weights), delimiter=',', fmt='%0f')
    print("Finished.")


################################################################################
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=r"../../../../Datasets/smallNORB_data")
    p.add_argument("--log_dir", default=r"../../logs/20210609135430")
    p.add_argument("--output_file",
        default=r"../../logs/20210609135430/final_branch_weights.txt")
    p.add_argument("--input_pipeline", default=5, type=int)
    p.add_argument("--merge_strategy", default=2, type=float)
    p.add_argument("--use_hvcs", default=True, type=bool)
    p.add_argument("--hvc_type", default=2, type=int)
    p.add_argument("--hvc_dims", default=[96, 144, 192], type=int)
    p.add_argument("--total_convolutions", default=11, type=int)
    p.add_argument("--branches_after", default=[4, 7, 10])
    a = p.parse_args()

    go(data_dir=a.data_dir, log_dir=a.log_dir, output_file=a.output_file,
       input_pipeline=a.input_pipeline, merge_strategy=a.merge_strategy,
       use_hvcs=a.use_hvcs, hvc_type=a.hvc_type, hvc_dims=a.hvc_dims,
       total_convolutions=a.total_convolutions, branches_after=a.branches_after)
