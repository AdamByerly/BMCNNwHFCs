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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import argparse
import numpy as np
from PIL import Image
from python.input.MNIST_input_pipeline import MNIST
import tensorflow as tf


def go(batch_size, data_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    in_pipe = MNIST(data_dir, False, 1)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        i = 0
        for features, labels in in_pipe. \
                get_validation_dataset(1).batch(batch_size):
            for j in range(min(batch_size, len(features))):
                fname = os.path.join(output_dir,
                        str(i)+'('+str(np.argmax(labels[j]))+').gif')
                Image.fromarray((1-np.squeeze(features[j]))*255).save(fname)
                i += 1

        print("Finished.")


################################################################################
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", default=120, type=int)
    p.add_argument("--data_dir", default=r"../../../../Datasets/mnist_data")
    p.add_argument("--output_dir", default=r"../../MNIST_eval_digits")
    a = p.parse_args()

    go(batch_size=a.batch_size, data_dir=a.data_dir, output_dir=a.output_dir)
