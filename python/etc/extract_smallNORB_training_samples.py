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
import matplotlib.pyplot as plt
from python.input.smallNORB_input_pipeline import smallNORB
import tensorflow as tf


def start_image():
    return plt.figure(figsize=(14.7, 14.7))


def write_image(image, fig_pos):
    image = tf.squeeze(image)

    plt.subplot(10, 10, 1 + fig_pos)
    plt.imshow(image, cmap='Greys')
    plt.axis('off')


def finish_image(fig, filename):
    plt.savefig(filename)
    plt.close(fig)


def go(data_dir, output_dir):

    in_pipe = smallNORB(data_dir, True, 48, 32)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_count = 0
    for features, labels in in_pipe.\
            get_training_dataset(1, False).batch(100):
        fig = start_image()
        for i in range(100):
            write_image(features[i], i)
        finish_image(fig, os.path.join(output_dir, "images_{}.png")
            .format(image_count))
        image_count += 1
        if image_count >= 5:
            break


################################################################################
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=r"../../../../Datasets/smallNORB_data")
    p.add_argument("--output_dir", default=r"../../smallNORB_training_samples")
    a = p.parse_args()

    go(data_dir=a.data_dir, output_dir=a.output_dir)
