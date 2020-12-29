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

import argparse
import numpy as np
from python.training.train import make_tower
import tensorflow as tf
tf1 = tf.compat.v1


def go(input_shape, weights_file, classes, output_file_base, merge_strategy):
    tf1.disable_eager_execution()
    tf1.disable_control_flow_v2()
    tf1.logging.set_verbosity(tf1.logging.ERROR)
    tf1.reset_default_graph()

    ############################################################################
    # Tensorflow Graph
    ############################################################################
    print("Building Graph Operations...")
    with tf.device("/device:CPU:0"):  # set the default device to the CPU
        preds, _, _ = make_tower(
            features=tf.cast(np.zeros(shape=input_shape), tf.float32),
            labels=tf.cast(np.zeros(shape=[classes]), tf.float32),
            is_training=tf.convert_to_tensor(False),
            count_classes=classes, merge_strategy=merge_strategy)

    ############################################################################
    # Tensorflow session
    ############################################################################
    print("Starting Session...")
    with tf1.Session(config=tf1.ConfigProto(allow_soft_placement=True)) as sess:
        if weights_file is None:
            print("Cannot export without a weights file!")
            return

        print("Restoring weights file: {}".format(weights_file))
        tf1.train.Saver().restore(sess, weights_file)

        frozen_graph_def = tf1.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, [preds.op.name])

        with open(output_file_base+".pb", "wb") as f:
            f.write(frozen_graph_def.SerializeToString())

        print("Finished.")
        sess.close()


################################################################################
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_shape", default=[1, 28, 28, 1])
    p.add_argument("--merge_strategy", default=2, type=float)
    p.add_argument("--classes", default=10, type=float)
    p.add_argument("--weights_file",
        default="../../data/learnable_ones_init/weights"
                "/20201228150718/weights-1-latest-499")
    p.add_argument("--output_file_base",
        default="../../data/learnable_ones_init/model")
    a = p.parse_args()

    go(input_shape=a.input_shape,
        weights_file=a.weights_file, classes=a.classes,
        output_file_base=a.output_file_base,
        merge_strategy=a.merge_strategy)
