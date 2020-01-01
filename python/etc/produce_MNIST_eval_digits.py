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
import argparse
import numpy as np
from PIL import Image
from python.training.input_pipeline import InputPipeline
import tensorflow as tf
tf1 = tf.compat.v1


def go(batch_size, data_dir, output_dir):
    tf1.disable_eager_execution()
    tf1.disable_control_flow_v2()
    tf1.logging.set_verbosity(tf1.logging.ERROR)
    tf1.reset_default_graph()

    ############################################################################
    # Data feeds
    ############################################################################
    print("Setting up data feeds...")
    input_pipe  = InputPipeline(data_dir)
    eval_data   = input_pipe.get_validation_dataset(batch_size)

    ############################################################################
    # Tensorflow Graph
    ############################################################################
    print("Building Graph Operations...")
    with tf.device("/device:CPU:0"):  # set the default device to the CPU
        with tf.name_scope("input/data_itetator"):
            data_iterator = tf1.data.Iterator.from_structure(
                tf1.data.get_output_types(eval_data),
                tf1.data.get_output_shapes(eval_data))
            features, labels = data_iterator.get_next()

    ############################################################################
    # Tensorflow session
    ############################################################################
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Starting Session...")
    with tf1.Session(config=tf1.ConfigProto(allow_soft_placement=True)) as sess:
        tf1.global_variables_initializer().run()

        i = 1
        sess.run(data_iterator.make_initializer(eval_data))
        try:
            while True:
                imgs, ls = sess.run([features, labels])
                for j in range(min(batch_size, len(imgs))):
                    fname = os.path.join(output_dir,
                                str(i)+'('+str(np.argmax(ls[j]))+').gif')
                    Image.fromarray((1-np.squeeze(imgs[j]))*255).save(fname)
                    i += 1
        except tf.errors.OutOfRangeError:
            pass

        print("Finished.")
        sess.close()


################################################################################
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", default=120, type=int)
    p.add_argument("--data_dir", default=r"..\..\data\mnist_data")
    p.add_argument("--output_dir", default=r"..\..\data\images\all")
    a = p.parse_args()

    go(batch_size=a.batch_size, data_dir=a.data_dir, output_dir=a.output_dir)
