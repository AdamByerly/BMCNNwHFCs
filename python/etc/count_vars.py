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

import numpy as np
import tensorflow as tf
from python.training.train import make_tower
tf1 = tf.compat.v1


def get_var_count(creator, class_count, image_size, image_channels):
    tf1.disable_eager_execution()
    tf1.disable_control_flow_v2()
    tf1.reset_default_graph()
    model = creator(tf.zeros([1, image_size, image_size, image_channels]),
            tf.zeros([1, class_count]), tf.constant(True), class_count)
    with tf1.Session(config=tf1.ConfigProto(allow_soft_placement=True)) as sess:
        tf1.global_variables_initializer().run()
        variable_names = [v.name for v in tf1.trainable_variables()]
        _, variables = sess.run([model, tf1.trainable_variables()])
        all_vars = 0
        bn_vars = 0
        conv_vars = 0
        cap_vars = 0
        for k, v in zip(variable_names, variables):
            all_vars += np.prod(v.shape)
            if k.find("bn") >= 0:
                bn_vars += np.prod(v.shape)
            elif k.find("conv") >= 0:
                conv_vars += np.prod(v.shape)
            elif k.find("cap") >= 0:
                cap_vars += np.prod(v.shape)
        return all_vars, bn_vars, conv_vars, cap_vars


vs = get_var_count(make_tower, 10, 28, 1)
print("Total Variable Count ........:  {:,}".format(vs[0]))
print("Batch Norm. Variable Count ..:  {:,}".format(vs[1]))
print("Convolutions Variable Count .:  {:,}".format(vs[2]))
print("Capsules Variable Count .....:  {:,}".format(vs[3]))
