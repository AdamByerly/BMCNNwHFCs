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
from python.models.nn_ops import caps_from_conv_xyz, hvc_from_xyz
from python.models.nn_ops import conv, caps_from_conv_zxy, hvc_from_zxy
from python.models.nn_ops import flatten, fc_w_batch_norm, batch_norm
from python.models.variable import get_batch_norm_variable
from python.models.variable import get_conv_3x3_batch_norm_variable
from python.models.variable import get_hvc_from_xyz_batch_norm_variable
from python.models.variable import get_hvc_from_zxy_batch_norm_variable
import tensorflow as tf


class SmallImageBranchingMerging(Loggable):
    def __init__(self, count_classes, image_size, image_channels,
            merge_strategy, use_hvcs, hvc_type, initial_filters,
            filter_growth, hvc_dims, total_convolutions, branches_after,
            reconstruct_from_hvcs):
        Loggable.__init__(self)

        self._count_classes         = count_classes
        self._image_size            = image_size
        self._image_channels        = image_channels
        self._merge_strategy        = merge_strategy
        self._use_hvcs              = use_hvcs
        self._hvc_type              = hvc_type
        self._initial_filters       = initial_filters
        self._filter_growth         = filter_growth
        self._hvc_dims              = hvc_dims
        self._total_convolutions    = total_convolutions
        self._branches_after        = branches_after
        self._reconstruct_from_hvcs = reconstruct_from_hvcs

        with tf.name_scope("variables"):
            prior_size   = self._image_channels
            tensor_size  = self._image_size
            self.vars    = []
            conv_filters = self._initial_filters
            branch_no    = 0

            for var_idx in range(self._total_convolutions):
                self.vars.append(get_conv_3x3_batch_norm_variable(
                    "W_conv" + str(var_idx), prior_size, conv_filters))

                tensor_size -= 2

                if var_idx in self._branches_after:
                    self._create_branch_vars(branch_no,
                        var_idx, conv_filters, tensor_size**2)
                    branch_no += 1

                prior_size    = conv_filters
                conv_filters += self._filter_growth

            initer    = tf.ones_initializer()
            trainable = False
            if self._merge_strategy == 1:
                initer    = tf.random_normal_initializer()
                trainable = True
            elif self._merge_strategy == 2:
                trainable = True

            self.branch_weights = get_batch_norm_variable(
                shape=[len(self._branches_after)], name="branch_weights",
                initializer=initer, trainable=trainable)

            if self._reconstruct_from_hvcs:
                self._create_reconstruction_vars(tensor_size**2)

    def _create_branch_vars(self, branch_no,
            var_idx, conv_filters, filter_size):
        if self._use_hvcs:
            if self._hvc_type == 1:
                self.vars.append(
                    get_hvc_from_zxy_batch_norm_variable(
                        "W_ocap" + str(var_idx), (conv_filters*filter_size)
                        // self._hvc_dims[branch_no], self._count_classes,
                        self._hvc_dims[branch_no]))
            else:
                self.vars.append(
                    get_hvc_from_xyz_batch_norm_variable(
                        "W_ocap" + str(var_idx), (conv_filters*filter_size)
                        // self._hvc_dims[branch_no], self._count_classes,
                        self._hvc_dims[branch_no]))
        else:
            self.vars.append(get_batch_norm_variable(
                [conv_filters*filter_size, self._count_classes],
                "W_fc" + str(var_idx)))

    def _create_reconstruction_vars(self, filter_size):
        if self._hvc_type == 1:
            self.recon_fc_1 = get_batch_norm_variable(
                [filter_size*self._count_classes, 512], "recon_fc_1")
        else:
            self.recon_fc_1 = get_batch_norm_variable(
                [self._hvc_dims[-1] * self._count_classes, 512], "recon_fc_1")
        self.recon_fc_2 = get_batch_norm_variable([512, 1024], "recon_fc_2")
        self.recon_fc_3 = get_batch_norm_variable(
                [1024, self._image_size ** 2], "recon_fc_3")

    def get_all_trainable_variables(self):
        output = [v for v in self.vars for v in v.get_trainable_variables()]

        if self._reconstruct_from_hvcs:
            output += [v for v in self.recon_fc_1.get_trainable_variables()] \
                   +  [v for v in self.recon_fc_2.get_trainable_variables()] \
                   +  [v for v in self.recon_fc_3.get_trainable_variables()] \

        return output + [v for v in
            self.branch_weights.get_trainable_variables()]

    def get_all_savable_variables(self):
        output = [v for v in self.vars for v in v.get_savable_variables()]

        if self._reconstruct_from_hvcs:
            output += [v for v in self.recon_fc_1.get_savable_variables()] \
                   +  [v for v in self.recon_fc_2.get_savable_variables()] \
                   +  [v for v in self.recon_fc_3.get_savable_variables()] \

        return output + [v for v in
            self.branch_weights.get_savable_variables()]

    def forward(self, features, labels, is_training, perturbation=None):
        with tf.name_scope("forward"):
            recon        = None
            tensor_size  = self._image_size
            conv_filters = self._initial_filters
            branch_no    = 0
            var_counter  = 0
            logits_list  = []
            x            = features

            for op_idx in range(self._total_convolutions):
                x = conv("conv" + str(op_idx), is_training, x,
                        self.vars[var_counter])

                var_counter += 1
                tensor_size -= 2

                if op_idx in self._branches_after:
                    y = self._forward_thru_branch(x, self.vars[var_counter],
                        branch_no, op_idx, conv_filters, tensor_size**2,
                        is_training)
                    if self._use_hvcs:
                        z = tf.norm(y, axis=2, name="logits_" + str(op_idx))
                        logits_list.append(z)
                    else:
                        logits_list.append(y)

                    var_counter += 1
                    branch_no += 1

                conv_filters += self._filter_growth

            if self._use_hvcs and self._reconstruct_from_hvcs:
                recon = self._forward_thru_reconstruction(labels,
                            y, tensor_size**2, perturbation, is_training)

            with tf.name_scope("logits"):
                logits = tf.stack(logits_list, axis=2)
                logits = tf.multiply(logits, self.branch_weights.variable)
                logits = batch_norm("bn_logits",
                            is_training, logits, self.branch_weights)
                logits = tf.reduce_sum(logits, axis=2, name="logits")
                return logits, recon

    def _forward_thru_branch(self, in_tensor, variable,
            branch_no, op_idx, conv_filters, filter_size, is_training):
        if self._use_hvcs:
            if self._hvc_type == 1:
                x = caps_from_conv_zxy("pcap" + str(op_idx), in_tensor,
                        (conv_filters*filter_size)//self._hvc_dims[branch_no],
                        self._hvc_dims[branch_no])
                x = hvc_from_zxy("ocap" + str(op_idx), is_training, x, variable)
            else:
                x = caps_from_conv_xyz("pcap" + str(op_idx), in_tensor,
                        (conv_filters*filter_size)//self._hvc_dims[branch_no],
                        self._hvc_dims[branch_no])
                x = hvc_from_xyz("ocap" + str(op_idx), is_training, x, variable)
        else:
            x = flatten("flatten", in_tensor)
            x = fc_w_batch_norm("fc", is_training, x, variable, None)

        return x

    def _forward_thru_reconstruction(self, labels,
            in_tensor, filter_size, perturbation, is_training):
        with tf.name_scope("reconstruction"):
            capsule_mask_3d = tf.expand_dims(labels, -1)

            if self._hvc_type == 1:
                atom_mask = tf.tile(capsule_mask_3d, [1, 1, filter_size])
            else:
                atom_mask = tf.tile(capsule_mask_3d, [1, 1, self._hvc_dims[-1]])

            if perturbation is not None:
                in_tensor = in_tensor + perturbation

            capsules_masked = in_tensor * atom_mask

            if self._hvc_type == 1:
                recon = tf.reshape(capsules_masked,
                            [-1, filter_size*self._count_classes])
            else:
                recon = tf.reshape(capsules_masked,
                            [-1, self._hvc_dims[-1]*self._count_classes])

            recon = fc_w_batch_norm("recon_fc_1",
                        is_training, recon, self.recon_fc_1)
            recon = fc_w_batch_norm("recon_fc_2",
                        is_training, recon, self.recon_fc_2)
            recon = fc_w_batch_norm("recon_fc_3",
                        is_training, recon, self.recon_fc_3, tf.nn.tanh)
            recon = tf.reshape(recon, [-1, self._image_size,
                        self._image_size, self._image_channels])

            return recon
