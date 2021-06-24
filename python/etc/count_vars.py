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
from python.models.BranchingMerging import SmallImageBranchingMerging


def go(merge_strategy, use_hvcs, hvc_type, initial_filters, filter_growth,
       hvc_dims, total_convolutions, branches_after, reconstruct_from_hvcs,
       classes, image_size, image_channels):

    model = SmallImageBranchingMerging(classes, image_size, image_channels,
        merge_strategy, use_hvcs, hvc_type, initial_filters, filter_growth,
        hvc_dims, total_convolutions, branches_after, reconstruct_from_hvcs)

    all_vars, bn_vars, fc_vars, conv_vars, cap_vars, \
        branch_weight_vars, recon_vars, other_vars = 0, 0, 0, 0, 0, 0, 0, 0

    for v in model.get_all_trainable_variables():
        all_vars += np.prod(v.shape)
        if v.name.find("bn") >= 0:
            bn_vars += np.prod(v.shape)
        elif v.name.find("recon") >= 0:
            recon_vars += np.prod(v.shape)
        elif v.name.find("fc") >= 0:
            fc_vars += np.prod(v.shape)
        elif v.name.find("conv") >= 0:
            conv_vars += np.prod(v.shape)
        elif v.name.find("cap") >= 0:
            cap_vars += np.prod(v.shape)
        elif v.name.find("branch_weight") >= 0:
            branch_weight_vars += np.prod(v.shape)
        else:
            other_vars += np.prod(v.shape)

    print("Total Variable Count ...........:  {:,}".format(all_vars))
    print("Batch Norm. Variable Count .....:  {:,}".format(bn_vars))
    print("Convolutions Variable Count ....:  {:,}".format(conv_vars))
    print("Fully Connected Variable Count .:  {:,}".format(fc_vars))
    print("Capsules Variable Count ........:  {:,}".format(cap_vars))
    print("Branch Weight Variable Count ...:  {:,}".format(branch_weight_vars))
    print("Reconstruction Variable Count ..:  {:,}".format(recon_vars))
    print("Other Variable Count ...........:  {:,}".format(other_vars))


################################################################################
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--merge_strategy", default=2, type=float)
    p.add_argument("--use_hvcs", default=False, type=bool)
    p.add_argument("--hvc_type", default=2, type=int)
    p.add_argument("--initial_filters", default=64, type=int)
    p.add_argument("--filter_growth", default=16, type=int)
    p.add_argument("--hvc_dims", default=[128, 176, 224], type=int)
    p.add_argument("--total_convolutions", default=11, type=int)
    p.add_argument("--branches_after", default=[4, 7, 10])
    p.add_argument("--reconstruct_from_hvcs", default=False, type=bool)
    p.add_argument("--classes", default=10, type=int)
    p.add_argument("--image_size", default=32, type=int)
    p.add_argument("--image_channels", default=3, type=int)
    a = p.parse_args()

    go(merge_strategy=a.merge_strategy, use_hvcs=a.use_hvcs,
       hvc_type=a.hvc_type, initial_filters=a.initial_filters,
       filter_growth=a.filter_growth, hvc_dims=a.hvc_dims,
       total_convolutions=a.total_convolutions,
       branches_after=a.branches_after,
       reconstruct_from_hvcs=a.reconstruct_from_hvcs,
       classes=a.classes, image_size=a.image_size,
       image_channels=a.image_channels)
