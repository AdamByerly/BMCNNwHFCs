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
import shutil
import argparse


def get_weight_file_type(weight_file):
    dash_positions = [i for i, x in enumerate(weight_file) if x == '-']
    idx_of_dashes = [x for i, x in enumerate(dash_positions)]
    return weight_file[:idx_of_dashes[0]]


def get_weight_file_epoch(weight_file):
    dash_positions   = [i for i, x in enumerate(weight_file) if x == '-']
    idx_of_dashes    = [x for i, x in enumerate(dash_positions)]
    period_positions = [i for i, x in enumerate(weight_file) if x == '.']
    idx_of_period    = [x for i, x in enumerate(period_positions)]
    return weight_file[idx_of_dashes[0]+1:idx_of_period[0]]


def get_weight_types_and_files(path):
    weight_files = [f for f in os.listdir(path)
                    if f.startswith("best_top1")
                    or f.startswith("latest")]
    return list(set([get_weight_file_type(f)
                     for f in weight_files])), weight_files


def go(path):
    dirs = [d for d in os.listdir(path)
            if os.path.isdir( os.path.join(path, d))]

    for d in dirs:
        dpath = os.path.join(path, d)
        weight_types, weight_files = get_weight_types_and_files(dpath)

        for wt in weight_types:
            wt_path = os.path.join(path, wt)
            if not os.path.exists(wt_path):
                os.mkdir(wt_path)

            segregated_experiment_path = os.path.join(wt_path, d)
            if not os.path.exists(segregated_experiment_path):
                os.mkdir(segregated_experiment_path)

            type_files = [wf for wf in weight_files if wf.find(wt) >= 0]

            epochs = [get_weight_file_epoch(f) for f in type_files]
            max_epoch = max([int(e) for e in epochs])

            type_files = [f for f in type_files if f.find(str(max_epoch)) >= 0]

            for tf in type_files:
                orig_location = os.path.join(dpath, tf)
                new_location = os.path.join(segregated_experiment_path, tf)
                print('{} -> {}'.format(orig_location, new_location))
                shutil.copyfile(orig_location, new_location)


################################################################################
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="../../logs_ms0")
    a = p.parse_args()

    go(a.log_dir)
