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
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tensorflow.python.framework import tensor_util
from tensorflow.python.summary.summary_iterator import summary_iterator


def get_file_path(folder_path, tag):
    file_name = tag.replace("/", "_")\
                    .replace("-", "_")\
                    .replace(" ", "_") + '.csv'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)


def go(summary_dirs, output_dir):
    dirs  = os.listdir(summary_dirs)
    dirs  = [os.path.join(summary_dirs, dname) for dname in dirs]
    dirs  = [d for d in dirs if os.path.isdir(d)
             and len(list(Path(d).glob("events.out.tfevents.*"))) > 0]
    files = [os.path.join(d, list(Path(d).glob(
             "events.out.tfevents.*"))[0].name) for d in dirs]

    scalar_data = defaultdict(lambda: list([]))
    for i, f in enumerate(files):
        for e in summary_iterator(f):
            if e.step < 1:
                continue
            # noinspection PyProtectedMember
            val = e.summary.value._values[0]
            tag = val.tag
            if tag.find("Test") < 0:
                continue
            if len(scalar_data[tag]) <= i:
                scalar_data[tag].append(list())
            scalar_data[tag][i].append(tensor_util.MakeNdarray(val.tensor))

    for k in scalar_data.keys():
        events_count = np.max([len(l) for l in scalar_data[k]])
        # noinspection PyTypeChecker
        vals = np.full((events_count, len(dirs)), None)
        for x in range(len(dirs)):
            for y in range(len(scalar_data[k][x])):
                vals[y, x] = scalar_data[k][x][y]
        df = pd.DataFrame(vals, columns=[os.path.basename(d) for d in dirs])
        df.to_csv(get_file_path(output_dir, k))


################################################################################
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--event_data_dir", default=r"../../logs_ms0")
    p.add_argument("--output_dir", default=r"../../logs_ms0")
    a = p.parse_args()

    go(summary_dirs=a.event_data_dir, output_dir=a.output_dir)
