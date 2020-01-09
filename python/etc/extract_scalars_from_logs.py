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
import pandas as pd
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator \
    import EventAccumulator


def tabulate_events(dpath):
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload()
                         for dname in os.listdir(dpath)]
    tags = summary_iterators[0].Tags()['scalars']
    out = defaultdict(list)
    for tag in tags:
        out[tag].extend([[e.value for e
            in acc.Scalars(tag)] for acc in summary_iterators])
    return out, tags


def get_file_path(folder_path, tag):
    file_name = tag.replace("/", "_") + '.csv'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)


def go(summary_dirs, output_dir):
    dirs = os.listdir(summary_dirs)
    events, tags = tabulate_events(summary_dirs)
    for index, tag in enumerate(tags):
        events_count = np.max([len(l) for l in events[tag]])
        # noinspection PyTypeChecker
        # (It works exactly as we want)
        vals = np.full((events_count, len(dirs)), None)
        for x in range(len(dirs)):
            for y in range(len(events[tag][x])):
                vals[y, x] = events[tag][x][y]
        df = pd.DataFrame(vals, columns=dirs)
        df.to_csv(get_file_path(output_dir, tag))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--event_data_dir",
        default=r"..\..\data\learnable_ones_init\weights")
    p.add_argument("--output_dir", default=r"..\..\data\learnable_ones_init")
    a = p.parse_args()

    go(summary_dirs=a.event_data_dir, output_dir=a.output_dir)
