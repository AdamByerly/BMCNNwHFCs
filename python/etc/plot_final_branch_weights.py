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

import argparse
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt


def go(input_file, output_file):
    df = pd.read_csv(input_file, header=None)

    linestyle_cycler = (cycler("color", ["blue", "orange", "green"])
                        + cycler("linestyle", ["solid", "dotted", "dashed"]))

    plt.figure(figsize=(7.48, 5.92))
    plt.rc("axes", prop_cycle=linestyle_cycler)
    plt.plot(df[0])
    plt.plot(df[1])
    plt.plot(df[2])
    plt.legend(["Branch 1", "Branch 2", "Branch 3"])
    plt.xticks(ticks=range(len(df[0])),
        labels=sum([["", i+2] for i in range(0, 32, 2)], []))
    plt.xlabel("Trial #")
    plt.savefig(output_file, bbox_inches="tight")


################################################################################
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_file",
        default=r"../../logs_ms2/final_branch_weights.txt")
    p.add_argument("--output_file",
        default=r"../../logs_ms2/final_branch_weights.png")
    a = p.parse_args()

    go(input_file=a.input_file, output_file=a.output_file)
