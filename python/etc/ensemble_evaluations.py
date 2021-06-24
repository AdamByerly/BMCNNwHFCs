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

import glob
import shutil
import argparse
import numpy as np
from python.input.MNIST_input_pipeline import MNIST
from python.models.BranchingMerging import SmallImageBranchingMerging
import tensorflow as tf


def write_to_file(output_file, images_dir, image_output_dir,
        prediction_matrix, label_list, output_all_logits, output_images):
    if len(prediction_matrix) < 1:
        return  # must have been 0 models in the ensemble?!?!

    print("Determining which samples were predicted differently...")
    all_wrong_samples = []
    disagreeing_samples = []
    all_wrong_count = 0
    for i in range(0, len(prediction_matrix[0][1])):
        all_correct = True
        all_wrong = True
        true_value = label_list[i]
        if output_all_logits:
            first_value = np.argmax(prediction_matrix[0][1][i], axis=0)
        else:
            first_value = prediction_matrix[0][1][i]
        for j in range(0, len(prediction_matrix)):
            if output_all_logits:
                if true_value != np.argmax(prediction_matrix[j][1][i], axis=0):
                    all_correct = False
                if first_value != np.argmax(prediction_matrix[j][1][i], axis=0):
                    all_wrong = False
            else:
                if true_value != prediction_matrix[j][1][i]:
                    all_correct = False
                if first_value != prediction_matrix[j][1][i]:
                    all_wrong = False
        if not all_correct and not all_wrong:
            disagreeing_samples.append(i)
        elif not all_correct and all_wrong:
            all_wrong_samples.append(i)
            all_wrong_count += 1

    print("Slimming down to only those that were predicted differently...")
    label_list2 = []
    for i in range(0, len(disagreeing_samples)):
        label_list2.append(label_list[disagreeing_samples[i]])
    prediction_matrix2 = []
    for i in range(0, len(prediction_matrix)):
        prediction_matrix2.append((prediction_matrix[i][0], []))
        for j in range(0, len(disagreeing_samples)):
            prediction_matrix2[i][1].append(
                prediction_matrix[i][1][disagreeing_samples[j]])
    all_agreed_upon = len(prediction_matrix[0][1]) \
                      - len(disagreeing_samples)-all_wrong_count

    print("Writing ensemble combinations...")
    with open(output_file, 'w') as ofile:
        ofile.write(str(len(prediction_matrix[0][1]))+" ")
        ofile.write(str(all_agreed_upon)+" ")
        ofile.write(str(all_wrong_count)+" ")
        ofile.write(str(len(prediction_matrix))+"\n")
        for i in range(0, len(disagreeing_samples)):
            if i > 0:
                ofile.write(" ")
            ofile.write(str(disagreeing_samples[i]))
        ofile.write("\n")
        for i in range(0, len(label_list2)):
            if i > 0:
                ofile.write(" ")
            ofile.write(str(int(label_list2[i])))
        ofile.write("\n")
        for i in range(0, len(prediction_matrix2)):
            last = prediction_matrix2[i][0].rfind('\\')
            next_to_last = prediction_matrix2[i][0][:last].rfind('\\')
            ofile.write(prediction_matrix2[i][0][next_to_last+1:last])
            for j in range(0, len(prediction_matrix2[i][1])):
                if output_all_logits:
                    for k in range(0, len(prediction_matrix2[i][1][j])):
                        ofile.write(" ")
                        ofile.write(str(prediction_matrix2[i][1][j][k]))
                else:
                    ofile.write(" ")
                    ofile.write(str(prediction_matrix2[i][1][j]))
            ofile.write("\n")

    def copy_sample(sample_idx, sub_dir):
        dest_dir = os.path.join(image_output_dir, sub_dir)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        base_fn = os.path.join(images_dir, str(sample_idx))
        for src_fn in glob.iglob(f'{base_fn}(*).gif'):
            dest_fn = os.path.join(dest_dir, os.path.split(src_fn)[1])
            shutil.copy(src_fn, dest_fn)

    if output_images:
        print("Copying images not agreed upon to output dir...")
        for i in disagreeing_samples:
            copy_sample(i, "disagreeing")
        for i in all_wrong_samples:
            copy_sample(i, "all_wrong")


def go(data_dir, log_dir, output_file, image_output_dir, images_dir,
       output_all_logits, output_images, batch_size, merge_strategy,
       use_hvcs=True, hvc_type=1, hvc_dims=None, total_convolutions=None,
       branches_after=None):

    files = []
    for dirname, _, filenames in os.walk(log_dir):
        file = list(set([os.path.join(dirname,
            os.path.splitext(fn)[0]) for fn in filenames]))
        if len(file) > 0:
            files.append(file[0])

    prediction_matrix = []

    in_pipe = MNIST(data_dir, False, 1)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        print("Gathering evaluation digit labels...")

        label_list = np.empty((0, ))
        for _, ls in in_pipe. \
                get_validation_dataset(1).batch(batch_size):
            label_list = np.concatenate((label_list,
                            np.argmax(ls, axis=1)), axis=0)

        print("Building model...")

        model = SmallImageBranchingMerging(in_pipe.get_class_count(),
                    in_pipe.get_image_size(), in_pipe.get_image_channels(),
                    merge_strategy, use_hvcs, hvc_type, hvc_dims,
                    total_convolutions, branches_after, False)

        for weights_file in files:
            print("Restoring weights file: {}".format(weights_file))
            ckpt = tf.train.Checkpoint(
                    vars=model.get_all_savable_variables())
            ckpt.restore(weights_file).expect_partial()

            predictions = np.empty((0, in_pipe.get_class_count()))
            for fs, ls in in_pipe. \
                    get_validation_dataset(1).batch(batch_size):
                preds, _ = model.forward(fs, ls, False)
                predictions = np.concatenate((predictions, preds), axis=0)

            if output_all_logits:
                prediction_matrix.append((weights_file, predictions))
            else:
                prediction_matrix.append(
                    (weights_file, np.argmax(predictions, axis=1)))

        print("Graph evaluations complete.")

    print("Saving evaluation results...")
    write_to_file(output_file, images_dir, image_output_dir,
        prediction_matrix, label_list, output_all_logits, output_images)
    print("Finished.")


################################################################################
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=r"../../../../Datasets/mnist_data")
    p.add_argument("--log_dir", default=r"../../logs_ms1/best_top1")
    p.add_argument("--output_file",
        default=r"../../logs_ms1/ensemble_best_top1.txt")
    p.add_argument("--image_output_dir", default=r"../../logs_ms1/images")
    p.add_argument("--images_dir", default=r"../../MNIST_eval_digits")
    p.add_argument("--output_all_logits", default=False, type=bool)
    p.add_argument("--output_images", default=True, type=bool)
    p.add_argument("--batch_size", default=120, type=int)
    p.add_argument("--merge_strategy", default=1, type=float)
    p.add_argument("--use_hvcs", default=True, type=bool)
    p.add_argument("--hvc_type", default=2, type=int)
    p.add_argument("--hvc_dims", default=[64, 112, 160], type=int)
    p.add_argument("--total_convolutions", default=9, type=int)
    p.add_argument("--branches_after", default=[2, 5, 8])
    a = p.parse_args()

    go(data_dir=a.data_dir, log_dir=a.log_dir, output_file=a.output_file,
       image_output_dir=a.image_output_dir, images_dir=a.images_dir,
       output_all_logits=a.output_all_logits, output_images=a.output_images,
       batch_size=a.batch_size, merge_strategy=a.merge_strategy,
       use_hvcs=a.use_hvcs, hvc_type=a.hvc_type, hvc_dims=a.hvc_dims,
       total_convolutions=a.total_convolutions, branches_after=a.branches_after)
