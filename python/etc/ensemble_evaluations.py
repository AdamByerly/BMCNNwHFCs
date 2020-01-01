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
import glob
import shutil
import argparse
import numpy as np
from python.training.train import make_tower
from python.training.gpu_utils import make_towers
from python.training.input_pipeline import InputPipeline
import tensorflow as tf
tf1 = tf.compat.v1


def get_labels(classes, sess, labels_op,
        is_training_ph, validation_data, data_iterator):
    labels = np.empty((0, classes))
    sess.run(data_iterator.make_initializer(validation_data))
    try:
        while True:
            ls = sess.run([labels_op], feed_dict={is_training_ph: False})
            ls = np.squeeze(ls)
            labels = np.concatenate((labels, ls), axis=0)
    except tf.errors.OutOfRangeError:
        pass
    return labels


def evaluate(sess, classes, preds_op,
        is_training_ph, validation_data, data_iterator):
    predictions = np.empty((0, classes))
    sess.run(data_iterator.make_initializer(validation_data))
    try:
        while True:
            ps = sess.run([preds_op], feed_dict={is_training_ph: False})
            ps = np.squeeze(ps)
            predictions = np.concatenate((predictions, ps), axis=0)
    except tf.errors.OutOfRangeError:
        pass
    return predictions


def write_to_file(output_dir, images_dir,
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
    file_name = os.path.join(output_dir, 'ensemble_data.txt')
    with open(file_name, 'w') as output_file:
        output_file.write(str(len(prediction_matrix[0][1]))+" ")
        output_file.write(str(all_agreed_upon)+" ")
        output_file.write(str(all_wrong_count)+" ")
        output_file.write(str(len(prediction_matrix))+"\n")
        for i in range(0, len(disagreeing_samples)):
            if i > 0:
                output_file.write(" ")
            output_file.write(str(disagreeing_samples[i]))
        output_file.write("\n")
        for i in range(0, len(label_list2)):
            if i > 0:
                output_file.write(" ")
            output_file.write(str(label_list2[i]))
        output_file.write("\n")
        for i in range(0, len(prediction_matrix2)):
            last = prediction_matrix2[i][0].rfind('\\')
            next_to_last = prediction_matrix2[i][0][:last].rfind('\\')
            output_file.write(prediction_matrix2[i][0][next_to_last+1:last])
            for j in range(0, len(prediction_matrix2[i][1])):
                if output_all_logits:
                    for k in range(0, len(prediction_matrix2[i][1][j])):
                        output_file.write(" ")
                        output_file.write(str(prediction_matrix2[i][1][j][k]))
                else:
                    output_file.write(" ")
                    output_file.write(str(prediction_matrix2[i][1][j]))
            output_file.write("\n")

    def copy_sample(sample_idx, sub_dir):
        dest_dir = os.path.join(output_dir, sub_dir)
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


def go(batch_size, gpus, log_dir, output_dir, data_dir,
        images_dir, output_all_logits, output_images):
    tf1.disable_eager_execution()
    tf1.disable_control_flow_v2()
    tf1.logging.set_verbosity(tf1.logging.ERROR)
    tf1.reset_default_graph()

    ############################################################################
    # Data feeds
    ############################################################################
    print("Setting up data feeds...")
    input_pipe = InputPipeline(data_dir)
    classes = input_pipe.get_class_count()
    eval_data = input_pipe.get_validation_dataset(batch_size)

    ############################################################################
    # Tensorflow Graph
    ############################################################################
    print("Building Graph Operations...")
    with tf.device("/device:CPU:0"):  # set the default device to the CPU
        with tf.name_scope("input/placeholders"):
            is_training = tf1.placeholder(dtype=bool, name="is_training")

        with tf.name_scope("input/data_itetator"):
            data_iterator = tf1.data.Iterator.from_structure(
                tf1.data.get_output_types(eval_data),
                tf1.data.get_output_shapes(eval_data))
            features, labels = data_iterator.get_next()

        preds, logits, loss, labels, grads = make_towers(
            None, features, labels, gpus, make_tower,
            **{"is_training": is_training, "count_classes": classes})

    files = []
    for dirname, _, filenames in os.walk(log_dir):
        file = list(set([os.path.join(dirname,
            os.path.splitext(fn)[0]) for fn in filenames]))
        if len(file) > 0:
            files.append(file[0])

    prediction_matrix = []

    ############################################################################
    # Tensorflow session
    ############################################################################
    print("Starting Session...")
    with tf1.Session(config=tf1.ConfigProto(allow_soft_placement=True)) as sess:
        label_list = get_labels(classes, sess,
            labels, is_training, eval_data, data_iterator)
        label_list = np.argmax(label_list, axis=1)

        for weights_file in files:
            print("Restoring weights file: {}".format(weights_file))
            tf1.train.Saver().restore(sess, weights_file)

            predictions = evaluate(sess, classes,
                preds, is_training, eval_data, data_iterator)

            if output_all_logits:
                prediction_matrix.append((weights_file, predictions))
            else:
                prediction_matrix.append(
                    (weights_file, np.argmax(predictions, axis=1)))

        print("Graph evaluations complete.")
        sess.close()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving evaluation results...")
    write_to_file(output_dir, images_dir, prediction_matrix,
        label_list, output_all_logits, output_images)
    print("Finished.")


################################################################################
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", default=120, type=int)
    p.add_argument("--gpus", default=1, type=int)
    p.add_argument("--output_all_logits", default=False, type=bool)
    p.add_argument("--output_images", default=True, type=bool)
    p.add_argument("--data_dir", default=r"..\..\data\mnist_data")
    p.add_argument("--images_dir", default=r"..\..\data\images\all")
    p.add_argument("--log_dir",
        default=r"..\..\data\weights\learnable_ones_init")
    p.add_argument("--output_dir", default=r"..\..\data\learnable_ones_init")
    a = p.parse_args()

    go(batch_size=a.batch_size, gpus=a.gpus, log_dir=a.log_dir,
        output_dir=a.output_dir, data_dir=a.data_dir, images_dir=a.images_dir,
        output_all_logits=a.output_all_logits, output_images=a.output_images)
