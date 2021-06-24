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

from datetime import datetime
from python.train import go

# Merge Strategy 0 (equal branch weights, not learnable)
for i in range(32):
    go(run_name=datetime.now().strftime("%Y%m%d%H%M%S"), end_epoch=300,
       data_dir=r"../../../Datasets/mnist_data", input_pipeline=1,
       log_dir="../logs_ms0", batch_size=120, merge_strategy=0, loss_type=1,
       use_hvcs=True, hvc_type=2, hvc_dims=[64, 112, 160],
       use_augmentation=True, augmentation_type=1, total_convolutions=9,
       branches_after=[2, 5, 8], reconstruct_from_hvcs=True)

# Merge Strategy 1 (learnable branch weights, initialized randomly)
for i in range(32):
    go(run_name=datetime.now().strftime("%Y%m%d%H%M%S"), end_epoch=300,
       data_dir=r"../../../Datasets/mnist_data", input_pipeline=1,
       log_dir="../logs_ms1", batch_size=120, merge_strategy=1, loss_type=1,
       use_hvcs=True, hvc_type=2, hvc_dims=[64, 112, 160],
       use_augmentation=True, augmentation_type=1, total_convolutions=9,
       branches_after=[2, 5, 8], reconstruct_from_hvcs=True)

# Merge Strategy 2 (learnable branch weights, initialized to one)
for i in range(32):
    go(run_name=datetime.now().strftime("%Y%m%d%H%M%S"), end_epoch=300,
       data_dir=r"../../../Datasets/mnist_data", input_pipeline=1,
       log_dir="../logs_ms2", batch_size=120, merge_strategy=2, loss_type=1,
       use_hvcs=True, hvc_type=2, hvc_dims=[64, 112, 160],
       use_augmentation=True, augmentation_type=1, total_convolutions=9,
       branches_after=[2, 5, 8], reconstruct_from_hvcs=True)

# MarginPlusMeanSquaredError Loss
for i in range(10):
    go(run_name=datetime.now().strftime("%Y%m%d%H%M%S"), end_epoch=300,
       data_dir=r"../../../Datasets/mnist_data", input_pipeline=1,
       log_dir="../logs_lta", batch_size=120, merge_strategy=2, loss_type=3,
       use_hvcs=True, hvc_type=2, hvc_dims=[64, 112, 160],
       use_augmentation=True, augmentation_type=1, total_convolutions=9,
       branches_after=[2, 5, 8], reconstruct_from_hvcs=True)

# CategoricalCrossEntropyPlusMeanSquaredError Loss
for i in range(10):
    go(run_name=datetime.now().strftime("%Y%m%d%H%M%S"), end_epoch=300,
       data_dir=r"../../../Datasets/mnist_data", input_pipeline=1,
       log_dir="../logs_lta", batch_size=120, merge_strategy=2, loss_type=4,
       use_hvcs=True, hvc_type=2, hvc_dims=[64, 112, 160],
       use_augmentation=True, augmentation_type=1, total_convolutions=9,
       branches_after=[2, 5, 8], reconstruct_from_hvcs=True)

# No extra branches
for i in range(10):
    go(run_name=datetime.now().strftime("%Y%m%d%H%M%S"), end_epoch=300,
       data_dir=r"../../../Datasets/mnist_data", input_pipeline=1,
       log_dir="../logs_naa", batch_size=120, merge_strategy=2, loss_type=1,
       use_hvcs=True, hvc_type=2, hvc_dims=[160],
       use_augmentation=True, augmentation_type=1, total_convolutions=9,
       branches_after=[8], reconstruct_from_hvcs=True)

# No HVCs or extra branches
for i in range(10):
    go(run_name=datetime.now().strftime("%Y%m%d%H%M%S"), end_epoch=300,
       data_dir=r"../../../Datasets/mnist_data", input_pipeline=1,
       log_dir="../logs_naa", batch_size=120, merge_strategy=2, loss_type=1,
       use_hvcs=False, hvc_type=2, hvc_dims=[160],
       use_augmentation=True, augmentation_type=1, total_convolutions=9,
       branches_after=[8], reconstruct_from_hvcs=False)

# No HVCs
for i in range(10):
    go(run_name=datetime.now().strftime("%Y%m%d%H%M%S"), end_epoch=300,
       data_dir=r"../../../Datasets/mnist_data", input_pipeline=1,
       log_dir="../logs_naa", batch_size=120, merge_strategy=2, loss_type=1,
       use_hvcs=False, hvc_type=2, hvc_dims=[64, 112, 160],
       use_augmentation=True, augmentation_type=1, total_convolutions=9,
       branches_after=[2, 5, 8], reconstruct_from_hvcs=False)

# Random shifting (full margin) augmentation only
for i in range(10):
    go(run_name=datetime.now().strftime("%Y%m%d%H%M%S"), end_epoch=300,
       data_dir=r"../../../Datasets/mnist_data", input_pipeline=1,
       log_dir="../logs_daa", batch_size=120, merge_strategy=2, loss_type=1,
       use_hvcs=True, hvc_type=2, hvc_dims=[64, 112, 160],
       use_augmentation=True, augmentation_type=2, total_convolutions=9,
       branches_after=[2, 5, 8], reconstruct_from_hvcs=False)

# Random shifting (max 2) augmentation only
for i in range(10):
    go(run_name=datetime.now().strftime("%Y%m%d%H%M%S"), end_epoch=300,
       data_dir=r"../../../Datasets/mnist_data", input_pipeline=1,
       log_dir="../logs_daa", batch_size=120, merge_strategy=2, loss_type=1,
       use_hvcs=True, hvc_type=2, hvc_dims=[64, 112, 160],
       use_augmentation=True, augmentation_type=3, total_convolutions=9,
       branches_after=[2, 5, 8], reconstruct_from_hvcs=False)

# Fashion-MNIST - No HVCs or extra branches
for i in range(10):
    go(run_name=datetime.now().strftime("%Y%m%d%H%M%S"), end_epoch=300,
       data_dir=r"../../../Datasets/fashion_mnist_data", input_pipeline=1,
       log_dir="../logs_fshn", batch_size=120, merge_strategy=2, loss_type=1,
       use_hvcs=False, hvc_type=2, hvc_dims=[160],
       use_augmentation=True, augmentation_type=1, total_convolutions=9,
       branches_after=[8], reconstruct_from_hvcs=False)

# Fashion-MNIST
for i in range(10):
    go(run_name=datetime.now().strftime("%Y%m%d%H%M%S"), end_epoch=300,
       data_dir=r"../../../Datasets/fashion_mnist_data", input_pipeline=1,
       log_dir="../logs_fshn", batch_size=120, merge_strategy=2, loss_type=1,
       use_hvcs=True, hvc_type=2, hvc_dims=[64, 112, 160],
       use_augmentation=True, augmentation_type=1, total_convolutions=9,
       branches_after=[2, 5, 8], reconstruct_from_hvcs=False)

# Cifar10 - No HVCs or extra branches
for i in range(10):
    go(run_name=datetime.now().strftime("%Y%m%d%H%M%S"), end_epoch=300,
       data_dir=r"../../../Datasets/cifar10_data", input_pipeline=3,
       log_dir="../logs_c10", batch_size=120, merge_strategy=2, loss_type=1,
       use_hvcs=False, hvc_type=2, hvc_dims=[160],
       use_augmentation=True, augmentation_type=0, total_convolutions=9,
       branches_after=[8], reconstruct_from_hvcs=False)

# Cifar10
for i in range(10):
    go(run_name=datetime.now().strftime("%Y%m%d%H%M%S"), end_epoch=300,
       data_dir=r"../../../Datasets/cifar10_data", input_pipeline=3,
       log_dir="../logs_c10", batch_size=120, merge_strategy=2, loss_type=1,
       use_hvcs=True, hvc_type=2, hvc_dims=[64, 112, 160],
       use_augmentation=True, augmentation_type=0, total_convolutions=9,
       branches_after=[2, 5, 8], reconstruct_from_hvcs=False)

# Cifar100 - No HVCs or extra branches
for i in range(10):
    go(run_name=datetime.now().strftime("%Y%m%d%H%M%S"), end_epoch=300,
       data_dir=r"../../../Datasets/cifar100_data", input_pipeline=4,
       log_dir="../logs_c100", batch_size=120, merge_strategy=2, loss_type=1,
       use_hvcs=False, hvc_type=2, hvc_dims=[160],
       use_augmentation=True, augmentation_type=0, total_convolutions=9,
       branches_after=[8], reconstruct_from_hvcs=False)

# Cifar100
for i in range(10):
    go(run_name=datetime.now().strftime("%Y%m%d%H%M%S"), end_epoch=300,
       data_dir=r"../../../Datasets/cifar100_data", input_pipeline=4,
       log_dir="../logs_c100", batch_size=120, merge_strategy=2, loss_type=1,
       use_hvcs=True, hvc_type=2, hvc_dims=[64, 112, 160],
       use_augmentation=True, augmentation_type=0, total_convolutions=9,
       branches_after=[2, 5, 8], reconstruct_from_hvcs=False)
