/* Copyright 2021 Adam Byerly. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 ( the "License" );
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ===========================================================================*/

#ifndef KERNELS_H
#define KERNELS_H

#include "cuda_runtime.h"
#include "type_defs.h"

combo_eval_kernel_wrapper<int> get_majority_vote_func() ;
combo_eval_kernel_wrapper<float> get_product_func() ;
combo_eval_kernel_wrapper<float> get_sum_func() ;

__global__ void evaluate_combination_majority_vote(
    const EnsembleData<int>* ensemble_data,
    const int* permutation_masks, int* results ) ;

__global__ void evaluate_combination_with_logits_product(
    const EnsembleData<float>* ensemble_data,
    const int* permutation_masks, int* results ) ;

__global__ void evaluate_combination_with_logits_sum(
    const EnsembleData<float>* ensemble_data,
    const int* permutation_masks, int* results ) ;

#endif