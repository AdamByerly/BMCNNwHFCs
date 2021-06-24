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

#include "kernels.h"
#include "device_launch_parameters.h"

combo_eval_kernel_wrapper<int> get_majority_vote_func()
{
    return [&] ( int grid_size, int block_size,
        const EnsembleData<int>* ensemble_data, const int* permutation_masks, int* results )
    {
        evaluate_combination_majority_vote<<<grid_size, block_size>>>(
            ensemble_data, permutation_masks, results ) ;
    } ;
}

combo_eval_kernel_wrapper<float> get_product_func()
{
    return [] ( int grid_size, int block_size,
        const EnsembleData<float>* ensemble_data, const int* permutation_masks, int* results )
    {
        evaluate_combination_with_logits_product<<<grid_size, block_size>>>(
            ensemble_data, permutation_masks, results ) ;
    } ;
}

combo_eval_kernel_wrapper<float> get_sum_func()
{
    return [] ( int grid_size, int block_size,
        const EnsembleData<float>* ensemble_data, const int* permutation_masks, int* results )
    {
        evaluate_combination_with_logits_sum<<<grid_size, block_size>>>(
            ensemble_data, permutation_masks, results ) ;
    } ;
}

//TODO: benchmark unrolling the CLASS_COUNT loops in the kernels

__global__ void evaluate_combination_majority_vote(
    const EnsembleData<int>* ensemble_data,
    const int* permutation_masks, int* results )
{
    const auto experiment_count = ensemble_data->experiment_count ;
    const auto samples_to_check = ensemble_data->samples_to_check ;
    const auto predictions      = ensemble_data->predictions ;
    const auto truth            = ensemble_data->truth ;

    const auto idx              = blockIdx.x * blockDim.x + threadIdx.x ;
    const auto this_combi       = &permutation_masks[ idx * experiment_count ] ;
    const auto this_output      = &results[ idx ] ;

    auto majority_correct = 0 ;
    for( auto i = 0 ; i < samples_to_check ; ++i )
    {
        const auto this_truth = truth[ i ] ;

        auto count_correct = 0 ;

        int counts_incorrect[ CLASS_COUNT::value ] ;
        for( auto j = 0 ; j < CLASS_COUNT::value ; ++j )
            counts_incorrect[ j ] = 0 ;

        for( auto j = 0 ; j < experiment_count ; ++j )
        {
            if( ! this_combi[ j ] )
                continue ;

            const auto this_one = predictions[ j * samples_to_check + i ] ;
            count_correct += this_one == this_truth ? 1 : 0 ;

            for( auto k = 0 ; k < CLASS_COUNT::value ; ++k )
                counts_incorrect[ k ] += this_one == k && this_truth != k ;
        }

        auto increment_majority_count = true ;
        for( auto j = 0 ; j < CLASS_COUNT::value ; ++j )
            increment_majority_count &= count_correct > counts_incorrect[ j ] ;

        if( increment_majority_count )
            ++majority_correct ;
    }

    *this_output = ensemble_data->all_correct + majority_correct ;
}

__global__ void evaluate_combination_with_logits_product(
    const EnsembleData<float>* ensemble_data,
    const int* permutation_masks, int* results )
{
    const auto experiment_count = ensemble_data->experiment_count ;
    const auto samples_to_check = ensemble_data->samples_to_check ;
    const auto predictions      = ensemble_data->predictions ;
    const auto truth            = ensemble_data->truth ;

    const auto idx              = blockIdx.x * blockDim.x + threadIdx.x ;
    const auto this_combi       = &permutation_masks[ idx * experiment_count ] ;
    const auto this_output      = &results[ idx ] ;

    auto highest_probability_correct = 0 ;
    for( auto i = 0 ; i < samples_to_check ; ++i )
    {
        const auto this_truth = truth[ i ] ;

        float logits_product[ CLASS_COUNT::value ] ;
        for( auto j = 0 ; j < CLASS_COUNT::value ; ++j )
            logits_product[ j ] = 1 ;

        for( auto j = 0 ; j < experiment_count ; ++j )
        {
            if( ! this_combi[ j ] )
                continue ;

            for( auto k = 0 ; k < CLASS_COUNT::value ; ++k )
            {
                logits_product[ k ] *= predictions[
                    j * samples_to_check * CLASS_COUNT::value
                        + i * CLASS_COUNT::value + k ] ;
            }
        }

        auto idx_of_max = 0 ;
        for( auto k = 0 ; k < CLASS_COUNT::value ; ++k )
            if( logits_product[ k ] > logits_product[ idx_of_max ] )
                idx_of_max = k ;

        if( idx_of_max == this_truth )
            ++highest_probability_correct ;
    }

    *this_output = ensemble_data->all_correct + highest_probability_correct ;
}

__global__ void evaluate_combination_with_logits_sum(
    const EnsembleData<float>* ensemble_data,
    const int* permutation_masks, int* results )
{
    const auto experiment_count = ensemble_data->experiment_count ;
    const auto samples_to_check = ensemble_data->samples_to_check ;
    const auto predictions      = ensemble_data->predictions ;
    const auto truth            = ensemble_data->truth ;

    const auto idx              = blockIdx.x * blockDim.x + threadIdx.x ;
    const auto this_combi       = &permutation_masks[ idx * experiment_count ] ;
    const auto this_output      = &results[ idx ] ;

    auto highest_probability_correct = 0 ;
    for( auto i = 0 ; i < samples_to_check ; ++i )
    {
        const auto this_truth = truth[ i ] ;

        float logits_product[ CLASS_COUNT::value ] ;
        for( auto j = 0 ; j < CLASS_COUNT::value ; ++j )
            logits_product[ j ] = 0 ;

        for( auto j = 0 ; j < experiment_count ; ++j )
        {
            if( ! this_combi[ j ] )
                continue ;

            for( auto k = 0 ; k < CLASS_COUNT::value ; ++k )
            {
                logits_product[ k ] += predictions[
                    j * samples_to_check * CLASS_COUNT::value
                        + i * CLASS_COUNT::value + k ] ;
            }
        }

        auto idx_of_max = 0 ;
        for( auto k = 0 ; k < CLASS_COUNT::value ; ++k )
            if( logits_product[ k ] > logits_product[ idx_of_max ] )
                idx_of_max = k ;

        if( idx_of_max == this_truth )
            ++highest_probability_correct ;
    }

    *this_output = ensemble_data->all_correct + highest_probability_correct ;
}
