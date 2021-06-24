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

#include <thread>
#include <vector>
#include <cstring>
#include <utility>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include "main.h"
#include "Misc.h"
#include "kernels.h"

// ReSharper disable once CppParameterMayBeConst
int main( int argc, char *argv[] )
{
    try
    {
        DeviceInfo di ;
        if( di.get_device_count() < 1 )
            throw std::runtime_error( "No usable GPUs found!" ) ;

        if( argc < 5 )
            throw std::runtime_error(
                "Usage: EnsembleFind {results_file} {majority_vote|product|sum}"
                    " {accuracy_reporting_threshold} {output_file_name}" ) ;

        const auto file_name = argv[ 1 ] ;
        const auto ensemble_method = argv[ 2 ] ;
        const auto accuracy_reporting_threshold = atoi( argv[ 3 ] ) ;
        const auto output_file_name = argv[ 4 ] ;

        std::vector<std::string> experiment_ids ;
        if(    ! std::strcmp( ensemble_method, "product" )
            || ! std::strcmp( ensemble_method, "sum" ) )
        {
            EnsembleData<float> ed ;
            if( ! EnsembleData<float>::get_ensemble_data_w_logits(
                file_name, ed, experiment_ids, CLASS_COUNT::value ) )
                throw std::runtime_error( "Error parsing ensemble data!" ) ;

            ProcessData<float> pd( experiment_ids, ed,
                accuracy_reporting_threshold, output_file_name ) ;

            if( ! strcmp( ensemble_method, "product" ) )
                execute( pd, get_product_func() ) ;
            else
                execute( pd, get_sum_func() ) ;
        }
        else if( ! strcmp( ensemble_method, "majority_vote" ) )
        {
            EnsembleData<int> ed ;
            if( ! EnsembleData<int>::get_ensemble_data(
                file_name, ed, experiment_ids, CLASS_COUNT::value ) )
                throw std::runtime_error( "Error parsing ensemble data!" ) ;

            ProcessData<int> pd( experiment_ids, ed,
                accuracy_reporting_threshold, output_file_name ) ;
            execute( pd, get_majority_vote_func() ) ;
        }
    }
    catch( const std::exception& exc )
    {
        std::cerr << exc.what() << std::endl ;
    }

    return cuda_close() ;
}

template<typename T>
void execute( ProcessData<T>& pd,
    combo_eval_kernel_wrapper<T> combi_kernel_method )
{
    copy_ensemble_data_to_gpus( pd ) ;
    allocate_buffers( pd ) ;
    iterate_combinations( pd, combi_kernel_method ) ;
    free_buffers( pd ) ;
    free_ensemble_data_on_gpus( pd ) ;
}

template<typename T>
void iterate_combinations( ProcessData<T>& pd,
    combo_eval_kernel_wrapper<T> combi_kernel_method )
{
    const auto n = pd.host_ensemble_data.experiment_count ;

    for( auto r = 3 ; r <= n ; ++r )
    {
        std::vector<int> permutation_mask( n ) ;
        std::fill( permutation_mask.begin(), permutation_mask.begin() + r, 1 ) ;
        auto s = get_start_time() ;

        const auto combi_count = nCr( n, r ) ;
        const auto batch_size = std::min( combi_count,
            static_cast<unsigned long long>( pd.max_batch_size ) ) ;
        const auto batches = combi_count / batch_size
            + ( combi_count % batch_size == 0 ? 0 : 1 ) ;

        std::cout << "Evaluating all " << combi_count <<
            " ensemble combinations of " << r << " models..." << std::endl ;

        const auto stride = n * sizeof( int ) ;
        auto more_combis_exist = true ;
        for( auto i = 0 ; i < batches && more_combis_exist ; ++i )
        {
            if( i > 0 )
            {
                more_combis_exist = std::prev_permutation(
                    permutation_mask.begin(), permutation_mask.end() ) ;
            }

            auto batch_combi_count = 0 ;
            while( more_combis_exist )
            {
                memcpy( &pd.host_input_buffer[ batch_combi_count * n ],
                    permutation_mask.data(), stride ) ;
                ++batch_combi_count ;
                if( batch_combi_count >= batch_size )
                    break ;
                more_combis_exist = std::prev_permutation(
                    permutation_mask.begin(), permutation_mask.end() ) ;
            }

            test_combinations( pd, batch_combi_count, combi_kernel_method ) ;
        }

        std::cout << "Complete after " << get_time_length( s )
            << " seconds." << std::endl ;
    }

    auto acc_map = pd.output_marshaller->get_accuracy_count_map() ;

    std::cout << std::endl << "Accuracy Counts" ;
    std::cout << std::endl << "=================" << std::endl ;
    for( auto c : acc_map )
        std::cout << c.first << " : " << c.second << std::endl ;
}

template<typename T>
void test_combinations( ProcessData<T>& pd, const int batch_combination_count,
    combo_eval_kernel_wrapper<T>& combi_kernel_method )
{
    auto cs = cudaSetDevice( pd.preferred_gpu ) ;
    cuda_err_throw( cs, "[test_combinations] cudaSetDevice(1)" ) ;

    cs = cudaMemcpy( pd.device_input_buffer,
        pd.host_input_buffer, pd.input_buffer_size, cudaMemcpyHostToDevice ) ;
    cuda_err_throw( cs, "[test_combinations] cudaMemcpy(1)" ) ;

    combi_kernel_method(
        pd.max_batch_size/pd.combinations_per_gpu_block,
        pd.combinations_per_gpu_block, pd.device_ensemble_data,
        pd.device_input_buffer, pd.device_output_buffer ) ;

    cs = cudaGetLastError() ;
    cuda_err_throw( cs, "[test_combinations] kernel launch" ) ;

    cs = cudaDeviceSynchronize() ;
    cuda_err_throw( cs, "[test_combinations] cudaDeviceSynchronize" ) ;

    cs = cudaMemcpy( pd.host_output_buffer, pd.device_output_buffer,
            pd.output_buffer_size, cudaMemcpyDeviceToHost ) ;
        cuda_err_throw( cs, "[test_combinations] cudaMemcpy(2)" ) ;

    OutputData<T> od ;
    od.batch_count = batch_combination_count ;
    od.experiment_ids = &pd.experiment_ids ;
    od.ensemble_data = &pd.host_ensemble_data ;
    od.input_buffer_size = pd.input_buffer_size ;
    od.output_buffer_size = pd.output_buffer_size ;
    od.output_accuracy_threshold = pd.output_accuracy_threshold ;
    od.input_buffer = static_cast<int*>( malloc( od.input_buffer_size ) ) ;
    od.output_buffer = static_cast<int*>( malloc( od.output_buffer_size ) ) ;
    memcpy( od.output_buffer, pd.host_output_buffer, pd.output_buffer_size ) ;
    memcpy( od.input_buffer, pd.host_input_buffer, pd.input_buffer_size ) ;
    pd.output_data_queue->push( std::move( od ) ) ;

    while( pd.output_data_queue->size() >= pd.max_output_queue_depth )
        std::this_thread::yield() ;
}
