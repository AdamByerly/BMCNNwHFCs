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

#ifndef MAIN_H
#define MAIN_H

#include <string>
#include <iostream>
#include <stdexcept>
#include "type_defs.h"
#include "DeviceInfo.h"
#include "ProcessData.h"
#include "EnsembleData.h"
#include "OutputMarshaller.h"

template<typename T>
void execute( ProcessData<T>& pd,
    combo_eval_kernel_wrapper<T> combi_kernel_method ) ;

template<typename T>
void iterate_combinations( ProcessData<T>& pd,
    combo_eval_kernel_wrapper<T> combi_kernel_method ) ;

template<typename T>
void test_combinations( ProcessData<T>& pd, const int batch_combination_count,
    combo_eval_kernel_wrapper<T>& combi_kernel_method ) ;

[[gsl::suppress( 26812, justification: "The warning is coming from"
    " CUDA code provided by NVIDIA that we don't want to change." )]]
inline void cuda_err_throw( const cudaError_t status, const char* msg )
{
    if( status != cudaError::cudaSuccess )
        throw std::runtime_error( ( std::string( msg )
            + std::string( " failed: " )
            + cudaGetErrorString( status ) ).c_str() ) ;
}

inline int cuda_close()
{
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    const auto cs = cudaDeviceReset() ;
    if( cs != cudaSuccess )
    {
        std::cerr << "cudaDeviceReset failed: "
            << cudaGetErrorString( cs ) << std::endl ;
        return -1 ;
    }

    return 0 ;
}

template<typename T>
void copy_ensemble_data_to_gpus( ProcessData<T>& pd )
{
    //We need a copy of the host structure on the host such that its pointers
    // have addresses that are on the device.  This is so that we can actually
    // reference those pointers and copy from the host pointers into them.
    auto edc = pd.host_ensemble_data ;

    //Now copy from the host structure pointers into the device structure
    // pointers.  After this is done, we'll have a copy of the host structure
    // on the host which has pointers that are addresses on the device.
    auto cs = cudaSetDevice( pd.preferred_gpu ) ;
    cuda_err_throw( cs, "[copy_ensemble_data_to_gpus] cudaSetDevice(1)" ) ;

    cs = cudaMalloc( reinterpret_cast<void**>( &edc.truth ),
        pd.host_ensemble_data.samples_to_check * sizeof( int ) ) ;
    cuda_err_throw( cs, "[copy_ensemble_data_to_gpus] cudaMalloc(1)" ) ;

    cs = cudaMemcpy( edc.truth, pd.host_ensemble_data.truth,
        pd.host_ensemble_data.samples_to_check * sizeof( int ),
        cudaMemcpyHostToDevice ) ;
    cuda_err_throw( cs, "[copy_ensemble_data_to_gpus] cudaMemcpy(1)" ) ;

    cs = cudaMalloc( reinterpret_cast<void**>( &edc.predictions ),
        pd.host_ensemble_data.predictions_buffer_size ) ;
    cuda_err_throw( cs, "[copy_ensemble_data_to_gpus] cudaMalloc(2)" ) ;

    cs = cudaMemcpy( edc.predictions, pd.host_ensemble_data.predictions,
        pd.host_ensemble_data.predictions_buffer_size,
        cudaMemcpyHostToDevice ) ;
    cuda_err_throw( cs, "[copy_ensemble_data_to_gpus] cudaMemcpy(2)" ) ;

    //Now we copy the structure that is on the host,
    // but has pointers whose addresses are on the device to the device.
    cs = cudaMalloc( reinterpret_cast<void**>(
        &pd.device_ensemble_data ), sizeof( EnsembleData<int> ) ) ;
    cuda_err_throw( cs, "[copy_ensemble_data_to_gpus] cudaMalloc(3)" ) ;

    cs = cudaMemcpy( pd.device_ensemble_data, &edc,
        sizeof( EnsembleData<int> ), cudaMemcpyHostToDevice ) ;
    cuda_err_throw( cs, "[copy_ensemble_data_to_gpus] cudaMemcpy(3)" ) ;
}

template<typename T>
void free_ensemble_data_on_gpus( ProcessData<T>& pd )
{
    auto cs = cudaSetDevice( pd.preferred_gpu ) ;
    cuda_err_throw( cs, "[free_ensemble_data_on_gpus] cudaSetDevice" ) ;

    //Copy the structure from the GPU (which has pointers to addresses
    // on the GPU) to the host so that we have those pointers, and then
    // can go on to deallocate their memory.
    EnsembleData<int> edc ;
    cs = cudaMemcpy( &edc, pd.device_ensemble_data,
        sizeof( EnsembleData<int> ), cudaMemcpyDeviceToHost ) ;
    cuda_err_throw( cs, "[free_ensemble_data_on_gpus] cudaMemcpy(1)" ) ;

    cs = cudaFree( edc.truth ) ;
    cuda_err_throw( cs, "[free_ensemble_data_on_gpus] cudaFree(1)" ) ;

    cs = cudaFree( edc.predictions ) ;
    cuda_err_throw( cs, "[free_ensemble_data_on_gpus] cudaFree(2)" ) ;

    cs = cudaFree( pd.device_ensemble_data ) ;
    cuda_err_throw( cs, "[free_ensemble_data_on_gpus] cudaFree(3)" ) ;
}

template<typename T>
void allocate_buffers( ProcessData<T>& pd )
{
    pd.input_buffer_size = pd.max_batch_size
        * pd.host_ensemble_data.experiment_count * sizeof( int ) ;
    pd.output_buffer_size = pd.max_batch_size * sizeof( int ) ;

    pd.host_input_buffer = static_cast<int*>(
        malloc( pd.input_buffer_size ) ) ;
    pd.host_output_buffer = static_cast<int*>(
        malloc( pd.output_buffer_size ) ) ;

    auto cs = cudaSetDevice( pd.preferred_gpu ) ;
    cuda_err_throw( cs, "[allocate_buffers] cudaSetDevice" ) ;

    cs = cudaMalloc( reinterpret_cast<void**>(
        &pd.device_input_buffer ), pd.input_buffer_size ) ;
    cuda_err_throw( cs, "[allocate_buffers] cudaMalloc(1)" ) ;

    cs = cudaMalloc( reinterpret_cast<void**>(
        &pd.device_output_buffer ), pd.output_buffer_size ) ;
    cuda_err_throw( cs, "[allocate_buffers] cudaMalloc(2)" ) ;
}

template<typename T>
void free_buffers( ProcessData<T>& pd )
{
    auto cs = cudaSetDevice( pd.preferred_gpu ) ;
    cuda_err_throw( cs, "[free_buffers] cudaSetDevice" ) ;

    cs = cudaFree( pd.device_input_buffer ) ;
    cuda_err_throw( cs, "[free_buffers] cudaFree(1)" ) ;

    cs = cudaFree( pd.device_output_buffer ) ;
    cuda_err_throw( cs, "[free_buffers] cudaFree(2)" ) ;

    free( pd.host_output_buffer ) ;
    free( pd.host_input_buffer ) ;
}

#endif