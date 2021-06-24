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

#ifndef PROCESSDATA_H
#define PROCESSDATA_H

#include <string>
#include <vector>
#include <memory>
#include "OutputData.h"
#include "EnsembleData.h"
#include "ThreadSafeQueue.h"
#include "OutputMarshaller.h"

template<typename T>
struct ProcessData
{
    // ReSharper disable once CppPossiblyUninitializedMember
    // ReSharper disable CppParameterMayBeConst
    ProcessData( const std::vector<std::string>& experiment_ids,
            EnsembleData<T>& host_ensemble_data, int output_accuracy_threshold,
            std::string output_file_name, int preferred_gpu = 0, int combinations_per_gpu_block = 256,
            int max_batch_size = 1048576, int max_output_queue_depth = 1024)
        :       output_data_queue( std::make_unique<ThreadSafeQueue<OutputData<T>>>() )
            ,   output_marshaller( std::make_unique<OutputMarshaller<T>>( *output_data_queue, output_file_name ) )
            ,   experiment_ids( experiment_ids )
            ,   host_ensemble_data( host_ensemble_data )
            ,   output_accuracy_threshold( output_accuracy_threshold )
            ,   preferred_gpu( preferred_gpu )
            ,   combinations_per_gpu_block( combinations_per_gpu_block )
            ,   max_batch_size( max_batch_size )
            ,   max_output_queue_depth( max_output_queue_depth )
    {}
    // ReSharper restore CppParameterMayBeConst

    ProcessData( ProcessData const& ) = delete ;
    ProcessData& operator=( ProcessData const& ) = delete ;

    std::unique_ptr<ThreadSafeQueue<OutputData<T>>> output_data_queue          ;
    std::unique_ptr<OutputMarshaller<T>>            output_marshaller          ;
    const std::vector<std::string>&                 experiment_ids             ;
    EnsembleData<T>&                                host_ensemble_data         ;
    EnsembleData<T>*                                device_ensemble_data       ;
    int*                                            host_input_buffer          ;
    int*                                            host_output_buffer         ;
    int*                                            device_input_buffer        ;
    int*                                            device_output_buffer       ;
    int                                             input_buffer_size          ;
    int                                             output_buffer_size         ;
    int                                             max_batch_size             ;
    int                                             max_output_queue_depth     ;
    int                                             combinations_per_gpu_block ;
    int                                             preferred_gpu              ;
    int                                             output_accuracy_threshold  ;
    std::string                                     output_file_name           ;
} ;

#endif // PROCESSDATA_H
