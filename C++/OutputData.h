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

#ifndef OUTPUTDATA_H
#define OUTPUTDATA_H

#include <string>
#include <vector>
#include "EnsembleData.h"

template<typename T>
struct OutputData
{
    // ReSharper disable once CppPossiblyUninitializedMember
    OutputData() {}

    OutputData( OutputData const& ) = delete ;
    OutputData& operator=( OutputData const& ) = delete ;

    OutputData( OutputData&& ) = default ;
    OutputData& operator=( OutputData&& ) = default ;

    const std::vector<std::string>* experiment_ids            ;
    const EnsembleData<T>*          ensemble_data             ;
    int*                            input_buffer              ;
    int*                            output_buffer             ;
    int                             input_buffer_size         ;
    int                             output_buffer_size        ;
    int                             batch_count               ;
    int                             output_accuracy_threshold ;
} ;

#endif // OUTPUTDATA_H
