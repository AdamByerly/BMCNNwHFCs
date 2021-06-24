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

#ifndef OUTPUTMARSHALLER_H
#define OUTPUTMARSHALLER_H

#include <map>
#include <atomic>
#include <string>
#include <thread>
#include <cstdint>
#include <utility>
#include "Outputter.h"
#include "OutputData.h"
#include "OutputRecord.h"
#include "ThreadSafeQueue.h"

template<typename T>
class OutputMarshaller
{
    public:
        explicit OutputMarshaller(
                ThreadSafeQueue<OutputData<T>>& output_data_queue, std::string output_file_name )
            :       _output_data_queue( output_data_queue )
                ,   _outputter( _output_record_queue, output_file_name )
        {
            auto started = false ;
            try
            {
                _done_signal = false ;
                _output_thread = std::thread(
                    &OutputMarshaller::worker_thread, this, &_done_signal ) ;
                started = true ;
            }
            catch( ... )
            {
                if( started )
                    _done_signal = true ;
                throw ;
            }
        }

        OutputMarshaller( OutputMarshaller const& ) = delete ;
        OutputMarshaller& operator=( OutputMarshaller const& ) = delete ;

        ~OutputMarshaller()
        {
            _done_signal = true ;
            try
            {
                if( _output_thread.joinable() )
                    _output_thread.join() ;
            }
            catch( ... )
            {
                //an acceptable race condition
                // that is not a problem if it happened
            }
        }

        const std::map<uint32_t, uint64_t>& get_accuracy_count_map() const
        {
            return _accuracy_count_map ;
        }

    private:
        std::thread                      _output_thread       ;
        std::atomic_bool                 _done_signal         ;
        ThreadSafeQueue<OutputData<T>>&  _output_data_queue   ;
        ThreadSafeQueue<OutputRecord<T>> _output_record_queue ;
        Outputter<T>                     _outputter           ;
        std::map<uint32_t, uint64_t>     _accuracy_count_map  ;

        void worker_thread( std::atomic_bool* me_done )
        {
            while( ! *me_done || _output_data_queue.size() > 0 )
            {
                OutputData<T> od ;
                if( _output_data_queue.pop( od ) )
                    handle_item( od ) ;
                else
                    std::this_thread::yield() ;
            }
        }

        void handle_item( OutputData<T>& od )
        {
            for( auto i = 0 ; i < od.batch_count ; ++i )
            {
                const auto acc = od.output_buffer[ i ] ;

                _accuracy_count_map[ acc ] = _accuracy_count_map[ acc ] + 1 ;

                if( acc >= od.output_accuracy_threshold )
                {
                    OutputRecord<T> orec ;
                    orec.experiment_ids = od.experiment_ids ;
                    orec.ensemble_data = od.ensemble_data ;
                    orec.accuracy = acc ;
                    orec.ensemble_mask = static_cast<int*>( malloc(
                        orec.ensemble_data->experiment_count * sizeof( int ) ) ) ;
                    memcpy( orec.ensemble_mask,
                        &od.input_buffer[ orec.ensemble_data->experiment_count * i ],
                        orec.ensemble_data->experiment_count * sizeof( int ) ) ;
                    _output_record_queue.push( std::move( orec ) ) ;
                }
            }
            free( od.input_buffer ) ;
            free( od.output_buffer ) ;
        }
} ;

#endif // OUTPUTMARSHALLER_H
