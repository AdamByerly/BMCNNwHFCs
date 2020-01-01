/* Copyright 2019 Adam Byerly. All Rights Reserved.
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

#include <atomic>
#include <thread>
#include "Outputter.h"
#include "OutputData.h"
#include "OutputRecord.h"
#include "ThreadSafeQueue.h"

template<typename T>
class OutputMarshaller
{
    public:
        explicit OutputMarshaller(
                ThreadSafeQueue<OutputData<T>>& output_data_queue )
            :       _output_data_queue( output_data_queue )
                ,   _outputter( _output_record_queue )
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

    private:
        std::thread                      _output_thread       ;
        std::atomic_bool                 _done_signal         ;
        ThreadSafeQueue<OutputData<T>>&  _output_data_queue   ;
        ThreadSafeQueue<OutputRecord<T>> _output_record_queue ;
        Outputter<T>                     _outputter           ;

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
                if( acc >= od.output_accuracy_threshold )
                {
                    OutputRecord<T> or ;
                    or.experiment_ids = od.experiment_ids ;
                    or.ensemble_data = od.ensemble_data ;
                    or.accuracy = acc ;
                    or.ensemble_mask = static_cast<int*>( malloc(
                        or.ensemble_data->experiment_count * sizeof( int ) ) ) ;
                    memcpy( or.ensemble_mask,
                        &od.input_buffer[ or .ensemble_data->experiment_count * i ],
                        or.ensemble_data->experiment_count * sizeof( int ) ) ;
                    _output_record_queue.push( std::move( or ) ) ;
                }
            }
            free( od.input_buffer ) ;
            free( od.output_buffer ) ;
        }
} ;

#endif // OUTPUTMARSHALLER_H
