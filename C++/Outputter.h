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

#ifndef OUTPUTTER_H
#define OUTPUTTER_H

#include <atomic>
#include <thread>
#include <sstream>
#include <iostream>
#include "OutputRecord.h"
#include "ThreadSafeQueue.h"

template<typename T>
class Outputter
{
    public:
        explicit Outputter( ThreadSafeQueue<OutputRecord<T>>& output_queue )
            : _output_queue( output_queue )
        {
            auto started = false ;
            try
            {
                _done_signal = false ;
                _output_thread = std::thread(
                    &Outputter::worker_thread, this, &_done_signal ) ;
                started = true ;
            }
            catch( ... )
            {
                if( started )
                    _done_signal = true ;
                throw ;
            }
        }

        Outputter( Outputter const& ) = delete ;
        Outputter& operator=( Outputter const& ) = delete ;

        ~Outputter()
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
        std::thread                       _output_thread ;
        std::atomic_bool                  _done_signal   ;
        ThreadSafeQueue<OutputRecord<T>>& _output_queue  ;

        void worker_thread( std::atomic_bool* me_done ) const
        {
            while( ! *me_done || _output_queue.size() > 0 )
            {
                OutputRecord<T> or ;
                if( _output_queue.pop( or ) )
                    handle_item( or ) ;
                else
                    std::this_thread::yield() ;
            }
        }

        static void handle_item( OutputRecord<T>& or )
        {
            std::stringstream ss ;
            ss << or.accuracy << ": " ;
            for( auto i = 0 ; i < or.ensemble_data->experiment_count ; ++i )
                if( or.ensemble_mask[ i ] )
                    ss << ( *or.experiment_ids )[ i ] << "; " ;

            ss << std::endl ;
            std::cout << ss.str() ;
            //TODO: Do something with the output data
            free( or.ensemble_mask ) ;
        }
} ;

#endif // OUTPUTTER_H
