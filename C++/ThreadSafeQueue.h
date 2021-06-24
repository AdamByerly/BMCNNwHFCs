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

#ifndef THREADSAFEQUEUE_H
#define THREADSAFEQUEUE_H

#include <mutex>
#include <queue>
#include <thread>
#include <utility>

template<typename T>
class ThreadSafeQueue final
{
    private:
        mutable std::mutex    _sync_object ;
                std::queue<T> _queue       ;

    public:
        ThreadSafeQueue() {}
        ThreadSafeQueue( ThreadSafeQueue const& ) = delete ;
        ThreadSafeQueue& operator=( ThreadSafeQueue const& ) = delete ;

        size_t size()
        {
            std::lock_guard<std::mutex> lock( _sync_object ) ;
            return _queue.size() ;
        }

        void push( T to_push )
        {
            std::lock_guard<std::mutex> lock( _sync_object ) ;
            _queue.push( std::move( to_push ) ) ;
        }

        bool pop( T& popped )
        {
            std::unique_lock<std::mutex> lock( _sync_object, std::defer_lock_t() ) ;
            while( ! lock.try_lock() )
                std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) ) ;

            if( _queue.empty() )
                return false ;

            popped = std::move( _queue.front() ) ;
            _queue.pop() ;
            return true ;
        }
} ;

#endif // THREADSAFEQUEUE_H
