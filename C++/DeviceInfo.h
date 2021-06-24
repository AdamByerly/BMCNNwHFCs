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

#ifndef DEVICEINFO_H
#define DEVICEINFO_H

#include <tuple>
#include <vector>
#include <stdexcept>
#include "cuda_runtime_api.h"

class DeviceInfo
{
    private:
        int                                  _device_count        ;
        std::vector<size_t>                  _total_global_memory ;
        std::vector<std::tuple<int,int,int>> _max_grid_size       ;
        std::vector<std::tuple<int,int,int>> _max_block_size      ;

    public:
        DeviceInfo()
        {
            cudaGetDeviceCount( &_device_count ) ;
            for( auto i = 0 ; i < _device_count ; ++i )
            {
                cudaDeviceProp prop ;
                cudaGetDeviceProperties( &prop, i ) ;
                _total_global_memory.push_back( prop.totalGlobalMem ) ;
                _max_grid_size.push_back( std::tuple<int,int,int>( prop.maxGridSize[ 0 ],
                    prop.maxGridSize[ 1 ], prop.maxGridSize[ 2 ] ) ) ;
                _max_block_size.push_back( std::tuple<int, int, int>( prop.maxThreadsDim[ 0 ],
                    prop.maxThreadsDim[ 1 ], prop.maxThreadsDim[ 2 ] ) ) ;
            }
        }

        int get_device_count() const { return _device_count ; }

        size_t get_total_global_memory( const int device_idx ) const
        {
            return _total_global_memory[ device_idx ] ;
        }

        int get_device_max_grid_size( const int device_idx, const int dimension ) const
        {
            if( dimension == 0 )
                return std::get<0>( _max_grid_size[ device_idx ] ) ;
            if( dimension == 1 )
                return std::get<1>( _max_grid_size[ device_idx ] ) ;
            if( dimension == 2 )
                return std::get<2>( _max_grid_size[ device_idx ] ) ;
            throw std::runtime_error( "Invalid dimension requested!" ) ;
        }

        int get_device_max_block_size( const int device_idx, const int dimension ) const
        {
            if( dimension == 0 )
                return std::get<0>( _max_block_size[ device_idx ] ) ;
            if( dimension == 1 )
                return std::get<1>( _max_block_size[ device_idx ] ) ;
            if( dimension == 2 )
                return std::get<2>( _max_block_size[ device_idx ] ) ;
            throw std::runtime_error( "Invalid dimension requested!" ) ;
        }
} ;

#endif // DEVICEINFO_H
