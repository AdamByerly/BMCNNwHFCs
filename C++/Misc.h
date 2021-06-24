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

#ifndef MISC_H
#define MISC_H

#include <chrono>
#include <limits>
#include <stdexcept>

inline std::chrono::time_point<std::chrono::high_resolution_clock> get_start_time()
{
    return std::chrono::high_resolution_clock::now() ;
}

inline double get_time_length( std::chrono::time_point<std::chrono::high_resolution_clock>& start_time )
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - start_time ).count() / 1000.0 ;
}

inline unsigned long long gcd( unsigned long long x, unsigned long long y )
{
    while( y != 0 )
    {
        const auto t = x % y ;
        x = y ;
        y = t ;
    }
    return x ;
}

inline unsigned long long nCr( unsigned long long n, const unsigned long long r )
{
    if( r > n )
        throw std::invalid_argument( "invalid argument supplied to nCr" ) ;

    unsigned long long k = 1 ;
    for( unsigned long long d = 1 ; d <= r ; ++d, --n )
    {
        const auto g = gcd( k, d ) ;
        k /= g ;
        const auto t = n / ( d / g ) ;
        if( k > std::numeric_limits<unsigned long long>::max() / t )
            throw std::overflow_error( "overflow in nCr" ) ;
        k *= t ;
    }
    return k ;
}

#endif // MISC_H
