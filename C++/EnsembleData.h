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

#ifndef ENSEMBLEDATA_H
#define ENSEMBLEDATA_H

#include <string>
#include <vector>
#include <fstream>

template<typename T>
class EnsembleData
{
    public:
        int    class_count             = 0       ;
        int    total_samples           = 0       ;
        int    all_correct             = 0       ;
        int    all_wrong               = 0       ;
        int    samples_to_check        = 0       ;
        int    experiment_count        = 0       ;
        int*   truth                   = nullptr ;
        T*     predictions             = nullptr ;
        size_t predictions_buffer_size = 0       ;

        EnsembleData()
        {
            _do_delete = false ;
        }

        EnsembleData( const EnsembleData& src )
        {
            copy( *this, src ) ;
        }

        EnsembleData& operator=( const EnsembleData& rhs )
        {
            copy( *this, rhs ) ;
            return *this ;
        }

        static bool get_ensemble_data( const std::string& file_name,
            EnsembleData<int>& ed, std::vector<std::string>& experiment_ids, const int class_count )
        {
            try
            {
                auto in_file = get_ensemble_data_common( file_name, ed, class_count ) ;

                ed.predictions_buffer_size = ed.experiment_count
                    * ed.samples_to_check * sizeof( int ) ;
                ed.predictions = new int[ ed.predictions_buffer_size ] ;

                for( auto i = 0 ; i < ed.experiment_count ; ++i )
                {
                    std::string eid ;
                    in_file >> eid ;
                    experiment_ids.push_back( eid ) ;

                    for( auto j = 0 ; j < ed.samples_to_check ; ++j )
                        in_file >> ed.predictions[ i * ed.samples_to_check + j ] ;
                }

                return true ;
            }
            catch( ... )
            {
                return false ;
            }
        }

        static bool get_ensemble_data_w_logits( const std::string& file_name,
            EnsembleData<float>& ed, std::vector<std::string>& experiment_ids, const int class_count )
        {
            try
            {
                auto in_file = get_ensemble_data_common( file_name, ed, class_count ) ;

                ed.predictions_buffer_size = ed.experiment_count
                    * ed.samples_to_check * class_count * sizeof( float ) ;
                ed.predictions = new float[ ed.predictions_buffer_size ] ;

                for( auto i = 0 ; i < ed.experiment_count ; ++i )
                {
                    std::string eid ;
                    in_file >> eid ;
                    experiment_ids.push_back( eid ) ;

                    for( auto j = 0 ; j < ed.samples_to_check * class_count ; ++j )
                        in_file >> ed.predictions[ i * ed.samples_to_check * class_count + j ] ;
                }

                return true ;
            }
            catch( ... )
            {
                return false ;
            }
        }

        ~EnsembleData()
        {
            if( ! _do_delete )
                return ;

            if( truth != nullptr )
                delete[] truth ;
            if( predictions != nullptr )
                delete[] predictions ;
        }

    private:
        bool _do_delete = false ;

        static std::ifstream get_ensemble_data_common(
            const std::string& file_name, EnsembleData& ed, const int class_count )
        {
            ed._do_delete = true ;
            ed.class_count = class_count ;

            std::ifstream in_file( file_name ) ;

            in_file >> ed.total_samples ;
            in_file >> ed.all_correct ;
            in_file >> ed.all_wrong ;
            in_file >> ed.experiment_count ;
            ed.samples_to_check = ed.total_samples - ed.all_correct - ed.all_wrong ;

            // The indices of the disagreeing samples are unimportant for this process
            auto dummy = 0 ;
            for( auto i = 0 ; i < ed.samples_to_check ; ++i )
                in_file >> dummy ;

            ed.truth = new int[ ed.samples_to_check ] ;

            for( auto i = 0 ; i < ed.samples_to_check ; ++i )
                in_file >> ed.truth[ i ] ;

            return in_file ;
        }

        static void copy( EnsembleData& dest, const EnsembleData& src )
        {
            dest._do_delete       = false                ;
            dest.total_samples    = src.total_samples    ;
            dest.all_correct      = src.all_correct      ;
            dest.all_wrong        = src.all_wrong        ;
            dest.samples_to_check = src.samples_to_check ;
            dest.experiment_count = src.experiment_count ;
            dest.truth            = src.truth            ;
            dest.predictions      = src.predictions      ;
        }
} ;

#endif // ENSEMBLEDATA_H
