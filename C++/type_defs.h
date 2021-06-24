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

#ifndef TYPE_DEFS_H
#define TYPE_DEFS_H

#include <functional>
#include <type_traits>
#include "EnsembleData.h"

typedef std::integral_constant<int, 10> CLASS_COUNT ;

template<typename T>
using combo_eval_kernel_wrapper
    = std::function<void( int, int, const EnsembleData<T>*, const int*, int* )> ;

#endif
