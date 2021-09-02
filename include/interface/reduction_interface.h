/***************************************************************************
 *
 *  @license
 *  Copyright (C) Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCL-BLAS: BLAS implementation using SYCL
 *
 *  @filename reduction_interface.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_REDUCTION_INTERFACE_H
#define SYCL_BLAS_REDUCTION_INTERFACE_H

#include "executors/executor.h"
#include "operations/extension_trees.h"

namespace blas {

namespace internal {

template <typename operator_t, typename element_t, 
          typename executor_t, typename input_t, typename output_t, 
          typename index_t>
typename executor_t::policy_t::event_t _reduction(executor_t& ex, 
                                                  input_t buffer_in, 
                                                  output_t buffer_out,
                                                  index_t rows, index_t cols, int const ClSize, int const WgSize);

} // namespace internal

template <typename operator_t, typename element_t, 
          typename executor_t, typename input_t, typename output_t, 
          typename index_t>
typename executor_t::policy_t::event_t _reduction(executor_t& ex, 
                                                  input_t buffer_in, 
                                                  output_t buffer_out,
                                                  index_t rows, index_t cols, int const ClSize, int const WgSize) {
#ifdef SYCL_BLAS_USE_USM
  return internal::_reduction<operator_t, element_t>(ex, buffer_in, buffer_out, rows, rows, ClSize, WgSize);
#else
  return internal::_reduction<operator_t, element_t>(ex, ex.get_policy_handler().get_buffer(buffer_in),
                            ex.get_policy_handler().get_buffer(buffer_out),
                            rows, rows, ClSize, WgSize);
#endif
}

}  // namespace blas

#endif  // SYCL_BLAS_REDUCTION_INTERFACE_H
