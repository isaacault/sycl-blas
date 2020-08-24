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
 *  @filename arm_gpu.hpp
 *
 **************************************************************************/
#ifndef SYCL_BLAS_GEMM_ARM_GPU_BACKEND_HPP
#define SYCL_BLAS_GEMM_ARM_GPU_BACKEND_HPP
#include "interface/gemm_launcher.h"

namespace blas {
namespace gemm {
namespace backend {
template <bool _t_a, bool _t_b, bool is_beta_zero, typename executor_t,
          typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename executor_t::policy_t::event_t _gemm(
    executor_t& ex, index_t _M, index_t _N, index_t _K, element_t _alpha,
    container_0_t _a, index_t _lda, container_1_t _b, index_t _ldb,
    element_t _beta, container_2_t _c, index_t _ldc, index_t batch_size,
    gemm_batch_type_t batch_type) {
  if (batch_type == gemm_batch_type_t::interleaved) {
    return blas::Gemm_Launcher<
        64, false, false, false, 64, Tile<2, 2, 4, 4, 1, 1, 4, 4>, _t_a, _t_b,
        static_cast<int>(gemm_memory_t::no_local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 4,
        static_cast<int>(
            gemm_batch_type_t::interleaved)>::template _select_gemm(ex, _M, _N,
                                                                    _K, _alpha,
                                                                    _a, _lda,
                                                                    _b, _ldb,
                                                                    _beta, _c,
                                                                    _ldc,
                                                                    batch_size);
  } else if (_M == 128 && _K == 128 && _N == 784 && batch_size == 36) {
    // M: 128, K: 128, N: 784, Batch: 36
    if (_t_b) {
      // RHS transpose
      // Gemm <0, 0, 0, 0, Tile<8, 4, 4, 8, 1, 1, 1, 1>,
      // floatgemm_memory:no_local, gemm_algorithm:standard,
      // gemm_vectorization:full, vector size4, batch_type:strided>
      return blas::Gemm_Launcher<
          64, false, false, false, 64, Tile<8, 4, 4, 8>, _t_a, _t_b,
          static_cast<int>(gemm_memory_t::no_local),
          static_cast<int>(gemm_algorithm_t::standard),
          static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 4,
          static_cast<int>(
              gemm_batch_type_t::strided)>::template _select_gemm(ex, _M, _N,
                                                                  _K, _alpha,
                                                                  _a, _lda, _b,
                                                                  _ldb, _beta,
                                                                  _c, _ldc,
                                                                  batch_size);
    } else {
      // No transpose
      // Gemm <0, 0, 0, 0, Tile<4, 4, 4, 4, 1, 1, 1, 1>,
      // floatgemm_memory:no_local, gemm_algorithm:standard,
      // gemm_vectorization:partial, vector size8, batch_type:strided>
      return blas::Gemm_Launcher<
          64, false, false, false, 64, Tile<4, 4, 4, 4>, _t_a, _t_b,
          static_cast<int>(gemm_memory_t::no_local),
          static_cast<int>(gemm_algorithm_t::standard),
          static_cast<int>(gemm_vectorization_t::partial), is_beta_zero, 8,
          static_cast<int>(
              gemm_batch_type_t::strided)>::template _select_gemm(ex, _M, _N,
                                                                  _K, _alpha,
                                                                  _a, _lda, _b,
                                                                  _ldb, _beta,
                                                                  _c, _ldc,
                                                                  batch_size);
    }
  } else if (batch_size == 1) {
    if (_M == 1024 && _K == 512 && _N == 3136) {
      // M: 1024 K: 512, N: 3136, Batch: 1
      // Gemm <0, 0, 0, 0, Tile<4, 8, 8, 4, 1, 1, 1, 1>,
      // floatgemm_memory:no_local, gemm_algorithm:standard,
      // gemm_vectorization:partial, vector size2, batch_type:strided>
      return blas::Gemm_Launcher<
          64, false, false, false, 64, Tile<4, 8, 8, 4>, _t_a, _t_b,
          static_cast<int>(gemm_memory_t::no_local),
          static_cast<int>(gemm_algorithm_t::standard),
          static_cast<int>(gemm_vectorization_t::partial), is_beta_zero, 2,
          static_cast<int>(
              gemm_batch_type_t::strided)>::template _select_gemm(ex, _M, _N,
                                                                  _K, _alpha,
                                                                  _a, _lda, _b,
                                                                  _ldb, _beta,
                                                                  _c, _ldc,
                                                                  batch_size);
    } else if ((_M == 256 && _K == 2304 && _N == 784) ||
               (_M == 64 && _K == 576 && _N == 12544)) {
      // M: 64, K: 576, N: 12544, Batch: 1
      // M: 256, K: 2304, N: 784, Batch: 1
      // Gemm <0, 0, 0, 0, Tile<8, 4, 4, 8, 1, 1, 1, 1>,
      // floatgemm_memory:no_local, gemm_algorithm:standard,
      // gemm_vectorization:partial, vector size8, batch_type:strided>
      return blas::Gemm_Launcher<
          64, false, false, false, 64, Tile<8, 4, 4, 8>, _t_a, _t_b,
          static_cast<int>(gemm_memory_t::no_local),
          static_cast<int>(gemm_algorithm_t::standard),
          static_cast<int>(gemm_vectorization_t::partial), is_beta_zero, 8,
          static_cast<int>(
              gemm_batch_type_t::strided)>::template _select_gemm(ex, _M, _N,
                                                                  _K, _alpha,
                                                                  _a, _lda, _b,
                                                                  _ldb, _beta,
                                                                  _c, _ldc,
                                                                  batch_size);
    } else if (_N == 784 &&
               ((_M == 2048 && _K == 1024) || (_M == 512 && _K == 1024) ||
                (_M == 2048 && _K == 512))) {
      // M: 512, K: 1024, N: 784, Batch: 1
      // M: 2048, K: 1024, N: 784, Batch: 1
      // M: 2048, K: 512, N: 784, Batch: 1
      // Gemm <0, 0, 0, 0, Tile<4, 4, 4, 4, 1, 1, 1, 1>,
      // floatgemm_memory:no_local, gemm_algorithm:standard,
      // gemm_vectorization:full, vector size1, batch_type:strided>
      return blas::Gemm_Launcher<
          64, false, false, false, 64, Tile<4, 4, 4, 4>, _t_a, _t_b,
          static_cast<int>(gemm_memory_t::no_local),
          static_cast<int>(gemm_algorithm_t::standard),
          static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 1,
          static_cast<int>(
              gemm_batch_type_t::strided)>::template _select_gemm(ex, _M, _N,
                                                                  _K, _alpha,
                                                                  _a, _lda, _b,
                                                                  _ldb, _beta,
                                                                  _c, _ldc,
                                                                  batch_size);
    } else if (_M == 512 && _K == 2048 && _N == 784) {
      // M: 512, K: 2048, N: 784, Batch: 1
      // Gemm <0, 0, 0, 0, Tile<4, 4, 4, 4, 1, 1, 1, 1>,
      // floatgemm_memory:no_local, gemm_algorithm:standard,
      // gemm_vectorization:full, vector size2, batch_type:strided>
      return blas::Gemm_Launcher<
          64, false, false, false, 64, Tile<4, 4, 4, 4>, _t_a, _t_b,
          static_cast<int>(gemm_memory_t::no_local),
          static_cast<int>(gemm_algorithm_t::standard),
          static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 2,
          static_cast<int>(
              gemm_batch_type_t::strided)>::template _select_gemm(ex, _M, _N,
                                                                  _K, _alpha,
                                                                  _a, _lda, _b,
                                                                  _ldb, _beta,
                                                                  _c, _ldc,
                                                                  batch_size);
    }
  } else if (batch_size > 1 && !_t_a) {
    /* Tends to perform well for Winograd sizes (i.e. batched) */
    return blas::Gemm_Launcher<
        64, false, false, false, 64, Tile<4, 4, 8, 8>, _t_a, _t_b,
        static_cast<int>(gemm_memory_t::no_local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::partial), is_beta_zero, 2,
        static_cast<int>(
            gemm_batch_type_t::strided)>::template _select_gemm(ex, _M, _N, _K,
                                                                _alpha, _a,
                                                                _lda, _b, _ldb,
                                                                _beta, _c, _ldc,
                                                                batch_size);
  } else if (!_t_a) {
    /* Does well on most im2col or 1x1 convolutions, or is within 10% of
     * best kernel. */
    return blas::Gemm_Launcher<
        64, false, false, false, 64, Tile<4, 4, 4, 4>, _t_a, _t_b,
        static_cast<int>(gemm_memory_t::no_local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::partial), is_beta_zero, 2,
        static_cast<int>(
            gemm_batch_type_t::strided)>::template _select_gemm(ex, _M, _N, _K,
                                                                _alpha, _a,
                                                                _lda, _b, _ldb,
                                                                _beta, _c, _ldc,
                                                                batch_size);
  } else {
    return blas::Gemm_Launcher<
        128, false, false, false, 64, Tile<4, 8, 16, 8>, _t_a, _t_b,
        static_cast<int>(gemm_memory_t::no_local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::partial), is_beta_zero, 4,
        static_cast<int>(
            gemm_batch_type_t::strided)>::template _select_gemm(ex, _M, _N, _K,
                                                                _alpha, _a,
                                                                _lda, _b, _ldb,
                                                                _beta, _c, _ldc,
                                                                batch_size);
  }
}  // namespace backend
}  // namespace backend
}  // namespace gemm
}  // namespace blas
#endif
