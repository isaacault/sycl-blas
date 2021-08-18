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
 *  @filename blas2_ger_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<int, int, scalar_t, int, int, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  int m;
  int n;
  int lda_mul;
  int incX;
  int incY;
  scalar_t alpha;
  std::tie(m, n, alpha, incX, incY, lda_mul) = combi;
  int lda = m * lda_mul;

#ifdef SYCL_BLAS_USE_USM
  using data_t = scalar_t;
#else
  using data_t = utils::data_storage_t<scalar_t>;
#endif

  // Input matrix
  std::vector<data_t> a_v(m * incX);
  // Input Vector
  std::vector<data_t> b_v(n * incY);
  // output Vector
  std::vector<data_t> c_m_gpu_result(lda * n, scalar_t(10));
  // output system vector
  std::vector<data_t> c_m_cpu(lda * n, scalar_t(10));
  fill_random(a_v);
  fill_random(b_v);

  // SYSTEM GER
  reference_blas::ger(m, n, static_cast<data_t>(alpha), a_v.data(), incX,
                      b_v.data(), incY, c_m_cpu.data(), lda);

  auto q = make_queue();
  test_executor_t ex(q);

#ifdef SYCL_BLAS_USE_USM
  data_t* v_a_gpu = cl::sycl::malloc_device<data_t>(m * incX, q);
  data_t* v_b_gpu = cl::sycl::malloc_device<data_t>(n * incY, q);
  data_t* m_c_gpu = cl::sycl::malloc_device<data_t>(lda * n, q);

  q.memcpy(v_a_gpu, a_v.data(), sizeof(data_t) * m * incX).wait();
  q.memcpy(v_b_gpu, b_v.data(), sizeof(data_t) * n * incY).wait();
  q.memcpy(m_c_gpu, c_m_gpu_result.data(), sizeof(data_t) * lda * n).wait();
#else
  auto v_a_gpu = utils::make_quantized_buffer<scalar_t>(ex, a_v);
  auto v_b_gpu = utils::make_quantized_buffer<scalar_t>(ex, b_v);
  auto m_c_gpu = utils::make_quantized_buffer<scalar_t>(ex, c_m_gpu_result);
#endif

  // SYCLger
  auto ev = _ger(ex, m, n, alpha, v_a_gpu, incX, v_b_gpu, incY, m_c_gpu, lda);
#ifdef SYCL_BLAS_USE_USM
  ex.get_policy_handler().wait(ev);
#endif

  auto event =
#ifdef SYCL_BLAS_USE_USM
      q.memcpy(c_m_gpu_result.data(), m_c_gpu, sizeof(data_t) * lda * n);
#else
      utils::quantized_copy_to_host<scalar_t>(ex, m_c_gpu, c_m_gpu_result);
#endif
  ex.get_policy_handler().wait({event});

  const bool isAlmostEqual =
      utils::compare_vectors<data_t, scalar_t>(c_m_gpu_result, c_m_cpu);
  ASSERT_TRUE(isAlmostEqual);

#ifdef SYCL_BLAS_USE_USM
  cl::sycl::free(v_a_gpu, q);
  cl::sycl::free(v_b_gpu, q);
  cl::sycl::free(m_c_gpu, q);
#endif
}

#ifdef STRESS_TESTING
const auto combi =
    ::testing::Combine(::testing::Values(11, 65, 255, 1023, 1024 * 1024),  // m
                       ::testing::Values(14, 63, 257, 1010, 1024 * 1024),  // n
                       ::testing::Values(0.0, 1.0, 1.5),  // alpha
                       ::testing::Values(1, 2),           // incX
                       ::testing::Values(1, 3),           // incY
                       ::testing::Values(1, 2)            // lda_mul
    );
#else
// For the purpose of travis and other slower platforms, we need a faster test
const auto combi = ::testing::Combine(::testing::Values(1023),  // m
                                      ::testing::Values(14),  // n
                                      ::testing::Values(0.0),  // alpha
                                      ::testing::Values(2),         // incX
                                      ::testing::Values(3),         // incY
                                      ::testing::Values(2)          // lda_mul
);
#endif

BLAS_REGISTER_TEST(Ger, combination_t, combi);
