/***************************************************************************
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
 *  @filename sycl_policy_handler.cpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_POLICY_HANDLER_CPP
#define SYCL_BLAS_POLICY_HANDLER_CPP
#include "operations/blas_constants.h"
// the templated methods
#include "policy/sycl_policy_handler.hpp"
namespace blas {

#define INSTANTIATE_TEMPLATE_METHODS(element_t)                                \
  template element_t *PolicyHandler<codeplay_policy>::allocate<element_t>(     \
      size_t num_elements) const;                                              \
  template void PolicyHandler<codeplay_policy>::deallocate<element_t>(         \
      element_t * p) const;                                                    \
  template BufferIterator<element_t, codeplay_policy>                          \
      PolicyHandler<codeplay_policy>::get_buffer<element_t>(element_t * ptr)   \
          const;                                                               \
  template BufferIterator<element_t, codeplay_policy>                          \
  PolicyHandler<codeplay_policy>::get_buffer<element_t>(                       \
      BufferIterator<element_t, codeplay_policy> buff) const;                  \
  template typename codeplay_policy::default_accessor_t<                       \
      typename ValueType<element_t>::type, cl::sycl::access::mode::read_write> \
      PolicyHandler<codeplay_policy>::get_range_access<                        \
          cl::sycl::access::mode::read_write, element_t>(element_t * vptr);    \
                                                                               \
  template typename codeplay_policy::default_accessor_t<                       \
      typename ValueType<element_t>::type, cl::sycl::access::mode::read_write> \
  PolicyHandler<codeplay_policy>::get_range_access<                            \
      element_t, cl::sycl::access::mode::read_write>(                          \
      BufferIterator<element_t, codeplay_policy> buff);                        \
  template typename codeplay_policy::event_t                                   \
  PolicyHandler<codeplay_policy>::copy_to_device<element_t>(                   \
      const element_t *src, element_t *dst, size_t size);                      \
                                                                               \
  template typename codeplay_policy::event_t                                   \
  PolicyHandler<codeplay_policy>::copy_to_device<element_t>(                   \
      const element_t *src, BufferIterator<element_t, codeplay_policy> dst,    \
      size_t size);                                                            \
  template typename codeplay_policy::event_t                                   \
  PolicyHandler<codeplay_policy>::copy_to_host<element_t>(                     \
      element_t * src, element_t * dst, size_t size);                          \
                                                                               \
  template typename codeplay_policy::event_t                                   \
  PolicyHandler<codeplay_policy>::fill<element_t>(                             \
      BufferIterator<element_t, codeplay_policy> buff, element_t value,        \
      size_t size);                                                            \
                                                                               \
  template typename codeplay_policy::event_t                                   \
  PolicyHandler<codeplay_policy>::copy_to_host<element_t>(                     \
      BufferIterator<element_t, codeplay_policy> src, element_t * dst,         \
      size_t size);                                                            \
  template ptrdiff_t PolicyHandler<codeplay_policy>::get_offset<element_t>(    \
      const element_t *ptr) const;                                             \
                                                                               \
  template ptrdiff_t PolicyHandler<codeplay_policy>::get_offset<element_t>(    \
      BufferIterator<element_t, codeplay_policy> ptr) const;

INSTANTIATE_TEMPLATE_METHODS(float)

#ifdef BLAS_DATA_TYPE_DOUBLE
INSTANTIATE_TEMPLATE_METHODS(double)
#endif  // BLAS_DATA_TYPE_DOUBLE

#ifdef BLAS_DATA_TYPE_HALF
INSTANTIATE_TEMPLATE_METHODS(cl::sycl::half)
#endif  // BLAS_DATA_TYPE_HALF

#define INSTANTIATE_TEMPLATE_METHODS_SPECIAL(ind, val)                        \
  template IndexValueTuple<ind, val>                                          \
      *PolicyHandler<codeplay_policy>::allocate<IndexValueTuple<ind, val>>(   \
          size_t num_elements) const;                                         \
  template void                                                               \
      PolicyHandler<codeplay_policy>::deallocate<IndexValueTuple<ind, val>>(  \
          IndexValueTuple<ind, val> * p) const;                               \
  template BufferIterator<IndexValueTuple<ind, val>, codeplay_policy>         \
      PolicyHandler<codeplay_policy>::get_buffer<IndexValueTuple<ind, val>>(  \
          IndexValueTuple<ind, val> * ptr) const;                             \
  template BufferIterator<IndexValueTuple<ind, val>, codeplay_policy>         \
  PolicyHandler<codeplay_policy>::get_buffer<IndexValueTuple<ind, val>>(      \
      BufferIterator<IndexValueTuple<ind, val>, codeplay_policy> buff) const; \
  template typename codeplay_policy::default_accessor_t<                      \
      typename ValueType<IndexValueTuple<ind, val>>::type,                    \
      cl::sycl::access::mode::read_write>                                     \
      PolicyHandler<codeplay_policy>::get_range_access<                       \
          cl::sycl::access::mode::read_write, IndexValueTuple<ind, val>>(     \
          IndexValueTuple<ind, val> * vptr);                                  \
                                                                              \
  template typename codeplay_policy::default_accessor_t<                      \
      typename ValueType<IndexValueTuple<ind, val>>::type,                    \
      cl::sycl::access::mode::read_write>                                     \
  PolicyHandler<codeplay_policy>::get_range_access<                           \
      IndexValueTuple<ind, val>, cl::sycl::access::mode::read_write>(         \
      BufferIterator<IndexValueTuple<ind, val>, codeplay_policy> buff);       \
  template typename codeplay_policy::event_t                                  \
  PolicyHandler<codeplay_policy>::copy_to_device<IndexValueTuple<ind, val>>(  \
      const IndexValueTuple<ind, val> *src, IndexValueTuple<ind, val> *dst,   \
      size_t size);                                                           \
                                                                              \
  template typename codeplay_policy::event_t                                  \
  PolicyHandler<codeplay_policy>::copy_to_device<IndexValueTuple<ind, val>>(  \
      const IndexValueTuple<ind, val> *src,                                   \
      BufferIterator<IndexValueTuple<ind, val>, codeplay_policy> dst,         \
      size_t size);                                                           \
  template typename codeplay_policy::event_t                                  \
  PolicyHandler<codeplay_policy>::copy_to_host<IndexValueTuple<ind, val>>(    \
      IndexValueTuple<ind, val> * src, IndexValueTuple<ind, val> * dst,       \
      size_t size);                                                           \
                                                                              \
  template typename codeplay_policy::event_t                                  \
  PolicyHandler<codeplay_policy>::copy_to_host<IndexValueTuple<ind, val>>(    \
      BufferIterator<IndexValueTuple<ind, val>, codeplay_policy> src,         \
      IndexValueTuple<ind, val> * dst, size_t size);                          \
  template ptrdiff_t                                                          \
  PolicyHandler<codeplay_policy>::get_offset<IndexValueTuple<ind, val>>(      \
      const IndexValueTuple<ind, val> *ptr) const;                            \
                                                                              \
  template ptrdiff_t                                                          \
  PolicyHandler<codeplay_policy>::get_offset<IndexValueTuple<ind, val>>(      \
      BufferIterator<IndexValueTuple<ind, val>, codeplay_policy> ptr) const;

INSTANTIATE_TEMPLATE_METHODS_SPECIAL(int, float)
INSTANTIATE_TEMPLATE_METHODS_SPECIAL(long, float)
INSTANTIATE_TEMPLATE_METHODS_SPECIAL(long long, float)

#ifdef BLAS_DATA_TYPE_DOUBLE
INSTANTIATE_TEMPLATE_METHODS_SPECIAL(int, double)
INSTANTIATE_TEMPLATE_METHODS_SPECIAL(long, double)
INSTANTIATE_TEMPLATE_METHODS_SPECIAL(long long, double)
#endif  // BLAS_DATA_TYPE_DOUBLE

#ifdef BLAS_DATA_TYPE_HALF
INSTANTIATE_TEMPLATE_METHODS_SPECIAL(int, cl::sycl::half)
INSTANTIATE_TEMPLATE_METHODS_SPECIAL(long, cl::sycl::half)
INSTANTIATE_TEMPLATE_METHODS_SPECIAL(long long, cl::sycl::half)
#endif  // BLAS_DATA_TYPE_HALF

#ifdef BLAS_ENABLE_CONST_INPUT
#define INSTANTIATE_CONST_TEMPLATE_METHODS(element_t)                        \
  template BufferIterator<element_t, codeplay_policy>                        \
      PolicyHandler<codeplay_policy>::get_buffer<element_t>(element_t * ptr) \
          const;                                                             \
  template BufferIterator<element_t, codeplay_policy>                        \
  PolicyHandler<codeplay_policy>::get_buffer<element_t>(                     \
      BufferIterator<element_t, codeplay_policy> buff) const;

INSTANTIATE_CONST_TEMPLATE_METHODS(float const)

#ifdef BLAS_DATA_TYPE_DOUBLE
INSTANTIATE_CONST_TEMPLATE_METHODS(double const)
#endif  // BLAS_DATA_TYPE_DOUBLE

#ifdef BLAS_DATA_TYPE_HALF
INSTANTIATE_CONST_TEMPLATE_METHODS(cl::sycl::half const)
#endif  // BLAS_DATA_TYPE_HALF
#endif // BLAS_ENABLE_CONST_INPUT

}  // namespace blas
#endif
