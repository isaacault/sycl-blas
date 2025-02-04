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
 *  @filename blas_operators.h
 *
 **************************************************************************/

// NO H for this one as this one is internal. but all the macro will be
// generated by cmake in cpp file
#ifndef SYCL_BLAS_OPERATORS_H
#define SYCL_BLAS_OPERATORS_H

namespace blas {
struct Operators;

// A template for getting the return type of a blas operator
// This is special cased for the CollapseIndex operator, which returns a
// different type than its input
template <typename operator_t, typename rhs_t>
struct ResolveReturnType {
  using type = rhs_t;
};

struct CollapseIndexTupleOperator;
template <typename rhs_t>
struct ResolveReturnType<CollapseIndexTupleOperator, rhs_t> {
  using type = typename rhs_t::value_t;
};

struct AddOperator;
struct ProductOperator;
struct DivisionOperator;
struct MaxOperator;
struct MinOperator;
struct AbsoluteAddOperator;

}  // namespace blas

#endif
