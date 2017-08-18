/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2016 Codeplay Software Limited
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
 *  @filename blas1_trees.hpp
 *
 **************************************************************************/

#ifndef BLAS1_TREES_HPP
#define BLAS1_TREES_HPP

#include <stdexcept>
#include <vector>

#include <operations/blas_operators.hpp>
#include <views/view_sycl.hpp>

namespace blas {
namespace internal {

/*! DetectScalar.
 * @brief Class specialization used to detect scalar values in ScalarOp nodes.
 * When the value is not an integral basic type,
 * it is assumed to be a vector and the first value
 * is used.
 */
template <typename T>
struct DetectScalar {
  static typename T::value_type get_scalar(T &opSCL) { return opSCL.eval(0); }
};

/*! DetectScalar.
 * @brief See Detect Scalar.
 */
template <>
struct DetectScalar<int> {
  using T = int;
  static T get_scalar(T &scalar) { return scalar; }
};

/*! DetectScalar.
 * @brief See Detect Scalar.
 */
template <>
struct DetectScalar<float> {
  using T = float;
  static T get_scalar(T &scalar) { return scalar; }
};

/*! DetectScalar.
 * @brief See Detect Scalar.
*/
template <>
struct DetectScalar<double> {
  using T = double;
  static T get_scalar(T &scalar) { return scalar; }
};

/*! DetectScalar.
 * @brief See Detect Scalar.
*/
template <>
struct DetectScalar<std::complex<float>> {
  using T = std::complex<float>;
  static T get_scalar(T &scalar) { return scalar; }
};

/*! DetectScalar.
 * @brief See Detect Scalar.
*/
template <>
struct DetectScalar<std::complex<double>> {
  using T = std::complex<double>;
  static T get_scalar(T &scalar) { return scalar; }
};

/*! get_scalar.
 * @brief Template autodecuction function for DetectScalar.
*/
template <typename T>
auto get_scalar(T &scl) -> decltype(DetectScalar<T>::get_scalar(scl)) {
  return DetectScalar<T>::get_scalar(scl);
}
}  // namespace internal

/** Join.
 * @brief Joins both sides of the expression in the single kernel.
 */
template <class LHS, class RHS>
struct Join {
  LHS l;
  RHS r;

  using value_type = typename RHS::value_type;

  Join(LHS &_l, RHS _r) : l(_l), r(_r){};

  // PROBLEM: Only the RHS size is considered. If LHS size is different??
  // If it is smaller, eval function will crash
  size_t getSize() { return r.getSize(); }

//  inline __attribute__((always_inline))
  bool valid_thread (cl::sycl::nd_item<1> ndItem) {
    return ((ndItem.get_global(0) < getSize()));
  }

  value_type eval(size_t i) {
    l.eval(i);
    return r.eval(i);
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
};

/** Assign.
 */
template <class LHS, class RHS>
struct Assign {
  LHS l;
  RHS r;

  using value_type = typename RHS::value_type;

  Assign(LHS &_l, RHS _r) : l(_l), r(_r){};

  // PROBLEM: Only the RHS size is considered. If LHS size is different??
  // If it is smaller, eval function will crash
  size_t getSize() { return r.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) {
    return ((ndItem.get_global(0) < getSize()));
  }
  value_type eval(size_t i) {
    auto val = l.eval(i) = r.eval(i);
    return val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
};

/*! DoubleAssign.
 */
template <class LHS1, class LHS2, class RHS1, class RHS2>
struct DobleAssign {
  LHS1 l1;
  LHS2 l2;
  RHS1 r1;
  RHS2 r2;

 public:
  using value_type = typename RHS1::value_type;

  DobleAssign(LHS1 &_l1, LHS2 &_l2, RHS1 _r1, RHS2 _r2)
      : l1(_l1), l2(_l2), r1(_r1), r2(_r2){};

  // PROBLEM: Only the RHS size is considered. If LHS size is different??
  // If it is smaller, eval function will crash
  size_t getSize() { return r2.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) {
    return ((ndItem.get_global(0) < getSize()));
  }
  value_type eval(size_t i) {
    auto val1 = r1.eval(i);
    auto val2 = r2.eval(i);
    l1.eval(i) = val1;
    l2.eval(i) = val2;
    return val1;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
};

/*!ScalarOp.
 * @brief Implements an scalar operation.
 * (e.g alpha OP x, with alpha scalar and x vector)
 */
template <typename Operator, typename SCL, typename RHS>
struct ScalarOp {
  using value_type = typename RHS::value_type;
  SCL scl;
  RHS r;

  ScalarOp(SCL _scl, RHS &_r) : scl(_scl), r(_r){};

  size_t getSize() { return r.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) {
    return ((ndItem.get_global(0) < getSize()));
  }
  value_type eval(size_t i) {
    return Operator::eval(internal::get_scalar(scl), r.eval(i));
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
};

/*! UnaryOp.
 * Implements a Unary Operation ( Op(z), e.g. z++), with z a vector.
 */
template <typename Operator, typename RHS>
struct UnaryOp {
  using value_type = typename RHS::value_type;
  RHS r;

  UnaryOp(RHS &_r) : r(_r){};

  size_t getSize() { return r.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) {
    return ((ndItem.get_global(0) < getSize()));
  }
  value_type eval(size_t i) { return Operator::eval(r.eval(i)); }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
};

/*! BinaryOp.
 * @brief Implements a Binary Operation (x OP z) with x and z vectors.
 */
template <typename Operator, typename LHS, typename RHS>
struct BinaryOp {
  using value_type = typename RHS::value_type;
  LHS l;
  RHS r;

  BinaryOp(LHS &_l, RHS &_r) : l(_l), r(_r){};

  // PROBLEM: Only the RHS size is considered. If LHS size is different??
  // If it is smaller, eval function will crash
  size_t getSize() { return r.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) {
    return ((ndItem.get_global(0) < getSize()));
  }
  value_type eval(size_t i) { return Operator::eval(l.eval(i), r.eval(i)); }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
};

/*! TupleOp.
 * @brief Implements a Tuple Operation (map (\x -> [i, x]) vector).
 */
template <typename RHS>
struct TupleOp {
  using value_type = IndVal<typename RHS::value_type>;
  RHS r;

  TupleOp(RHS &_r) : r(_r) {}

  size_t getSize() { return r.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) {
    return ((ndItem.get_global(0) < getSize()));
  }
  value_type eval(size_t i) { return value_type(i, r.eval(i)); }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
};

/*! AssignReduction.
 * @brief Implements the reduction operation for assignments (in the form y = x)
 *  with y a scalar and x a subexpression tree.
 */
// #define TRACE_ERROR 1
// #define TWO_LOCAL_ADDS
#ifdef TWO_LOCAL_ADDS
  #define NUM_LOCAL_ADDS 2
#else
  #define NUM_LOCAL_ADDS 1
#endif
template <typename Operator, unsigned int interLoop, class LHS, class RHS>
struct AssignReduction {
  using value_type = typename RHS::value_type;
  LHS l;
  RHS r;
  size_t blqS;  // block  size
  size_t grdS;  // grid  size
  static const size_t intLoop = interLoop;

  AssignReduction(LHS &_l, RHS &_r, size_t _blqS, size_t _grdS)
#ifdef TRACE_ERROR
    : l(_l), r(_r), blqS(_blqS), grdS(_grdS){ printf ("ZZZ %lu %lu\n", blqS, grdS);};
#else
    : l(_l), r(_r), blqS(_blqS), grdS(_grdS){ };
#endif

  size_t getSize() { return r.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(size_t i) {
    size_t vecS = r.getSize();
    size_t frs_thrd = NUM_LOCAL_ADDS * blqS * i;
    size_t lst_thrd = ((frs_thrd + blqS) > vecS) ? vecS : (frs_thrd + blqS);
    // Reduction across the grid
#ifdef TRACE_ERROR
    printf ("YYY %lu %lu\n", blqS, grdS);
#endif
    value_type val = Operator::init(r);
    for (size_t j = frs_thrd; j < lst_thrd; j++) {
      value_type local_val = Operator::init(r);
      for (size_t k = j; k < vecS; k += NUM_LOCAL_ADDS * grdS) {
        local_val = Operator::eval(local_val, r.eval(k));
#ifdef TWO_LOCAL_ADDS
        if (k + blqS < vecS) {
          local_val = Operator::eval(local_val, r.eval(k + blqS));
        }
#endif
      }
      // Reduction inside the block
      val = Operator::eval(val, local_val);
    }
    return l.eval(i) = val;
  }
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t glbalSz = ndItem.get_num_groups(0) * localSz;

    size_t vecS = r.getSize();

//    if (groupid == 0) printf ("blqS = %lu , localSz = %lu\n", blqS, localSz);
//    if (groupid == 0) printf ("blqS = %lu , grdS = %lu , vecS = %lu\n", blqS, grdS, vecS);
//    printf ("Hola %lu \n", localid);
    // Reduction across the grid
    value_type val = Operator::init(r);
//    for (size_t k = frs_thrd; k < vecS; k += NUM_LOCAL_ADDS * grdS) {
    if (interLoop == 1) {
      size_t frs_thrd = NUM_LOCAL_ADDS * groupid * localSz + localid;
      for (size_t k = frs_thrd; k < vecS; k += NUM_LOCAL_ADDS * glbalSz) {
  //      printf ("localid = %lu , k = %lu , num = %f\n", localid, k, r.eval(k));
  //        printf ("YES!!!!\n");
        val = Operator::eval(val, r.eval(k));
  #ifdef TWO_LOCAL_ADDS
  //    if ((k + blqS < vecS)) {
      if ((k + localSz < vecS)) {
  //      val = Operator::eval(val, r.eval(k + blqS));
        val = Operator::eval(val, r.eval(k + localSz));
        }
  #endif
      }
    } else {
//      printf ("YES!!!!\n");
      size_t frs_thrd = interLoop * (groupid * localSz + localid);
      for (size_t k = frs_thrd; k < vecS; k += interLoop * glbalSz) {
        for (size_t k_int=k; k_int<std::min(k+interLoop,vecS);k_int++) {
//        printf ("localid = %lu , k = %lu , num = %f\n", localid, k_int, r.eval(k));
          val = Operator::eval(val, r.eval(k_int));
        }
      }
    }
#ifdef TRACE_ERROR
//    if (blqS == grdS) printf ("Hola %lu -> %20.10f\n", localid, val);
//    if (groupid == 0) printf ("XXX %lu %lu %lu\n", groupid, blqS, grdS);
    if (groupid == 0) printf ("XXX %lu %lu %lu\n", groupid, blqS, glbalSz);
#endif

    scratch[localid] = val;
    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Reduction inside the block
    for (size_t offset = localSz >> 1; offset > 0; offset >>= 1) {
      if (localid < offset) {
        scratch[localid] =
            Operator::eval(scratch[localid], scratch[localid + offset]);
      }
      // This barrier is mandatory to be sure the data are on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
    }
#ifdef TRACE_ERROR
//    if (groupid == 0) printf ("XYY %lu %lu %lu\n", groupid, blqS, grdS);
    if (groupid == 0) printf ("XYY %lu %lu %lu\n", groupid, blqS, glbalSz);
#endif
    if (localid == 0) {
      l.eval(groupid) = scratch[localid];
//      if (blqS == grdS) printf ("Fin %lu -> %20.10f\n", groupid, scratch[localid]);
    }
#ifdef TRACE_ERROR
//    if (groupid == 0) printf ("XYZ %lu %lu %lu\n", groupid, blqS, grdS);
    if (groupid == 0) printf ("XYZ %lu %lu %lu\n", groupid, blqS, glbalSz);
#endif
    return l.eval(groupid);
  }
};

template <typename Operator, unsigned int interLoop=1, typename LHS, typename RHS>
AssignReduction<Operator, interLoop, LHS, RHS> make_AssignReduction(LHS &l, RHS &r,
                                                             size_t blqS,
                                                             size_t grdS) {
  return AssignReduction<Operator, interLoop, LHS, RHS>(l, r, blqS, grdS);
}

template <typename LHS, typename RHS>
auto make_addAssignReduction(LHS &l, RHS &r, size_t blqS, size_t grdS)
    -> decltype(make_AssignReduction<addOp2_struct>(l, r, blqS, grdS)) {
  return make_AssignReduction<addOp2_struct>(l, r, blqS, grdS);
}

template <typename LHS, typename RHS>
auto make_prdAssignReduction(LHS &l, RHS &r, size_t blqS, size_t grdS)
    -> decltype(make_AssignReduction<prdOp2_struct>(l, r, blqS, grdS)) {
  return make_AssignReduction<prdOp2_struct>(l, r, blqS, grdS);
}

template <typename LHS, typename RHS>
auto make_addAbsAssignReduction(LHS &l, RHS &r, size_t blqS, size_t grdS)
    -> decltype(make_AssignReduction<addAbsOp2_struct>(l, r, blqS, grdS)) {
  return make_AssignReduction<addAbsOp2_struct>(l, r, blqS, grdS);
}

template <typename LHS, typename RHS>
auto make_maxIndAssignReduction(LHS &l, RHS &r, size_t blqS, size_t grdS)
    -> decltype(make_AssignReduction<maxIndOp2_struct>(l, r, blqS, grdS)) {
  return make_AssignReduction<maxIndOp2_struct>(l, r, blqS, grdS);
}

template <typename LHS, typename RHS>
auto make_minIndAssignReduction(LHS &l, RHS &r, size_t blqS, size_t grdS)
    -> decltype(make_AssignReduction<minIndOp2_struct>(l, r, blqS, grdS)) {
  return make_AssignReduction<minIndOp2_struct>(l, r, blqS, grdS);
}

/*

*/

template <typename Operator, class LHS, class RHS>
struct ReducOp {
  using value_type = typename LHS::value_type;
  LHS l;
  RHS r;
  size_t blqS;  // block  size
  size_t grdS;  // grid  size

  ReducOp(LHS &_l, RHS &_r, size_t _blqS, size_t _grdS)
    : l(_l), r(_r), blqS(_blqS), grdS(_grdS) {}
/*
  size_t getSize() { return r.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  auto applyOperator(LHS val1, LHS val2) { return Operator::eval(val1, val2); }

  value_type eval(size_t i) { return value_type(i, r.eval(i)); }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
*/
};

//#define ONE_LOOP 1
/*! AssignReduction_2Ops.
 * @brief Implements the reduction operation for assignments (in the form y = x)
 *  with y a scalar and x a subexpression tree.
 */
template <typename Operator, unsigned int interLoop, class LHS1, class RHS1, class LHS2, class RHS2>
struct AssignReduction_2Ops {
  using value_type = typename RHS1::value_type;
  LHS1 l1;
  RHS1 r1;
  LHS2 l2;
  RHS2 r2;
  size_t blqS;  // block  size
  size_t grdS;  // grid  size
  static const size_t intLoop = interLoop;

  AssignReduction_2Ops(LHS1 &_l1, RHS1 &_r1, LHS2 &_l2, RHS2 &_r2,
                          size_t _blqS, size_t _grdS)
      : l1(_l1), r1(_r1), l2(_l2), r2(_r2),blqS(_blqS), grdS(_grdS){};

  size_t getSize() { return r1.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(size_t i) {
    size_t vecS = r1.getSize();
    size_t frs_thrd = NUM_LOCAL_ADDS * blqS * i;
    size_t lst_thrd = ((frs_thrd + blqS) > vecS) ? vecS : (frs_thrd + blqS);
    // Reduction across the grid
    value_type val1 = Operator::init(r1);
    value_type val2 = Operator::init(r2);
    for (size_t j = frs_thrd; j < lst_thrd; j++) {
      value_type local_val1 = Operator::init(r1);
      value_type local_val2 = Operator::init(r2);
      for (size_t k = j; k < vecS; k += NUM_LOCAL_ADDS * grdS) {
        local_val1 = Operator::eval(local_val1, r1.eval(k));
        local_val2 = Operator::eval(local_val2, r2.eval(k));
#ifdef TWO_LOCAL_ADDS
        if (k + blqS < vecS) {
          local_val1 = Operator::eval(local_val1, r1.eval(k + blqS));
          local_val2 = Operator::eval(local_val2, r2.eval(k + blqS));
        }
#endif
      }
      // Reduction inside the block
      val1 = Operator::eval(val1, local_val1);
      val2 = Operator::eval(val2, local_val2);
    }
    l1.eval(i) = val1;
    return l2.eval(i) = val2;
  }
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
  template <typename sharedT>
//  value_type eval(sharedT scratch1, sharedT scratch2, cl::sycl::nd_item<1> ndItem) {
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t glbalSz = ndItem.get_num_groups(0) * localSz;

    size_t vecS = r1.getSize();

//    sharedT scratch1 = sharedT(scratch, 0, 1, localSz);
//    sharedT scratch2 = sharedT(scratch, 0, 1, localSz);
    size_t shft1 = 0;
    size_t shft2 = localSz;

    // Reduction across the grid
    value_type val1 = Operator::init(r1);
    value_type val2 = Operator::init(r2);
    if (interLoop == 1) {
      size_t frs_thrd = NUM_LOCAL_ADDS * groupid * localSz + localid;
#ifdef ONE_LOOP
      for (size_t k = frs_thrd; k < vecS; k += NUM_LOCAL_ADDS * glbalSz) {
        val1 = Operator::eval(val1, r1.eval(k));
        val2 = Operator::eval(val2, r2.eval(k));
  #ifdef TWO_LOCAL_ADDS
        if ((k + blqS < vecS)) {
//          val1 = Operator::eval(val1, r1.eval(k + blqS));
          val1 = Operator::eval(val1, r1.eval(k + localSz));
//          val2 = Operator::eval(val2, r2.eval(k + blqS));
          val2 = Operator::eval(val2, r2.eval(k + localSz));
        }
  #endif
      }
#else
//      printf ("YES(2)!!\n");
      for (size_t k = frs_thrd; k < vecS; k += NUM_LOCAL_ADDS * glbalSz) {
        val1 = Operator::eval(val1, r1.eval(k));
      }
      for (size_t k = frs_thrd; k < vecS; k += NUM_LOCAL_ADDS * glbalSz) {
        val2 = Operator::eval(val2, r2.eval(k));
      }
#endif
    } else {
      size_t frs_thrd = interLoop * (groupid * localSz + localid);
#ifdef ONE_LOOP
      for (size_t k = frs_thrd; k < vecS; k += interLoop * glbalSz) {
        for (size_t k_int=k; k_int<std::min(k+interLoop,vecS);k_int++) {
//        printf ("localid = %lu , k = %lu , num = %f\n", localid, k_int, r.eval(k));
          val1 = Operator::eval(val1, r1.eval(k_int));
          val2 = Operator::eval(val2, r2.eval(k_int));
        }
      }
#else
      for (size_t k = frs_thrd; k < vecS; k += interLoop * glbalSz) {
        for (size_t k_int=k; k_int<std::min(k+interLoop,vecS);k_int++) {
//        printf ("localid = %lu , k = %lu , num = %f\n", localid, k_int, r.eval(k));
          val1 = Operator::eval(val1, r1.eval(k_int));
        }
      }
      for (size_t k = frs_thrd; k < vecS; k += interLoop * glbalSz) {
        for (size_t k_int=k; k_int<std::min(k+interLoop,vecS);k_int++) {
//        printf ("localid = %lu , k = %lu , num = %f\n", localid, k_int, r.eval(k));
          val2 = Operator::eval(val2, r2.eval(k_int));
        }
      }
#endif
    }

//    scratch1[localid] = val1;
//    scratch2[localid] = val2;
    scratch[shft1+localid] = val1;
    scratch[shft2+localid] = val2;
    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Reduction inside the block
    for (size_t offset = localSz >> 1; offset > 0; offset >>= 1) {
      if (localid < offset) {
        scratch[shft1+localid] =
            Operator::eval(scratch[shft1+localid], scratch[shft1+localid + offset]);
        scratch[shft2+localid] =
            Operator::eval(scratch[shft2+localid], scratch[shft2+localid + offset]);
      }
      // This barrier is mandatory to be sure the data are on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
    }
    if (localid == 0) {
      l1.eval(groupid) = scratch[shft1+localid];
      l2.eval(groupid) = scratch[shft2+localid];
    }
    return l2.eval(groupid);
  }
};

template <typename Operator, unsigned int interLoop=1,
                             typename LHS1, typename RHS1,
                             typename LHS2, typename RHS2>
AssignReduction_2Ops<Operator, interLoop, LHS1, RHS1, LHS2, RHS2>
            make_AssignReduction_2Ops(LHS1 &l1, RHS1 &r1, LHS2 &l2, RHS2 &r2,
                                                             size_t blqS,
                                                             size_t grdS) {
  return AssignReduction_2Ops<Operator, interLoop, LHS1, RHS1, LHS2, RHS2>
                                               (l1, r1, l2, r2, blqS, grdS);
}

/*! AssignReduction_4Ops.
 * @brief Implements the reduction operation for assignments (in the form y = x)
 *  with y a scalar and x a subexpression tree.
 */
template <typename Operator, unsigned int interLoop,
                             class LHS1, class RHS1, class LHS2, class RHS2,
                             class LHS3, class RHS3, class LHS4, class RHS4>
struct AssignReduction_4Ops {
  using value_type = typename RHS1::value_type;
  LHS1 l1;
  RHS1 r1;
  LHS2 l2;
  RHS2 r2;
  LHS2 l3;
  RHS2 r3;
  LHS2 l4;
  RHS2 r4;
  size_t blqS;  // block  size
  size_t grdS;  // grid  size
  static const size_t intLoop = interLoop;

  AssignReduction_4Ops(LHS1 &_l1, RHS1 &_r1, LHS2 &_l2, RHS2 &_r2,
                       LHS3 &_l3, RHS3 &_r3, LHS4 &_l4, RHS4 &_r4,
                          size_t _blqS, size_t _grdS)
      : l1(_l1), r1(_r1), l2(_l2), r2(_r2),
        l3(_l3), r3(_r3), l4(_l4), r4(_r4),blqS(_blqS), grdS(_grdS){};

  size_t getSize() { return r1.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(size_t i) {
    size_t vecS = r1.getSize();
    size_t frs_thrd = NUM_LOCAL_ADDS * blqS * i;
    size_t lst_thrd = ((frs_thrd + blqS) > vecS) ? vecS : (frs_thrd + blqS);
    // Reduction across the grid
    value_type val1 = Operator::init(r1);
    value_type val2 = Operator::init(r2);
    value_type val3 = Operator::init(r3);
    value_type val4 = Operator::init(r4);
    for (size_t j = frs_thrd; j < lst_thrd; j++) {
      value_type local_val1 = Operator::init(r1);
      value_type local_val2 = Operator::init(r2);
      value_type local_val3 = Operator::init(r3);
      value_type local_val4 = Operator::init(r4);
      for (size_t k = j; k < vecS; k += NUM_LOCAL_ADDS * grdS) {
        local_val1 = Operator::eval(local_val1, r1.eval(k));
        local_val2 = Operator::eval(local_val2, r2.eval(k));
        local_val3 = Operator::eval(local_val3, r3.eval(k));
        local_val4 = Operator::eval(local_val4, r4.eval(k));
#ifdef TWO_LOCAL_ADDS
        if (k + blqS < vecS) {
          local_val1 = Operator::eval(local_val1, r1.eval(k + blqS));
          local_val2 = Operator::eval(local_val2, r2.eval(k + blqS));
          local_val3 = Operator::eval(local_val3, r3.eval(k + blqS));
          local_val4 = Operator::eval(local_val4, r4.eval(k + blqS));
        }
#endif
      }
      // Reduction inside the block
      val1 = Operator::eval(val1, local_val1);
      val2 = Operator::eval(val2, local_val2);
      val3 = Operator::eval(val3, local_val3);
      val4 = Operator::eval(val4, local_val4);
    }
    l1.eval(i) = val1;
    l2.eval(i) = val2;
    l3.eval(i) = val3;
    return l4.eval(i) = val4;
  }
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
  template <typename sharedT>
//  value_type eval(sharedT scratch1, sharedT scratch2, cl::sycl::nd_item<1> ndItem) {
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t glbalSz = ndItem.get_num_groups(0) * localSz;

    size_t vecS = r1.getSize();

//    sharedT scratch1 = sharedT(scratch, 0, 1, localSz);
//    sharedT scratch2 = sharedT(scratch, 0, 1, localSz);
    size_t shft1 = 0;
    size_t shft2 = localSz;
    size_t shft3 = shft2 + localSz;
    size_t shft4 = shft3 + localSz;

    // Reduction across the grid
    value_type val1 = Operator::init(r1);
    value_type val2 = Operator::init(r2);
    value_type val3 = Operator::init(r3);
    value_type val4 = Operator::init(r4);
    if (interLoop == 1) {
      size_t frs_thrd = NUM_LOCAL_ADDS * groupid * localSz + localid;
#ifdef ONE_LOOP
      for (size_t k = frs_thrd; k < vecS; k += NUM_LOCAL_ADDS * grdS) {
        val1 = Operator::eval(val1, r1.eval(k));
        val2 = Operator::eval(val2, r2.eval(k));
        val3 = Operator::eval(val3, r3.eval(k));
        val4 = Operator::eval(val4, r4.eval(k));
  #ifdef TWO_LOCAL_ADDS
        if ((k + blqS < vecS)) {
//          val1 = Operator::eval(val1, r1.eval(k + blqS));
          val1 = Operator::eval(val1, r1.eval(k + localSz));
//          val2 = Operator::eval(val2, r2.eval(k + blqS));
          val2 = Operator::eval(val2, r2.eval(k + localSz));
//          val3 = Operator::eval(val3, r3.eval(k + blqS));
          val3 = Operator::eval(val3, r3.eval(k + localSz));
//          val4 = Operator::eval(val4, r4.eval(k + blqS));
          val4 = Operator::eval(val4, r4.eval(k + localSz));
        }
  #endif
      }
#else
//      printf ("YES(4)!!\n");
      for (size_t k = frs_thrd; k < vecS; k += NUM_LOCAL_ADDS * grdS) {
        val1 = Operator::eval(val1, r1.eval(k));
      }
      for (size_t k = frs_thrd; k < vecS; k += NUM_LOCAL_ADDS * grdS) {
        val2 = Operator::eval(val2, r2.eval(k));
      }
      for (size_t k = frs_thrd; k < vecS; k += NUM_LOCAL_ADDS * grdS) {
        val3 = Operator::eval(val3, r3.eval(k));
      }
      for (size_t k = frs_thrd; k < vecS; k += NUM_LOCAL_ADDS * grdS) {
        val4 = Operator::eval(val4, r4.eval(k));
      }
#endif
    } else {
      size_t frs_thrd = interLoop * (groupid * localSz + localid);
#ifdef ONE_LOOP
      for (size_t k = frs_thrd; k < vecS; k += interLoop * glbalSz) {
        for (size_t k_int=k; k_int<std::min(k+interLoop,vecS);k_int++) {
//        printf ("localid = %lu , k = %lu , num = %f\n", localid, k_int, r.eval(k));
          val1 = Operator::eval(val1, r1.eval(k_int));
          val2 = Operator::eval(val2, r2.eval(k_int));
          val3 = Operator::eval(val3, r3.eval(k_int));
          val4 = Operator::eval(val4, r4.eval(k_int));
        }
      }
#else
      for (size_t k = frs_thrd; k < vecS; k += interLoop * glbalSz) {
        for (size_t k_int=k; k_int<std::min(k+interLoop,vecS);k_int++) {
          val1 = Operator::eval(val1, r1.eval(k_int));
        }
      }
      for (size_t k = frs_thrd; k < vecS; k += interLoop * glbalSz) {
        for (size_t k_int=k; k_int<std::min(k+interLoop,vecS);k_int++) {
          val2 = Operator::eval(val2, r2.eval(k_int));
        }
      }
      for (size_t k = frs_thrd; k < vecS; k += interLoop * glbalSz) {
        for (size_t k_int=k; k_int<std::min(k+interLoop,vecS);k_int++) {
          val3 = Operator::eval(val3, r3.eval(k_int));
        }
      }
      for (size_t k = frs_thrd; k < vecS; k += interLoop * glbalSz) {
        for (size_t k_int=k; k_int<std::min(k+interLoop,vecS);k_int++) {
          val4 = Operator::eval(val4, r4.eval(k_int));
        }
      }
#endif
    }

//    scratch1[localid] = val1;
//    scratch2[localid] = val2;
    scratch[shft1+localid] = val1;
    scratch[shft2+localid] = val2;
    scratch[shft3+localid] = val3;
    scratch[shft4+localid] = val4;
    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Reduction inside the block
    for (size_t offset = localSz >> 1; offset > 0; offset >>= 1) {
      if (localid < offset) {
        scratch[shft1+localid] =
            Operator::eval(scratch[shft1+localid], scratch[shft1+localid + offset]);
        scratch[shft2+localid] =
            Operator::eval(scratch[shft2+localid], scratch[shft2+localid + offset]);
        scratch[shft3+localid] =
            Operator::eval(scratch[shft3+localid], scratch[shft3+localid + offset]);
        scratch[shft4+localid] =
            Operator::eval(scratch[shft4+localid], scratch[shft4+localid + offset]);
      }
      // This barrier is mandatory to be sure the data are on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
    }
    if (localid == 0) {
      l1.eval(groupid) = scratch[shft1+localid];
      l2.eval(groupid) = scratch[shft2+localid];
      l3.eval(groupid) = scratch[shft3+localid];
      l4.eval(groupid) = scratch[shft4+localid];
    }
    return l4.eval(groupid);
  }
};

template <typename Operator, unsigned int interLoop=1,
                             typename LHS1, typename RHS1,
                             typename LHS2, typename RHS2,
                             typename LHS3, typename RHS3,
                             typename LHS4, typename RHS4>
AssignReduction_4Ops<Operator, interLoop, LHS1, RHS1, LHS2, RHS2, LHS3, RHS3, LHS4, RHS4>
            make_AssignReduction_4Ops(LHS1 &l1, RHS1 &r1, LHS2 &l2, RHS2 &r2,
                                      LHS3 &l3, RHS3 &r3, LHS4 &l4, RHS4 &r4,
                                                             size_t blqS,
                                                             size_t grdS) {
  return AssignReduction_4Ops<Operator, interLoop,
                                        LHS1, RHS1, LHS2, RHS2,
                                        LHS3, RHS3, LHS4, RHS4>
                                               (l1, r1, l2, r2,
                                                l3, r3, l4, r4, blqS, grdS);
}
/*!
@brief Template function for constructing operation nodes based on input
tempalte and function arguments. Non-specialised case for N reference operands.
@tparam operationT Operation type of the operation node.
@tparam operandsTN Operand types of the oeration node.
@param operands Reference operands of the operation node.
@return Constructed operation node.
*/
template <template <class...> class operationT, typename... operandsTN>
operationT<operandsTN...> make_op(operandsTN &... operands) {
  return operationT<operandsTN...>(operands...);
}

/*!
@brief Template function for constructing operation nodes based on input
tempalte and function arguments. Specialised case for an operator and N
reference operands.
@tparam operationT Operation type of the operation node.
@tparam operatorT Operator type of the operation node.
@tparam operandsTN Operand types of the operation node.
@param operands Reference operands of the operation node.
@return Constructed operation node.
*/
template <template <class...> class operationT, typename operatorT,
          typename... operandsTN>
operationT<operatorT, operandsTN...> make_op(operandsTN &... operands) {
  return operationT<operatorT, operandsTN...>(operands...);
}

/*!
@brief Template function for constructing operation nodes based on input
tempalte and function arguments. Specialised case for an operator, a single by
value operand and N reference operands.
@tparam operationT Operation type of the operation node.
@tparam operatorT Operator type of the operation node.
@tparam operandT0 Operand type of the first by value operand of the operation
node.
@tparam operandsTN Operand types of the subsequent reference operands of the
operation node.
@param operand0 First by value operand of the operation node.
@param operands Subsequent reference operands of the oepration node.
@return Constructed operation node.
*/
template <template <class...> class operationT, typename operatorT,
          typename operandT0, typename... operandsTN>
operationT<operatorT, operandT0, operandsTN...> make_op(
    operandT0 operand0, operandsTN &... operands) {
  return operationT<operatorT, operandT0, operandsTN...>(operand0, operands...);
}

}  // namespace blas

#endif  // BLAS1_TREES_HPP
