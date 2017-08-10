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
 *  @filename blas2_trees.hpp
 *
 **************************************************************************/

#ifndef BLAS2_TREES_HPP
#define BLAS2_TREES_HPP

#include <stdexcept>
#include <vector>

#include <operations/blas2_trees.hpp>
#include <operations/blas_operators.hpp>
#include <views/view_sycl.hpp>

namespace blas {

/**** ADD A SET OF COLUMNS, 1 ROW PER THREAD ****/
template <class RHS>
struct AddSetColumns {
  using value_type = typename RHS::value_type;

  RHS r;

  AddSetColumns(RHS &_r) : r(_r){};

  size_t getSize() { return r.getSizeR(); }

  value_type eval(size_t i) {
    auto dimC = r.getSizeC();

    auto val = iniAddOp1_struct::eval(r.eval(0));
    for (size_t j = 0; j < dimC; j++) {
      val += r.eval(i, j);
    }
    return val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }

};

template <class RHS> AddSetColumns<RHS> make_addSetColumns(RHS &r) {
  return AddSetColumns<RHS>(r);
}

/**** GEMV BY ROWS 1 ROW x 1 BLOCK ****/
template <unsigned int interLoop, class LHS, class RHS1, class RHS2>
struct GemvR_1Row_1WG {
  LHS  l;
  RHS1 r1;
  RHS2 r2;

  using value_type = typename RHS2::value_type;

  GemvR_1Row_1WG(LHS &_l,RHS1 &_r1, RHS2 &_r2)
    : l(_l), r1(_r1), r2(_r2) {};

  size_t getSize() { return r1.getSizeR(); }

  value_type eval(size_t i) { // NOT VERIFIED
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t j = 0; j < dim; j++) {
      val += r1.eval(i, j) * r2.eval(j);
    }
    return l.eval(i) = val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }

  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t glbalSz = ndItem.get_global_range(0);

    size_t vecS = r2.getSize();

    value_type val = addOp2_struct::init(r2);

    if (interLoop == 1) {
      size_t frs_thrd = localid;
      for (size_t k = frs_thrd; k < vecS; k += localSz) {
//        auto prod = prdOp2_struct::eval(r1.eval(groupid,k),r2.eval(k));
//        val = addOp2_struct::eval(val, prod);
        val += r1.eval(groupid,k) * r2.eval(k);
      }
    } else { // NOT VERIFIED
      size_t frs_thrd = interLoop * (groupid * localSz + localid);
      for (size_t k = frs_thrd; k < vecS; k += interLoop * glbalSz) {
        for (size_t k_int=k; k_int<std::min(k+interLoop,vecS);k_int++) {
          val = addOp2_struct::eval(val, r1.eval(groupid,k_int));
        }
      }
    }

    shrMem[localid] = val;
    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Reduction inside the block
    for (size_t offset = localSz >> 1; offset > 0; offset >>= 1) {
      if (localid < offset) {
//        shrMem[localid] =
//            addOp2_struct::eval(shrMem[localid], shrMem[localid + offset]);
        shrMem[localid] += shrMem[localid + offset];
      }
      // This barrier is mandatory to be sure the data are on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
    }
    if (localid == 0) {
      l.eval(groupid) = shrMem[localid];
    }

    return l.eval(groupid);
  }
};

template <unsigned int interLoop=1, typename LHS, typename RHS1, typename RHS2>
GemvR_1Row_1WG<interLoop, LHS, RHS1, RHS2> make_GemvR_1Row_1WG(LHS &l, RHS1 &r1,
                                                              RHS2 &r2) {
  return GemvR_1Row_1WG<interLoop, LHS, RHS1, RHS2>(l, r1, r2);
}

/**** GEMV BY ROWS 1 ROW x 1 BLOCK, WITHOUT LOCAL ADDITION ****/
template <unsigned int interLoop, class LHS, class RHS1, class RHS2>
struct GemvR_1Row_1WG_NoRed {
  LHS  l;
  RHS1 r1;
  RHS2 r2;

  using value_type = typename RHS2::value_type;

  GemvR_1Row_1WG_NoRed(LHS &_l,RHS1 &_r1, RHS2 &_r2)
    : l(_l), r1(_r1), r2(_r2) {};

  size_t getSize() { return l.getSize(); }

  value_type eval(size_t i) { // NOT VERIFIED
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t j = 0; j < dim; j++) {
      val += r1.eval(i, j) * r2.eval(j);
    }
    return l.eval(i) = val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t glbalSz = ndItem.get_num_groups(0) * localSz;

    size_t vecS = r2.getSize();

    value_type val = addOp2_struct::init(r2);

    if (interLoop == 1) {
      size_t frs_thrd = localid;
      for (size_t k = frs_thrd; k < vecS; k += localSz) {
        auto prod = prdOp2_struct::eval(r1.eval(groupid,k),r2.eval(k));
        val = addOp2_struct::eval(val, prod);
      }
    } else { // NOT VERIFIED
      size_t frs_thrd = interLoop * (groupid * localSz + localid);
      for (size_t k = frs_thrd; k < vecS; k += interLoop * glbalSz) {
        for (size_t k_int=k; k_int<std::min(k+interLoop,vecS);k_int++) {
          val = addOp2_struct::eval(val, r1.eval(groupid,k_int));
        }
      }
    }

    return l.eval(groupid,localid) = val;
  }
};

template <unsigned int interLoop=1, typename LHS, typename RHS1, typename RHS2>
GemvR_1Row_1WG_NoRed<interLoop, LHS, RHS1, RHS2>
            make_GemvR_1Row_1WG_NoRed(LHS &l, RHS1 &r1, RHS2 &r2) {
  return GemvR_1Row_1WG_NoRed<interLoop, LHS, RHS1, RHS2>(l, r1, r2);
}

/**** GEMV BY ROWS 1 ROW x N BLOCK ****/
template <unsigned int interLoop, class LHS, class RHS1, class RHS2>
struct GemvR_1Row_NWG {
  LHS  l;
  RHS1 r1;
  RHS2 r2;
  size_t nWG_col;

  using value_type = typename RHS2::value_type;

  GemvR_1Row_NWG(LHS &_l,RHS1 &_r1, RHS2 &_r2, size_t &_nWG_col)
    : l(_l), r1(_r1), r2(_r2), nWG_col(_nWG_col){};

  size_t getSize() { return r1.getSizeR(); }

  value_type eval(size_t i) { // NOT VERIFIED
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t j = 0; j < dim; j++) {
      val += r1.eval(i, j) * r2.eval(j);
    }
    return l.eval(i) = val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }

  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);
    size_t glbalSz = ndItem.get_global_range(0);

    size_t dimR  = r1.getSizeR();
    size_t dimC  = r1.getSizeC();
    size_t blqSz = (groupSz + nWG_col - 1) / nWG_col;  // number of "real" workgroups

    size_t blqidR = groupid / nWG_col;  // row bloq id of the current workgroup
    size_t blqidC = groupid % nWG_col;  // col blq id of the current workgroup

    size_t vecS = r2.getSize();

    value_type val = addOp2_struct::init(r2);

    if (interLoop == 1) {
      size_t frs_thrd = blqidC * localSz + localid;
      for (size_t k = frs_thrd; k < vecS; k += localSz*nWG_col) {
        auto prod = prdOp2_struct::eval(r1.eval(blqidR,k),r2.eval(k));
        val = addOp2_struct::eval(val, prod);
      }
    } else { // NOT VERIFIED
      size_t frs_thrd = interLoop * (groupid * localSz + localid);
      for (size_t k = frs_thrd; k < vecS; k += interLoop * glbalSz) {
        for (size_t k_int=k; k_int<std::min(k+interLoop,vecS);k_int++) {
          val = addOp2_struct::eval(val, r1.eval(groupid,k_int));
        }
      }
    }

    shrMem[localid] = val;
    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Reduction inside the block
    for (size_t offset = localSz >> 1; offset > 0; offset >>= 1) {
      if (localid < offset) {
        shrMem[localid] =
            addOp2_struct::eval(shrMem[localid], shrMem[localid + offset]);
      }
      // This barrier is mandatory to be sure the data are on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
    }
    if (localid == 0) {
      l.eval(blqidR,blqidC) = shrMem[localid];
    }

    return l.eval(blqidR,blqidC);
  }
};

template <unsigned int interLoop=1, typename LHS, typename RHS1, typename RHS2>
GemvR_1Row_NWG<interLoop, LHS, RHS1, RHS2>
    make_GemvR_1Row_NWG(LHS &l, RHS1 &r1, RHS2 &r2, size_t nWG_col) {
  return GemvR_1Row_NWG<interLoop, LHS, RHS1, RHS2>(l, r1, r2, nWG_col);
}

/**** GEMV BY ROWS M ROWS x N BLOCK ****/
#define GROUP_ROWS 1 // Not useful for GEMV by rows
//#define SHARED_ACCESS 1
template <unsigned int interLoop, class LHS, class RHS1, class RHS2>
struct GemvR_MRow_NWG {
  LHS  l;
  RHS1 r1;
  RHS2 r2;
  size_t n_rows;
  size_t nWG_col;

  using value_type = typename RHS2::value_type;

  GemvR_MRow_NWG(LHS &_l,RHS1 &_r1, RHS2 &_r2, size_t &_n_rows, size_t &_nWG_col)
    : l(_l), r1(_r1), r2(_r2), n_rows(_n_rows), nWG_col(_nWG_col){};

  size_t getSize() { return r1.getSizeR(); }

  value_type eval(size_t i) { // NOT VERIFIED
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t j = 0; j < dim; j++) {
      val += r1.eval(i, j) * r2.eval(j);
    }
    return l.eval(i) = val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }

  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);
    size_t glbalid = ndItem.get_global(0);
    size_t glbalSz = ndItem.get_global_range(0);

    size_t dimR = r1.getSizeR();
    size_t dimC = r1.getSizeC();
    size_t blqSz = (groupSz + nWG_col - 1) / nWG_col;  // number of "real" workgroups

    size_t blqidR = groupid / nWG_col;  // row bloq id of the current workgroup
    size_t blqidC = groupid % nWG_col;  // col blq id of the current workgroup

    size_t vecS = r2.getSize();

    size_t frs_row = blqidR*n_rows;

    value_type val = addOp2_struct::init(r2);
#ifdef GROUP_ROWS
    size_t num_rows = 0;
//    for (size_t row=0, id_row=blqidR*n_rows;
    for (size_t row=0, id_row=frs_row;
          (row<n_rows) && (id_row<dimR); row++, id_row++, num_rows++) {
  #ifdef SHARED_ACCESS
      shrMem[row*localSz+localid] = val;
  #else
      shrMem[row+n_rows*localid] = val;
  #endif
    }
#endif
    if (interLoop == 1) {
      size_t frs_thrd = blqidC * localSz + localid;
//      if (localid == 0)
//        printf("%lu -> %lu , %lu\n", glbalid, frs_thrd, blqidR);
//      return 0.0;
#ifdef GROUP_ROWS
      for (size_t k = frs_thrd; k < vecS; k += localSz*nWG_col) {
        auto elm = r2.eval(k);
//        for (size_t row=0, id_row=blqidR*n_rows;
        for (size_t row=0, id_row=frs_row;
              (row<n_rows); row++, id_row++) {
          auto prod = prdOp2_struct::eval(r1.eval(id_row,k),elm);
  #ifdef SHARED_ACCESS
          shrMem[row*localSz+localid] =
                  addOp2_struct::eval(shrMem[row*localSz+localid], prod);
  #else
          shrMem[row+n_rows*localid] =
                  addOp2_struct::eval(shrMem[row+n_rows*localid], prod);
  #endif
        }
      }
#else
//      size_t id_row = blqidR*n_rows;
      size_t id_row = frs_row;
      for (size_t row=0; (row<n_rows); row++) {
        val = addOp2_struct::init(r2);
        for (size_t k = frs_thrd; k < vecS; k += localSz*nWG_col) {
          auto prod = prdOp2_struct::eval(r1.eval(id_row,k),r2.eval(k));
          val = addOp2_struct::eval(val, prod);
        }
  #ifdef SHARED_ACCESS
        shrMem[row*localSz+localid] = val;
  #else
        shrMem[row+n_rows*localid] = val;
  #endif
        id_row++;
      }
#endif
    } else { // NOT VERIFIED
      size_t frs_thrd = interLoop * (groupid * localSz + localid);
      for (size_t k = frs_thrd; k < vecS; k += interLoop * glbalSz) {
        for (size_t k_int=k; k_int<std::min(k+interLoop,vecS);k_int++) {
          val = addOp2_struct::eval(val, r1.eval(groupid,k_int));
        }
      }
    }

    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);
//    if (blqidR == 0)
//      printf ("%lu -> (%f , %f , %f , %f)\n",
//              glbalid, shrMem[localid], shrMem[localid+localSz],
//              shrMem[localid+2*localSz], shrMem[localid+3*localSz]);
    // Reduction inside the block
    for (size_t offset = localSz >> 1; offset > 0; offset >>= 1) {
      if (localid < offset) {
        for (size_t row=0; row<n_rows; row++) {
#ifdef SHARED_ACCESS
          shrMem[row*localSz+localid] =
              addOp2_struct::eval(shrMem[row*localSz+localid],
                                  shrMem[row*localSz+localid+offset]);
#else
          shrMem[row+n_rows*localid] =
              addOp2_struct::eval(shrMem[row+n_rows*localid],
                                  shrMem[row+n_rows*(localid+offset)]);
#endif
        }
      }
      // This barrier is mandatory to be sure the data are on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
    }
    if (localid == 0) {
//      if (blqidR == 0)
//        printf ("%lu -> (%f , %f , %f , %f)\n",
//                glbalid, shrMem[localid], shrMem[localid+localSz],
//                shrMem[localid+2*localSz], shrMem[localid+3*localSz]);
//      size_t id_row=blqidR*n_rows;
      size_t id_row=frs_row;
//      for (size_t row=0, id_row=blqidR*n_rows;(row<n_rows); row++, id_row++) {
      for (size_t row=0; row<n_rows; row++) {
#ifdef SHARED_ACCESS
        l.eval(id_row,blqidC) = shrMem[row*localSz];
#else
        l.eval(id_row,blqidC) = shrMem[row];
#endif
        id_row++;
      }
    }

    return l.eval(blqidR*n_rows,blqidC);
  }
};

template <unsigned int interLoop=1, typename LHS, typename RHS1, typename RHS2>
GemvR_MRow_NWG<interLoop, LHS, RHS1, RHS2>
    make_GemvR_MRow_NWG(LHS &l, RHS1 &r1, RHS2 &r2, size_t n_rows, size_t nWG_col) {
  return GemvR_MRow_NWG<interLoop, LHS, RHS1, RHS2>(l, r1, r2, n_rows, nWG_col);
}

/**** GEMV BY COLUMNS 1 ROW x 1 THREAD ****/
template <class RHS1, class RHS2>
struct GemvC_1Row_1Thread {
  RHS1 r1;
  RHS2 r2;
//  size_t mult;

  using value_type = typename RHS2::value_type;

  GemvC_1Row_1Thread(RHS1 &_r1, RHS2 &_r2) : r1(_r1), r2(_r2){};

  value_type eval(size_t i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t j = 0; j < dim; j++) {
//      val += r1.eval(i, j) * r2.eval(j);
      auto prod = prdOp2_struct::eval(r1.eval(i,j),r2.eval(j));
      val = addOp2_struct::eval(val, prod);
    }
    return val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
  size_t getSize() { return r1.getSizeR(); }
};

template <class RHS1, class RHS2>
GemvC_1Row_1Thread<RHS1, RHS2> make_GemvC_1Row_1Thread(RHS1 &r1, RHS2 &r2) {
  return GemvC_1Row_1Thread<RHS1, RHS2>(r1, r2);
}

/**** GEMV BY COLUMNS 1 ROW x 1 THREAD USING SHARED MEMORY ****/
template <class LHS, class RHS1, class RHS2, class RHS3>
struct GemvC_1Row_1Thread_ShMem {
  LHS l;
  using value_type = typename RHS2::value_type;
  value_type scl;

  RHS1 r1;
  RHS2 r2;
  RHS3 r3;

  GemvC_1Row_1Thread_ShMem(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2, RHS3 &_r3)
      : l(_l), scl(_scl), r1(_r1), r2(_r2), r3(_r3) {};

  size_t getSize() { return r1.getSizeR(); }

  value_type eval(size_t i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t j = 0; j < dim; j++) {
//      val += r1.eval(i, j) * r2.eval(j);
      auto prod = prdOp2_struct::eval(r1.eval(i,j),r2.eval(j));
      val = addOp2_struct::eval(val, prod);
    }
    auto prod = prdOp2_struct::eval(scl, val);
    return l.eval(i) = addOp2_struct::eval(prod, r3.eval(i));
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }

  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);
    size_t glbalid = ndItem.get_global(0);

    size_t dimR = r1.getSizeR();
    size_t dimC = r1.getSizeC();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t k=0; k<dimC; k+=localSz) {
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
      scratch[localid] = r2.eval(k+localid);
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
      for (size_t j=0; j<std::min(dimC-k,localSz); j++) {
        auto prod = prdOp2_struct::eval(r1.eval(glbalid,k+j),scratch[j]);
        val = addOp2_struct::eval(val, prod);
      }
    }
    auto prod = prdOp2_struct::eval(scl, val);
    return l.eval(glbalid) = addOp2_struct::eval(prod, r3.eval(glbalid));
  }
};

template <class LHS, class RHS1, class RHS2, class RHS3>
GemvC_1Row_1Thread_ShMem<LHS, RHS1, RHS2, RHS3> make_GemvC_1Row_1Thread_ShMem(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2, RHS3 &r3) {
  return GemvC_1Row_1Thread_ShMem<LHS, RHS1, RHS2, RHS3>(l, scl, r1, r2, r3);
}

/**** GEMV BY COLUMNS 1 ROW x 1 THREAD USING SHARED MEMORY MINIMIZING SYNCHRONIZATION ****/
/**** This option uses too much memory, failing when the local memory is completed ****/
template <class LHS, class RHS1, class RHS2, class RHS3>
struct GemvC_1Row_1Thread_ShMem_Full {
  LHS l;
  using value_type = typename RHS2::value_type;
  value_type scl;

  RHS1 r1;
  RHS2 r2;
  RHS3 r3;

  GemvC_1Row_1Thread_ShMem_Full(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2, RHS3 &_r3)
      : l(_l), scl(_scl), r1(_r1), r2(_r2), r3(_r3) {};

  size_t getSize() { return r1.getSizeR(); }

  value_type eval(size_t i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t j = 0; j < dim; j++) {
//      val += r1.eval(i, j) * r2.eval(j);
      auto prod = prdOp2_struct::eval(r1.eval(i,j),r2.eval(j));
      val = addOp2_struct::eval(val, prod);
    }
    auto prod = prdOp2_struct::eval(scl, val);
    return l.eval(i) = addOp2_struct::eval(prod, r3.eval(i));
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }

  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);
    size_t glbalid = ndItem.get_global(0);

    size_t dimR = r1.getSizeR();
    size_t dimC = r1.getSizeC();

    for (size_t k=0; k<dimC; k+=localSz) {
      if ((k+localid) < dimC) scratch[k+localid] = r2.eval(k+localid);
    }

    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    if (glbalid < dimR) {
      for (size_t k=0; k<dimC; k++) {
        auto prod = prdOp2_struct::eval(r1.eval(glbalid,k),scratch[k]);
        val = addOp2_struct::eval(val, prod);
      }
      auto prod = prdOp2_struct::eval(scl, val);
      l.eval(glbalid) = val = addOp2_struct::eval(prod, r3.eval(glbalid));
    }
    return val;
  }
};

template <class LHS, class RHS1, class RHS2, class RHS3>
GemvC_1Row_1Thread_ShMem_Full<LHS, RHS1, RHS2, RHS3> make_GemvC_1Row_1Thread_ShMem_Full(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2, RHS3 &r3) {
  return GemvC_1Row_1Thread_ShMem_Full<LHS, RHS1, RHS2, RHS3>(l, scl, r1, r2, r3);
}

/**** GEMV BY COLUMNS 1 ROW x M THREADS ****/
template <class LHS, class RHS1, class RHS2, class RHS3>
struct GemvC_1Row_MThreads {
  LHS l;
  using value_type = typename RHS2::value_type;
  value_type scl;

  RHS1 r1;
  RHS2 r2;
  RHS3 r3;
  size_t nThr;

  GemvC_1Row_MThreads(LHS &_l, value_type _scl, RHS1 &_r1,
                            RHS2 &_r2, RHS3 &_r3, size_t &_nThr)
      : l(_l), scl(_scl), r1(_r1), r2(_r2), r3(_r3), nThr(_nThr) {};

  size_t getSize() { return r1.getSizeR(); }

  value_type eval(size_t i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t j = 0; j < dim; j++) {
//      val += r1.eval(i, j) * r2.eval(j);
      auto prod = prdOp2_struct::eval(r1.eval(i,j),r2.eval(j));
      val = addOp2_struct::eval(val, prod);
    }
    return val;
  }

  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);
    size_t glbalid = ndItem.get_global(0);

    size_t dimR = r1.getSizeR();
    size_t dimC = r1.getSizeC();

    size_t rowSz = (localSz + nThr - 1) / nThr;
    size_t colSz = (dimC    + nThr - 1) / nThr;

    size_t idWFR = (localid % rowSz);
    size_t idWFC = (localid / rowSz);

    size_t rowid = (groupid * rowSz) + idWFR;
    size_t colid = colSz * idWFC;

//    if (idWFR == 0)
//      printf ("%lu -> (%lu,%lu) - (%lu,%lu) - (%lu,%lu)\n",
//              glbalid, groupid, localid, rowSz, colSz, rowid, colid);
    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t k=colid; k<std::min(dimC,colid+colSz); k++) {
      auto prod = prdOp2_struct::eval(r1.eval(rowid,k),r2.eval(k));
      val = addOp2_struct::eval(val, prod);
    }

    scratch[localid] = val;
    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Reduction inside the block
    for (size_t offset = nThr >> 1; offset > 0; offset >>= 1) {
      if ((rowid < dimR) && (idWFC < offset)) {
        scratch[localid] += scratch[localid + offset * rowSz];
      }
      // This barrier is mandatory to be sure the data are on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
    }
    // The result is stored in lhs
    if ((rowid < dimR) && (idWFC == 0)) {
//      l.eval(rowid) = scl * scratch[localid] + r3.eval(rowid);
      auto prod = prdOp2_struct::eval(scl, scratch[localid]);
      l.eval(rowid) = addOp2_struct::eval(prod, r3.eval(rowid));
    }

    return val;
  }
};

template <class LHS, class RHS1, class RHS2, class RHS3>
GemvC_1Row_MThreads<LHS, RHS1, RHS2, RHS3> make_GemvC_1Row_MThreads(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2, RHS3 &r3, size_t nThr) {
  return GemvC_1Row_MThreads<LHS, RHS1, RHS2, RHS3>(l, scl, r1, r2, r3, nThr);
}

/**** GEMV BY COLUMNS 1 ROW x M THREADS USING SHARED MEMORY****/
template <class LHS, class RHS1, class RHS2, class RHS3>
struct GemvC_1Row_MThreads_ShMem {
  LHS l;
  using value_type = typename RHS2::value_type;
  value_type scl;

  RHS1 r1;
  RHS2 r2;
  RHS3 r3;
  size_t nThr;

  GemvC_1Row_MThreads_ShMem(LHS &_l, value_type _scl, RHS1 &_r1,
                            RHS2 &_r2, RHS3 &_r3, size_t &_nThr)
      : l(_l), scl(_scl), r1(_r1), r2(_r2), r3(_r3), nThr(_nThr) {};

  size_t getSize() { return r1.getSizeR(); }

  value_type eval(size_t i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t j = 0; j < dim; j++) {
//      val += r1.eval(i, j) * r2.eval(j);
      auto prod = prdOp2_struct::eval(r1.eval(i,j),r2.eval(j));
      val = addOp2_struct::eval(val, prod);
    }
    return val;
  }

  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);
    size_t glbalid = ndItem.get_global(0);

    size_t dimR = r1.getSizeR();
    size_t dimC = r1.getSizeC();

    size_t rowSz = (localSz + nThr - 1) / nThr;
    size_t colSz = (dimC    + nThr - 1) / nThr;

    size_t idWFR = (localid % rowSz);
    size_t idWFC = (localid / rowSz);

    size_t rowid = (groupid * rowSz) + idWFR;
    size_t colid = colSz * idWFC;

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t k=colid; k<std::min(colid+colSz,dimC); k+=rowSz) {
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
//      scratch[localid] = r2.eval(k+idWFR);
      scratch[localid] = ((k+idWFR)<dimC)?r2.eval(k+idWFR):0.0;
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
      if (rowid < dimR) {
        for (size_t j=k; j<std::min(k+rowSz,std::min(colid+colSz,dimC)); j++) {
          auto prod = prdOp2_struct::eval(r1.eval(rowid,j),scratch[idWFC*rowSz+j-k]);
          val = addOp2_struct::eval(val, prod);
        }
      }
    }

    ndItem.barrier(cl::sycl::access::fence_space::local_space);
    scratch[localid] = val;
    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Reduction inside the block
    for (size_t offset = nThr >> 1; offset > 0; offset >>= 1) {
      if ((rowid < dimR) && (idWFC < offset)) {
//        scratch[localid] += scratch[localid + offset * rowSz];
        scratch[localid] = addOp2_struct::eval(scratch[localid],
                                            scratch[localid + offset * rowSz]);
      }
      // This barrier is mandatory to be sure the data are on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
    }
    // The result is stored in lhs
    if ((rowid < dimR) && (idWFC == 0)) {
//      l.eval(rowid) = scl * scratch[localid] + r3.eval(rowid);
      auto prod = prdOp2_struct::eval(scl, scratch[localid]);
      l.eval(rowid) = addOp2_struct::eval(prod, r3.eval(rowid));
    }

    return val;
  }
};

template <class LHS, class RHS1, class RHS2, class RHS3>
GemvC_1Row_MThreads_ShMem<LHS, RHS1, RHS2, RHS3> make_GemvC_1Row_MThreads_ShMem(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2, RHS3 &r3, size_t nThr) {
  return GemvC_1Row_MThreads_ShMem<LHS, RHS1, RHS2, RHS3>(l, scl, r1, r2, r3, nThr);
}

/**** GEMV BY COLUMNS 1 ROW x M THREADS USING SHARED MEMORY, WITHOUT LOCAL ADDITION ****/
template <class LHS, class RHS1, class RHS2>
struct GemvC_1Row_MThreads_ShMem_NoRed {
  LHS l;
  using value_type = typename RHS2::value_type;

  RHS1 r1;
  RHS2 r2;
  size_t nThr;

  GemvC_1Row_MThreads_ShMem_NoRed(LHS &_l, RHS1 &_r1, RHS2 &_r2, size_t &_nThr)
      : l(_l), r1(_r1), r2(_r2), nThr(_nThr) {};

  size_t getSize() { return r1.getSizeR(); }

  value_type eval(size_t i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t j = 0; j < dim; j++) {
//      val += r1.eval(i, j) * r2.eval(j);
      auto prod = prdOp2_struct::eval(r1.eval(i,j),r2.eval(j));
      val = addOp2_struct::eval(val, prod);
    }
    return val;
  }

  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);
    size_t glbalid = ndItem.get_global(0);

    size_t dimR = r1.getSizeR();
    size_t dimC = r1.getSizeC();

    size_t rowSz = (localSz + nThr - 1) / nThr;
    size_t colSz = (dimC    + nThr - 1) / nThr;

    size_t idWFR = (localid % rowSz);
    size_t idWFC = (localid / rowSz);

    size_t rowid = (groupid * rowSz) + idWFR;
    size_t colid = colSz * idWFC;

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t k=colid; k<std::min(dimC,colid+colSz); k+=rowSz) {
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
//      scratch[localid] = r2.eval(k+idWFR);
      scratch[localid] = ((k+idWFR)<dimC)?r2.eval(k+idWFR):0.0;
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
      for (size_t j=k; j<std::min(k+rowSz,std::min(colid+colSz,dimC)); j++) {
//        auto prod = prdOp2_struct::eval(r1.eval(rowid,j),r2.eval(j));
        auto prod = prdOp2_struct::eval(r1.eval(rowid,j),scratch[idWFC*rowSz+j-k]);
        val = addOp2_struct::eval(val, prod);
      }
    }

    if (rowid < dimR) l.eval(rowid,idWFC) = val;

    return val;
  }
};

template <class LHS, class RHS1, class RHS2>
GemvC_1Row_MThreads_ShMem_NoRed<LHS, RHS1, RHS2> make_GemvC_1Row_MThreads_ShMem_NoRed(
    LHS &l, RHS1 &r1, RHS2 &r2, size_t nThr) {
  return GemvC_1Row_MThreads_ShMem_NoRed<LHS, RHS1, RHS2>(l, r1, r2, nThr);
}

/**** GEMV BY COLUMNS 1 ROW x M BLOCKS ****/
template <class LHS, class RHS1, class RHS2>
struct GemvC_1Row_MBlocks {
  LHS l;
  using value_type = typename RHS2::value_type;

  RHS1 r1;
  RHS2 r2;
  size_t nBlq;

  GemvC_1Row_MBlocks(LHS &_l, RHS1 &_r1, RHS2 &_r2, size_t &_nBlq)
      : l(_l), r1(_r1), r2(_r2), nBlq(_nBlq) {};

  size_t getSize() { return r1.getSizeR(); }

  value_type eval(size_t i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t j = 0; j < dim; j++) {
//      val += r1.eval(i, j) * r2.eval(j);
      auto prod = prdOp2_struct::eval(r1.eval(i,j),r2.eval(j));
      val = addOp2_struct::eval(val, prod);
    }
    return val;
  }

//  template <typename sharedT>
//  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);
    size_t glbalid = ndItem.get_global(0);

    size_t dimR = r1.getSizeR();
    size_t dimC = r1.getSizeC();

    size_t rowSz = localSz;
    size_t colSz = (dimC    + nBlq - 1) / nBlq;

//    size_t idWFR = localid;
    size_t dimWF = (groupSz + nBlq - 1) / nBlq;
//    size_t idWFR = (groupid + dimWF - 1) / dimWF;
    size_t idWFR = (groupid / nBlq);
    size_t idWFC = (groupid % nBlq);

//    size_t rowid = (idWFR * rowSz) + idWFR;
    size_t rowid = (idWFR * rowSz) + localid;
    size_t colid = colSz * idWFC;

//    if (localid == 0) printf ("%lu\n", groupid);
//    if (idWFC == nBlq-1)
//      printf ("%lu -> (%lu,%lu) - (%lu,%lu) - (%lu,%lu)\n",
//              glbalid, groupid, localid, rowSz, colSz, rowid, colid);
    auto val = iniAddOp1_struct::eval(r2.eval(0));
    if (rowid < dimR) {
      for (size_t k=colid; k<std::min(dimC,colid+colSz); k++) {
        auto prod = prdOp2_struct::eval(r1.eval(rowid,k),r2.eval(k));
        val = addOp2_struct::eval(val, prod);
      }
      l.eval(rowid,idWFC) = val;
    }

//    if (rowid < dimR) l.eval(rowid,idWFC) = val;
    return val;
  }
};

template <class LHS, class RHS1, class RHS2>
GemvC_1Row_MBlocks<LHS, RHS1, RHS2> make_GemvC_1Row_MBlocks(
    LHS &l, RHS1 &r1, RHS2 &r2, size_t nBlq) {
  return GemvC_1Row_MBlocks<LHS, RHS1, RHS2>(l, r1, r2, nBlq);
}

/**** GEMV BY COLUMNS 1 ROW x M BLOCKS USING SHARED MEMORY ****/
template <class LHS, class RHS1, class RHS2>
struct GemvC_1Row_MBlocks_ShMem {
  LHS l;
  using value_type = typename RHS2::value_type;

  RHS1 r1;
  RHS2 r2;
  size_t nBlq;

  GemvC_1Row_MBlocks_ShMem(LHS &_l, RHS1 &_r1, RHS2 &_r2, size_t &_nBlq)
      : l(_l), r1(_r1), r2(_r2), nBlq(_nBlq) {};

  size_t getSize() { return r1.getSizeR(); }

  value_type eval(size_t i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t j = 0; j < dim; j++) {
//      val += r1.eval(i, j) * r2.eval(j);
      auto prod = prdOp2_struct::eval(r1.eval(i,j),r2.eval(j));
      val = addOp2_struct::eval(val, prod);
    }
    return val;
  }

  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);
    size_t glbalid = ndItem.get_global(0);

    size_t dimR = r1.getSizeR();
    size_t dimC = r1.getSizeC();

    size_t rowSz = localSz;
    size_t colSz = (dimC    + nBlq - 1) / nBlq;

//    size_t idWFR = localid;
    size_t dimWF = (groupSz + nBlq - 1) / nBlq;
//    size_t idWFR = (groupid + dimWF - 1) / dimWF;
    size_t idWFR = (groupid / nBlq);
    size_t idWFC = (groupid % nBlq);

//    size_t rowid = (idWFR * rowSz) + idWFR;
    size_t rowid = (idWFR * rowSz) + localid;
    size_t colid = colSz * idWFC;

//    if (localid == 0) printf ("%lu\n", groupid);
//    if (idWFR == 0)
//      printf ("%lu -> (%lu,%lu) - (%lu,%lu) - (%lu,%lu)\n",
//              glbalid, groupid, localid, rowSz, colSz, rowid, colid);
    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t k=colid; k<std::min(colid+colSz,dimC); k+=rowSz) {
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
//      scratch[localid] = ((k+localid)<dimC)?r2.eval(k+localid):0.0;
      scratch[localid] = r2.eval(k+localid);
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
      if (rowid < dimR) {
        for (size_t j=k; j<std::min(k+rowSz,std::min(colid+colSz,dimC)); j++) {
//          auto prod = prdOp2_struct::eval(r1.eval(rowid,j),r2.eval(j));
          auto prod = prdOp2_struct::eval(r1.eval(rowid,j),scratch[j-k]);
          val = addOp2_struct::eval(val, prod);
        }
      }
    }

    if (rowid < dimR) l.eval(rowid,idWFC) = val;
//    l.eval(idWFC,rowid) = val;
    return val;
  }
};

template <class LHS, class RHS1, class RHS2>
GemvC_1Row_MBlocks_ShMem<LHS, RHS1, RHS2> make_GemvC_1Row_MBlocks_ShMem(
    LHS &l, RHS1 &r1, RHS2 &r2, size_t nBlq) {
  return GemvC_1Row_MBlocks_ShMem<LHS, RHS1, RHS2>(l, r1, r2, nBlq);
}

/**** GEMV BY COLUMNS 1 ROW x M BLOCKS USING SHARED MEMORY MINIMIZING SYNCHRONIZATION ****/
template <class LHS, class RHS1, class RHS2>
struct GemvC_1Row_MBlocks_ShMem_Full {
  LHS l;
  using value_type = typename RHS2::value_type;

  RHS1 r1;
  RHS2 r2;
  size_t nBlq;

  GemvC_1Row_MBlocks_ShMem_Full(LHS &_l, RHS1 &_r1, RHS2 &_r2, size_t &_nBlq)
      : l(_l), r1(_r1), r2(_r2), nBlq(_nBlq) {};

  size_t getSize() { return r1.getSizeR(); }

  value_type eval(size_t i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t j = 0; j < dim; j++) {
//      val += r1.eval(i, j) * r2.eval(j);
      auto prod = prdOp2_struct::eval(r1.eval(i,j),r2.eval(j));
      val = addOp2_struct::eval(val, prod);
    }
    return val;
  }

  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);
//    size_t glbalid = ndItem.get_global(0);

    size_t dimR = r1.getSizeR();
    size_t dimC = r1.getSizeC();

    size_t rowSz = (dimR < localSz)? dimR: localSz;
    size_t colSz = (dimC    + nBlq - 1) / nBlq;

//    size_t idWFR = localid;
    size_t dimWF = (groupSz + nBlq - 1) / nBlq;
//    size_t idWFR = (groupid + dimWF - 1) / dimWF;
//    size_t idWFR = (groupid / nBlq);
//    size_t idWFC = (groupid % nBlq);
    size_t idWFR = (groupid % dimWF);
    size_t idWFC = (groupid / dimWF);

//    size_t rowid = (idWFR * rowSz) + idWFR;
    size_t rowid = (idWFR * rowSz) + localid;
    size_t colid = colSz * idWFC;

//    if (idWFR == 0)
//      printf ("%lu -> (%lu,%lu) - (%lu,%lu) - (%lu,%lu)\n",
//              glbalid, groupid, localid, rowSz, colSz, rowid, colid);
    size_t j;
//    for (size_t k=colid+localid, j=localid; k<std::min(dimC,colid+colSz); k+=rowSz, j+=rowSz) {
    j = localid;
//    for (size_t k=colid+localid; k<std::min(dimC,colid+colSz); k+=rowSz) {
    for (size_t k=colid+localid; k<std::min(colid+colSz,dimC); k+=rowSz) {
//    for (size_t j=colid+localid, k=localid; j<std::min(dimC,colid+colSz); j+=rowSz,k+=rowSz) {
//      scratch[k+localid-colid] = r2.eval(k+localid);
//      scratch[k-colid] = (k<dimC)?r2.eval(k):0.0;
//      scratch[k-colid] = r2.eval(k);
      scratch[j] = r2.eval(k);
      j+=rowSz;
//      scratch[k] = r2.eval(j);
    }

//    if (rowid == (dimR - 1))
//      printf ("%lu -> (%lu,%lu) - (%lu,%lu) - (%lu,%lu)\n",
//              glbalid, groupid, localid, rowSz, colSz, rowid, colid);

    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    if (rowid < dimR) {
      j = 0;
      for (size_t k=colid; k<std::min(colid+colSz,dimC); k++) {
//      for (size_t k=colid, j=0; k<std::min(dimC,colid+colSz); k++, j++) {
  //    for (size_t j=colid, k=0; k<std::min(dimC,colid+colSz); j++,k++) {
  //      auto prod = prdOp2_struct::eval(r1.eval(rowid,k),r2.eval(k));
  //      auto prod = prdOp2_struct::eval(r1.eval(rowid,j),scratch[k]);
  //      auto prod = prdOp2_struct::eval(r1.eval(rowid,k),scratch[k-colid]);
  //      val = addOp2_struct::eval(val, prod);
//        val += r1.eval(rowid,k) * scratch[k-colid];
//        val += r1.eval(rowid,k) * scratch[j];
        val += r1.eval(rowid,k) * scratch[j++];
      }
      l.eval(rowid,idWFC) = val;
    }

//    if (rowid < dimR) l.eval(rowid,idWFC) = val;
    return val;
  }
};

template <class LHS, class RHS1, class RHS2>
GemvC_1Row_MBlocks_ShMem_Full<LHS, RHS1, RHS2> make_GemvC_1Row_MBlocks_ShMem_Full(
    LHS &l, RHS1 &r1, RHS2 &r2, size_t nBlq) {
  return GemvC_1Row_MBlocks_ShMem_Full<LHS, RHS1, RHS2>(l, r1, r2, nBlq);
}

/*! PrdRowMatVct.
 * @brief CLASSICAL DOT PRODUCT GEMV
 * Each thread computes a dot product, If
 * the matrix is column-major the accesses are coalescent.
 */
template <class RHS1, class RHS2>
struct PrdRowMatVct {
  RHS1 r1;
  RHS2 r2;
  size_t mult;

  using value_type = typename RHS2::value_type;

  PrdRowMatVct(RHS1 &_r1, RHS2 &_r2) : r1(_r1), r2(_r2){};

  value_type eval(size_t i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t j = 0; j < dim; j++) {
      val += r1.eval(i, j) * r2.eval(j);
    }
    return val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
  size_t getSize() { return r1.getSizeR(); }
};

template <class RHS1, class RHS2>
PrdRowMatVct<RHS1, RHS2> make_prdRowMatVct(RHS1 &r1, RHS2 &r2) {
  return PrdRowMatVct<RHS1, RHS2>(r1, r2);
}

/** PrdRowMatVctMult
 * @brief MULTITHREAD DOT PRODUCT GEMV
 * P threads compute a dot product
 * If the matrix is column-major the accesses are coalescent.
 */
template <class LHS, class RHS1, class RHS2, class RHS3>
struct PrdRowMatVctMult {
  LHS l;
  using value_type = typename RHS2::value_type;
  value_type scl;

  RHS1 r1;
  RHS2 r2;
  RHS3 r3;
  size_t nThr;

  PrdRowMatVctMult(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2, RHS3 &_r3,
                   size_t _nThr)
      : l(_l), scl(_scl), r1(_r1), r2(_r2), r3(_r3), nThr{_nThr} {};

  value_type eval(size_t i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t j = 0; j < dim; j++) {
      val += r1.eval(i, j) * r2.eval(j);
    }
    l.eval(i) = scl * val + r3.eval(i);
    return val;
  }

  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);

    size_t dimR = r1.getSizeR();
    size_t dimC = r1.getSizeC();

    size_t rowSz = (localSz / nThr);  // number of rows per each workgroup
    size_t rowid = groupid * rowSz + localid % rowSz;  // rowid of the thread

    size_t colid = localid / rowSz;  // first column on which thread works

    // Local computations
//    if ((localid == 0) && (groupid == 0))
//      printf ("nThr = %lu, localSz = %lu , groupSz = %lu\n",
//                nThr, localSz, groupSz);
    auto val = iniAddOp1_struct::eval(r2.eval(0));
    if (rowid < dimR) {
      for (size_t j = colid; j < dimC; j += nThr) {
        val += r1.eval(rowid, j) * r2.eval(j);
      }
    }

    scratch[localid] = val;
    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Reduction inside the block
    for (size_t offset = nThr >> 1; offset > 0; offset >>= 1) {
      if ((rowid < dimR) && (colid < offset)) {
        scratch[localid] += scratch[localid + offset * rowSz];
      }
      // This barrier is mandatory to be sure the data are on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
    }
    // The result is stored in lhs
    if ((rowid < dimR) && (colid == 0)) {
      l.eval(rowid) = scl * scratch[localid] + r3.eval(rowid);
    }
    return val;
  }

  size_t getSize() { return r1.getSizeR(); }
};

template <class LHS, class RHS1, class RHS2, class RHS3>
PrdRowMatVctMult<LHS, RHS1, RHS2, RHS3> make_prdRowMatVctMult(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2, RHS3 &r3,
    size_t nThr) {
  return PrdRowMatVctMult<LHS, RHS1, RHS2, RHS3>(l, scl, r1, r2, r3, nThr);
}

/*! PrdRowMatCvtMultShm.
 * @brief TWO KERNELS DOT PRODUCT GEMV
 * FIRST KERNEL: THE LOCAL COMPUTATIONS ARE MADE
 * The common data are copied to the scratch vector,
 * and later the computation finalizes.
 */
template <class LHS, class RHS1, class RHS2>
struct PrdRowMatVctMultShm {
  LHS l;
  RHS1 r1;
  RHS2 r2;
  size_t nThr;

  using value_type = typename RHS2::value_type;

  PrdRowMatVctMultShm(LHS &_l, RHS1 &_r1, RHS2 &_r2, size_t _nThr)
      : l(_l), r1(_r1), r2(_r2), nThr{_nThr} {};

  value_type eval(size_t i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t j = 0; j < dim; j++) {
      val += r1.eval(i, j) * r2.eval(j);
    }
    l.eval(i) += val;
    return val;
  }

  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);

    size_t dimR = r1.getSizeR();
    size_t dimC = r1.getSizeC();

    size_t blqSz = (groupSz + nThr - 1) / nThr;  // number of "real" workgroups
    size_t blqidR = groupid % blqSz;  // 1st row id of the current workgroup
    size_t blqidC = groupid / blqSz;  // col bloq id of the current workgroup

    size_t rowSz =
        (dimR < localSz) ? dimR : localSz;  // number of rows per each workgroup
    size_t colSz =
        (dimC + nThr - 1) / nThr;  // number of columns per each thread

    size_t rowid = blqidR * rowSz + localid;  // rowid of the current thread
    size_t colid = blqidC * colSz;  // first column of the current thread

    size_t k;

//    printf ("(%3.3lu,%3.3lu)->(%3.3lu,%3.3lu)\n", groupid, localid, rowid, colid);
//    if ((blqidC == 0) && (blqidR == 0) && (localid == 0))
//      printf ("rowSz = %lu\n", rowSz);
    // Copying  to the scratch
    k = localid;
    for (size_t j = colid + localid; j < std::min(colid+colSz,dimC); j += rowSz) {
      scratch[k] = r2.eval(j);
      k += rowSz;
    }
//    if ((blqidC == 0) && (blqidR == 0) && (localid == 0))
//      printf ("kk = %lu\n", kk);
    // This barrier is mandatory to be sure the data are on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Local computation
    auto val = iniAddOp1_struct::eval(r2.eval(0));
    if (rowid < dimR) {
      k = 0;
      for (size_t j = colid; j < std::min(colid+colSz,dimC); j++) {
        val += r1.eval(rowid, j) * scratch[k++];
      }
      // The result is stored in lhs
      l.eval(rowid, blqidC) = val;
    }

    return val;
  }

  size_t getSize() { return r1.getSizeR(); }
};

template <class LHS, class RHS1, class RHS2>
PrdRowMatVctMultShm<LHS, RHS1, RHS2> make_prdRowMatVctMultShm(LHS &l, RHS1 &r1,
                                                              RHS2 &r2,
                                                              size_t nThr) {
  return PrdRowMatVctMultShm<LHS, RHS1, RHS2>(l, r1, r2, nThr);
}

/*! AddPrdRowMatVctMultShm.
 * @brief SECOND KERNEL: REDUCTION OF THE LOCAL COMPUTATIONS
 */
template <class LHS, class RHS1, class RHS2>
struct AddPrdRowMatVctMultShm {
  using value_type = typename RHS2::value_type;

  LHS l;
  value_type scl;
  RHS1 r1;
  RHS2 r2;

  AddPrdRowMatVctMultShm(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2)
      : l(_l), scl(_scl), r1(_r1), r2(_r2){};

  value_type eval(size_t i) {
    auto dimC = r1.getSizeC();

//    if (i==0) printf ("dimC = %lu\n", dimC);
    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t j = 0; j < dimC; j++) {
      val += r1.eval(i, j);
    }
    l.eval(i) = scl * val + r2.eval(i);
    return val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }

  size_t getSize() { return r1.getSizeR(); }
};

template <class LHS, class RHS1, class RHS2>
AddPrdRowMatVctMultShm<LHS, RHS1, RHS2> make_addPrdRowMatVctMultShm(
    LHS &l, typename RHS1::value_type &scl, RHS1 &r1, RHS2 &r2) {
  return AddPrdRowMatVctMultShm<LHS, RHS1, RHS2>(l, scl, r1, r2);
}

/*! RedRowMatVct.
 * @brief CLASSICAL AXPY GEMV
 */
// #define ORIGINAL_CODE 1
template <class RHS1, class RHS2>
struct RedRowMatVct {
  RHS1 r1;
  RHS2 r2;
  size_t warpSize;

  using value_type = typename RHS2::value_type;

  RedRowMatVct(RHS1 &_r1, RHS2 &_r2, size_t _warpSize)
      : r1(_r1), r2(_r2), warpSize(_warpSize){};

#if ORIGINAL_CODE
  value_type eval(size_t i) {
    auto dim = r2.getSize();
    value_type v[warpSize];
    for (size_t w = 0; w < warpSize; w++) {
      auto valWI = iniAddOp1_struct::eval(r2.eval(0));
      for (size_t j = w; j < dim; j += warpSize) {
        valWI += r1.eval(i, j) * r2.eval(j);
      }
      v[w] = valWI;
    }
    auto valWG = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t w = 0; w < warpSize; w++) {
      valWG += v[w];
    }
    return valWG;
  }
#else
  value_type eval(size_t i) {
    auto dim = r2.getSize();
    auto valWG = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t j = 0; j < dim; j++) {
      valWG += r1.eval(i, j) * r2.eval(j);
    }
    return valWG;
  }
#endif  // ORIGINAL_CODE

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }

#if BLAS_EXPERIMENTAL
  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    size_t Pieces = 2;

    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);
    size_t globalid = ndItem.get_global(0);
    size_t globalSz = ndItem.get_global_range(0);

    size_t dimR = r1.getSizeR();
    size_t dimC = r1.getSizeC();

    size_t blqSz = groupSz;  // number of workgroups
    // row blq id of the current workgroup
    size_t blqidR = (groupid + (Pieces * blqSz) - 1) / (Pieces * blqSz);
    size_t blqidC =
        groupid % (Pieces * blqSz);  // 1st col id of the current workgroup

    // number of columns per each workgroup
    size_t colSz = (dimC < (Pieces * localSz)) ? dimC : Pieces * localSz;
    // number of rows per each thread
    size_t rowSz = (dimR + blqidR - 1) / blqidR;

    size_t colid = blqidC * colSz + localid;  // colid of the current thread
    size_t rowid = blqidR * rowSz;            // first row of the current thread

    value_type val;
#if BLAS_EXPERIMENTAL
    // Local computations
    while (rowid < dimR) {
      auto val = iniAddOp1_struct::eval(r2.eval(0));
      for (size_t j = colid; j < dimC; j += colSz) {
        val += r1.eval(rowid, j) * r2.eval(j);
      }
      scratch[localid] = val;
      // This barrier is mandatory to be sure the data is on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
      // Reduction inside the block
      for (size_t offset = nThr >> 1; offset > 0; offset >>= 1) {
        if ((rowid < dimR) && (colid < offset)) {
          scratch[localid] += scratch[localid + offset];
        }
        // This barrier is mandatory to be sure the data are on the shared
        // memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);
      }
      // The result is stored in lhs
      if ((rowid < dimR) && (colid == 0)) {
        l.eval(rowid, blqidC) = scl * scratch[localid] + r3.eval(rowid);
      }
      rowid += rowSz;
    }
#endif  // BLAS_EXPERIMENTAL
    return val;
  }
#endif  // BLAS_EXPERIMENTAL

#if BLAS_EXPERIMENTAL
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
#endif  // BLAS_EXPERIMENTAL
  size_t getSize() { return r1.getSizeR(); }
};

template <class RHS1, class RHS2>
RedRowMatVct<RHS1, RHS2> make_redRowMatVct(RHS1 &r1, RHS2 &r2,
                                           size_t warpSize) {
  return RedRowMatVct<RHS1, RHS2>(r1, r2, warpSize);
}

/*! ModifRank1.
 * @brief RANK 1 UPDATE
 */
template <class RHS1, class RHS2, class RHS3>
struct ModifRank1 {
  RHS1 r1;
  RHS2 r2;
  RHS3 r3;

  using value_type = typename RHS2::value_type;

  ModifRank1(RHS1 &_r1, RHS2 &_r2, RHS3 &_r3) : r1(_r1), r2(_r2), r3(_r3){};

  value_type eval(size_t i) {
    auto size = (r1.getAccess()) ? r1.getSizeC() : r1.getSizeR();
    auto row = (r1.getAccess()) ? (i / size) : (i % size);
    auto col = (r1.getAccess()) ? (i % size) : (i / size);

    auto val = r2.eval(row) * r3.eval(col);

    return val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }

  size_t getSize() { return r1.getSize(); }
};

template <class RHS1, class RHS2, class RHS3>
ModifRank1<RHS1, RHS2, RHS3> make_modifRank1(RHS1 &r1, RHS2 &r2, RHS3 &r3) {
  return ModifRank1<RHS1, RHS2, RHS3>(r1, r2, r3);
}

}  // namespace blas

#endif  // BLAS2_TREES_HPP
