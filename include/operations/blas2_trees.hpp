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

  bool valid_thread (cl::sycl::nd_item<1> ndItem) {
    return ((ndItem.get_global(0) < getSize()));
  }

  value_type eval(size_t i) {
    auto dimR = r.getSizeR();
    auto dimC = r.getSizeC();

    auto val = iniAddOp1_struct::eval(r.eval(0));
    if (i < dimR) {
      for (size_t j = 0; j < dimC; j++) {
        val += r.eval(i, j);
      }
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

/**** GEMV BY ROWS M ROWS x N BLOCK ****/
//#define GROUP_ROWS 1 // Not useful for GEMV by rows
template <unsigned int interLoop, bool Lower, bool Diag, bool Upper, bool Unit,
          class LHS, class RHS1, class RHS2>
struct Gemv_Row {
  LHS  l;
  RHS1 r1;
  RHS2 r2;
  size_t nWG_row;
  size_t nWG_col;
  size_t shrMemSize;

  using value_type = typename RHS2::value_type;

  Gemv_Row(LHS &_l,RHS1 &_r1, RHS2 &_r2, size_t &_nWG_row, size_t &_nWG_col, size_t &_shrMemSize)
    : l(_l), r1(_r1), r2(_r2), nWG_row(_nWG_row), nWG_col(_nWG_col), shrMemSize(_shrMemSize) {};

  size_t getSize() { return r1.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(size_t i) { // NOT VERIFIED
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t j = 0; j < dim; j++) {
//      val += r1.eval(i, j) * r2.eval(j);
      auto prod = prdOp2_struct::eval(r1.eval(i,j),r2.eval(j));
      val = addOp2_struct::eval(val, prod);
    }
    return l.eval(i) = val;
  }
/*
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
*/
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);
    size_t glbalid = ndItem.get_global(0);
    size_t glbalSz = ndItem.get_global_range(0);

    size_t dimR = r1.getSizeR();
    size_t dimC = r1.getSizeC();

    size_t rowSz = (dimR + nWG_row - 1) / nWG_row;
    size_t colSz = (dimC<localSz)? localSz: (dimC + nWG_col - 1) / nWG_col;
    size_t shrSz = shrMemSize / localSz;

    size_t idWFR = groupid / nWG_col;  // row bloq id of the current workgroup
    size_t idWFC = groupid % nWG_col;  // col blq id of the current workgroup
//    size_t idWFR = (groupid % nWG_row);
//    size_t idWFC = (groupid / nWG_row);
    size_t dimWFC = ((dimC + (localSz*nWG_col) - 1) /
                             (localSz*nWG_col)) * localSz;

    size_t frs_row = idWFR*rowSz;
    size_t lst_row = std::min(dimR,frs_row+rowSz);

    size_t frs_col = idWFC * dimWFC + interLoop*localid;
    size_t lst_col = std::min(dimC,frs_col+dimWFC);

    size_t id_col_thr = idWFC * localSz + localid;

/*
    printf ("(%lu) -> (%lu,%lu) - (%lu,%lu) - (%lu)\n",
        glbalid, frs_row, lst_row, frs_col, lst_col, id_col_thr);
*/
//    if ((!Lower) && (glbalid == 0)) printf ("Upper\n");
//    if ((!Upper) && (glbalid == 0)) printf ("Lower\n");
//    if ((Diag)   && (glbalid == 0)) printf ("Diag\n");
//    if ((Unit)   && (glbalid == 0)) printf ("Unit\n");

    value_type val = addOp2_struct::init(r2);
    // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
    // TO SOLVE IT, USE GLOBAL VALUES OF frs_col AND lst_col
    if ((!Upper && (((idWFC*dimWFC)+((!Diag)?1:0))>(lst_row-1))) ||
        (!Lower && ((frs_row+((!Diag)?1:0))>((idWFC*dimWFC+dimWFC)-1)))) {
//          if (localid == 0)
//            printf ("%lu -> (%lu,%lu) - (%lu,%lu)\n",
//                      glbalid, idWFR*dimWFR, idWFR*dimWFR+dimWFR, frs_col, lst_col);
//      value_type val = iniAddOp1_struct::eval(r2.eval(0));
      for (size_t rowid = frs_row; rowid < lst_row; rowid += localSz) {
        l.eval(rowid,id_col_thr) = val;
      }
    } else {

  #ifdef GROUP_ROWS
  //      for (size_t row=0, id_row=rowid; row<blqSz; row++, id_row++) {
      for (size_t id_row=frs_row;(id_row<lst_row); id_row++) {
        l.eval(id_row,id_col_thr) = val;
      }
  #endif
      if (interLoop == 1) {
  #ifdef GROUP_ROWS
        for (size_t id_col = frs_col; id_col < lst_col; id_col += localSz) {
  //      for (size_t id_col = frs_col; id_col < dimC; id_col += localSz*nWG_col) {
          auto elm = r2.eval(id_col);
          for (size_t row=0, id_row=frs_row; (id_row<lst_row); row++, id_row++) {
  //          for (size_t row=0, id_row=rowid; (row<blqSz); row++, id_row++) {
            if (Lower && Upper && Diag && !Unit) {
              auto prod = prdOp2_struct::eval(r1.eval(id_row,id_col),elm);
              l.eval(id_row,id_col_thr) =
                      addOp2_struct::eval(l.eval(id_row,id_col_thr), prod);
            } else {
//              if ((Lower && ((id_col+((Diag&&Unit)?1:0)) <= id_row)) ||
              if ((Lower && ((id_col+((!Diag||Unit)?1:0)) <= id_row)) ||
                (Upper && (id_col >= (id_row+((!Diag||Unit)?1:0))))) {
//                (Upper && (id_col >= (id_row+((Diag&&Unit)?1:0))))) {
                auto prod = prdOp2_struct::eval(r1.eval(id_row,id_col),elm);
                l.eval(id_row,id_col_thr) =
                        addOp2_struct::eval(l.eval(id_row,id_col_thr), prod);
              }
              if (Diag && Unit && (id_row == id_col)) {
                l.eval(id_row,id_col_thr) =
                        addOp2_struct::eval(l.eval(id_row,id_col_thr),
                                            r1.eval(id_row,id_col));
              }
            }
        }
  #else
  //        for (size_t row=0, id_row=rowid; row<blqSz; row++, id_row++) {
        if (id_col_thr < dimC) {
//          printf ("(%lu) -> (%lu,%lu) - (%lu,%lu) - (%lu)\n",
//              glbalid, frs_row, lst_row, frs_col, lst_col, id_col_thr);
          for (size_t row=0, id_row=frs_row; (id_row<lst_row); row++, id_row++) {
            val = addOp2_struct::init(r2);
            for (size_t id_col = frs_col; id_col < lst_col; id_col += localSz) {
              if (Lower && Upper && Diag && !Unit) {
                auto prod = prdOp2_struct::eval(r1.eval(id_row,id_col),r2.eval(id_col));
                val = addOp2_struct::eval(val, prod);
//                val += (r1.eval(id_row,id_col) * r2.eval(id_col));
              } else {
//                if ((Lower && ((id_col+((Diag&&Unit)?1:0)) <= id_row)) ||
                if ((Lower && ((id_col+((!Diag||Unit)?1:0)) <= id_row)) ||
                    (Upper && (id_col >= (id_row+((!Diag||Unit)?1:0))))) {
//                    (Upper && (id_col >= (id_row+((Diag&&Unit)?1:0))))) {
                  auto prod = prdOp2_struct::eval(r1.eval(id_row,id_col),r2.eval(id_col));
                  val = addOp2_struct::eval(val, prod);
                }
                if (Diag && Unit && (id_row == id_col)) {
                  val = addOp2_struct::eval(val, r1.eval(id_row,id_col));
                }
              }
            }
            l.eval(id_row,id_col_thr) = val;
          }
        }
  #endif
      } else {
        for (size_t row=0, id_row=frs_row; (id_row<lst_row); row++, id_row++) {
  //        for (size_t row=0, id_row=rowid; row<blqSz; row++, id_row++) {
          val = addOp2_struct::init(r2);
          for (size_t id_col = frs_col; id_col < lst_col; id_col += localSz*interLoop) {
            auto lst_k_int = std::min(id_col+interLoop,lst_col);
//            for (size_t k_int=((Lower)?id_col:std::max(row+((Diag&&Unit)?1:0),id_col));
            for (size_t k_int=((Lower)?id_col:std::max(row+((!Diag||Unit)?1:0),id_col));
                        k_int<((Upper)?lst_k_int:std::min(row+((!Diag||Unit)?0:1),lst_k_int)); k_int++) {
//                        k_int<((Upper)?lst_k_int:std::min(row+((Diag&&Unit)?0:1),lst_k_int)); k_int++) {
//            for (size_t k_int=id_col; k_int<std::min(id_col+interLoop,lst_col);k_int++) {
              auto prod = prdOp2_struct::eval(r1.eval(id_row,k_int),r2.eval(k_int));
              val = addOp2_struct::eval(val, prod);
            }
          }
          l.eval(id_row,id_col_thr) = val;
        }
      }
    }
//    return addOp2_struct::init(r2);
    return val;
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

    size_t rowSz = (dimR + nWG_row - 1) / nWG_row;
    size_t colSz = (dimC<localSz)? localSz: (dimC + nWG_col - 1) / nWG_col;
    size_t shrSz = shrMemSize / localSz;

    size_t idWFR = groupid / nWG_col;  // row bloq id of the current workgroup
    size_t idWFC = groupid % nWG_col;  // col blq id of the current workgroup
//    size_t idWFR = (groupid % nWG_row);
//    size_t idWFC = (groupid / nWG_row);
//    size_t dimWFC = ((dimC + (localSz*nWG_col*interLoop) - 1) /
//                             (localSz*nWG_col*interLoop)) * (localSz*interLoop);
    size_t dimWFC = ((dimC + (localSz*nWG_col) - 1) /
                             (localSz*nWG_col)) * localSz;
//    size_t dimWFC = ((dimC+(localSz*nWG_col)-1)/(localSz*nWG_col)) * localSz;

    size_t frs_row = idWFR*rowSz;
    size_t lst_row = std::min(dimR,frs_row+rowSz);

//    size_t frs_col = interLoop * (idWFC * dimWFC + localid);
    size_t frs_col = idWFC * dimWFC + interLoop*localid;
//    size_t lst_col = std::min(dimC,frs_col+dimWFC*interLoop);
    size_t lst_col = std::min(dimC,frs_col+dimWFC);

//    if (glbalid == 0)
//      printf ("A , (%lu) -> (%lu,%lu) - (%lu,%lu)\n",
//        glbalid, frs_row, lst_row, frs_col, lst_col);

//    if ((!Upper) && (glbalid == 0)) printf ("Lower\n");
//    if ((!Lower) && (glbalid == 0)) printf ("Upper\n");
//    if ((Diag)   && (glbalid == 0)) printf ("Diag\n");
//    if ((Unit)   && (glbalid == 0)) printf ("Unit\n");


    // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
    // TO SOLVE IT, USE GLOBAL VALUES OF frs_col AND lst_col
    if ((!Upper && (((idWFC*dimWFC)+((!Diag)?1:0))>(lst_row-1))) ||
        (!Lower && ((frs_row+((!Diag)?1:0))>((idWFC*dimWFC+dimWFC)-1)))) {
//          if (localid == 0)
//            printf ("%lu -> (%lu,%lu) - (%lu,%lu)\n",
//                      glbalid, idWFR*dimWFR, idWFR*dimWFR+dimWFR, frs_col, lst_col);
      if (localid == 0) {
        value_type val = iniAddOp1_struct::eval(r2.eval(0));
        for (size_t rowid = frs_row; rowid < lst_row; rowid ++) {
          l.eval(rowid,idWFC) = val;
        }
      }
    } else {

      for (size_t rowid=frs_row; rowid<lst_row; rowid+=shrSz) {
  /*
        if (rowid > frs_row)
          // This barrier is mandatory to be sure the data is on the shared memory
          ndItem.barrier(cl::sycl::access::fence_space::local_space);
  */
        value_type val = addOp2_struct::init(r2);
        auto blqSz = std::min(shrSz,lst_row-rowid);
  #ifdef GROUP_ROWS
  //      for (size_t row=0, id_row=frs_row;(id_row<lst_row); row++, id_row++) {
        for (size_t row=0, id_row=rowid; row<blqSz; row++, id_row++) {
          shrMem[row*localSz+localid] = val;
        }
  #endif
        if (interLoop == 1) {
  #ifdef GROUP_ROWS
          for (size_t id_col = frs_col; id_col < lst_col; id_col += localSz) {
    //      for (size_t id_col = frs_col; id_col < dimC; id_col += localSz*nWG_col) {
            auto elm = r2.eval(id_col);
  //          for (size_t row=0, id_row=frs_row; (id_row<lst_row); row++, id_row++) {
            for (size_t row=0, id_row=rowid; (row<blqSz); row++, id_row++) {
              if (Lower && Upper && Diag && !Unit) {
                auto prod = prdOp2_struct::eval(r1.eval(id_row,id_col),elm);
                shrMem[row*localSz+localid] =
                        addOp2_struct::eval(shrMem[row*localSz+localid], prod);
              } else {
//                if ((Lower && ((id_col+((Diag&&Unit)?1:0)) <= id_row)) ||
                if ((Lower && ((id_col+((!Diag||Unit)?1:0)) <= id_row)) ||
                    (Upper && (id_col >= (id_row+((!Diag||Unit)?1:0))))) {
//                    (Upper && (id_col >= (id_row+((Diag&&Unit)?1:0))))) {
                  auto prod = prdOp2_struct::eval(r1.eval(id_row,id_col),elm);
                  shrMem[row*localSz+localid] =
                          addOp2_struct::eval(shrMem[row*localSz+localid], prod);
                }
                if (Diag && Unit && (id_row == id_col)) {
                  shrMem[row*localSz+localid] =
                          addOp2_struct::eval(shrMem[row*localSz+localid],
                                               r1.eval(id_row,id_col));
                }
              }
            }
          }
  #else
  //        for (size_t row=0, id_row=frs_row; (id_row<lst_row); row++, id_row++) {
          for (size_t row=0, id_row=rowid; row<blqSz; row++, id_row++) {
//            val = addOp2_struct::init(r2);
            val = (Diag && Unit && ((id_row >= frs_col) && (id_row < lst_col) &&
			             (((id_row-frs_col)%localSz) == 0)))?
                    	r1.eval(id_row,id_row): addOp2_struct::init(r2);
            for (size_t id_col = frs_col; id_col < lst_col; id_col += localSz) {
              if (Lower && Upper && Diag && !Unit) {
                auto prod = prdOp2_struct::eval(r1.eval(id_row,id_col),r2.eval(id_col));
                val = addOp2_struct::eval(val, prod);
              } else {
//                if ((Lower && ((id_col+((Diag&&Unit)?1:0)) <= id_row)) ||
                if ((Lower && ((id_col+((!Diag||Unit)?1:0)) <= id_row)) ||
                    (Upper && (id_col >= (id_row+((!Diag||Unit)?1:0))))) {
//                    (Upper && (id_col >= (id_row+((Diag&&Unit)?1:0))))) {
                  auto prod = prdOp2_struct::eval(r1.eval(id_row,id_col),r2.eval(id_col));
                  val = addOp2_struct::eval(val, prod);
                }
/*
                if (Diag && Unit && (id_row == id_col)) {
                  val = addOp2_struct::eval(val, r1.eval(id_row,id_col));
                }
*/
              }
            }
            shrMem[row*localSz+localid] = val;
  //          printf ("B , (%lu) - (%f)\n", glbalid, val);
          }
  #endif
        } else {
          for (size_t row=0, id_row=rowid; row<blqSz; row++, id_row++) {
            val = addOp2_struct::init(r2);
            for (size_t id_col = frs_col; id_col < lst_col; id_col += localSz*interLoop) {
              for (size_t k_int=id_col; k_int<std::min(id_col+interLoop,lst_col);k_int++) {
                if (Lower && Upper && Diag && !Unit) {
                  auto prod = prdOp2_struct::eval(r1.eval(id_row,k_int),r2.eval(k_int));
                  val = addOp2_struct::eval(val, prod);
                } else {
//                  if ((Lower && ((id_col+((Diag&&Unit)?1:0)) <= id_row)) ||
                  if ((Lower && ((id_col+((!Diag||Unit)?1:0)) <= id_row)) ||
                      (Upper && (id_col >= (id_row+((!Diag||Unit)?1:0))))) {
//                      (Upper && (id_col >= (id_row+((Diag&&Unit)?1:0))))) {
                    auto prod = prdOp2_struct::eval(r1.eval(id_row,k_int),r2.eval(k_int));
                    val = addOp2_struct::eval(val, prod);
                  }
                  if (Diag && Unit && (id_row == id_col)) {
                    val = addOp2_struct::eval(val, r1.eval(id_row,k_int));
                  }
                }
              }
            }
            shrMem[row*localSz+localid] = val;
          }
        }

        // This barrier is mandatory to be sure the data is on the shared memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);
        // Reduction inside the block
        for (size_t offset = localSz >> 1; offset > 0; offset >>= 1) {
          if (localid < offset) {
  //          for (size_t row=0, id_row=frs_row; (id_row<lst_row); row++, id_row++) {
            for (size_t row=0, id_row=rowid; row<blqSz; row++, id_row++) {
              shrMem[row*localSz+localid] =
                  addOp2_struct::eval(shrMem[row*localSz+localid],
                                      shrMem[row*localSz+localid+offset]);
            }
          }
          // This barrier is mandatory to be sure the data are on the shared memory
          ndItem.barrier(cl::sycl::access::fence_space::local_space);
        }
        if (localid == 0) {
  //        for (size_t row=0, id_row=frs_row; (id_row<lst_row); row++, id_row++) {
          for (size_t row=0, id_row=rowid; row<blqSz; row++, id_row++) {
            l.eval(id_row,idWFC) = shrMem[row*localSz];
          }
        }
      }
    }

    return addOp2_struct::init(r2);
  }
};

template <unsigned int interLoop=1,
          bool Lower=true, bool Diag=true, bool Upper=true, bool Unit=false,
          typename LHS, typename RHS1, typename RHS2>
Gemv_Row<interLoop, Lower, Diag, Upper, Unit, LHS, RHS1, RHS2>
    make_Gemv_Row(LHS &l, RHS1 &r1, RHS2 &r2, size_t nWG_row, size_t nWG_col, size_t shrMemSize) {
  return Gemv_Row<interLoop, Lower, Diag, Upper, Unit, LHS, RHS1, RHS2>(l, r1, r2, nWG_row, nWG_col, shrMemSize);
}

/**** GEMV BY COLUMNS 1 ROW x M BLOCKS USING PROPERLY THE SHARED MEMORY ****/

//template <class LHS, class RHS1, class RHS2>
template <bool Lower, bool Diag, bool Upper, bool Unit,
          class LHS, class RHS1, class RHS2>
struct Gemv_Col {
  LHS l;
  RHS1 r1;
  RHS2 r2;
  size_t nWG_row;
  size_t nWG_col;
  size_t shrMemSize;

  using value_type = typename RHS2::value_type;

  Gemv_Col(LHS &_l, RHS1 &_r1, RHS2 &_r2, size_t &_nWG_row, size_t &_nWG_col, size_t &_shrMemSize)
      : l(_l), r1(_r1), r2(_r2), nWG_row(_nWG_row), nWG_col(_nWG_col), shrMemSize(_shrMemSize) {};

  size_t getSize() { return r1.getSizeR(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(size_t i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t j = 0; j < dim; j++) {
//      val += r1.eval(i, j) * r2.eval(j);
      auto prod = prdOp2_struct::eval(r1.eval(i,j),r2.eval(j));
      val = addOp2_struct::eval(val, prod);
    }
    return l.eval(i) = val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);
    size_t glbalid = ndItem.get_global(0);

    size_t dimR = r1.getSizeR();
    size_t dimC = r1.getSizeC();

//    size_t rowSz = (dimR < localSz)? dimR:
//                    (dimR + nWG_row - 1) / nWG_row;
    size_t colSz = (dimC + nWG_col - 1) / nWG_col;

//    size_t idWFR = (groupid / nWG_col);
//    size_t idWFC = (groupid % nWG_col);
    size_t idWFR = (groupid % nWG_row);
    size_t idWFC = (groupid / nWG_row);
    size_t dimWFR = (dimR + (localSz*nWG_row) - 1) / (localSz*nWG_row) * localSz;

    size_t frs_row = idWFR * dimWFR + localid;
    size_t lst_row = std::min(dimR,frs_row + dimWFR);

    size_t frs_col = idWFC*colSz;
    size_t lst_col = std::min(dimC,frs_col+colSz);

//    if ((!Upper) && (glbalid == 0)) printf ("Lower\n");
//    if ((!Lower) && (glbalid == 0)) printf ("Upper\n");
//    if ((Diag)   && (glbalid == 0)) printf ("Diag\n");
//    if ((Unit)   && (glbalid == 0)) printf ("Unit\n");

    // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
    // TO SOLVE IT, USE GLOBAL VALUES OF frs_row AND lst_row
    if ((!Upper && ((frs_col+((!Diag)?1:0))>((idWFR*dimWFR+dimWFR)-1))) ||
        (!Lower && ((idWFR*dimWFR+((!Diag)?1:0))>(lst_col-1)))) {
    //      if (localid == 0)
    //        printf ("%lu -> (%lu,%lu) - (%lu,%lu)\n",
    //                  glbalid, idWFR*dimWFR, idWFR*dimWFR+dimWFR, frs_col, lst_col);
      auto val = iniAddOp1_struct::eval(r2.eval(0));
      for (size_t rowid = frs_row; rowid < lst_row; rowid += localSz) {
        l.eval(rowid,idWFC) = val;
      }
    } else {
      // The product is computed
      for (size_t rowid = frs_row; rowid < lst_row; rowid += localSz) {
        // The initial value of val is different for the first iteration
  //      auto val = iniAddOp1_struct::eval(r2.eval(0));
        auto val = (Diag && Unit && ((rowid >= frs_col) && (rowid < lst_col)))?
                    r1.eval(rowid,rowid): iniAddOp1_struct::eval(r2.eval(0));
  //      for (size_t id_col=frs_col; id_col<lst_col; id_col++) {
  //      if (glbalid == 0)
  //        printf ("(%lu) -> (%lu, %lu, %lu) - (%lu, %lu) - (%lu,%lu)\n",
  //                  glbalid, rowid, frs_row, lst_row, frs_col, lst_col,
  //                  ((Lower)?frs_col:std::max(rowid+((Diag&&Unit)?1:0),frs_col)),
  //                  ((Upper)?lst_col:std::min(rowid+((Diag&&Unit)?0:1),lst_col)));
//        for (size_t id_col=((Lower)?frs_col:std::max(rowid+((Diag&&Unit)?1:0),frs_col));
        for (size_t id_col=((Lower)?frs_col:std::max(rowid+((!Diag||Unit)?1:0),frs_col));
                    id_col<((Upper)?lst_col:std::min(rowid+((!Diag||Unit)?0:1),lst_col)); id_col++) {
//                    id_col<((Upper)?lst_col:std::min(rowid+((Diag&&Unit)?0:1),lst_col)); id_col++) {
          auto prod = prdOp2_struct::eval(r1.eval(rowid,id_col), r2.eval(id_col));
          val = addOp2_struct::eval(val, prod);
  //        val += r1.eval(rowid,id_col) * r2.eval(id_col);
        }
        // The result is stored in the correct component
        l.eval(rowid,idWFC) = val;
      }
    }

    return l.eval(frs_row,idWFC);
  }

  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);
    size_t glbalid = ndItem.get_global(0);

    size_t dimR = r1.getSizeR();
    size_t dimC = r1.getSizeC();

    size_t rowSz = (dimR < localSz)? dimR: localSz;
    size_t colSz = (dimC + nWG_col - 1) / nWG_col;

//    size_t idWFR = (groupid / nWG_col);
//    size_t idWFC = (groupid % nWG_col);
    size_t idWFR = (groupid % nWG_row);
    size_t idWFC = (groupid / nWG_row);
    size_t dimWFR = (dimR + (localSz*nWG_row) - 1) / (localSz*nWG_row) * localSz;

    size_t frs_row = idWFR * dimWFR + localid;
    size_t lst_row = std::min(dimR,frs_row + dimWFR);

    size_t frs_col = idWFC*colSz;
    size_t lst_col = std::min(dimC,frs_col+colSz);

//    if ((!Upper) && (glbalid == 0)) printf ("Lower\n");
//    if ((!Lower) && (glbalid == 0)) printf ("Upper\n");
//    if ((Unit)   && (glbalid == 0)) printf ("Unit\n");

    // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
    // TO SOLVE IT, USE GLOBAL VALUES OF frs_row AND lst_row
    if ((!Upper && ((frs_col+((!Diag)?1:0))>((idWFR*dimWFR+dimWFR)-1))) ||
        (!Lower && ((idWFR*dimWFR+((!Diag)?1:0))>(lst_col-1)))) {
//      if (localid == 0)
//        printf ("%lu -> (%lu,%lu) - (%lu,%lu)\n",
//                  glbalid, idWFR*dimWFR, idWFR*dimWFR+dimWFR, frs_col, lst_col);
      auto val = iniAddOp1_struct::eval(r2.eval(0));
      for (size_t rowid = frs_row; rowid < lst_row; rowid += localSz) {
        l.eval(rowid,idWFC) = val;
      }
    } else {
      // The computation are made in blocks of shrMemSize elements
      for (size_t colid=frs_col; colid<lst_col; colid+=shrMemSize) {
        if (colid > frs_col)
          // This barrier is mandatory to be sure the data is on the shared memory
          ndItem.barrier(cl::sycl::access::fence_space::local_space);

        auto blqSz = std::min(shrMemSize,lst_col-colid);
        // Copy a block of elements of vector r2 to the shared memory,
        // executing the expresion tree if it is needed
        for (size_t col=localid; (col<blqSz); col+=localSz) {
          shrMem[col] = r2.eval(colid+col);
        }
        // This barrier is mandatory to be sure the data is on the shared memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);

        // The product is computed
        for (size_t rowid = frs_row; rowid < lst_row; rowid += localSz) {
          // The initial value of val is different for the first iteration
/*
          auto val = (colid == frs_col)?
                      iniAddOp1_struct::eval(r2.eval(0)):
                      l.eval(rowid,idWFC);
*/
          auto val = ((colid == frs_col)?
                      iniAddOp1_struct::eval(r2.eval(0)):
                      l.eval(rowid,idWFC))+
                      ((Diag && Unit && ((rowid >= colid) && (rowid < colid+blqSz)))?
                    	r1.eval(rowid,rowid): iniAddOp1_struct::eval(r2.eval(0)));
          for (size_t id_col=colid, col=0; col<blqSz; id_col++, col++) {
            if (Lower && Upper && Diag && !Unit) {
              auto prod = prdOp2_struct::eval(r1.eval(rowid,id_col), shrMem[col]);
              val = addOp2_struct::eval(val, prod);
  //            val += r1.eval(rowid,id_col) * shrMem[col];
            } else {
  //            if (glbalid == 1)
  //              printf ("(%lu) -> (%lu, %lu, %lu) - (%lu, %lu, %lu) - (%lu,%lu)\n",
  //                        glbalid, rowid, frs_row, lst_row, id_col, frs_col, lst_col,
  //                        (id_col+((Diag&&Unit)?1:0)), (rowid+((Diag&&Unit)?1:0)));
//              if ((Lower && ((id_col+((Diag&&Unit)?1:0)) <= rowid)) ||
              if ((Lower && ((id_col+((!Diag||Unit)?1:0)) <= rowid)) ||
                  (Upper && (id_col >= (rowid+((!Diag||Unit)?1:0))))) {
//                  (Upper && (id_col >= (rowid+((Diag&&Unit)?1:0))))) {
                auto prod = prdOp2_struct::eval(r1.eval(rowid,id_col), shrMem[col]);
                val = addOp2_struct::eval(val, prod);
              }
/*
              if (Diag && Unit && (rowid == id_col)) {
                val = addOp2_struct::eval(val, r1.eval(rowid,id_col));
              }
*/
            }
          }
          // The result is stored in the correct component
          l.eval(rowid,idWFC) = val;
        }
      }
    }
    return l.eval(frs_row,idWFC);
  }
};

//template <class LHS, class RHS1, class RHS2>
template <bool Lower=true, bool Diag=true, bool Upper=true, bool Unit=false,
          class LHS, class RHS1, class RHS2>
Gemv_Col<Lower, Diag, Upper, Unit, LHS, RHS1, RHS2> make_Gemv_Col(
    LHS &l, RHS1 &r1, RHS2 &r2, size_t nWG_row, size_t nWG_col, size_t shrMemSize) {
  return Gemv_Col<Lower, Diag, Upper, Unit, LHS, RHS1, RHS2>(l, r1, r2, nWG_row, nWG_col, shrMemSize);
}

/**** GER BY ROWS M ROWS x N BLOCK USING PROPERLY THE SHARED MEMORY ****/
//template <class LHS, class RHS1, class RHS2>
template <bool Single, bool Lower, bool Diag, bool Upper,
          class LHS, class RHS1, class RHS2>
struct Ger_Row {
  LHS  l;
  RHS1 r1;
  RHS2 r2;
  size_t nWG_row;
  size_t nWG_col;
  size_t shrMemSize;

  using value_type = typename RHS2::value_type;
  value_type scl;

  Ger_Row(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2, size_t &_nWG_row, size_t &_nWG_col, size_t &_shrMemSize)
    : l(_l), scl(_scl), r1(_r1), r2(_r2), nWG_row(_nWG_row), nWG_col(_nWG_col), shrMemSize(_shrMemSize) { };

  size_t getSize() { return r1.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(size_t i) {
    auto size = (l.getAccess()) ? l.getSizeC() : l.getSizeR();
    auto row = (l.getAccess()) ? (i / size) : (i % size);
    auto col = (l.getAccess()) ? (i % size) : (i / size);

    auto val = scl * r1.eval(row) * r2.eval(col);

    return l.eval(i) += val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);
    size_t glbalid = ndItem.get_global(0);
    // size_t glbalSz = ndItem.get_global_range(0);

    size_t dimR = l.getSizeR();
    size_t dimC = l.getSizeC();

    size_t rowSz = (dimR + nWG_row - 1) / nWG_row;
    size_t colSz = (dimR<localSz)? localSz: (dimC + nWG_col - 1) / nWG_col;
    size_t shrSz = shrMemSize;

    size_t idWFR = (groupid % nWG_row);
    size_t idWFC = (groupid / nWG_row);
//    size_t idWFR  = groupid % nWG_row;  // row bloq id of the current workgroup
//    size_t idWFC  = groupid / nWG_row;  // col blq id of the current workgroup
    size_t dimWFC = (dimC + (localSz*nWG_col) - 1) / (localSz*nWG_col) * localSz;

    size_t frs_row = idWFR*rowSz;
    size_t lst_row = std::min(dimR,frs_row+rowSz);

    size_t frs_col = idWFC * dimWFC + localid;
    size_t lst_col = std::min(dimC,frs_col+dimWFC);

//    if ((!Upper) && (glbalid == 0)) printf ("Lower\n");
//    if ((!Lower) && (glbalid == 0)) printf ("Upper\n");
//    if ((Diag)   && (glbalid == 0)) printf ("Diag\n");

    // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
    // TO SOLVE IT, USE GLOBAL VALUES OF frs_col AND lst_col
    if ((!Upper && (((idWFC*dimWFC)+((!Diag)?1:0))>(lst_row-1))) ||
        (!Lower && ((frs_row+((!Diag)?1:0))>((idWFC*dimWFC+dimWFC)-1)))) {
      ;
    } else if (Single) {
      for (size_t colid = frs_col; colid < lst_col; colid += localSz) {
        auto val = scl * r2.eval(colid);
        for (size_t id_row=frs_row, row=0; id_row<lst_row; id_row++, row++) {
          if (Lower && Upper && Diag) {
            l.eval(id_row,colid) += r1.eval(id_row) * val;
          } else {
            if ((Lower && ((colid+((!Diag)?1:0)) <= id_row)) ||
              (Upper && (colid >= (id_row+((!Diag)?1:0))))) {
                l.eval(id_row,colid) += r1.eval(id_row) * val;
            }
          }
        }
      }
    } else {
      for (size_t colid = frs_col; colid < lst_col; colid += localSz) {
        auto val1 = scl * r1.eval(colid);
        auto val2 = scl * r2.eval(colid);
        for (size_t id_row=frs_row, row=0; id_row<lst_row; id_row++, row++) {
          if (Lower && Upper && Diag) {
            l.eval(id_row,colid) += r1.eval(id_row) * val2 + val1 * r2.eval(id_row);
          } else {
            if ((Lower && ((colid+((!Diag)?1:0)) <= id_row)) ||
              (Upper && (colid >= (id_row+((!Diag)?1:0))))) {
                l.eval(id_row,colid) += r1.eval(id_row) * val2 +
                                        r2.eval(id_row) * val1;
            }
          }
        }
      }
    }

    return l.eval(frs_row,frs_col);
  }

  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);
    size_t glbalid = ndItem.get_global(0);
    // size_t glbalSz = ndItem.get_global_range(0);

    size_t dimR = l.getSizeR();
    size_t dimC = l.getSizeC();

    size_t rowSz = (dimR + nWG_row - 1) / nWG_row;
    size_t colSz = (dimR<localSz)? localSz: (dimC + nWG_col - 1) / nWG_col;
    size_t shrSz = shrMemSize;

    size_t idWFR = (groupid % nWG_row);
    size_t idWFC = (groupid / nWG_row);
//    size_t idWFR  = groupid % nWG_row;  // row bloq id of the current workgroup
//    size_t idWFC  = groupid / nWG_row;  // col blq id of the current workgroup

    size_t dimWFC = (dimC + (localSz*nWG_col) - 1) / (localSz*nWG_col) * localSz;

    size_t frs_row = idWFR*rowSz;
    size_t lst_row = std::min(dimR,frs_row+rowSz);

    size_t frs_col = idWFC * dimWFC + localid;
    size_t lst_col = std::min(dimC,frs_col+dimWFC);

//    if ((!Upper) && (glbalid == 0)) printf ("Lower\n");
//    if ((!Lower) && (glbalid == 0)) printf ("Upper\n");
//    if ((Diag)   && (glbalid == 0)) printf ("Diag\n");

    // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
    // TO SOLVE IT, USE GLOBAL VALUES OF frs_col AND lst_col
    if ((!Upper && (((idWFC*dimWFC)+((!Diag)?1:0))>(lst_row-1))) ||
        (!Lower && ((frs_row+((!Diag)?1:0))>((idWFC*dimWFC+dimWFC)-1)))) {
      ;
    } else if (Single) {
      for (size_t rowid=frs_row; rowid<lst_row; rowid+=shrSz) {
        if (rowid > frs_row)
          // This barrier is mandatory to be sure the data is on the shared memory
          ndItem.barrier(cl::sycl::access::fence_space::local_space);

        auto blqSz = std::min(shrSz,lst_row-rowid);
        for (size_t row=localid, id_row=rowid+localid; (row<blqSz); row+=localSz, id_row+=localSz) {
          shrMem[row] = scl * r1.eval(id_row);
        }

        // This barrier is mandatory to be sure the data is on the shared memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);

        for (size_t colid = frs_col; (colid<lst_col); colid += localSz) {
          auto val = r2.eval(colid);
          for (size_t id_row=rowid, row=0; row<blqSz; id_row++, row++) {
            if (Lower && Upper && Diag) {
              l.eval(id_row,colid) += shrMem[row] * val;
            } else {
              if ((Lower && ((colid+((!Diag)?1:0)) <= id_row)) ||
                  (Upper && (colid >= (id_row+((!Diag)?1:0))))) {
                l.eval(id_row,colid) += shrMem[row] * val;
              }
            }
          }
        }
      }
    } else {
      auto shrSz1 = (shrSz / 2);
      for (size_t rowid=frs_row; rowid<lst_row; rowid+=shrSz) {
        if (rowid > frs_row)
          // This barrier is mandatory to be sure the data is on the shared memory
          ndItem.barrier(cl::sycl::access::fence_space::local_space);

        auto blqSz = std::min(shrSz1,lst_row-rowid);
        for (size_t row=localid, id_row=rowid+localid; (row<blqSz); row+=localSz, id_row+=localSz) {
          shrMem[       row] = scl * r1.eval(id_row);
          shrMem[shrSz1+row] = scl * r2.eval(id_row);
        }

        // This barrier is mandatory to be sure the data is on the shared memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);

        for (size_t colid = frs_col; (colid<lst_col); colid += localSz) {
          auto val1 = r1.eval(colid);
          auto val2 = r2.eval(colid);
          for (size_t id_row=rowid, row=0; row<blqSz; id_row++, row++) {
            if (Lower && Upper && Diag) {
              l.eval(id_row,colid) += shrMem[       row] * val2 +
                                      shrMem[shrSz1+row] * val1;
            } else {
              if ((Lower && ((colid+((!Diag)?1:0)) <= id_row)) ||
                  (Upper && (colid >= (id_row+((!Diag)?1:0))))) {
                l.eval(id_row,colid) += shrMem[       row] * val2 +
                                        shrMem[shrSz1+row] * val1;
              }
            }
          }
        }
      }
    }

    return shrMem[0];
  }

};

//template <class LHS, class RHS1, class RHS2>
template <bool Single=true, bool Lower=true, bool Diag=true, bool Upper=true,
          class LHS, class RHS1, class RHS2>
Ger_Row<Single, Lower, Diag, Upper, LHS, RHS1, RHS2> make_Ger_Row(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2, size_t nWG_row, size_t nWG_col, size_t shrMemSize) {
  return Ger_Row<Single, Lower, Diag, Upper, LHS, RHS1, RHS2>(l, scl, r1, r2, nWG_row, nWG_col, shrMemSize);
}

/**** GER BY COLUMNS M ROWS x N BLOCK USING PROPERLY THE SHARED MEMORY ****/
//template <class LHS, class RHS1, class RHS2>
template <bool Single, bool Lower, bool Diag, bool Upper,
          class LHS, class RHS1, class RHS2>
struct Ger_Col {
  LHS  l;
  RHS1 r1;
  RHS2 r2;
  size_t nWG_row;
  size_t nWG_col;
  size_t shrMemSize;

  using value_type = typename RHS2::value_type;
  value_type scl;

  Ger_Col(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2, size_t &_nWG_row, size_t &_nWG_col, size_t &_shrMemSize)
    : l(_l), scl(_scl), r1(_r1), r2(_r2), nWG_row(_nWG_row), nWG_col(_nWG_col), shrMemSize(_shrMemSize) { };

  size_t getSize() { return r1.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(size_t i) {
    auto size = (l.getAccess()) ? l.getSizeC() : l.getSizeR();
    auto row = (l.getAccess()) ? (i / size) : (i % size);
    auto col = (l.getAccess()) ? (i % size) : (i / size);

    auto val = scl * r1.eval(row) * r2.eval(col);

    return l.eval(i) += val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);
    size_t glbalid = ndItem.get_global(0);
    // size_t glbalSz = ndItem.get_global_range(0);

    size_t dimR = l.getSizeR();
    size_t dimC = l.getSizeC();

    size_t rowSz = (dimR + nWG_row - 1) / nWG_row;
    size_t colSz = (dimR<localSz)? localSz: (dimC + nWG_col - 1) / nWG_col;
    size_t shrSz = shrMemSize;

    size_t idWFR  = groupid % nWG_row;  // row bloq id of the current workgroup
    size_t idWFC  = groupid / nWG_row;  // col blq id of the current workgroup
//    size_t idWFR  = groupid % nWG_row;  // row bloq id of the current workgroup
//    size_t idWFC  = groupid / nWG_row;  // col blq id of the current workgroup
    size_t dimWFR = (dimR + (localSz*nWG_row) - 1) / (localSz*nWG_row) * localSz;

    size_t frs_row = idWFR * dimWFR + localid;
    size_t lst_row = std::min(dimR,frs_row + dimWFR);

    size_t frs_col = idWFC*colSz;
    size_t lst_col = std::min(dimC,frs_col+colSz);

//    if ((!Upper) && (glbalid == 0)) printf ("Lower\n");
//    if ((!Lower) && (glbalid == 0)) printf ("Upper\n");
//    if ((Diag)   && (glbalid == 0)) printf ("Diag\n");

    if ((!Upper && ((frs_col+((!Diag)?1:0))>((idWFR*dimWFR+dimWFR)-1))) ||
        (!Lower && ((idWFR*dimWFR+((!Diag)?1:0))>(lst_col-1)))) {
      ;
    } else if (Single) {
      for (size_t id_row = frs_row; id_row < lst_row; id_row += localSz) {
        auto val = scl * r1.eval(id_row);
//        for (size_t id_col=frs_col, col=0; id_col<lst_col; id_col++, col++) {
        for (size_t id_col=((Lower)?frs_col:std::max(id_row+((!Diag)?1:0),frs_col));
                    id_col<((Upper)?lst_col:std::min(id_row+((!Diag)?0:1),lst_col)); id_col++) {
          l.eval(id_row,id_col) += val * r2.eval(id_col);
        }
      }
    } else {
//      if (glbalid == 0) printf ("HERE\n");
      for (size_t id_row = frs_row; id_row < lst_row; id_row += localSz) {
        auto val1 = scl * r1.eval(id_row);
        auto val2 = scl * r2.eval(id_row);
//        for (size_t id_col=frs_col, col=0; id_col<lst_col; id_col++, col++) {
        for (size_t id_col=((Lower)?frs_col:std::max(id_row+((!Diag)?1:0),frs_col));
                    id_col<((Upper)?lst_col:std::min(id_row+((!Diag)?0:1),lst_col)); id_col++) {
          l.eval(id_row,id_col) += val1 * r2.eval(id_col) +
                                   val2 * r1.eval(id_col);
        }
      }
    }

    return l.eval(frs_row,frs_col);
  }

  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);
    size_t glbalid = ndItem.get_global(0);
    // size_t glbalSz = ndItem.get_global_range(0);

    size_t dimR = l.getSizeR();
    size_t dimC = l.getSizeC();

    size_t rowSz = (dimR + nWG_row - 1) / nWG_row;
    size_t colSz = (dimR<localSz)? localSz: (dimC + nWG_col - 1) / nWG_col;
    size_t shrSz = shrMemSize;

    size_t idWFR  = groupid % nWG_row;  // row bloq id of the current workgroup
    size_t idWFC  = groupid / nWG_row;  // col blq id of the current workgroup
//    size_t idWFR  = groupid % nWG_row;  // row bloq id of the current workgroup
//    size_t idWFC  = groupid / nWG_row;  // col blq id of the current workgroup
    size_t dimWFR = (dimR + (localSz*nWG_row) - 1) / (localSz*nWG_row) * localSz;

    size_t frs_row = idWFR * dimWFR + localid;
    size_t lst_row = std::min(dimR,frs_row + dimWFR);

    size_t frs_col = idWFC*colSz;
    size_t lst_col = std::min(dimC,frs_col+colSz);

//    if ((!Upper) && (glbalid == 0)) printf ("Lower\n");
//    if ((!Lower) && (glbalid == 0)) printf ("Upper\n");
//    if ((Diag)   && (glbalid == 0)) printf ("Diag\n");

    // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
    // TO SOLVE IT, USE GLOBAL VALUES OF frs_row AND lst_row
    if ((!Upper && ((frs_col+((!Diag)?1:0))>((idWFR*dimWFR+dimWFR)-1))) ||
        (!Lower && ((idWFR*dimWFR+((!Diag)?1:0))>(lst_col-1)))) {
      ;
    } else if (Single) {
      // The computation are made in blocks of shrMemSize elements
      for (size_t colid=frs_col; colid<lst_col; colid+=shrMemSize) {
        if (colid > frs_col)
          // This barrier is mandatory to be sure the data is on the shared memory
          ndItem.barrier(cl::sycl::access::fence_space::local_space);

        auto blqSz = std::min(shrMemSize,lst_col-colid);

        for (size_t col=localid; (col<blqSz); col+=localSz) {
          shrMem[col] = scl * r2.eval(colid+col);
        }

        // This barrier is mandatory to be sure the data is on the shared memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);

        for (size_t id_row = frs_row; id_row < lst_row; id_row += localSz) {
          auto val = r1.eval(id_row);
  //        for (size_t id_col=frs_col, col=0; id_col<std::min(dimC,frs_col+colSz); id_col++, col++) {
          for (size_t id_col=colid, col=0; col<blqSz; id_col++, col++) {
            if (Lower && Upper && Diag) {
              l.eval(id_row,id_col) += val * shrMem[col];
            } else {
              if ((Lower && ((id_col+((!Diag)?1:0)) <= id_row)) ||
                  (Upper && (id_col >= (id_row+((!Diag)?1:0))))) {
                l.eval(id_row,id_col) += val * shrMem[col];
              }
            }
          }
        }
      }
    } else {
      auto shrSz1 = (shrMemSize / 2);
      // The computation are made in blocks of shrMemSize/shrSz1 elements
//      for (size_t colid=frs_col; colid<lst_col; colid+=shrMemSize) {
      for (size_t colid=frs_col; colid<lst_col; colid+=shrSz1) {
//        if (glbalid == 0)
//          printf ("(%lu) -> (%lu,%lu,%lu), (%lu,%lu)\n",
//                  glbalid, colid, frs_col, lst_col, shrMemSize, shrSz1);
        if (colid > frs_col)
          // This barrier is mandatory to be sure the data is on the shared memory
          ndItem.barrier(cl::sycl::access::fence_space::local_space);

        auto blqSz = std::min(shrSz1,lst_col-colid);

        for (size_t col=localid; (col<blqSz); col+=localSz) {
          shrMem[       col] = scl * r1.eval(colid+col);
          shrMem[shrSz1+col] = scl * r2.eval(colid+col);
        }

        // This barrier is mandatory to be sure the data is on the shared memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);

        for (size_t id_row = frs_row; id_row < lst_row; id_row += localSz) {
          auto val1 = r1.eval(id_row);
          auto val2 = r2.eval(id_row);
  //        for (size_t id_col=frs_col, col=0; id_col<std::min(dimC,frs_col+colSz); id_col++, col++) {
          for (size_t id_col=colid, col=0; col<blqSz; id_col++, col++) {
            if (Lower && Upper && Diag) {
              l.eval(id_row,id_col) += val1 * shrMem[shrSz1+col] +
                                       val2 * shrMem[       col];
            } else {
              if ((Lower && ((id_col+((!Diag)?1:0)) <= id_row)) ||
                  (Upper && (id_col >= (id_row+((!Diag)?1:0))))) {
                l.eval(id_row,id_col) += val1 * shrMem[shrSz1+col] +
                                         val2 * shrMem[       col];
              }
            }
          }
        }
      }
    }

    return shrMem[0];
  }

};

//template <class LHS, class RHS1, class RHS2>
template <bool Single=true, bool Lower=true, bool Diag=true, bool Upper=true,
          class LHS, class RHS1, class RHS2>
Ger_Col<Single, Lower, Diag, Upper, LHS, RHS1, RHS2> make_Ger_Col(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2, size_t nWG_row, size_t nWG_col, size_t shrMemSize) {
  return Ger_Col<Single, Lower, Diag, Upper, LHS, RHS1, RHS2>(l, scl, r1, r2, nWG_row, nWG_col, shrMemSize);
}

/**************************************************/
/*************** PREVIOUS VERSIONS ****************/
/**************************************************/

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

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

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

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

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

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

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

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

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

  size_t getSize() { return r1.getSizeR(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

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

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

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

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

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

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

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

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

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

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

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

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

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

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

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

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

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

  size_t getSize() { return r1.getSizeR(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

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

  size_t getSize() { return r1.getSizeR(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

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

  size_t getSize() { return r1.getSizeR(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

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

  size_t getSize() { return r1.getSizeR(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

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

  size_t getSize() { return r1.getSizeR(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

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

  size_t getSize() { return r1.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

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

};

template <class RHS1, class RHS2, class RHS3>
ModifRank1<RHS1, RHS2, RHS3> make_modifRank1(RHS1 &r1, RHS2 &r2, RHS3 &r3) {
  return ModifRank1<RHS1, RHS2, RHS3>(r1, r2, r3);
}

template <class LHS, class RHS1, class RHS2>
struct Ger_1Row_1WG {
  LHS l;
  RHS1 r1;
  RHS2 r2;

  using value_type = typename RHS2::value_type;
  value_type scl;

  Ger_1Row_1WG(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2)
    : l(_l), scl(_scl), r1(_r1), r2(_r2) { };

  size_t getSize() { return r1.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(size_t i) {
    auto size = (l.getAccess()) ? l.getSizeC() : l.getSizeR();
    auto row = (l.getAccess()) ? (i / size) : (i % size);
    auto col = (l.getAccess()) ? (i % size) : (i / size);

    auto val = scl * r1.eval(row) * r2.eval(col);

    return l.eval(i) += val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);
    size_t glbalid = ndItem.get_global(0);

    size_t dimR = l.getSizeR();
    size_t dimC = l.getSizeC();

    value_type val = scl * r1.eval(groupid);

//    if (localid == 0)
//      printf ("%lu -> %f = %f * %f , %f , %f\n",
//          glbalid, val, scl, r1.eval(0), r2.eval(0), l.eval(groupid,0));
    size_t frs_thrd = localid;
    for (size_t k = frs_thrd; k < dimC; k += localSz) {
//        auto prod = prdOp2_struct::eval(scl,r2.eval(k));
//        l.eval(groupid,k) = addOp2_struct::eval(l.eval(groupid,k), prod);
      l.eval(groupid,k) += val * r2.eval(k);
    }
//    if (localid == 0)
//      printf ("%lu ->  %f\n", glbalid, l.eval(groupid,0));

    return val;
  }

};

template <class LHS, class RHS1, class RHS2>
Ger_1Row_1WG<LHS, RHS1, RHS2> make_Ger_1Row_1WG(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2) {
  return Ger_1Row_1WG<LHS, RHS1, RHS2>(l, scl, r1, r2);
}

template <class LHS, class RHS1, class RHS2>
struct Ger_MRow_NWG {
  LHS l;
  RHS1 r1;
  RHS2 r2;
  size_t n_rows;
  size_t nWG_col;

  using value_type = typename RHS2::value_type;
  value_type scl;

  Ger_MRow_NWG(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2, size_t &_n_rows, size_t &_nWG_col)
    : l(_l), scl(_scl), r1(_r1), r2(_r2), n_rows(_n_rows), nWG_col(_nWG_col) { };

  size_t getSize() { return r1.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(size_t i) {
    auto size = (l.getAccess()) ? l.getSizeC() : l.getSizeR();
    auto row = (l.getAccess()) ? (i / size) : (i % size);
    auto col = (l.getAccess()) ? (i % size) : (i / size);

    auto val = scl * r1.eval(row) * r2.eval(col);

    return l.eval(i) += val;
  }

  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);
    size_t glbalid = ndItem.get_global(0);
    // size_t glbalSz = ndItem.get_global_range(0);

    size_t dimR = l.getSizeR();
    size_t dimC = l.getSizeC();

    size_t nWG_row = (groupSz + nWG_col - 1) / nWG_col;  // number of "row" workgroups
    size_t blqidR  = groupid % nWG_row;  // row bloq id of the current workgroup
    size_t blqidC  = groupid / nWG_row;  // col blq id of the current workgroup

    size_t dimWFC = (dimC + (localSz*nWG_col) - 1) / (localSz*nWG_col) * localSz;

//    size_t blqidR = groupid / nWG_col;  // row bloq id of the current workgroup
//    size_t blqidC = groupid % nWG_col;  // col blq id of the current workgroup

    size_t frs_row = blqidR*n_rows;

    for (size_t row=localid; (row<n_rows); row+=localSz) {
      shrMem[row] = scl * r1.eval(frs_row+row);
//      shrMem[row] = ((frs_row+row)<dimR)?(scl*r1.eval(frs_row+row)):0.0;
    }

    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    size_t frs_thrd = blqidC * dimWFC + localid;
    size_t lst_thrd = std::min(dimC,frs_thrd + dimWFC);
    for (size_t k = frs_thrd; k < lst_thrd; k += localSz) {
      auto val = r2.eval(k);
//      for (size_t row=0; row<n_rows; row++) {
      for (size_t id_row=frs_row, row=0; id_row<std::min(dimR,frs_row+n_rows); id_row++, row++) {
//      if (localid == 0)
//        printf ("%lu -> (%lu,%lu,%lu) , %f = %f * %f , %f , %f\n",
//              glbalid, frs_row, frs_thrd, lst_thrd, shrMem[row], scl, r1.eval(frs_row), r2.eval(frs_thrd), l.eval(groupid,frs_thrd));
//        size_t id_row = frs_row+row;
//        if (id_row < dimR) {
    //        auto prod = prdOp2_struct::eval(scl,r2.eval(k));
    //        l.eval(id_row,k) = addOp2_struct::eval(l.eval(id_row,k), prod);
          l.eval(id_row,k) += shrMem[row] * val;
//        }
//      if (localid == dimC-1)
//        printf ("%lu ->  %f\n", glbalid, l.eval(groupid,frs_thrd));
      }
    }

    return shrMem[0];
  }

};

template <class LHS, class RHS1, class RHS2>
Ger_MRow_NWG<LHS, RHS1, RHS2> make_Ger_MRow_NWG(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2, size_t n_rows, size_t nWG_col) {
  return Ger_MRow_NWG<LHS, RHS1, RHS2>(l, scl, r1, r2, n_rows, nWG_col);
}

template <class LHS, class RHS1, class RHS2>
struct Ger_1Row_1Thread {
  LHS l;
  RHS1 r1;
  RHS2 r2;

  using value_type = typename RHS2::value_type;
  value_type scl;

  Ger_1Row_1Thread(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2)
    : l(_l), scl(_scl), r1(_r1), r2(_r2) { };

  size_t getSize() { return r1.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(size_t i) {
    auto size = (l.getAccess()) ? l.getSizeC() : l.getSizeR();
    auto row = (l.getAccess()) ? (i / size) : (i % size);
    auto col = (l.getAccess()) ? (i % size) : (i / size);

    auto val = scl * r1.eval(row) * r2.eval(col);

    return l.eval(i) += val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);
    size_t glbalid = ndItem.get_global(0);

    size_t dimR = l.getSizeR();
    size_t dimC = l.getSizeC();

    size_t row_id = groupid*localSz+localid;

    value_type val = scl * r1.eval(row_id);

//    if (localid == 0)
//      printf ("%lu -> %f = %f * %f , %f , %f\n",
//          glbalid, val, scl, r1.eval(0), r2.eval(0), l.eval(groupid,0));
    if (row_id < dimR) {
      for (size_t k = 0; k < dimC; k ++) {
  //        auto prod = prdOp2_struct::eval(scl,r2.eval(k));
  //        l.eval(groupid,k) = addOp2_struct::eval(l.eval(groupid,k), prod);
        l.eval(row_id,k) += val * r2.eval(k);
      }
    }
//    if (localid == 0)
//      printf ("%lu ->  %f\n", glbalid, l.eval(groupid,0));

    return val;
  }

};

template <class LHS, class RHS1, class RHS2>
Ger_1Row_1Thread<LHS, RHS1, RHS2> make_Ger_1Row_1Thread(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2) {
  return Ger_1Row_1Thread<LHS, RHS1, RHS2>(l, scl, r1, r2);
}

template <class LHS, class RHS1, class RHS2>
struct Ger_1Row_NWG_ShMem {
  LHS l;
  RHS1 r1;
  RHS2 r2;
  size_t n_cols;
  size_t nWG_row;

  using value_type = typename RHS2::value_type;
  value_type scl;

  Ger_1Row_NWG_ShMem(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2, size_t &_n_cols, size_t &_nWG_row)
    : l(_l), scl(_scl), r1(_r1), r2(_r2), n_cols(_n_cols), nWG_row(_nWG_row) { };

  size_t getSize() { return r1.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(size_t i) {
    auto size = (l.getAccess()) ? l.getSizeC() : l.getSizeR();
    auto row = (l.getAccess()) ? (i / size) : (i % size);
    auto col = (l.getAccess()) ? (i % size) : (i / size);

    auto val = scl * r1.eval(row) * r2.eval(col);

    return l.eval(i) += val;
  }

  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);
    size_t glbalid = ndItem.get_global(0);
    // size_t glbalSz = ndItem.get_global_range(0);

    size_t dimR = l.getSizeR();
    size_t dimC = l.getSizeC();

    size_t nWG_col = (groupSz + nWG_row - 1) / nWG_row;  // number of "col" workgroups
    size_t blqidR  = groupid % nWG_row;  // row bloq id of the current workgroup
    size_t blqidC  = groupid / nWG_row;  // col blq id of the current workgroup

    size_t dimWFR = (dimR + (localSz*nWG_row) - 1) / (localSz*nWG_row) * localSz;

//    size_t blqidR = groupid / nWG_col;  // row bloq id of the current workgroup
//    size_t blqidC = groupid % nWG_col;  // col blq id of the current workgroup

    size_t frs_col = blqidC*n_cols;

    for (size_t col=localid; (col<n_cols); col+=localSz) {
      shrMem[col] = scl * r2.eval(frs_col+col);
//      shrMem[row] = ((frs_row+row)<dimR)?(scl*r1.eval(frs_row+row)):0.0;
    }

    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    size_t frs_row = blqidR * dimWFR + localid;
    size_t lst_row = std::min(dimR,frs_row + dimWFR);
    for (size_t k = frs_row; k < lst_row; k += localSz) {
      auto val = r1.eval(k);
      for (size_t id_col=frs_col, col=0; id_col<std::min(dimC,frs_col+n_cols); id_col++, col++) {
//      if (localid == 0)
//        printf ("%lu -> (%lu,%lu,%lu) , %f = %f * %f , %f , %f\n",
//              glbalid, frs_row, frs_thrd, lst_thrd, shrMem[row], scl, r1.eval(frs_row), r2.eval(frs_thrd), l.eval(groupid,frs_thrd));
    //        auto prod = prdOp2_struct::eval(scl,r2.eval(k));
    //        l.eval(id_row,k) = addOp2_struct::eval(l.eval(id_row,k), prod);
        l.eval(k,id_col) += val * shrMem[col];
      }
//      if (localid == dimC-1)
//        printf ("%lu ->  %f\n", glbalid, l.eval(groupid,frs_thrd));
    }

    return shrMem[0];
  }

};

template <class LHS, class RHS1, class RHS2>
Ger_1Row_NWG_ShMem<LHS, RHS1, RHS2> make_Ger_1Row_NWG_ShMem(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2, size_t n_rows, size_t nWG_col) {
  return Ger_1Row_NWG_ShMem<LHS, RHS1, RHS2>(l, scl, r1, r2, n_rows, nWG_col);
}

}  // namespace blas

#endif  // BLAS2_TREES_HPP
