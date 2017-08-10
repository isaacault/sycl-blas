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
 *  @filename blas2_interface_sycl.hpp
 *
 **************************************************************************/

#ifndef BLAS2_INTERFACE_SYCL_HPP
#define BLAS2_INTERFACE_SYCL_HPP

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <executors/executor_sycl.hpp>
#include <operations/blas1_trees.hpp>

namespace blas {

/**** MATRIX VECTOR PRODUCT ****/

//#define OPT 3  // ACTIVE CASE FOR THE COLUMN ACCESS

/*! _gemv.
 * @brief Implementation of the General Matrix Vector product.
 */
template <unsigned int OPT, typename ExecutorType, typename T, typename ContainerT>
void _gemv(Executor<ExecutorType> ex, std::string _Trans, size_t _M, size_t _N,
           T _alpha, matrix_view<T, ContainerT> _mA, size_t _lda,
           vector_view<T, ContainerT> _vx, size_t _incx, T _beta,
           vector_view<T, ContainerT> _vy, size_t _incy) {
  if ((_Trans[0] != 'n') && (_Trans[0] != 't') && (_Trans[0] != 'c') &&
      (_Trans[0] != 'N') && (_Trans[0] != 'T') && (_Trans[0] != 'C'))
    std::cout << "Erroneous parameter" << std::endl;
  int accessOpr = ((_Trans[0] == 'n') || (_Trans[0] == 'N'));
  size_t M = _M;
  size_t N = _N;
  auto my_mA =
      matrix_view<T, ContainerT>(_mA, _M, _N, accessOpr, _lda, _mA.getDisp());
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, N);
  auto my_vy = vector_view<T, ContainerT>(_vy, _vy.getDisp(), _incy, M);
#ifdef VERBOSE
  std::cout << "alpha = " << _alpha << " , beta = " << _beta << std::endl;
  my_mA.printH("MA");
  my_vx.printH("VX");
  my_vy.printH("VY");
#endif  // VERBOSE
  if (my_mA.getAccess()) {  // ROWS ACCESS
    if (OPT == 1){
  #ifdef VERBOSE
  //    std::cout << "ROWS_2" << std::setprecision(15) << "M = " << _M
      std::cout << "ROWS_1" << "M = " << _M
                << " N = " << _N << std::endl;
  #endif  // VERBOSE
      auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
      auto redRowMatVectOp = make_redRowMatVct(my_mA, my_vx, 1);
      auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, redRowMatVectOp);
      auto addOp = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
      auto assignOp = make_op<Assign>(my_vy, addOp);
  #ifdef BLAS_EXPERIMENTAL
      ex.execute(assignOp, M);
  #endif  // BLAS_EXPERIMENTAL
      ex.execute(assignOp, 256);
    } else if (OPT == 11) {  // GEMV BY ROWS 1 ROW x 1 BLOCK
#ifdef VERBOSE
  //    std::cout << "ROWS_2" << std::setprecision(15) << "M = " << _M
      std::cout << "ROWS_11" << "M = " << _M
                << " N = " << _N << std::endl;
#endif  // VERBOSE
//      std::cout << "alpha = " << _alpha << " , beta = " << _beta << std::endl;
//      my_mA.printH("MA");
//      my_vx.printH("VX");
//      my_vy.printH("VY");
      size_t nWG_col = 1;
      size_t localSize = 256;
      ContainerT valT1(nWG_col * M);
      auto mat1 = matrix_view<T, ContainerT>(valT1, 0, M, nWG_col);

      auto gemvR = make_GemvR_1Row_1WG (mat1, my_mA, my_vx);
      ex.execute(gemvR, localSize, M*localSize, localSize);
//      mat1.printH("MAT1");

      auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
      auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, mat1);
      auto addOp = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
      auto assignOp = make_op<Assign>(my_vy, addOp);
      ex.execute(assignOp, localSize);
    } else if (OPT == 12) {  // GEMV BY ROWS 1 ROW x 1 BLOCK WITHOUT LOCAL ADDITION
#ifdef VERBOSE
  //    std::cout << "ROWS_2" << std::setprecision(15) << "M = " << _M
      std::cout << "ROWS_12" << "M = " << _M
                << " N = " << _N << std::endl;
#endif  // VERBOSE
//      std::cout << "alpha = " << _alpha << " , beta = " << _beta << std::endl;
//      my_mA.printH("MA");
//      my_vx.printH("VX");
//      my_vy.printH("VY");
      size_t nWG_col = 1;
      size_t localSize = (M < 256)?M:256;
      ContainerT valT1(localSize * nWG_col * M);
      auto mat1 = matrix_view<T, ContainerT>(valT1, 0, M, nWG_col*localSize);
      auto mat2 = matrix_view<T, ContainerT>(valT1, 0, M, M);

      auto gemvR = make_GemvR_1Row_1WG_NoRed (mat1, my_mA, my_vx);
      ex.execute(gemvR, localSize, localSize*M);
//      mat1.printH("MAT1");
//      mat2.printH("MAT2");

      auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
      auto addMOp = make_addSetColumns(mat1);
      auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, addMOp);
      auto addOp = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
      auto assignOp = make_op<Assign>(my_vy, addOp);
      ex.execute(assignOp, localSize);
    } else if (OPT == 13) {  // GEMV BY ROWS 1 ROW x nWG_col BLOCK
#ifdef VERBOSE
      std::cout << "ROWS_13" << "M = " << _M
                << " N = " << _N << std::endl;
#endif  // VERBOSE
//      std::cout << "alpha = " << _alpha << " , beta = " << _beta << std::endl;
//      my_mA.printH("MA");
//      my_vx.printH("VX");
//      my_vy.printH("VY");
      size_t nWG_col = 4;
      size_t localSize = 256;
      ContainerT valT1(nWG_col * M);
      auto mat1 = matrix_view<T, ContainerT>(valT1, 0, M, nWG_col);

      auto gemvR = make_GemvR_1Row_NWG (mat1, my_mA, my_vx, nWG_col);
      ex.execute(gemvR, localSize, M*nWG_col*localSize, localSize);
//      mat1.printH("MAT1");

      auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
      auto addMOp = make_addSetColumns(mat1);
      auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, addMOp);
      auto addOp = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
      auto assignOp = make_op<Assign>(my_vy, addOp);
      ex.execute(assignOp, localSize);
    } else if (OPT == 14) {  // GEMV BY ROWS M ROW x nWG_col BLOCK
#ifdef VERBOSE
  //    std::cout << "ROWS_2" << std::setprecision(15) << "M = " << _M
      std::cout << "ROWS_14" << "M = " << _M
                << " N = " << _N << std::endl;
#endif  // VERBOSE
//      std::cout << "alpha = " << _alpha << " , beta = " << _beta << std::endl;
//      my_mA.printH("MA");
//      my_vx.printH("VX");
//      my_vy.printH("VY");
      size_t nWG_col = 4;
      size_t n_rows = 4;
      size_t localSize = 256;
      ContainerT valT1(nWG_col * M);
      auto mat1 = matrix_view<T, ContainerT>(valT1, 0, M, nWG_col);
      auto nWG_row = (M + n_rows - 1) / n_rows;

      auto gemvR = make_GemvR_MRow_NWG (mat1, my_mA, my_vx, n_rows, nWG_col);
      ex.execute(gemvR, localSize, nWG_row*nWG_col*localSize, localSize*n_rows);

      auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
      auto addMOp = make_addSetColumns(mat1);
      auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, addMOp);
      auto addOp = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
      auto assignOp = make_op<Assign>(my_vy, addOp);
      ex.execute(assignOp, localSize);
    }
  } else { // COLUMN ACCESS
    if (OPT == 1) {  // Sure solution
      #ifdef VERBOSE
      std::cout << "COLS_1" << std::endl;
      #endif  // VERBOSE
      auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
      auto prdRowMatVectOp = make_prdRowMatVct(my_mA, my_vx);
      auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, prdRowMatVectOp);
      auto addOp = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
      auto assignOp = make_op<Assign>(my_vy, addOp);
      #ifdef BLAS_EXPERIMENTAL
      ex.execute(assignOp, M);
      #endif  // BLAS_EXPERIMENTAL
      ex.execute(assignOp);
    } else if (OPT == 2) {  // First improvement
  #ifdef VERBOSE
      std::cout << "COLS_2" << std::endl;
  #endif  // VERBOSE
//      auto nThr = 8;
      auto nThr = 4;
      auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
      auto prdRowMatVectOp =
          make_prdRowMatVctMult(my_vy, _alpha, my_mA, my_vx, scalOp1, nThr);
  //    auto localSize = 32;  // NOT FINAL VALUE
//      auto localSize = 32;  // NOT FINAL VALUE
      auto localSize = 256;  // NOT FINAL VALUE
      auto nWG = (M + localSize - 1) / localSize;
      auto gridSize = localSize * nThr * nWG;
//      ex.execute(prdRowMatVectOp, localSize * nThr, gridSize, localSize * nThr);
      ex.execute(prdRowMatVectOp, localSize, gridSize, localSize);
    } else if (OPT == 3) {
  #ifdef VERBOSE
      std::cout << "COLS_3" << std::endl;
  #endif  // VERBOSE
      auto nThr = 16;
      ContainerT valT1(nThr * M);
      auto mat1 = matrix_view<T, ContainerT>(valT1, 0, M, nThr);
      auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
  #ifdef BLAS_EXPERIMENTAL
      auto val1 = vector_view<T, ContainerT>(valT1, 0, 1, nThr * M);
      auto mat1 = matrix_view<T, ContainerT>(valT1, M, nThr);
      auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
      auto prdRowMatVectOp = make_prdRowMatVctMultShm(val1, my_mA, my_vx, nThr);
  #endif  // BLAS_EXPERIMENTAL
      auto prdRowMatVectOp = make_prdRowMatVctMultShm(mat1, my_mA, my_vx, nThr);
  //    auto localSize = 32;  // NOT FINAL VALUE
      auto localSize = 256;  // NOT FINAL VALUE
      auto nWG = (M + localSize - 1) / localSize;
      auto gridSize = localSize * nThr * nWG;
      ex.execute(prdRowMatVectOp, localSize, gridSize, (N + nThr - 1) / nThr);
  //    ex.execute(prdRowMatVectOp, localSize, gridSize, localSize);
  #ifdef VERBOSE
      my_vy.printH("VY");
  #endif
  #ifdef VERBOSE
      mat1.printH("MAT1");
  #endif  // VERBOSE
      auto addPrdOp = make_addPrdRowMatVctMultShm(my_vy, _alpha, mat1, scalOp1);
  #ifdef BLAS_EXPERIMENTAL
      ex.execute(addPrdOp, localSize, localSize);
  #endif  // BLAS_EXPERIMENTAL
  //    ex.execute(addPrdOp, localSize, nWG*localSize, 0);
      ex.execute(addPrdOp, localSize);
    } else if (OPT == 11) {  // Sure solution
  #ifdef VERBOSE
      std::cout << "COLS_11" << std::endl;
  #endif  // VERBOSE
      auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
  //    auto prdRowMatVectOp = make_prdRowMatVct(my_mA, my_vx);
      auto prdRowMatVectOp = make_GemvC_1Row_1Thread(my_mA, my_vx);
      auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, prdRowMatVectOp);
      auto addOp = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
      auto assignOp = make_op<Assign>(my_vy, addOp);
  #ifdef BLAS_EXPERIMENTAL
      ex.execute(assignOp, M);
  #endif  // BLAS_EXPERIMENTAL
      auto localSize = 256;  // NOT FINAL VALUE
      ex.execute(assignOp, localSize);
    } else if (OPT == 12) {  // Sure solution
  #ifdef VERBOSE
      std::cout << "COLS_12" << std::endl;
  #endif  // VERBOSE
      auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
  //    auto prdRowMatVectOp = make_prdRowMatVct(my_mA, my_vx);
      auto prdRowMatVectOp =
            make_GemvC_1Row_1Thread_ShMem(my_vy, _alpha, my_mA, my_vx, scalOp1);
      auto localSize = 256;  // NOT FINAL VALUE
      auto nWG = (M + localSize - 1) / localSize;
      auto gridSize = localSize *  nWG;
      ex.execute(prdRowMatVectOp, localSize , gridSize, localSize);
    } else if (OPT == 13) {
  #ifdef VERBOSE
      std::cout << "COLS_13" << std::endl;
  #endif  // VERBOSE
      auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
  //    auto prdRowMatVectOp = make_prdRowMatVct(my_mA, my_vx);
      auto prdRowMatVectOp =
            make_GemvC_1Row_1Thread_ShMem_Full(my_vy, _alpha, my_mA, my_vx, scalOp1);
      auto localSize = 256;  // NOT FINAL VALUE
      auto nWG = (M + localSize - 1) / localSize;
      auto gridSize = localSize *  nWG;
//      ex.execute(prdRowMatVectOp, localSize, gridSize, N);
    } else if (OPT == 14) {
  #ifdef VERBOSE
      std::cout << "COLS_14" << std::endl;
  #endif  // VERBOSE
      auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
  //    auto prdRowMatVectOp = make_prdRowMatVct(my_mA, my_vx);
      auto nThr = 4;  // NOT FINAL VALUE
      auto prdRowMatVectOp =
            make_GemvC_1Row_MThreads(my_vy, _alpha, my_mA, my_vx, scalOp1, nThr);
      auto localSize = 256;  // NOT FINAL VALUE
      auto nWG = (M + localSize - 1) / localSize;
      auto gridSize = localSize *  nWG * nThr;
      ex.execute(prdRowMatVectOp, localSize , gridSize, localSize);
    } else if (OPT == 15) {
  #ifdef VERBOSE
      std::cout << "COLS_15" << std::endl;
  #endif  // VERBOSE
      auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
  //    auto prdRowMatVectOp = make_prdRowMatVct(my_mA, my_vx);
      auto nThr = 4;  // NOT FINAL VALUE
      auto prdRowMatVectOp =
            make_GemvC_1Row_MThreads_ShMem(my_vy, _alpha, my_mA, my_vx, scalOp1, nThr);
      auto localSize = 256;  // NOT FINAL VALUE
      auto nWG = (M + localSize - 1) / localSize;
      auto gridSize = localSize *  nWG * nThr;
      ex.execute(prdRowMatVectOp, localSize, gridSize, localSize);
    } else if (OPT == 16) {
  #ifdef VERBOSE
      std::cout << "COLS_16" << std::endl;
  #endif  // VERBOSE
      auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
  //    auto prdRowMatVectOp = make_prdRowMatVct(my_mA, my_vx);
      auto nThr = 4;  // NOT FINAL VALUE
  //    auto nThr = 16;
      ContainerT valT1(nThr * M);
      auto mat1 = matrix_view<T, ContainerT>(valT1, 0, M, nThr);
      auto prdRowMatVectOp =
            make_GemvC_1Row_MThreads_ShMem_NoRed(mat1, my_mA, my_vx, nThr);
      auto localSize = 256;  // NOT FINAL VALUE
      auto nWG = (M + localSize - 1) / localSize;
      auto gridSize = localSize *  nWG * nThr;
      ex.execute(prdRowMatVectOp, localSize , gridSize, localSize);
      auto addMOp = make_addSetColumns(mat1);
      auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, addMOp);
      auto addOp = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
      auto assignOp = make_op<Assign>(my_vy, addOp);
      ex.execute(assignOp, localSize);
    } else if (OPT == 17) {
#ifdef VERBOSE
  //    std::cout << "ROWS_2" << std::setprecision(15) << "M = " << _M
      std::cout << "ROWS_17" << "M = " << _M
                << " N = " << _N << std::endl;
#endif  // VERBOSE
      size_t nBlq = 16;
      ContainerT valT1(nBlq * M);
      auto mat1 = matrix_view<T, ContainerT>(valT1, 0, M, nBlq);

      auto localSize = 256;  // NOT FINAL VALUE
      auto nWG = (M + localSize - 1) / localSize;
//      auto nWG = (M + (localSize*nBlq) - 1) / (localSize*nBlq);
//      auto dimWGR = (nWG + nBlq - 1) / nBlq;
      auto dimWGR = nWG;
      auto gridSize = localSize *  dimWGR * nBlq;
      auto gemvC = make_GemvC_1Row_MBlocks (mat1, my_mA, my_vx, nBlq);
      ex.execute(gemvC, localSize, gridSize);
//      mat1.printH("MAT1");

      auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
      auto addMOp = make_addSetColumns(mat1);
      auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, addMOp);
      auto addOp = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
      auto assignOp = make_op<Assign>(my_vy, addOp);
      ex.execute(assignOp, localSize);
    } else if (OPT == 18) {
#ifdef VERBOSE
  //    std::cout << "ROWS_2" << std::setprecision(15) << "M = " << _M
      std::cout << "ROWS_18" << "M = " << _M
                << " N = " << _N << std::endl;
#endif  // VERBOSE
      size_t nBlq = 16;
      ContainerT valT1(nBlq * M);
      auto mat1 = matrix_view<T, ContainerT>(valT1, 0, M, nBlq);

      auto localSize = 256;  // NOT FINAL VALUE
      auto nWG = (M + localSize - 1) / localSize;
//      auto nWG = (M + (localSize*nBlq) - 1) / (localSize*nBlq);
//      auto dimWGR = (nWG + nBlq - 1) / nBlq;
      auto dimWGR = nWG;
      auto gridSize = localSize *  dimWGR * nBlq;
      auto gemvC = make_GemvC_1Row_MBlocks_ShMem (mat1, my_mA, my_vx, nBlq);
      ex.execute(gemvC, localSize, gridSize, localSize);
//      mat1.printH("MAT1");

      auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
      auto addMOp = make_addSetColumns(mat1);
      auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, addMOp);
      auto addOp = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
      auto assignOp = make_op<Assign>(my_vy, addOp);
      ex.execute(assignOp, localSize);
    } else if (OPT == 19) {
#ifdef VERBOSE
  //    std::cout << "ROWS_2" << std::setprecision(15) << "M = " << _M
      std::cout << "ROWS_19" << "M = " << _M
                << " N = " << _N << std::endl;
#endif  // VERBOSE
      size_t nBlq = 16;
      ContainerT valT1(nBlq * M);
      auto mat1 = matrix_view<T, ContainerT>(valT1, 0, M, nBlq);

      auto localSize = 256;  // NOT FINAL VALUE
      auto nWG = (M + localSize - 1) / localSize;
//      auto dimWGR = (nWG + nBlq - 1) / nBlq;
      auto dimWGR = nWG;
      auto gridSize = localSize *  dimWGR * nBlq;
      auto gemvC = make_GemvC_1Row_MBlocks_ShMem_Full (mat1, my_mA, my_vx, nBlq);
      ex.execute(gemvC, localSize, gridSize, (N + nBlq - 1) / nBlq);
//      mat1.printH("MAT1");
/**/
      auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
      auto addMOp = make_addSetColumns(mat1);
      auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, addMOp);
      auto addOp = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
      auto assignOp = make_op<Assign>(my_vy, addOp);
      ex.execute(assignOp, localSize);
/*
      auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
      auto addProdOp = make_addPrdRowMatVctMultShm(my_vy, _alpha, mat1, scalOp1);
      ex.execute(addProdOp, localSize);
*/
    }
  }
#ifdef VERBOSE
  my_vy.printH("RES");
#endif
}

/**** RANK 1 MODIFICATION ****/

template <unsigned int OPT, typename ExecutorType, typename T, typename ContainerT>
void _ger(Executor<ExecutorType> ex, size_t _M, size_t _N, T _alpha,
          vector_view<T, ContainerT> _vx, size_t _incx,
          vector_view<T, ContainerT> _vy, size_t _incy,
          matrix_view<T, ContainerT> _mA, size_t _lda) {
  int accessOpr = true;
  size_t M = _M;
  size_t N = _N;
  auto my_mA =
      matrix_view<T, ContainerT>(_mA, _M, _N, accessOpr, _lda, _mA.getDisp());
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, M);
  auto my_vy = vector_view<T, ContainerT>(_vy, _vy.getDisp(), _incy, N);
  if (my_mA.getAccess()) {  // ROWS ACCESS
    if (OPT == 1) {
#ifdef VERBOSE
      std::cout << "alpha = " << _alpha << std::endl;
      my_mA.printH("MA");
      my_vx.printH("VX");
      my_vy.printH("VY");
#endif
      auto modifOp = make_modifRank1(my_mA, my_vx, my_vy);
      auto scalOp = make_op<ScalarOp, prdOp2_struct>(_alpha, modifOp);
      auto addOp = make_op<BinaryOp, addOp2_struct>(my_mA, scalOp);
      auto assignOp = make_op<Assign>(my_mA, addOp);
      ex.execute(assignOp);
#ifdef VERBOSE
      my_vy.printH("VY");
#endif
    } else if (OPT == 11) {
      std::cout << "GER_ROW_11 = " << std::endl;
#ifdef VERBOSE
      std::cout << "alpha = " << _alpha << std::endl;
      my_mA.printH("MA");
      my_vx.printH("VX");
      my_vy.printH("VY");
#endif
      auto localSize = 256;  // NOT FINAL VALUE
      auto assignOp = make_Ger_1Row_1WG(my_mA, _alpha, my_vx, my_vy);
      ex.execute(assignOp, localSize, M);
#ifdef VERBOSE
      my_vy.printH("VY");
#endif
    }
  } else { // COLUMN ACCESS
    if (OPT == 1) {
#ifdef VERBOSE
      std::cout << "alpha = " << _alpha << std::endl;
      my_mA.printH("MA");
      my_vx.printH("VX");
      my_vy.printH("VY");
#endif
      auto modifOp = make_modifRank1(my_mA, my_vx, my_vy);
      auto scalOp = make_op<ScalarOp, prdOp2_struct>(_alpha, modifOp);
      auto addOp = make_op<BinaryOp, addOp2_struct>(my_mA, scalOp);
      auto assignOp = make_op<Assign>(my_mA, addOp);
      ex.execute(assignOp);
#ifdef VERBOSE
      my_vy.printH("VY");
#endif
    } else if (OPT == 11) {
      std::cout << "GER_COL_11 = " << std::endl;
    #ifdef VERBOSE
      std::cout << "alpha = " << _alpha << std::endl;
      my_mA.printH("MA");
      my_vx.printH("VX");
      my_vy.printH("VY");
    #endif
      auto localSize = 256;  // NOT FINAL VALUE
      auto assignOp = make_Ger_1Row_1WG(my_mA, _alpha, my_vx, my_vy);
      ex.execute(assignOp, localSize, M*localSize);
    #ifdef VERBOSE
      my_vy.printH("VY");
    #endif
    }
  }
}

}  // namespace blas

#endif  // BLAS2_INTERFACE_SYCL_HPP
