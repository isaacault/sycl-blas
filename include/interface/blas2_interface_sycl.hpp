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
//#include <algorithm>

#include <executors/executor_sycl.hpp>
#include <operations/blas1_trees.hpp>

namespace blas {

/**** MATRIX VECTOR PRODUCT ****/

/*! _GEMV.
 * @brief Implementation of the General Matrix Vector product.
 */
template <unsigned int _localSize = 0, unsigned int _shrMemSize = 0,
          unsigned int _n_rows_WG = 0, unsigned int _n_cols_WG = 0,
          typename ExecutorType, typename T, typename ContainerT>
void _GEMV(Executor<ExecutorType> ex, std::string _Trans, size_t _M, size_t _N,
           T _alpha, matrix_view<T, ContainerT> _mA, size_t _lda,
           vector_view<T, ContainerT> _vx, size_t _incx, T _beta,
           vector_view<T, ContainerT> _vy, size_t _incy) {
  if ((_Trans[0] != 'n') && (_Trans[0] != 't') && (_Trans[0] != 'c') &&
      (_Trans[0] != 'N') && (_Trans[0] != 'T') && (_Trans[0] != 'C'))
    std::cout << "Erroneous parameter" << std::endl;
  int accessOpr = ((_Trans[0] == 'n') || (_Trans[0] == 'N'));
  unsigned int M = _M;
  unsigned int N = _N;
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
//      std::cout << "ROWS_CASE"  << std::endl;
    const auto interLoop=1;
    const auto localSize = (_localSize == 0)? 256: _localSize;
    const auto n_rows_WG = (_n_rows_WG == 0)? 1: std::min(M,_n_rows_WG);
    const auto n_cols_WG = (_n_cols_WG == 0)? N: std::min(N,_n_cols_WG);
    const auto shrMemSize = (_localSize == 0)? localSize: _shrMemSize;

    const auto nWG_col = (N - 1) / n_cols_WG + 1;
    const auto nWG_row = (M - 1) / n_rows_WG + 1;

//    const auto scratchSize = ((shrMemSize==0)?localSize:1)*nWG_col;
    const auto scratchSize = ((shrMemSize==0)?std::min(N,localSize):1)*nWG_col;
    ContainerT valT1(M * scratchSize);
    auto mat1 = matrix_view<T, ContainerT>(valT1, 0, M, scratchSize);

#ifdef VERBOSE
    std::cout << "ROWS_CASE: "
              << "M = " << M << " , N = " << N
              << " , localSize = "  << localSize
              << " , n_rows_WG = "  << n_rows_WG
              << " , n_cols_WG = "  << n_cols_WG
              << " , shrMemSize = " << shrMemSize
              << " , scratchSize = "  << scratchSize
              << std::endl;
#endif  // VERBOSE

    auto gridSize = localSize * nWG_row * nWG_col;
    auto gemvR = make_Gemv_Row<interLoop> (mat1, my_mA, my_vx, nWG_row, nWG_col, shrMemSize);
    if (shrMemSize == 0) {
      ex.execute(gemvR, localSize, gridSize);
    } else {
      ex.execute(gemvR, localSize, gridSize, shrMemSize);
    }
//    mat1.printH("MAT1");

    auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
    auto addMOp = make_addSetColumns(mat1);
    auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, addMOp);
    auto addOp = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
    auto assignOp = make_op<Assign>(my_vy, addOp);
    ex.execute(assignOp, localSize);

  } else {
//      std::cout << "COLS_CASE"  << std::endl;
    const auto localSize  = (_localSize == 0)? 256: _localSize;
    const auto n_rows_WG  = (_n_rows_WG == 0)? localSize: std::min(M,_n_rows_WG);
    const auto n_cols_WG  = (_n_cols_WG == 0)? localSize: std::min(N,_n_cols_WG);;
    const auto shrMemSize = (_localSize == 0)? localSize: _shrMemSize;

    const auto nWG_col = (N - 1) / n_cols_WG + 1;
    const auto nWG_row = (M - 1) / n_rows_WG + 1;

    const auto scratchSize = nWG_col;
    ContainerT valT1(M * scratchSize);
    auto mat1 = matrix_view<T, ContainerT>(valT1, 0, M, scratchSize);

#ifdef VERBOSE
    std::cout << "COLS_CASE: "
              << "M = " << M << " , N = " << N
              << " , localSize = "  << localSize
              << " , n_rows_WG = "  << n_rows_WG
              << " , n_cols_WG = "  << n_cols_WG
              << " , shrMemSize = " << shrMemSize
              << " , scratchSize = "  << scratchSize
              << std::endl;
#endif  // VERBOSE

    auto gridSize = localSize * nWG_row * nWG_col;
    auto gemvC = make_Gemv_Col (mat1, my_mA, my_vx, nWG_row, nWG_col, shrMemSize);
    if (shrMemSize == 0) {
      ex.execute(gemvC, localSize, gridSize);
    } else {
      ex.execute(gemvC, localSize, gridSize, shrMemSize);
    }

    auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
    auto addMOp = make_addSetColumns(mat1);
    auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, addMOp);
    auto addOp = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
    auto assignOp = make_op<Assign>(my_vy, addOp);
    ex.execute(assignOp, localSize);
  }
}

/*! _TRMV.
 * @brief Implementation of the Triangular Matrix Vector product.
 */
template <unsigned int _localSize = 0, unsigned int _shrMemSize = 0,
          unsigned int _n_rows_WG = 0, unsigned int _n_cols_WG = 0,
          typename ExecutorType, typename T, typename ContainerT>
void _TRMV(Executor<ExecutorType> ex, std::string _Uplo,
           std::string _Trans, std::string _Diag,
           size_t _N, matrix_view<T, ContainerT> _mA, size_t _lda,
           vector_view<T, ContainerT> _vx, size_t _incx) {
  if ((_Trans[0] != 'n') && (_Trans[0] != 't') && (_Trans[0] != 'c') &&
      (_Trans[0] != 'N') && (_Trans[0] != 'T') && (_Trans[0] != 'C') &&
      (_Uplo[0]  != 'u') && (_Uplo[0]  != 'l') &&
      (_Uplo[0]  != 'U') && (_Uplo[0]  != 'L') &&
      (_Diag[0]  != 'u') && (_Diag[0]  != 'n') &&
      (_Diag[0]  != 'U') && (_Diag[0]  != 'N'))
    std::cout << "Erroneous parameter" << std::endl;
  int accessOpr = ((_Trans[0] == 'n') || (_Trans[0] == 'N'));
  int triangOpr = (accessOpr)?((_Uplo[0] == 'u') || (_Uplo[0] == 'U')):
                  	      ((_Uplo[0] == 'l') || (_Uplo[0] == 'L'));
  int unitDiag  = ((_Diag[0] == 'u') || (_Diag[0] == 'U'));
  unsigned int N = _N;
  auto my_mA =
      matrix_view<T, ContainerT>(_mA, _N, _N, accessOpr, _lda, _mA.getDisp());
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, N);
#ifdef VERBOSE
  my_mA.printH("MA");
  my_vx.printH("VX");
#endif  // VERBOSE
//  std::cout << "Trans = " << _Trans << " , accessOpr = " << accessOpr
//            << " , Uplo = " << _Uplo << " , triangOpr = " << triangOpr
//            << std::endl;
  if (my_mA.getAccess()) {  // ROWS ACCESS
//      std::cout << "ROWS_CASE"  << std::endl;
    const auto interLoop=1;
    const auto localSize = (_localSize == 0)? 256: _localSize;
    const auto n_rows_WG = (_n_rows_WG == 0)? 1: std::min(N,_n_rows_WG);
    const auto n_cols_WG = (_n_cols_WG == 0)? N: std::min(N,_n_cols_WG);
    const auto shrMemSize = (_localSize == 0)? localSize: _shrMemSize;

    const auto nWG_col = (N - 1) / n_cols_WG + 1;
    const auto nWG_row = (N - 1) / n_rows_WG + 1;

//    const auto scratchSize = ((shrMemSize==0)?localSize:1)*nWG_col;
    const auto scratchSize = ((shrMemSize==0)?std::min(N,localSize):1)*nWG_col;
    ContainerT valT1(N * scratchSize);
    auto mat1 = matrix_view<T, ContainerT>(valT1, 0, N, scratchSize);

#ifdef VERBOSE
    std::cout << "ROWS_CASE: "
              << "N = " << N
              << " , localSize = "  << localSize
              << " , n_rows_WG = "  << n_rows_WG
              << " , n_cols_WG = "  << n_cols_WG
              << " , shrMemSize = " << shrMemSize
              << " , scratchSize = "  << scratchSize
              << std::endl;
#endif  // VERBOSE

    auto gridSize = localSize * nWG_row * nWG_col;
//    auto gemvR = make_Gemv_Row<interLoop> (mat1, my_mA, my_vx, nWG_row, nWG_col, shrMemSize);
//    if (shrMemSize == 0) {
//      ex.execute(gemvR, localSize, gridSize);
//    } else {
//      ex.execute(gemvR, localSize, gridSize, shrMemSize);
//    }
    if (triangOpr == 1) {
      if (unitDiag == 1) {
        auto gemvR = make_Gemv_Row<interLoop,false,true,true, true> (mat1, my_mA, my_vx, nWG_row, nWG_col, shrMemSize);
        if (shrMemSize == 0) {
          ex.execute(gemvR, localSize, gridSize);
        } else {
          ex.execute(gemvR, localSize, gridSize, shrMemSize);
        }
      } else {
        auto gemvR = make_Gemv_Row<interLoop,false,true,true> (mat1, my_mA, my_vx, nWG_row, nWG_col, shrMemSize);
        if (shrMemSize == 0) {
          ex.execute(gemvR, localSize, gridSize);
        } else {
          ex.execute(gemvR, localSize, gridSize, shrMemSize);
        }

      }
    } else {
      if (unitDiag == 1) {
        auto gemvR = make_Gemv_Row<interLoop,true,true,false, true> (mat1, my_mA, my_vx, nWG_row, nWG_col, shrMemSize);
        if (shrMemSize == 0) {
          ex.execute(gemvR, localSize, gridSize);
        } else {
          ex.execute(gemvR, localSize, gridSize, shrMemSize);
        }
      } else {
        auto gemvR = make_Gemv_Row<interLoop,true,true,false> (mat1, my_mA, my_vx, nWG_row, nWG_col, shrMemSize);
        if (shrMemSize == 0) {
          ex.execute(gemvR, localSize, gridSize);
        } else {
          ex.execute(gemvR, localSize, gridSize, shrMemSize);
        }
      }
    }

//    mat1.printH("MAT1");

    auto addMOp = make_addSetColumns(mat1);
    auto assignOp = make_op<Assign>(my_vx, addMOp);
    ex.execute(assignOp, localSize);

  } else {
//      std::cout << "COLS_CASE"  << std::endl;
    const auto localSize  = (_localSize == 0)? 256: _localSize;
    const auto n_rows_WG  = (_n_rows_WG == 0)? localSize: std::min(N,_n_rows_WG);;
    const auto n_cols_WG  = (_n_cols_WG == 0)? localSize: std::min(N,_n_cols_WG);;
    const auto shrMemSize = (_localSize == 0)? localSize: _shrMemSize;

    const auto nWG_row = (N - 1) / n_rows_WG + 1;
    const auto nWG_col = (N - 1) / n_cols_WG + 1;

    const auto scratchSize = nWG_col;
    ContainerT valT1(N * scratchSize);
    auto mat1 = matrix_view<T, ContainerT>(valT1, 0, N, scratchSize);

#ifdef VERBOSE
    std::cout << "COLS_CASE: "
              << "N = " << N
              << " , localSize = "  << localSize
              << " , n_rows_WG = "  << n_rows_WG
              << " , n_cols_WG = "  << n_cols_WG
              << " , shrMemSize = " << shrMemSize
              << " , scratchSize = "  << scratchSize
              << std::endl;
#endif  // VERBOSE

    auto gridSize = localSize * nWG_row * nWG_col;
//    auto gemvC = make_Gemv_Col (mat1, my_mA, my_vx, nWG_row, nWG_col, shrMemSize);
    if (triangOpr == 1) {
      if (unitDiag == 1) {
        auto gemvC = make_Gemv_Col<false,true,true, true> (mat1, my_mA, my_vx, nWG_row, nWG_col, shrMemSize);
        if (shrMemSize == 0) {
          ex.execute(gemvC, localSize, gridSize);
        } else {
          ex.execute(gemvC, localSize, gridSize, shrMemSize);
        }
      } else {
        auto gemvC = make_Gemv_Col<false,true,true> (mat1, my_mA, my_vx, nWG_row, nWG_col, shrMemSize);
        if (shrMemSize == 0) {
          ex.execute(gemvC, localSize, gridSize);
        } else {
          ex.execute(gemvC, localSize, gridSize, shrMemSize);
        }

      }
    } else {
      if (unitDiag == 1) {
        auto gemvC = make_Gemv_Col<true,true,false, true> (mat1, my_mA, my_vx, nWG_row, nWG_col, shrMemSize);
        if (shrMemSize == 0) {
          ex.execute(gemvC, localSize, gridSize);
        } else {
          ex.execute(gemvC, localSize, gridSize, shrMemSize);
        }
      } else {
        auto gemvC = make_Gemv_Col<true,true,false> (mat1, my_mA, my_vx, nWG_row, nWG_col, shrMemSize);
        if (shrMemSize == 0) {
          ex.execute(gemvC, localSize, gridSize);
        } else {
          ex.execute(gemvC, localSize, gridSize, shrMemSize);
        }
      }
    }

    auto addMOp = make_addSetColumns(mat1);
    auto assignOp = make_op<Assign>(my_vx, addMOp);
    ex.execute(assignOp, localSize);
  }
}

/*! _SYMV.
 * @brief Implementation of the Symmetric Matrix Vector product.
 */
/*
ssymv 	( 	character  	UPLO,
   integer  	N,
   real  	ALPHA,
   real, dimension(lda,*)  	A,
   integer  	LDA,
   real, dimension(*)  	X,
   integer  	INCX,
   real  	BETA,
   real, dimension(*)  	Y,
   integer  	INCY
 ) 	*/
template <unsigned int _localSize = 0, unsigned int _shrMemSize = 0,
          unsigned int _n_rows_WG = 0, unsigned int _n_cols_WG = 0,
          typename ExecutorType, typename T, typename ContainerT>
void _SYMV(Executor<ExecutorType> ex, std::string _Uplo, size_t _N,
           T _alpha, matrix_view<T, ContainerT> _mA, size_t _lda,
           vector_view<T, ContainerT> _vx, size_t _incx, T _beta,
           vector_view<T, ContainerT> _vy, size_t _incy) {
  if ((_Uplo[0]  != 'u') && (_Uplo[0]  != 'l') &&
      (_Uplo[0]  != 'U') && (_Uplo[0]  != 'L'))
    std::cout << "Erroneous parameter" << std::endl;
//  int accessOpr = ((_Trans[0] == 'n') || (_Trans[0] == 'N'));
  int accessOpr = 1;
  int triangOpr = ((_Uplo[0] == 'u') || (_Uplo[0] == 'U'));
  unsigned int N = _N;
  auto my_mA =
      matrix_view<T, ContainerT>(_mA, _N, _N, accessOpr, _lda, _mA.getDisp());
  auto my_mAT =
          matrix_view<T, ContainerT>(_mA, _N, _N, 0, _lda, _mA.getDisp());
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, N);
  auto my_vy = vector_view<T, ContainerT>(_vy, _vy.getDisp(), _incy, N);
#ifdef VERBOSE
  std::cout << "alpha = " << _alpha << " , beta = " << _beta << std::endl;
  my_mA.printH("MA");
  my_vx.printH("VX");
  my_vy.printH("VY");
#endif  // VERBOSE

  const auto interLoop=1;

  const auto localSize  = (_localSize == 0)? 256: _localSize;
  const auto shrMemSize = (_localSize == 0)? localSize: _shrMemSize;

  const auto n_rows_WG_R = (_n_rows_WG == 0)? 1: std::min(N,_n_rows_WG);
  const auto n_cols_WG_R = (_n_cols_WG == 0)? N: std::min(N,_n_cols_WG);
//  const auto n_cols_WG_R = std::max(localSize,((_n_cols_WG == 0)? N:_n_cols_WG));
  const auto nWG_row_R = (N - 1) / n_rows_WG_R + 1;
  const auto nWG_col_R = (N - 1) / n_cols_WG_R + 1;
  auto gridSize_R = localSize * nWG_row_R * nWG_col_R;

  const auto n_rows_WG_C  = (_n_rows_WG == 0)? localSize: _n_rows_WG;
  const auto n_cols_WG_C  = (_n_cols_WG == 0)? localSize: _n_cols_WG;
//  const auto n_rows_WG_C  = (_n_rows_WG == 0)? localSize: std::min(N,_n_rows_WG);
//  const auto n_cols_WG_C  = (_n_cols_WG == 0)? localSize: std::min(N,_n_cols_WG);
//  const auto n_cols_WG_C  = std::min(N,((_n_cols_WG == 0)? localSize:_n_cols_WG));
  const auto nWG_row_C = (N - 1) / n_rows_WG_C + 1;
  const auto nWG_col_C = (N - 1) / n_cols_WG_C + 1;
  auto gridSize_C = localSize * nWG_row_C * nWG_col_C;

  const auto scratchSize_R = ((shrMemSize==0)?std::min(N,localSize):1)*nWG_col_R;
  ContainerT valTR(N * scratchSize_R);
  auto matR = matrix_view<T, ContainerT>(valTR, 0, N, scratchSize_R);

  const auto scratchSize_C = nWG_col_C;
  ContainerT valTC(N * scratchSize_C);
  auto matC = matrix_view<T, ContainerT>(valTC, 0, N, scratchSize_C);

  if (my_mA.getAccess()) {  // ROWS ACCESS
//      std::cout << "ROWS_CASE"  << std::endl;
/*
    const auto interLoop=1;
    const auto localSize = (_localSize == 0)? 256: _localSize;
    const auto n_rows_WG = (_n_rows_WG == 0)? 1: std::min(N,_n_rows_WG);
    const auto n_cols_WG = (_n_cols_WG == 0)? N: std::min(N,_n_cols_WG);
    const auto shrMemSize = (_localSize == 0)? localSize: _shrMemSize;

    const auto nWG_col = (N - 1) / n_cols_WG + 1;
    const auto nWG_row = (N - 1) / n_rows_WG + 1;

//    const auto scratchSize = ((shrMemSize==0)?localSize:1)*nWG_col;
    const auto scratchSize = ((shrMemSize==0)?std::min(N,localSize):1)*nWG_col;
    ContainerT valT1(2 * N * scratchSize);
    auto mat1 = matrix_view<T, ContainerT>(valT1, 0, N, scratchSize);
    auto mat2 = matrix_view<T, ContainerT>(valT1, N*scratchSize, N, scratchSize);
*/
#ifdef VERBOSE
    std::cout << "SYMV_ROWS: "
              << "N = " << N
              << " , localSize = "  << localSize
              << " , shrMemSize = " << shrMemSize
              << " , n_rows_WG_R = "  << n_rows_WG_R
              << " , n_cols_WG_R = "  << n_cols_WG_R
              << " , scratchSize_R = "  << scratchSize_R
              << " , n_rows_WG_C = "  << n_rows_WG_C
              << " , n_cols_WG_C = "  << n_cols_WG_C
              << " , scratchSize_C = "  << scratchSize_C
              << std::endl;
#endif  // VERBOSE

//    my_mA.printH("MAT");
//    my_vx.printH("VX");
//    my_vy.printH("VY");

//    auto gridSize = localSize * nWG_row * nWG_col;
//    auto gemvR = make_Gemv_Row<interLoop> (mat1, my_mA, my_vx, nWG_row, nWG_col, shrMemSize);
//    if (shrMemSize == 0) {
//      ex.execute(gemvR, localSize, gridSize);
//    } else {
//      ex.execute(gemvR, localSize, gridSize, shrMemSize);
//    }
/* */
    if (triangOpr == 1) {
      auto gemvR = make_Gemv_Row<interLoop,false,true,true> (matR, my_mA, my_vx, nWG_row_R, nWG_col_R, shrMemSize);
//      auto gemvC = make_Gemv_Col<true,false,false> (matC, my_mA, my_vx, nWG_row_C, nWG_col_C, shrMemSize);
      auto gemvC = make_Gemv_Col<true,false,false> (matC, my_mAT, my_vx, nWG_row_C, nWG_col_C, shrMemSize);
      if (shrMemSize == 0) {
        ex.execute(gemvR, localSize, gridSize_R);
        ex.execute(gemvC, localSize, gridSize_C);
      } else {
        ex.execute(gemvR, localSize, gridSize_R, shrMemSize);
        ex.execute(gemvC, localSize, gridSize_C, shrMemSize);
      }
    } else {
      auto gemvR = make_Gemv_Row<interLoop,true,true,false> (matR, my_mA, my_vx, nWG_row_R, nWG_col_R, shrMemSize);
//      auto gemvC = make_Gemv_Col<false,false,true> (matC, my_mA, my_vx, nWG_row_C, nWG_col_C, shrMemSize);
      auto gemvC = make_Gemv_Col<false,false,true> (matC, my_mAT, my_vx, nWG_row_C, nWG_col_C, shrMemSize);
      if (shrMemSize == 0) {
        ex.execute(gemvR, localSize, gridSize_R);
        ex.execute(gemvC, localSize, gridSize_C);
      } else {
        ex.execute(gemvR, localSize, gridSize_R, shrMemSize);
        ex.execute(gemvC, localSize, gridSize_C, shrMemSize);
      }
    }
/* */

//    matR.printH("MATR");
//    matC.printH("MATC");

    auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
    auto addMOpR = make_addSetColumns(matR);
    auto addMOpC = make_addSetColumns(matC);
    auto addMOp  = make_op<BinaryOp, addOp2_struct>(addMOpR, addMOpC);
    auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, addMOp);
    auto addOp = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
    auto assignOp = make_op<Assign>(my_vy, addOp);
    ex.execute(assignOp, localSize);

  } else {
//      std::cout << "COLS_CASE"  << std::endl;
/*
    const auto interLoop=1;
    const auto localSize  = (_localSize == 0)? 256: _localSize;
    const auto n_rows_WG  = (_n_rows_WG == 0)? localSize: _n_rows_WG;
    const auto n_cols_WG  = (_n_cols_WG == 0)? localSize: _n_cols_WG;
    const auto shrMemSize = (_localSize == 0)? localSize: _shrMemSize;

    const auto nWG_col = (N - 1) / n_cols_WG + 1;
    const auto nWG_row = (N - 1) / n_rows_WG + 1;

    const auto scratchSize = nWG_col;
    ContainerT valT1(2 * N * scratchSize);
    auto mat1 = matrix_view<T, ContainerT>(valT1, 0, N, scratchSize);
    auto mat2 = matrix_view<T, ContainerT>(valT1, N*scratchSize, N, scratchSize);
*/
#ifdef VERBOSE
    std::cout << "SYMV_COLS: "
              << "N = " << N
              << " , localSize = "  << localSize
              << " , shrMemSize = " << shrMemSize
              << " , n_rows_WG_R = "  << n_rows_WG_R
              << " , n_cols_WG_R = "  << n_cols_WG_R
              << " , scratchSize_R = "  << scratchSize_R
              << " , n_rows_WG_C = "  << n_rows_WG_C
              << " , n_cols_WG_C = "  << n_cols_WG_C
              << " , scratchSize_C = "  << scratchSize_C
              << std::endl;
#endif  // VERBOSE

/* */
    if (triangOpr == 1) {
      auto gemvC = make_Gemv_Col<false,true,true> (matC, my_mA, my_vx, nWG_row_C, nWG_col_C, shrMemSize);
      auto gemvR = make_Gemv_Row<interLoop,true,false,false> (matR, my_mAT, my_vx, nWG_row_R, nWG_col_R, shrMemSize);
      if (shrMemSize == 0) {
        ex.execute(gemvC, localSize, gridSize_C);
        ex.execute(gemvR, localSize, gridSize_R);
      } else {
        ex.execute(gemvC, localSize, gridSize_C, shrMemSize);
        ex.execute(gemvR, localSize, gridSize_R, shrMemSize);
      }
    } else {
      auto gemvC = make_Gemv_Col<true,true,false> (matC, my_mA, my_vx, nWG_row_C, nWG_col_C, shrMemSize);
      auto gemvR = make_Gemv_Row<interLoop,false,false,true> (matR, my_mAT, my_vx, nWG_row_R, nWG_col_R, shrMemSize);
      if (shrMemSize == 0) {
        ex.execute(gemvC, localSize, gridSize_C);
        ex.execute(gemvR, localSize, gridSize_R);
      } else {
        ex.execute(gemvC, localSize, gridSize_C, shrMemSize);
        ex.execute(gemvR, localSize, gridSize_R, shrMemSize);
      }

    }
/* */
/* */
    auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
    auto addMOpR = make_addSetColumns(matR);
    auto addMOpC = make_addSetColumns(matC);
    auto addMOp  = make_op<BinaryOp, addOp2_struct>(addMOpR, addMOpC);
    auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, addMOp);
    auto addOp = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
    auto assignOp = make_op<Assign>(my_vy, addOp);
    ex.execute(assignOp, localSize);
/*  */
  }
}

/*! _GER.
 * @brief Implementation of the rank 1 operation
 */

template <unsigned int _localSize = 0, unsigned int _shrMemSize = 0,
          unsigned int _n_rows_WG = 0, unsigned int _n_cols_WG = 0,
          typename ExecutorType, typename T, typename ContainerT>
void _GER(Executor<ExecutorType> ex, size_t _M, size_t _N, T _alpha,
          vector_view<T, ContainerT> _vx, size_t _incx,
          vector_view<T, ContainerT> _vy, size_t _incy,
          matrix_view<T, ContainerT> _mA, size_t _lda) {
  int accessOpr = true;
  unsigned int M = _M;
  unsigned int N = _N;
  auto my_mA =
      matrix_view<T, ContainerT>(_mA, _M, _N, accessOpr, _lda, _mA.getDisp());
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, M);
  auto my_vy = vector_view<T, ContainerT>(_vy, _vy.getDisp(), _incy, N);
  if (my_mA.getAccess()) {  // ROWS ACCESS
    const auto localSize = (_localSize == 0)? 256: _localSize;
    const auto n_rows_WG = (_n_rows_WG == 0)? 1: std::min(M,_n_rows_WG);;
    const auto n_cols_WG = (_n_cols_WG == 0)? N: std::min(N,_n_cols_WG);;
    const auto shrMemSize = (_localSize == 0)? localSize: _shrMemSize;

    const auto nWG_row = (M - 1) / n_rows_WG + 1;
    const auto nWG_col = (N - 1) / n_cols_WG + 1;

#ifdef VERBOSE
    std::cout << "GER_ROWS: "
              << "M = " << M << " , N = " << N
              << " , localSize = "  << localSize
              << " , n_rows_WG = "  << n_rows_WG
              << " , n_cols_WG = "  << n_cols_WG
              << " , shrMemSize = " << shrMemSize
              << std::endl;
#endif  // VERBOSE

    auto gridSize = localSize * nWG_row * nWG_col;
    auto assignOp = make_Ger_Row(my_mA, _alpha, my_vx, my_vy, nWG_row, nWG_col, shrMemSize);
    if (shrMemSize == 0) {
      ex.execute(assignOp, localSize, gridSize);
    } else {
      ex.execute(assignOp, localSize, gridSize, shrMemSize);
    }

//    ex.execute(assignOp, localSize, nWG_row*localSize*nWG_col, std::max(localSize,n_rows_WG));
  } else { // COLUMN ACCESS
    const auto localSize = (_localSize == 0)? 256: _localSize;
    const auto n_rows_WG = (_n_rows_WG == 0)? localSize: std::min(M,_n_rows_WG);;
    const auto n_cols_WG = (_n_cols_WG == 0)? localSize: std::min(N,_n_cols_WG);;
    const auto shrMemSize = (_localSize == 0)? localSize: _shrMemSize;

    const auto nWG_row = (M - 1) / n_rows_WG + 1;
    const auto nWG_col = (N - 1) / n_cols_WG + 1;

#ifdef VERBOSE
    std::cout << "GER_COLS: "
              << "M = " << M << " , N = " << N
              << " , localSize = "  << localSize
              << " , n_rows_WG = "  << n_rows_WG
              << " , n_cols_WG = "  << n_cols_WG
              << " , shrMemSize = " << shrMemSize
              << std::endl;
#endif  // VERBOSE

    auto gridSize = localSize * nWG_row * nWG_col;
    auto assignOp = make_Ger_Col(my_mA, _alpha, my_vx, my_vy, nWG_row, nWG_col, shrMemSize);
    if (shrMemSize == 0) {
      ex.execute(assignOp, localSize, gridSize);
    } else {
      ex.execute(assignOp, localSize, gridSize, shrMemSize);
    }
  }
}

/*! _SYR.
 * @brief Implementation of the rank 1 operation
 */
/*
ssyr 	( 	character  	UPLO,
   integer  	N,
   real  	ALPHA,
   real, dimension(*)  	X,
   integer  	INCX,
   real, dimension(lda,*)  	A,
   integer  	LDA
 )
*/
template <unsigned int _localSize = 0, unsigned int _shrMemSize = 0,
          unsigned int _n_rows_WG = 0, unsigned int _n_cols_WG = 0,
          typename ExecutorType, typename T, typename ContainerT>
void _SYR(Executor<ExecutorType> ex, std::string _Uplo,
          size_t _N, T _alpha,
          vector_view<T, ContainerT> _vx, size_t _incx,
          matrix_view<T, ContainerT> _mA, size_t _lda) {
  int accessOpr = true;
  int triangOpr = ((_Uplo[0] == 'u') || (_Uplo[0] == 'U'));
  unsigned int N = _N;
  auto my_mA =
      matrix_view<T, ContainerT>(_mA, _N, _N, accessOpr, _lda, _mA.getDisp());
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, N);
  if (my_mA.getAccess()) {  // ROWS ACCESS
    const auto localSize = (_localSize == 0)? 256: _localSize;
    const auto n_rows_WG = (_n_rows_WG == 0)? 1: std::min(N,_n_rows_WG);
    const auto n_cols_WG = (_n_cols_WG == 0)? N: std::min(N,_n_cols_WG);;
    const auto shrMemSize = (_localSize == 0)? localSize: _shrMemSize;

    const auto nWG_row = (N - 1) / n_rows_WG + 1;
    const auto nWG_col = (N - 1) / n_cols_WG + 1;

#ifdef VERBOSE
    std::cout << "SYR_ROWS: "
              << "N = " << N
              << " , localSize = "  << localSize
              << " , n_rows_WG = "  << n_rows_WG
              << " , n_cols_WG = "  << n_cols_WG
              << " , shrMemSize = " << shrMemSize
              << std::endl;
#endif  // VERBOSE

//    my_mA.printH("MAT");
//    my_vx.printH("VX ");

    if (triangOpr) {
      auto gridSize = localSize * nWG_row * nWG_col;
      auto assignOp = make_Ger_Row<true,false,true,true>(my_mA, _alpha, my_vx, my_vx, nWG_row, nWG_col, shrMemSize);
      if (shrMemSize == 0) {
        ex.execute(assignOp, localSize, gridSize);
      } else {
        ex.execute(assignOp, localSize, gridSize, shrMemSize);
      }
    } else {
      auto gridSize = localSize * nWG_row * nWG_col;
      auto assignOp = make_Ger_Row<true,true,true,false>(my_mA, _alpha, my_vx, my_vx, nWG_row, nWG_col, shrMemSize);
      if (shrMemSize == 0) {
        ex.execute(assignOp, localSize, gridSize);
      } else {
        ex.execute(assignOp, localSize, gridSize, shrMemSize);
      }
    }
//    my_mA.printH("MAT");

//    ex.execute(assignOp, localSize, nWG_row*localSize*nWG_col, std::max(localSize,n_rows_WG));
  } else { // COLUMN ACCESS
    const auto localSize = (_localSize == 0)? 256: _localSize;
    const auto n_rows_WG = (_n_rows_WG == 0)? localSize: std::min(N,_n_rows_WG);
    const auto n_cols_WG = (_n_cols_WG == 0)? localSize: std::min(N,_n_cols_WG);;
    const auto shrMemSize = (_localSize == 0)? localSize: _shrMemSize;

    const auto nWG_row = (N - 1) / n_rows_WG + 1;
    const auto nWG_col = (N - 1) / n_cols_WG + 1;

#ifdef VERBOSE
    std::cout << "SYR_COLS: "
              << "N = " << N
              << " , localSize = "  << localSize
              << " , n_rows_WG = "  << n_rows_WG
              << " , n_cols_WG = "  << n_cols_WG
              << " , shrMemSize = " << shrMemSize
              << std::endl;
#endif  // VERBOSE

//    my_mA.printH("MAT");
//    my_vx.printH("VX ");

    auto gridSize = localSize * nWG_row * nWG_col;
    if (triangOpr) {
      auto assignOp = make_Ger_Col<true,false,true,true>(my_mA, _alpha, my_vx, my_vx, nWG_row, nWG_col, shrMemSize);
      if (shrMemSize == 0) {
        ex.execute(assignOp, localSize, gridSize);
      } else {
        ex.execute(assignOp, localSize, gridSize, shrMemSize);
      }
    } else {
      auto assignOp = make_Ger_Col<true,true,true,false>(my_mA, _alpha, my_vx, my_vx, nWG_row, nWG_col, shrMemSize);
      if (shrMemSize == 0) {
        ex.execute(assignOp, localSize, gridSize);
      } else {
        ex.execute(assignOp, localSize, gridSize, shrMemSize);
      }
    }
//    my_mA.printH("MAT");
  }
}

/*
ssyr2 	( 	character  	UPLO,
		integer  	N,
		real  	ALPHA,
		real, dimension(*)  	X,
		integer  	INCX,
		real, dimension(*)  	Y,
		integer  	INCY,
		real, dimension(lda,*)  	A,
		integer  	LDA
	)
*/
template <unsigned int _localSize = 0, unsigned int _shrMemSize = 0,
          unsigned int _n_rows_WG = 0, unsigned int _n_cols_WG = 0,
          typename ExecutorType, typename T, typename ContainerT>
void _SYR2(Executor<ExecutorType> ex, std::string _Uplo,
          size_t _N, T _alpha,
          vector_view<T, ContainerT> _vx, size_t _incx,
          vector_view<T, ContainerT> _vy, size_t _incy,
          matrix_view<T, ContainerT> _mA, size_t _lda) {
  int accessOpr = true;
  int triangOpr = ((_Uplo[0] == 'u') || (_Uplo[0] == 'U'));
  unsigned int N = _N;
  auto my_mA =
      matrix_view<T, ContainerT>(_mA, _N, _N, accessOpr, _lda, _mA.getDisp());
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, N);
  auto my_vy = vector_view<T, ContainerT>(_vy, _vy.getDisp(), _incy, N);
  if (my_mA.getAccess()) {  // ROWS ACCESS
    const auto localSize = (_localSize == 0)? 256: _localSize;
    const auto n_rows_WG = (_n_rows_WG == 0)? 1: std::min(N,_n_rows_WG);
    const auto n_cols_WG = (_n_cols_WG == 0)? N: std::min(N,_n_cols_WG);;
    const auto shrMemSize = (_localSize == 0)? 2*localSize: _shrMemSize;

    const auto nWG_row = (N - 1) / n_rows_WG + 1;
    const auto nWG_col = (N - 1) / n_cols_WG + 1;

#ifdef VERBOSE
    std::cout << "SYR2_ROWS: "
              << "N = " << N
              << " , localSize = "  << localSize
              << " , n_rows_WG = "  << n_rows_WG
              << " , n_cols_WG = "  << n_cols_WG
              << " , shrMemSize = " << shrMemSize
              << std::endl;
#endif  // VERBOSE

    my_mA.printH("MAT");
    my_vx.printH("VX ");
    my_vy.printH("VY ");

    if (triangOpr) {
      auto gridSize = localSize * nWG_row * nWG_col;
      auto assignOp = make_Ger_Row<false,false,true,true>(my_mA, _alpha, my_vx, my_vy, nWG_row, nWG_col, shrMemSize);
      if (shrMemSize == 0) {
        ex.execute(assignOp, localSize, gridSize);
      } else {
        ex.execute(assignOp, localSize, gridSize, shrMemSize);
      }
    } else {
      auto gridSize = localSize * nWG_row * nWG_col;
      auto assignOp = make_Ger_Row<false,true,true,false>(my_mA, _alpha, my_vx, my_vy, nWG_row, nWG_col, shrMemSize);
      if (shrMemSize == 0) {
        ex.execute(assignOp, localSize, gridSize);
      } else {
        ex.execute(assignOp, localSize, gridSize, shrMemSize);
      }
    }
    my_mA.printH("MAT");

//    ex.execute(assignOp, localSize, nWG_row*localSize*nWG_col, std::max(localSize,n_rows_WG));
  } else { // COLUMN ACCESS
    const auto localSize = (_localSize == 0)? 256: _localSize;
    const auto n_rows_WG = (_n_rows_WG == 0)? localSize: std::min(N,_n_rows_WG);
    const auto n_cols_WG = (_n_cols_WG == 0)? localSize: std::min(N,_n_cols_WG);;
    const auto shrMemSize = (_localSize == 0)? 2*localSize: _shrMemSize;

    const auto nWG_row = (N - 1) / n_rows_WG + 1;
    const auto nWG_col = (N - 1) / n_cols_WG + 1;

#ifdef VERBOSE
    std::cout << "SYR2_COLS: "
              << "N = " << N
              << " , localSize = "  << localSize
              << " , n_rows_WG = "  << n_rows_WG
              << " , n_cols_WG = "  << n_cols_WG
              << " , shrMemSize = " << shrMemSize
              << std::endl;
#endif  // VERBOSE

//    my_mA.printH("MAT");
//    my_vx.printH("VX ");
//    my_vy.printH("VY ");

    auto gridSize = localSize * nWG_row * nWG_col;
    if (triangOpr) {
      auto assignOp = make_Ger_Col<false,false,true,true>(my_mA, _alpha, my_vx, my_vy, nWG_row, nWG_col, shrMemSize);
      if (shrMemSize == 0) {
        ex.execute(assignOp, localSize, gridSize);
      } else {
        ex.execute(assignOp, localSize, gridSize, shrMemSize);
      }
    } else {
      auto assignOp = make_Ger_Col<false,true,true,false>(my_mA, _alpha, my_vx, my_vy, nWG_row, nWG_col, shrMemSize);
      if (shrMemSize == 0) {
        ex.execute(assignOp, localSize, gridSize);
      } else {
        ex.execute(assignOp, localSize, gridSize, shrMemSize);
      }
    }
//    my_mA.printH("MAT");
  }
}

/**************************************************/
/*************** PREVIOUS VERSIONS ****************/
/**************************************************/

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
    } else if (OPT == 20) {
      const auto interLoop=1;
//      std::cout << "ROWS_20"  << std::endl;
    #ifdef VERBOSE
      std::cout << "ROWS_20" << "M = " << _M
                << " N = " << _N << std::endl;
    #endif  // VERBOSE
      const auto localSize = 256;  // NOT FINAL VALUE
      const auto n_rows_WG = localSize;
//      const auto n_rows_WG = 1;
//      const auto n_cols_WG = localSize;
      const auto n_cols_WG = N;
//      auto n_cols_WG = 4;
//      auto shrMemSize = std::max(localSize,n_cols_WG);
//      auto shrMemSize = localSize*n_rows_WG;
      const auto shrMemSize = localSize;
//      const auto shrMemSize = 0;

//      auto nWG_col = (N + n_cols_WG - 1) / n_cols_WG;
      auto nWG_col = (N - 1) / n_cols_WG + 1;
//      auto nWG_row = (M + n_rows_WG - 1) / n_rows_WG;
      auto nWG_row = (M - 1) / n_rows_WG + 1;
//      std::cout << "nWG_row = " << nWG_row << " , "
//                << "nWG_col = " << nWG_col << std::endl;

      auto scratchSize = ((shrMemSize==0)?localSize:1)*nWG_col;
      ContainerT valT1(M * scratchSize);
      auto mat1 = matrix_view<T, ContainerT>(valT1, 0, M, scratchSize);

//      std::cout << "(A) scratchSize = "  << scratchSize
//                << " , shrMemSize = " << shrMemSize << std::endl;

      auto gridSize = localSize * nWG_row * nWG_col;
      auto gemvR = make_Gemv_Row<interLoop> (mat1, my_mA, my_vx, nWG_row, nWG_col, shrMemSize);
      if (shrMemSize == 0) {
        ex.execute(gemvR, localSize, gridSize);
      } else {
        ex.execute(gemvR, localSize, gridSize, shrMemSize);
      }
//      mat1.printH("MA");

//      std::cout << "(B) scratchSize = "  << scratchSize
//                << " , shrMemSize = " << shrMemSize << std::endl;

      auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
      auto addMOp = make_addSetColumns(mat1);
      auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, addMOp);
      auto addOp = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
      auto assignOp = make_op<Assign>(my_vy, addOp);
      ex.execute(assignOp, localSize);
//      std::cout << "(C) scratchSize = "  << scratchSize
//                << " , shrMemSize = " << shrMemSize << std::endl;

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
      std::cout << "COLS_17" << "M = " << _M
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
      std::cout << "COLS_18" << "M = " << _M
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
      std::cout << "COLS_19" << "M = " << _M
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
    } else if (OPT == 20) {
//      std::cout << "COLS_20"  << std::endl;
    #ifdef VERBOSE
      std::cout << "COLS_20" << "M = " << _M
                << " N = " << _N << std::endl;
    #endif  // VERBOSE
      auto localSize = 256;  // NOT FINAL VALUE
      auto n_rows_WG = localSize;
      auto n_cols_WG = localSize;
//      auto n_cols_WG = 4;
//      auto shrMemSize = std::max(localSize,n_cols_WG);
      auto shrMemSize = localSize;

//      auto nWG_col = (N + n_cols_WG - 1) / n_cols_WG;
      auto nWG_col = (N - 1) / n_cols_WG + 1;
//      auto nWG_row = (M + n_rows_WG - 1) / n_rows_WG;
      auto nWG_row = (M - 1) / n_rows_WG + 1;
//      std::cout << "nWG_row = " << nWG_row << " , "
//                << "nWG_col = " << nWG_col << std::endl;

      ContainerT valT1(M * nWG_col);
      auto mat1 = matrix_view<T, ContainerT>(valT1, 0, M, nWG_col);

      auto gridSize = localSize * nWG_row * nWG_col;
      auto gemvC = make_Gemv_Col (mat1, my_mA, my_vx, nWG_row, nWG_col, shrMemSize);
      ex.execute(gemvC, localSize, gridSize, shrMemSize);
//      ex.execute(gemvC, localSize, gridSize);
//      mat1.printH("MA");

      auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
      auto addMOp = make_addSetColumns(mat1);
      auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, addMOp);
      auto addOp = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
      auto assignOp = make_op<Assign>(my_vy, addOp);
      ex.execute(assignOp, localSize);
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
#ifdef VERBOSE
      std::cout << "GER_ROW_11 = " << std::endl;
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
} else if (OPT == 12) {
#ifdef VERBOSE
      std::cout << "GER_COL_12 = " << std::endl;
      std::cout << "alpha = " << _alpha << std::endl;
      my_mA.printH("MA");
      my_vx.printH("VX");
      my_vy.printH("VY");
#endif
      auto localSize = 64;  // NOT FINAL VALUE
      auto n_rows_WG = 1;
    //      auto n_rows_WG = localSize;
      auto n_cols_WG = localSize;
    //      auto n_cols_WG = N;
      auto nWG_col = (N + n_cols_WG - 1) / n_cols_WG;
      auto nWG_row = (M + n_rows_WG - 1) / n_rows_WG;
//      std::cout << "n_rows_WG = " << n_rows_WG
//                << " , nWG_row = " << nWG_row
//                << " , nWG_col = " << nWG_col
//                << std::endl;
      auto assignOp = make_Ger_MRow_NWG(my_mA, _alpha, my_vx, my_vy, n_rows_WG, nWG_col);
//      ex.execute(assignOp, localSize, nWG_row*localSize*nWG_col, n_rows_WG);
      ex.execute(assignOp, localSize, nWG_row*localSize*nWG_col, std::max(localSize,n_rows_WG));
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
//      ex.execute(assignOp);
      ex.execute(assignOp, 256);
#ifdef VERBOSE
      my_vy.printH("VY");
#endif
    } else if (OPT == 11) {
    #ifdef VERBOSE
      std::cout << "GER_COL_11 = " << std::endl;
      std::cout << "alpha = " << _alpha << std::endl;
      my_mA.printH("MA");
      my_vx.printH("VX");
      my_vy.printH("VY");
    #endif
      auto localSize = 256;  // NOT FINAL VALUE
      auto assignOp = make_Ger_1Row_1Thread(my_mA, _alpha, my_vx, my_vy);
      ex.execute(assignOp, localSize, M*localSize);
    #ifdef VERBOSE
      my_vy.printH("VY");
    #endif
    } else if (OPT == 12) {
    #ifdef VERBOSE
      std::cout << "GER_COL_12 = " << std::endl;
      std::cout << "alpha = " << _alpha << std::endl;
      my_mA.printH("MA");
      my_vx.printH("VX");
      my_vy.printH("VY");
    #endif
      auto localSize = 64;  // NOT FINAL VALUE
//      auto n_rows_WG = 1;
      auto n_rows_WG = localSize;
      auto n_cols_WG = localSize;
//      auto n_cols_WG = N;
      auto nWG_col = (N + n_cols_WG - 1) / n_cols_WG;
      auto nWG_row = (M + n_rows_WG - 1) / n_rows_WG;
//      std::cout << "n_rows_WG = " << n_rows_WG
//                << " , nWG_row = " << nWG_row
//                << " , nWG_col = " << nWG_col
//                << std::endl;
      auto assignOp = make_Ger_1Row_NWG_ShMem(my_mA, _alpha, my_vx, my_vy, n_cols_WG, nWG_row);
//      ex.execute(assignOp, localSize, nWG_row*localSize*nWG_col, n_rows_WG);
      ex.execute(assignOp, localSize, nWG_row*localSize*nWG_col, std::max(localSize,n_cols_WG));
    #ifdef VERBOSE
      my_vy.printH("VY");
    #endif
    }
  }
}

}  // namespace blas

#endif  // BLAS2_INTERFACE_SYCL_HPP
