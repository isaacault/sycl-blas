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
 *  @filename blas2_tree_executor.hpp
 *
 **************************************************************************/

#ifndef BLAS2_TREE_EXECUTOR_HPP
#define BLAS2_TREE_EXECUTOR_HPP

#include <stdexcept>

#include <CL/sycl.hpp>

#include <executors/executor_base.hpp>
#include <operations/blas1_trees.hpp>
#include <operations/blas2_trees.hpp>
#include <operations/blas3_trees.hpp>
#include <views/view_sycl.hpp>

namespace blas {

/** Evaluate.
 */
template <typename Tree>
struct Evaluate;

/**** ADD A SET OF COLUMNS, 1 ROW PER THREAD ****/
/*! Evaluate<AddSetColumns>.
 * @brief See Evaluate.
 */
template <typename RHS>
struct Evaluate<AddSetColumns<RHS>> {
 using value_type = typename RHS::value_type;
 using rhs_type = typename Evaluate<RHS>::type;
 using input_type = AddSetColumns<RHS>;
 using type = AddSetColumns<rhs_type>;

 static type convert_to(input_type v, cl::sycl::handler &h) {
   auto rhs = Evaluate<RHS>::convert_to(v.r, h);
   return type(rhs);
 }
};

/**** GEMV BY ROWS 1 ROW x M BLOCKS USING PROPERLY THE SHARED MEMORY ****/
/*! Evaluate<Gemv_Row>.
* @brief See Evaluate.
*/
template <unsigned int interLoop, bool Lower, bool Diag, bool Upper, bool Unit,
          typename LHS, typename RHS1, typename RHS2>
struct Evaluate<Gemv_Row<interLoop, Lower, Diag, Upper, Unit, LHS, RHS1, RHS2>> {
  using value_type = typename RHS2::value_type;
  using lhs_type = typename Evaluate<LHS>::type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using cont_type = typename Evaluate<LHS>::cont_type;
  using input_type = Gemv_Row<interLoop, Lower, Diag, Upper, Unit, LHS, RHS1, RHS2>;
  using type = Gemv_Row<interLoop, Lower, Diag, Upper, Unit,
                        lhs_type, rhs1_type, rhs2_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Evaluate<LHS>::convert_to(v.l, h);
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    return type(lhs, rhs1, rhs2, v.nWG_row, v.nWG_col, v.shrMemSize);
  }
};

/**** GEMV BY COLUMNS 1 ROW x M BLOCKS USING PROPERLY THE SHARED MEMORY ****/
/*! Evaluate<Gemv_Col>.
* @brief See Evaluate.
*/
//template <typename LHS, typename RHS1, typename RHS2>
template <bool Lower, bool Diag, bool Upper, bool Unit,
          class LHS, class RHS1, class RHS2>
struct Evaluate<Gemv_Col<Lower, Diag, Upper, Unit, LHS, RHS1, RHS2>> {
  using value_type = typename RHS2::value_type;
  using lhs_type = typename Evaluate<LHS>::type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using cont_type = typename Evaluate<LHS>::cont_type;
  using input_type = Gemv_Col<Lower, Diag, Upper, Unit, LHS, RHS1, RHS2>;
  using type = Gemv_Col<Lower, Diag, Upper, Unit, lhs_type, rhs1_type, rhs2_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Evaluate<LHS>::convert_to(v.l, h);
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    return type(lhs, rhs1, rhs2, v.nWG_row, v.nWG_col, v.shrMemSize);
  }
};

/**** GER BY ROWS M ROWS x N BLOCK USING PROPERLY THE SHARED MEMORY ****/
/*! Evaluate<Gemv_Col>.
* @brief See Evaluate.
*/
template <typename LHS, typename RHS1, typename RHS2>
struct Evaluate<Ger_Row<LHS, RHS1, RHS2>> {
  using value_type = typename RHS2::value_type;
  using lhs_type = typename Evaluate<LHS>::type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using cont_type = typename Evaluate<LHS>::cont_type;
  using input_type = Ger_Row<LHS, RHS1, RHS2>;
  using type = Ger_Row<lhs_type, rhs1_type, rhs2_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Evaluate<LHS>::convert_to(v.l, h);
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    return type(lhs, v.scl, rhs1, rhs2, v.nWG_row, v.nWG_col, v.shrMemSize);
  }
};

/**** GER BY ROWS M ROWS x N BLOCK USING PROPERLY THE SHARED MEMORY ****/
/*! Evaluate<Gemv_Col>.
* @brief See Evaluate.
*/
template <typename LHS, typename RHS1, typename RHS2>
struct Evaluate<Ger_Col<LHS, RHS1, RHS2>> {
  using value_type = typename RHS2::value_type;
  using lhs_type = typename Evaluate<LHS>::type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using cont_type = typename Evaluate<LHS>::cont_type;
  using input_type = Ger_Col<LHS, RHS1, RHS2>;
  using type = Ger_Col<lhs_type, rhs1_type, rhs2_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Evaluate<LHS>::convert_to(v.l, h);
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    return type(lhs, v.scl, rhs1, rhs2, v.nWG_row, v.nWG_col, v.shrMemSize);
  }
};

/**********************************************************************/
/************************* TEST VERSIONS ******************************/
/**********************************************************************/


/**** GEMV BY ROWS 1 ROW x 1 BLOCK ****/
/*! Evaluate<GemvR_1Row_1WG>.
 * @brief See Evaluate.
 */
template <unsigned int interLoop, typename LHS, typename RHS1, typename RHS2>
struct Evaluate<GemvR_1Row_1WG<interLoop, LHS, RHS1, RHS2>> {
 using value_type = typename RHS2::value_type;
 using lhs_type = typename Evaluate<LHS>::type;
 using rhs1_type = typename Evaluate<RHS1>::type;
 using rhs2_type = typename Evaluate<RHS2>::type;
 using cont_type = typename Evaluate<LHS>::cont_type;
 using input_type = GemvR_1Row_1WG<interLoop, LHS, RHS1, RHS2>;
 using type = GemvR_1Row_1WG<interLoop, lhs_type, rhs1_type, rhs2_type>;

 static type convert_to(input_type v, cl::sycl::handler &h) {
   auto lhs = Evaluate<LHS>::convert_to(v.l, h);
   auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
   auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
   return type(lhs, rhs1, rhs2);
 }
};

/**** GEMV BY ROWS 1 ROW x 1 BLOCK, WITHOUT LOCAL ADDITION ****/
/*! Evaluate<GemvR_1Row_1WG_NoRed>.
 * @brief See Evaluate.
 */
template <unsigned int interLoop, typename LHS, typename RHS1, typename RHS2>
struct Evaluate<GemvR_1Row_1WG_NoRed<interLoop, LHS, RHS1, RHS2>> {
 using value_type = typename RHS2::value_type;
 using lhs_type = typename Evaluate<LHS>::type;
 using rhs1_type = typename Evaluate<RHS1>::type;
 using rhs2_type = typename Evaluate<RHS2>::type;
 using cont_type = typename Evaluate<LHS>::cont_type;
 using input_type = GemvR_1Row_1WG_NoRed<interLoop, LHS, RHS1, RHS2>;
 using type = GemvR_1Row_1WG_NoRed<interLoop, lhs_type, rhs1_type, rhs2_type>;

 static type convert_to(input_type v, cl::sycl::handler &h) {
   auto lhs = Evaluate<LHS>::convert_to(v.l, h);
   auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
   auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
   return type(lhs, rhs1, rhs2);
 }
};

/**** GEMV BY ROWS 1 ROW x N BLOCK ****/
/*! Evaluate<GemvR_1Row_NWG>.
 * @brief See Evaluate.
 */
template <unsigned int interLoop, typename LHS, typename RHS1, typename RHS2>
struct Evaluate<GemvR_1Row_NWG<interLoop, LHS, RHS1, RHS2>> {
 using value_type = typename RHS2::value_type;
 using lhs_type = typename Evaluate<LHS>::type;
 using rhs1_type = typename Evaluate<RHS1>::type;
 using rhs2_type = typename Evaluate<RHS2>::type;
 using cont_type = typename Evaluate<LHS>::cont_type;
 using input_type = GemvR_1Row_NWG<interLoop, LHS, RHS1, RHS2>;
 using type = GemvR_1Row_NWG<interLoop, lhs_type, rhs1_type, rhs2_type>;

 static type convert_to(input_type v, cl::sycl::handler &h) {
   auto lhs = Evaluate<LHS>::convert_to(v.l, h);
   auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
   auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
   return type(lhs, rhs1, rhs2, v.nWG_col);
 }
};

/**** GEMV BY ROWS M ROW x N BLOCK ****/
/*! Evaluate<GemvR_MRow_NWG>.
 * @brief See Evaluate.
 */
template <unsigned int interLoop, typename LHS, typename RHS1, typename RHS2>
struct Evaluate<GemvR_MRow_NWG<interLoop, LHS, RHS1, RHS2>> {
 using value_type = typename RHS2::value_type;
 using lhs_type = typename Evaluate<LHS>::type;
 using rhs1_type = typename Evaluate<RHS1>::type;
 using rhs2_type = typename Evaluate<RHS2>::type;
 using cont_type = typename Evaluate<LHS>::cont_type;
 using input_type = GemvR_MRow_NWG<interLoop, LHS, RHS1, RHS2>;
 using type = GemvR_MRow_NWG<interLoop, lhs_type, rhs1_type, rhs2_type>;

 static type convert_to(input_type v, cl::sycl::handler &h) {
   auto lhs = Evaluate<LHS>::convert_to(v.l, h);
   auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
   auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
   return type(lhs, rhs1, rhs2, v.n_rows, v.nWG_col);
 }
};

/**** GEMV BY COLUMNS 1 ROW x 1 THREAD ****/
/*! Evaluate<GemvC_1Row_1Thread>.
 * @brief See Evaluate.
 */
template <typename RHS1, typename RHS2>
struct Evaluate<GemvC_1Row_1Thread<RHS1, RHS2>> {
  using value_type = typename RHS2::value_type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using input_type = GemvC_1Row_1Thread<RHS1, RHS2>;
  using type = GemvC_1Row_1Thread<rhs1_type, rhs2_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    return type(rhs1, rhs2);
  }
};


/**** GEMV BY COLUMNS 1 ROW x 1 THREAD USING SHARED MEMORY ****/
/*! Evaluate<GemvC_1Row_1Thread_ShMem>.
* @brief See Evaluate.
*/
template <typename LHS, typename RHS1, typename RHS2, typename RHS3>
struct Evaluate<GemvC_1Row_1Thread_ShMem<LHS, RHS1, RHS2, RHS3>> {
  using value_type = typename RHS2::value_type;
  using lhs_type = typename Evaluate<LHS>::type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using rhs3_type = typename Evaluate<RHS3>::type;
  using cont_type = typename Evaluate<LHS>::cont_type;
  using input_type = GemvC_1Row_1Thread_ShMem<LHS, RHS1, RHS2, RHS3>;
  using type = GemvC_1Row_1Thread_ShMem<lhs_type, rhs1_type, rhs2_type, rhs3_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Evaluate<LHS>::convert_to(v.l, h);
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    auto rhs3 = Evaluate<RHS3>::convert_to(v.r3, h);
    return type(lhs, v.scl, rhs1, rhs2, rhs3);
  }
};

/**** GEMV BY COLUMNS 1 ROW x 1 THREAD USING SHARED MEMORY MINIMIZING SYNCHRONIZATION ****/
/**** This option uses too much memory, failing when the local memory is completed ****/
/*! Evaluate<GemvC_1Row_1Thread_ShMem_Full>.
* @brief See Evaluate.
*/
template <typename LHS, typename RHS1, typename RHS2, typename RHS3>
struct Evaluate<GemvC_1Row_1Thread_ShMem_Full<LHS, RHS1, RHS2, RHS3>> {
  using value_type = typename RHS2::value_type;
  using lhs_type = typename Evaluate<LHS>::type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using rhs3_type = typename Evaluate<RHS3>::type;
  using cont_type = typename Evaluate<LHS>::cont_type;
  using input_type = GemvC_1Row_1Thread_ShMem_Full<LHS, RHS1, RHS2, RHS3>;
  using type = GemvC_1Row_1Thread_ShMem_Full<lhs_type, rhs1_type, rhs2_type, rhs3_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Evaluate<LHS>::convert_to(v.l, h);
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    auto rhs3 = Evaluate<RHS3>::convert_to(v.r3, h);
    return type(lhs, v.scl, rhs1, rhs2, rhs3);
  }
};

/**** GEMV BY COLUMNS 1 ROW x M THREADS ****/
/*! Evaluate<GemvC_1Row_MThreads>.
* @brief See Evaluate.
*/
template <typename LHS, typename RHS1, typename RHS2, typename RHS3>
struct Evaluate<GemvC_1Row_MThreads<LHS, RHS1, RHS2, RHS3>> {
  using value_type = typename RHS2::value_type;
  using lhs_type = typename Evaluate<LHS>::type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using rhs3_type = typename Evaluate<RHS3>::type;
  using cont_type = typename Evaluate<LHS>::cont_type;
  using input_type = GemvC_1Row_MThreads<LHS, RHS1, RHS2, RHS3>;
  using type = GemvC_1Row_MThreads<lhs_type, rhs1_type, rhs2_type, rhs3_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Evaluate<LHS>::convert_to(v.l, h);
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    auto rhs3 = Evaluate<RHS3>::convert_to(v.r3, h);
    return type(lhs, v.scl, rhs1, rhs2, rhs3, v.nThr);
  }
};

/**** GEMV BY COLUMNS 1 ROW x M THREADS USING SHARED MEMORY****/
/*! Evaluate<GemvC_1Row_MThreads_ShMem>.
* @brief See Evaluate.
*/
template <typename LHS, typename RHS1, typename RHS2, typename RHS3>
struct Evaluate<GemvC_1Row_MThreads_ShMem<LHS, RHS1, RHS2, RHS3>> {
  using value_type = typename RHS2::value_type;
  using lhs_type = typename Evaluate<LHS>::type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using rhs3_type = typename Evaluate<RHS3>::type;
  using cont_type = typename Evaluate<LHS>::cont_type;
  using input_type = GemvC_1Row_MThreads_ShMem<LHS, RHS1, RHS2, RHS3>;
  using type = GemvC_1Row_MThreads_ShMem<lhs_type, rhs1_type, rhs2_type, rhs3_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Evaluate<LHS>::convert_to(v.l, h);
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    auto rhs3 = Evaluate<RHS3>::convert_to(v.r3, h);
    return type(lhs, v.scl, rhs1, rhs2, rhs3, v.nThr);
  }
};

/**** GEMV BY COLUMNS 1 ROW x M THREADS USING SHARED MEMORY, WITHOUT LOCAL ADDITION ****/
/*! Evaluate<GemvC_1Row_MThreads_ShMem>.
* @brief See Evaluate.
*/
template <typename LHS, typename RHS1, typename RHS2>
struct Evaluate<GemvC_1Row_MThreads_ShMem_NoRed<LHS, RHS1, RHS2>> {
  using value_type = typename RHS2::value_type;
  using lhs_type = typename Evaluate<LHS>::type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using cont_type = typename Evaluate<LHS>::cont_type;
  using input_type = GemvC_1Row_MThreads_ShMem_NoRed<LHS, RHS1, RHS2>;
  using type = GemvC_1Row_MThreads_ShMem_NoRed<lhs_type, rhs1_type, rhs2_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Evaluate<LHS>::convert_to(v.l, h);
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    return type(lhs, rhs1, rhs2, v.nThr);
  }
};

/**** GEMV BY COLUMNS 1 ROW x M BLOCKS ****/
/*! Evaluate<GemvC_1Row_MBlocks>.
* @brief See Evaluate.
*/
template <typename LHS, typename RHS1, typename RHS2>
struct Evaluate<GemvC_1Row_MBlocks<LHS, RHS1, RHS2>> {
  using value_type = typename RHS2::value_type;
  using lhs_type = typename Evaluate<LHS>::type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using cont_type = typename Evaluate<LHS>::cont_type;
  using input_type = GemvC_1Row_MBlocks<LHS, RHS1, RHS2>;
  using type = GemvC_1Row_MBlocks<lhs_type, rhs1_type, rhs2_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Evaluate<LHS>::convert_to(v.l, h);
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    return type(lhs, rhs1, rhs2, v.nBlq);
  }
};

/**** GEMV BY COLUMNS 1 ROW x M BLOCKS USING SHARED MEMORY ****/
/*! Evaluate<GemvC_1Row_MBlocks_ShMem>.
* @brief See Evaluate.
*/
template <typename LHS, typename RHS1, typename RHS2>
struct Evaluate<GemvC_1Row_MBlocks_ShMem<LHS, RHS1, RHS2>> {
  using value_type = typename RHS2::value_type;
  using lhs_type = typename Evaluate<LHS>::type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using cont_type = typename Evaluate<LHS>::cont_type;
  using input_type = GemvC_1Row_MBlocks_ShMem<LHS, RHS1, RHS2>;
  using type = GemvC_1Row_MBlocks_ShMem<lhs_type, rhs1_type, rhs2_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Evaluate<LHS>::convert_to(v.l, h);
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    return type(lhs, rhs1, rhs2, v.nBlq);
  }
};

/**** GEMV BY COLUMNS 1 ROW x M BLOCKS USING SHARED MEMORY MINIMIZING SYNCHRONIZATION ****/
/*! Evaluate<GemvC_1Row_MBlocks_ShMem_Full>.
* @brief See Evaluate.
*/
template <typename LHS, typename RHS1, typename RHS2>
struct Evaluate<GemvC_1Row_MBlocks_ShMem_Full<LHS, RHS1, RHS2>> {
  using value_type = typename RHS2::value_type;
  using lhs_type = typename Evaluate<LHS>::type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using cont_type = typename Evaluate<LHS>::cont_type;
  using input_type = GemvC_1Row_MBlocks_ShMem_Full<LHS, RHS1, RHS2>;
  using type = GemvC_1Row_MBlocks_ShMem_Full<lhs_type, rhs1_type, rhs2_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Evaluate<LHS>::convert_to(v.l, h);
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    return type(lhs, rhs1, rhs2, v.nBlq);
  }
};

/**************************************************/
/*************** PREVIOUS VERSIONS ****************/
/**************************************************/


/*! Evaluate<Ger_1Row_1WG>
 * @brief See Evaluate.
 */
template <typename LHS, typename RHS1, typename RHS2>
struct Evaluate<Ger_1Row_1WG<LHS, RHS1, RHS2>> {
  using lhs_type  = typename Evaluate<LHS>::type;
  using value_type = typename LHS::value_type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using input_type = Ger_1Row_1WG<LHS, RHS1, RHS2>;
  using type = Ger_1Row_1WG<lhs_type, rhs1_type, rhs2_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs  = Evaluate<LHS>::convert_to(v.l , h);
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    return type(lhs, v.scl, rhs1, rhs2);
  }
};

/*! Evaluate<Ger_MRow_NWG>
 * @brief See Evaluate.
 */
template <typename LHS, typename RHS1, typename RHS2>
struct Evaluate<Ger_MRow_NWG<LHS, RHS1, RHS2>> {
  using lhs_type  = typename Evaluate<LHS>::type;
  using value_type = typename LHS::value_type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using input_type = Ger_MRow_NWG<LHS, RHS1, RHS2>;
  using type = Ger_MRow_NWG<lhs_type, rhs1_type, rhs2_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs  = Evaluate<LHS>::convert_to(v.l , h);
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    return type(lhs, v.scl, rhs1, rhs2, v.n_rows, v.nWG_col);
  }
};

/*! Evaluate<Ger_1Row_1Thread>
 * @brief See Evaluate.
 */
template <typename LHS, typename RHS1, typename RHS2>
struct Evaluate<Ger_1Row_1Thread<LHS, RHS1, RHS2>> {
 using lhs_type  = typename Evaluate<LHS>::type;
 using value_type = typename LHS::value_type;
 using rhs1_type = typename Evaluate<RHS1>::type;
 using rhs2_type = typename Evaluate<RHS2>::type;
 using input_type = Ger_1Row_1Thread<LHS, RHS1, RHS2>;
 using type = Ger_1Row_1Thread<lhs_type, rhs1_type, rhs2_type>;

 static type convert_to(input_type v, cl::sycl::handler &h) {
   auto lhs  = Evaluate<LHS>::convert_to(v.l , h);
   auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
   auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
   return type(lhs, v.scl, rhs1, rhs2);
 }
};

/*! Evaluate<Ger_MRow_NWG>
 * @brief See Evaluate.
 */
template <typename LHS, typename RHS1, typename RHS2>
struct Evaluate<Ger_1Row_NWG_ShMem<LHS, RHS1, RHS2>> {
  using lhs_type  = typename Evaluate<LHS>::type;
  using value_type = typename LHS::value_type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using input_type = Ger_1Row_NWG_ShMem<LHS, RHS1, RHS2>;
  using type = Ger_1Row_NWG_ShMem<lhs_type, rhs1_type, rhs2_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs  = Evaluate<LHS>::convert_to(v.l , h);
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    return type(lhs, v.scl, rhs1, rhs2, v.n_cols, v.nWG_row);
  }
};

/**********************************************************************/
/************************** OLD VERSIONS ******************************/
/**********************************************************************/

/**** CLASSICAL DOT PRODUCT GEMV ****/
/*! Evaluate<PrdRowMatVct>.
 * @brief See Evaluate.
 */
template <typename RHS1, typename RHS2>
struct Evaluate<PrdRowMatVct<RHS1, RHS2>> {
  using value_type = typename RHS2::value_type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using input_type = PrdRowMatVct<RHS1, RHS2>;
  using type = PrdRowMatVct<rhs1_type, rhs2_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    return type(rhs1, rhs2);
  }
};

/*! Evaluate<PrdRowMatVctMult>.
 * @brief See Evaluate.
 */
template <typename LHS, typename RHS1, typename RHS2, typename RHS3>
struct Evaluate<PrdRowMatVctMult<LHS, RHS1, RHS2, RHS3>> {
  using value_type = typename RHS2::value_type;
  using lhs_type = typename Evaluate<LHS>::type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using rhs3_type = typename Evaluate<RHS3>::type;
  using cont_type = typename Evaluate<LHS>::cont_type;
  using input_type = PrdRowMatVctMult<LHS, RHS1, RHS2, RHS3>;
  using type = PrdRowMatVctMult<lhs_type, rhs1_type, rhs2_type, rhs3_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Evaluate<LHS>::convert_to(v.l, h);
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    auto rhs3 = Evaluate<RHS3>::convert_to(v.r3, h);
    return type(lhs, v.scl, rhs1, rhs2, rhs3, v.nThr);
  }
};

/*! Evaluate<PrdRowMatVctMultShm>.
 * @brief See Evaluate.
 */
template <typename LHS, typename RHS1, typename RHS2>
struct Evaluate<PrdRowMatVctMultShm<LHS, RHS1, RHS2>> {
  using value_type = typename RHS2::value_type;
  using lhs_type = typename Evaluate<LHS>::type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using cont_type = typename Evaluate<LHS>::cont_type;
  using input_type = PrdRowMatVctMultShm<LHS, RHS1, RHS2>;
  using type = PrdRowMatVctMultShm<lhs_type, rhs1_type, rhs2_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Evaluate<LHS>::convert_to(v.l, h);
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    return type(lhs, rhs1, rhs2, v.nThr);
  }
};

/*! Evaluate<AddPrdRowMatVctMultShm>.
 * @brief See Evaluate.
 */
template <typename LHS, typename RHS1, typename RHS2>
struct Evaluate<AddPrdRowMatVctMultShm<LHS, RHS1, RHS2>> {
  using value_type = typename RHS2::value_type;
  using lhs_type = typename Evaluate<LHS>::type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using cont_type = typename Evaluate<LHS>::cont_type;
  using input_type = AddPrdRowMatVctMultShm<LHS, RHS1, RHS2>;
  using type = AddPrdRowMatVctMultShm<lhs_type, rhs1_type, rhs2_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Evaluate<LHS>::convert_to(v.l, h);
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    return type(lhs, v.scl, rhs1, rhs2);
  }
};

/*! Evaluate<RedRowMatVct>.
 * @brief See Evaluate.
 */
template <typename RHS1, typename RHS2>
struct Evaluate<RedRowMatVct<RHS1, RHS2>> {
  using value_type = typename RHS2::value_type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using input_type = RedRowMatVct<RHS1, RHS2>;
  using type = RedRowMatVct<rhs1_type, rhs2_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    return type(rhs1, rhs2, v.warpSize);
  }
};

/*! Evaluate<ModifRank1>
 * @brief See Evaluate.
 */
template <typename RHS1, typename RHS2, typename RHS3>
struct Evaluate<ModifRank1<RHS1, RHS2, RHS3>> {
  using value_type = typename RHS2::value_type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using rhs3_type = typename Evaluate<RHS3>::type;
  using input_type = ModifRank1<RHS1, RHS2, RHS3>;
  using type = ModifRank1<rhs1_type, rhs2_type, rhs3_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    auto rhs3 = Evaluate<RHS2>::convert_to(v.r3, h);
    return type(rhs1, rhs2, rhs3);
  }
};

}  // namespace blas

#endif  // BLAS2_TREE_EXECUTOR_HPP
