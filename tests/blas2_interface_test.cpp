#include <algorithm>
#include <cstdlib>
#include <interface/blas1_interface_sycl.hpp>
#include <interface/blas2_interface_sycl.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace cl::sycl;
using namespace blas;

#define DEF_SIZE_VECT 1200
#define ERROR_ALLOWED 1.0E-8
#define RANDOM_DATA 1
#define SHOW_VALUES   1

#define EXECUTED_ON_GPU 1
// #define EXECUTING_FLOAT

#define SHOW_TIMES 1  // If it exists, the code prints the execution time
                      // The ... should be changed by the corresponding routine
#define NUMBER_REPEATS 6  // Number of times the computations are made
// If it is greater than 1, the compile time is not considered


#ifdef EXECUTED_ON_GPU
  #define DEFAULT_ACCESS false
#else
  #define DEFAULT_ACCESS true
#endif

#define ROW_TEST "Tr"
#define COL_TEST "No"

// #define TRM_UNIT 1
#ifdef TRM_UNIT
  #define UNT_TEST "Un"
#else
  #define UNT_TEST "No"
#endif

// #define MATRIX_VECTOR_PRODUCT 1

#ifdef EXECUTING_FLOAT
  #define BASETYPE float
  #define CONS_ROW1 1.5F
  #define CONS_ROW2 2.0F
  #define CONS_COL1 0.5F
  #define CONS_COL2 2.5F
  #define CONS_SYM1 0.4F
  #define CONS_SYM2 3.5F
  #define CONS_GER  3.0F
  #define CONS_SYR  4.0F
  #define CONS_SYR2 4.5F
#else
  #define BASETYPE double
  #define CONS_ROW1 1.5
  #define CONS_ROW2 2.0
  #define CONS_COL1 0.5
  #define CONS_COL2 2.5
  #define CONS_SYM1 0.4
  #define CONS_SYM2 3.5
  #define CONS_GER  3.0
  #define CONS_SYR  4.0
  #define CONS_SYR2 4.5
#endif

// INITIAL MATRIZ VECTOR PRODUCT

template <typename ExecutorType, typename T, typename ContainerT>
void _gemv0(Executor<ExecutorType> ex, std::string _Trans, size_t _M, size_t _N,
            T _alpha, matrix_view<T, ContainerT> _mA, size_t _lda,
            vector_view<T, ContainerT> _vx, size_t _incx, T _beta,
            vector_view<T, ContainerT> _vy, size_t _incy) {
  if ((_Trans[0] != 'n') && (_Trans[0] != 'N') && (_Trans[0] != 't') &&
      (_Trans[0] != 'T') && (_Trans[0] != 'c') && (_Trans[0] != 'C'))
    std::cout << "Erroneous parameter" << std::endl;
  bool accessOpr = ((_Trans[0] == 'n') || (_Trans[0] == 'N'));
  size_t M = (accessOpr ? _M : _N);
  size_t N = (accessOpr ? _N : _M);
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
  if (my_mA.getAccess()) {
    auto scalOp = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
    auto assignOp = make_op<Assign>(my_vy, scalOp);
    ex.execute(assignOp);
    auto disp = my_mA.getDisp();
    for (size_t i = 0; i < M; i++) {
      auto my_row = vector_view<T, ContainerT>(my_mA.getData(), disp, 1, N);
      auto my_rs = vector_view<T, ContainerT>(my_vy, i + my_vy.getDisp(), 1, 1);
      auto scl = my_rs.eval(0);
      auto prdOp1 = make_op<BinaryOp, prdOp2_struct>(my_row, my_vx);
      auto localSize = 256;
      auto nWG = 128;
#ifdef SYCL_CODE
      auto assignOp1 =
          make_addAssignReduction(my_rs, prdOp1, localSize, localSize * nWG);
      ex.reduce(assignOp1);
#else   // SYCL_CODE
      ContainerT valT1(nWG);
      auto val1 = vector_view<T, ContainerT>(valT1, 0, 1, nWG);
      auto assignOp11 =
          make_addAssignReduction(val1, prdOp1, localSize, localSize * nWG);
      ex.execute(assignOp11);
      auto assignOp1 = make_addAssignReduction(my_rs, val1, localSize, nWG);
      ex.execute(assignOp1);
#endif  // SYCL_CODE
      auto prdOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, my_rs);
      auto addOp2 = make_op<ScalarOp, addOp2_struct>(scl, prdOp2);
      auto assignOp2 = make_op<Assign>(my_rs, addOp2);
      ex.execute(assignOp2);
      disp += _lda;
    }
  } else {
    auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
    auto assignOp1 = make_op<Assign>(my_vy, scalOp1);
    ex.execute(assignOp1);
    auto disp = my_mA.getDisp();
    for (size_t j = 0; j < N; j++) {
      auto my_col = vector_view<T, ContainerT>(my_mA.getData(), disp, 1, M);
      auto scalOp2 =
          make_op<ScalarOp, prdOp2_struct>(_alpha * my_vx.eval(j), my_col);
      auto addOp2 = make_op<BinaryOp, addOp2_struct>(my_vy, scalOp2);
      auto assignOp2 = make_op<Assign>(my_vy, addOp2);
      ex.execute(assignOp2);
      disp += _lda;
    }
  }
#ifdef VERBOSE
  my_vy.printH("VY");
#endif  // VERBOSE
}

// INITIAL RANK 1 MODIFICATION

template <typename ExecutorType, typename T, typename ContainerT>
void _ger0(Executor<ExecutorType> ex, size_t _M, size_t _N, T _alpha,
           vector_view<T, ContainerT> _vx, size_t _incx,
           vector_view<T, ContainerT> _vy, size_t _incy,
           matrix_view<T, ContainerT> _mA, size_t _lda) {
  bool accessOpr = true;
  size_t M = _M;
  size_t N = _N;
  auto my_mA =
      matrix_view<T, ContainerT>(_mA, _M, _N, accessOpr, _lda, _mA.getDisp());
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, M);
  auto my_vy = vector_view<T, ContainerT>(_vy, _vy.getDisp(), _incy, N);
#ifdef VERBOSE
  std::cout << "alpha = " << _alpha << std::endl;
  my_mA.printH("MA");
  my_vx.printH("VX");
  my_vy.printH("VY");
#endif  // VERBOSE
  if (my_mA.getAccess()) {
    auto disp = my_mA.getDisp();
    for (size_t i = 0; i < M; i++) {
      auto my_row = vector_view<T, ContainerT>(my_mA.getData(), disp, 1, N);
      auto scalOp2 =
          make_op<ScalarOp, prdOp2_struct>(_alpha * my_vx.eval(i), my_vy);
      auto addOp2 = make_op<BinaryOp, addOp2_struct>(my_row, scalOp2);
      auto assignOp2 = make_assign(my_row, addOp2);
      ex.execute(assignOp2);
      disp += _lda;
    }
  } else {
    auto disp = my_mA.getDisp();
    for (size_t j = 0; j < N; j++) {
      auto my_col = vector_view<T, ContainerT>(my_mA.getData(), disp, 1, M);
      auto scalOp2 =
          make_op<ScalarOp, prdOp2_struct>(_alpha * my_vy.eval(j), my_vx);
      auto addOp2 = make_op<BinaryOp, addOp2_struct>(my_col, scalOp2);
      auto assignOp2 = make_assign(my_col, addOp2);
      ex.execute(assignOp2);
      disp += _lda;
    }
  }
#ifdef VERBOSE
  my_vy.printH("VY");
#endif  // VERBOSE
}

// TESTING ROUTINE

size_t TestingBLAS2(bool accessDev, size_t dim, size_t divSz, size_t shftR,
                    size_t shftC) {
  // CREATING DATA
  size_t dimR = dim / divSz;
  size_t dimC = dim * divSz;
  size_t dimL = ((accessDev) ? dimC : dimR);
  std::vector<BASETYPE> vM0(dimR * dimC);
  std::vector<BASETYPE> vM1(dimR * dimC);
  std::vector<BASETYPE> vX0(dimC);
  std::vector<BASETYPE> vY0(dimR);
  std::vector<BASETYPE> vX1(dimC);
  std::vector<BASETYPE> vY1(dimR);
  std::vector<BASETYPE> vX2(dimC);
  std::vector<BASETYPE> vY2(dimR);
  std::vector<BASETYPE> vR(9);
  std::vector<BASETYPE> vS(13);
  std::vector<BASETYPE> vT(3);
#ifdef SHOW_TIMES
  std::chrono::time_point<std::chrono::steady_clock> t_start, t_stop;
  std::chrono::duration<BASETYPE> t1_gmvR, t1_gmvC, t1_ger;
  std::chrono::duration<BASETYPE> t2_gmvR, t2_gmvC, t2_ger;
  std::chrono::duration<BASETYPE> t3_gmvR, t3_gmvC, t3_ger;
  std::chrono::duration<BASETYPE> t4_gmvR, t4_gmvC, t4_ger;
  std::chrono::duration<BASETYPE> t5_gmvR, t5_gmvC, t5_ger;
  std::chrono::duration<BASETYPE> t6_gmvR, t6_gmvC, t6_ger;
  std::chrono::duration<BASETYPE> t7_gmvR, t7_gmvC, t7_ger;
  std::chrono::duration<BASETYPE> t8_gmvR, t8_gmvC, t8_ger;
  std::chrono::duration<BASETYPE> t9_gmvR, t9_gmvC, t9_ger;
  std::chrono::duration<BASETYPE> t10_gmvR, t10_gmvC, t10_ger;
  std::chrono::duration<BASETYPE> t11_gmvR, t11_gmvC, t11_ger;
  std::chrono::duration<BASETYPE> t12_gmvR, t12_gmvC, t12_ger;
  std::chrono::duration<BASETYPE> t13_gmvR, t13_gmvC, t13_ger;
  std::vector<std::chrono::duration<BASETYPE>> v1_gmvR(NUMBER_REPEATS), v1_gmvC(NUMBER_REPEATS), v1_ger(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v2_gmvR(NUMBER_REPEATS), v2_gmvC(NUMBER_REPEATS), v2_ger(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v3_gmvR(NUMBER_REPEATS), v3_gmvC(NUMBER_REPEATS), v3_ger(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v4_gmvR(NUMBER_REPEATS), v4_gmvC(NUMBER_REPEATS), v4_ger(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v5_gmvR(NUMBER_REPEATS), v5_gmvC(NUMBER_REPEATS), v5_ger(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v6_gmvR(NUMBER_REPEATS), v6_gmvC(NUMBER_REPEATS), v6_ger(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v7_gmvR(NUMBER_REPEATS), v7_gmvC(NUMBER_REPEATS), v7_ger(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v8_gmvR(NUMBER_REPEATS), v8_gmvC(NUMBER_REPEATS), v8_ger(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v9_gmvR(NUMBER_REPEATS), v9_gmvC(NUMBER_REPEATS), v9_ger(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v10_gmvR(NUMBER_REPEATS), v10_gmvC(NUMBER_REPEATS), v10_ger(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v11_gmvR(NUMBER_REPEATS), v11_gmvC(NUMBER_REPEATS), v11_ger(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v12_gmvR(NUMBER_REPEATS), v12_gmvC(NUMBER_REPEATS), v12_ger(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v13_gmvR(NUMBER_REPEATS), v13_gmvC(NUMBER_REPEATS), v13_ger(NUMBER_REPEATS);
#endif

  // INITIALIZING DATA
  size_t vSeed, gap;
  BASETYPE minV, maxV;
#ifdef RANDOM_DATA
  vSeed = 1;
  minV = -10.0;
  maxV = 10.0;
  gap = (size_t)(maxV - minV + 1);
  srand(vSeed);
#else   // RANDOM_DATA
  minV = 1.00;
  maxV = 1.00;
#endif  // RANDOM_DATA
  std::for_each(std::begin(vM1), std::end(vM1), [&](BASETYPE &elem) {
#ifdef RANDOM_DATA
    elem = minV + (BASETYPE)(rand() % gap);
#else   // RANDOM_DATA
    elem = minV; minV += maxV;
#endif  // RANDOM_DATA

  });

#ifdef RANDOM_DATA
  vSeed = 1;
  minV = -10.0;
  maxV = 10.0;
  gap = (size_t)(maxV - minV + 1);
  srand(vSeed);
#else   // RANDOM_DATA
  minV = 1.00;
  maxV = 1.00;
#endif  // RANDOM_DATA
  std::for_each(std::begin(vX0), std::end(vX0), [&](BASETYPE &elem) {
#ifdef RANDOM_DATA
    elem = minV + (BASETYPE)(rand() % gap);
#else   // RANDOM_DATA
    elem = minV; minV += maxV;
#endif  // RANDOM_DATA
  });

#ifdef RANDOM_DATA
  vSeed = 2;
  minV = -10.0;
  maxV = 10.0;
  gap = (size_t)(maxV - minV + 1);
  srand(vSeed);
#else   // RANDOM_DATA
  minV = 1.00;
  maxV = 1.00;
#endif  // RANDOM_DATA
  std::for_each(std::begin(vX1), std::end(vX1), [&](BASETYPE &elem) {
#ifdef RANDOM_DATA
    elem = minV + (BASETYPE)(rand() % gap);
#else   // RANDOM_DATA
    elem = minV; minV += maxV;
#endif  // RANDOM_DATA
  });

#ifdef RANDOM_DATA
  vSeed = 3;
  minV = -10.0;
  maxV = 10.0;
  gap = (size_t)(maxV - minV + 1);
  srand(vSeed);
#else   // RANDOM_DATA
  minV = 1.00;
  maxV = 1.00;
#endif  // RANDOM_DATA
  std::for_each(std::begin(vY0), std::end(vY0), [&](BASETYPE &elem) {
#ifdef RANDOM_DATA
    elem = minV + (BASETYPE)(rand() % gap);
#else   // RANDOM_DATA
    elem = minV; minV += maxV;
#endif  // RANDOM_DATA
  });

#ifdef RANDOM_DATA
  vSeed = 4;
  minV = -10.0;
  maxV = 10.0;
  gap = (size_t)(maxV - minV + 1);
  srand(vSeed);
#else   // RANDOM_DATA
  minV = 1.00;
  maxV = 1.00;
#endif  // RANDOM_DATA
  std::for_each(std::begin(vY1), std::end(vY1), [&](BASETYPE &elem) {
#ifdef RANDOM_DATA
    elem = minV + (BASETYPE)(rand() % gap);
#else   // RANDOM_DATA
    elem = minV; minV += maxV;
#endif  // RANDOM_DATA
  });

  for (int i=0; i<3; i++) {
    vR[i] = 0.0;
    vS[i] = 0.0;
    vT[i] = 0.0;
  }
/*
  // CREATING HOST STRUCTURES
  matrix_view<BASETYPE, std::vector<BASETYPE>> v_M0(vM0, accessDev, dimR, dimC, true,
                                                dimL, 0);
  matrix_view<BASETYPE, std::vector<BASETYPE>> v_M1(vM1, accessDev, dimR, dimC, true,
                                                dimL, 0);
  vector_view<BASETYPE, std::vector<BASETYPE>> v_X0(vX0, 0, 1, dimC);
  vector_view<BASETYPE, std::vector<BASETYPE>> v_Y0(vY0, 0, 1, dimR);
  vector_view<BASETYPE, std::vector<BASETYPE>> v_R(vR, 0, 1, 1);
*/
  // COMPUTING THE RESULTS
  size_t returnVal = 0;
  BASETYPE res;

  BASETYPE addY = 0.0, auxY;
  for (size_t i = shftR; i < dimR; i++) {
//    vY2[i - shftR] = 1.5 * vY[i - shftR];
    auxY = CONS_ROW1 * vY1[i - shftR];
//    printf ("(AB) %f = 1.5 * %f\n", auxY, vY2[i - shftC]);
    for (size_t j = shftC; j < dimC; j++) {
      if (accessDev) {
//        vY2[i - shftR] += 2.0 * vM[dimC * i + j] * vY[j - shftC];
//        printf ("(A) %f += 2.0 * %f * %f\n", auxY, vM[dimC * i + j], vY[j - shftC]);
        auxY += CONS_ROW2 * vM1[dimC * i + j] * vY0[j - shftC];
      } else {
//        vY2[i - shftR] += 2.0 * vM[dimR * j + i] * vY[j - shftC];
//        printf ("(B) %f += 2.0 * %f * %f\n", auxY, vM[dimR * j + i], vY[j - shftC]);
        auxY += CONS_ROW2 * vM1[dimR * j + i] * vY0[j - shftC];
      }
    }
//    addY += vY2[i - shftR];
    addY += auxY;
//    printf("VY2(%lu) = %f\n", i, auxY);
  }
  for (size_t i = dimR - shftR; i < dimR; i++) {
#ifdef VERBOSE
    std::cout << "+" << vY[i] << std::endl;
#endif  // VERBOSE
    addY += vY1[i];
  }

  BASETYPE addX = 0.0, auxX;
  for (size_t j = shftC; j < dimC; j++) {
//    vX2[j - shftC] = 0.5 * vX2[j - shftC];
    auxX = CONS_COL1 * vX1[j - shftC];
    for (size_t i = shftR; i < dimR; i++) {
      if (accessDev) {
//        vX2[j - shftC] += 2.5 * vM[dimC * i + j] * vX[i - shftR];
        auxX += CONS_COL2 * vM1[dimC * i + j] * vX0[i - shftR];
      } else {
//        vX2[j - shftC] += 2.5 * vM[dimR * j + i] * vX[i - shftR];
        auxX += CONS_COL2 * vM1[dimR * j + i] * vX0[i - shftR];
      }
    }
//    addX += vX2[j - shftC];
//    printf("VX2(%lu) = %f\n", j, auxX);
    addX += auxX;
  }
  for (size_t j = dimC - shftC; j < dimC; j++) {
    addX += vX1[j];
  }

  BASETYPE addRng1 = 0.0;
  for (size_t i = 0; i < dimR; i++) {
    for (size_t j = 0; j < dimC; j++) {
      BASETYPE aux = 0.0, auxU = 0.0, auxL = 0.0;
//      addRng1 += (accessDev) ? vM1[dimC * i + j] : vM1[dimR * j + i];
      aux  += (accessDev) ? vM1[dimC * i + j] : vM1[dimR * j + i];
      if ((i >= shftR) && (j >= shftC)) {
//        addRng1 += (3.0 * vY0[i - shftR] * vX0[j - shftC]);
        aux += (CONS_GER * vY0[i - shftR] * vX0[j - shftC]);
      }
      addRng1  += aux ;
    }
  }

  // CREATING THE SYCL QUEUE AND EXECUTOR
  // cl::sycl::cpu_selector s; NOOOOO
#ifdef EXECUTED_ON_GPU
  cl::sycl::gpu_selector   s;
#else
  cl::sycl::intel_selector s;
#endif
  cl::sycl::queue q([=](cl::sycl::exception_list eL) {
    try {
      for (auto &e : eL) {
        std::rethrow_exception(e);
      }
    } catch (cl::sycl::exception &e) {
      std::cout << " E " << e.what() << std::endl;
    } catch (...) {
      std::cout << " An exception " << std::endl;
    }
  });
  Executor<SYCL> ex(q);

  {
    // CREATION OF THE BUFFERS
    buffer<BASETYPE, 1> bM0(vM0.data(), range<1>{vM0.size()});
    buffer<BASETYPE, 1> bM1(vM1.data(), range<1>{vM1.size()});
    buffer<BASETYPE, 1> bX0(vX0.data(), range<1>{vX0.size()});
    buffer<BASETYPE, 1> bY0(vY0.data(), range<1>{vY0.size()});
    buffer<BASETYPE, 1> bX1(vX1.data(), range<1>{vX1.size()});
    buffer<BASETYPE, 1> bY1(vY1.data(), range<1>{vY1.size()});
    buffer<BASETYPE, 1> bX2(vX2.data(), range<1>{vX2.size()});
    buffer<BASETYPE, 1> bY2(vY2.data(), range<1>{vY2.size()});
    buffer<BASETYPE, 1> bR(vR.data(), range<1>{vR.size()});
    buffer<BASETYPE, 1> bS(vS.data(), range<1>{vS.size()});
    buffer<BASETYPE, 1> bT(vT.data(), range<1>{vT.size()});

    // BUILDING A SYCL VIEW OF THE BUFFERS
    BufferMatrixView<BASETYPE> bmM0(bM0, accessDev, dimR, dimC);
    BufferMatrixView<BASETYPE> bmM1(bM1, accessDev, dimR, dimC);
    BufferVectorView<BASETYPE> bvV0(bM0);
    BufferVectorView<BASETYPE> bvX0(bX0);
    BufferVectorView<BASETYPE> bvY0(bY0);
    BufferVectorView<BASETYPE> bvX1(bX1);
    BufferVectorView<BASETYPE> bvY1(bY1);
    BufferVectorView<BASETYPE> bvX2(bX2);
    BufferVectorView<BASETYPE> bvY2(bY2);
    BufferVectorView<BASETYPE> bvR1(bR,0);
    BufferVectorView<BASETYPE> bvR2(bR,1);
    BufferVectorView<BASETYPE> bvR3(bR,2);
    BufferVectorView<BASETYPE> bvR4(bR,3);
    BufferVectorView<BASETYPE> bvR5(bR,4);
    BufferVectorView<BASETYPE> bvR6(bR,5);
    BufferVectorView<BASETYPE> bvS1(bS,0);
    BufferVectorView<BASETYPE> bvS2(bS,1);
    BufferVectorView<BASETYPE> bvS3(bS,2);
    BufferVectorView<BASETYPE> bvS4(bS,3);
    BufferVectorView<BASETYPE> bvS5(bS,4);
    BufferVectorView<BASETYPE> bvS6(bS,5);
    BufferVectorView<BASETYPE> bvS7(bS,6);
    BufferVectorView<BASETYPE> bvS8(bS,7);
    BufferVectorView<BASETYPE> bvS9(bS,8);
    BufferVectorView<BASETYPE> bvS10(bS,9);
    BufferVectorView<BASETYPE> bvS11(bS,10);
    BufferVectorView<BASETYPE> bvS12(bS,11);
    BufferVectorView<BASETYPE> bvS13(bS,12);
    BufferVectorView<BASETYPE> bvT1(bT,0);
    BufferVectorView<BASETYPE> bvT2(bT,1);
    BufferVectorView<BASETYPE> bvT3(bT,2);

    // EXECUTION OF THE ROUTINES
    for (int i = 0; i < NUMBER_REPEATS; i++) {
      /*****************************************/
      auto assign_M0 = make_op<Assign>(bmM0, bmM1);
      ex.execute(assign_M0); q.wait_and_throw();

      /*****************************************/
/* */
      auto assign_X2_1 = make_op<Assign>(bvX2, bvX1);
      ex.execute(assign_X2_1); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _gemv<1, SYCL>(ex, ROW_TEST, dimR - shftC, dimR - shftR, CONS_COL2, bmM0(shftR, shftC),
                  dimL, bvX0, 1, CONS_COL1, bvX2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t1_gmvR = t_stop - t_start;
      } else if (i > 0) {
        t1_gmvR += t_stop - t_start;
      } else {
        t1_gmvR = t_start - t_start;
      }
      v1_gmvR[i] = t_stop - t_start;
  #endif
      auto reducOpX_1 = make_addAssignReduction(bvR1, bvX2, 256, 256);
      ex.reduce(reducOpX_1); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_X2_11 = make_op<Assign>(bvX2, bvX1);
      ex.execute(assign_X2_11); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _gemv<11, SYCL>(ex, ROW_TEST, dimR - shftC, dimR - shftR, CONS_COL2, bmM0(shftR, shftC),
                  dimL, bvX0, 1, CONS_COL1, bvX2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t2_gmvR = t_stop - t_start;
      } else if (i > 0) {
        t2_gmvR += t_stop - t_start;
      } else {
        t2_gmvR = t_start - t_start;
      }
      v2_gmvR[i] = t_stop - t_start;
  #endif
      auto reducOpX_11 = make_addAssignReduction(bvR2, bvX2, 256, 256);
      ex.reduce(reducOpX_11); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_X2_12 = make_op<Assign>(bvX2, bvX1);
      ex.execute(assign_X2_12); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _gemv<12, SYCL>(ex, ROW_TEST, dimR - shftC, dimR - shftR, CONS_COL2, bmM0(shftR, shftC),
                  dimL, bvX0, 1, CONS_COL1, bvX2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t3_gmvR = t_stop - t_start;
      } else if (i > 0) {
        t3_gmvR += t_stop - t_start;
      } else {
        t3_gmvR = t_start - t_start;
      }
      v3_gmvR[i] = t_stop - t_start;
  #endif
      auto reducOpX_12 = make_addAssignReduction(bvR3, bvX2, 256, 256);
      ex.reduce(reducOpX_12); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_X2_13 = make_op<Assign>(bvX2, bvX1);
      ex.execute(assign_X2_13); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _gemv<13, SYCL>(ex, ROW_TEST, dimR - shftC, dimR - shftR, CONS_COL2, bmM0(shftR, shftC),
                  dimL, bvX0, 1, CONS_COL1, bvX2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t4_gmvR = t_stop - t_start;
      } else if (i > 0) {
        t4_gmvR += t_stop - t_start;
      } else {
        t4_gmvR = t_start - t_start;
      }
      v4_gmvR[i] = t_stop - t_start;
  #endif
      auto reducOpX_13 = make_addAssignReduction(bvR4, bvX2, 256, 256);
      ex.reduce(reducOpX_13); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_X2_14 = make_op<Assign>(bvX2, bvX1);
      ex.execute(assign_X2_14); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _gemv<14, SYCL>(ex, ROW_TEST, dimR - shftC, dimR - shftR, CONS_COL2, bmM0(shftR, shftC),
                  dimL, bvX0, 1, CONS_COL1, bvX2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t5_gmvR = t_stop - t_start;
      } else if (i > 0) {
        t5_gmvR += t_stop - t_start;
      } else {
        t5_gmvR = t_start - t_start;
      }
      v5_gmvR[i] = t_stop - t_start;
  #endif
      auto reducOpX_14 = make_addAssignReduction(bvR5, bvX2, 256, 256);
      ex.reduce(reducOpX_14); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_X2_20 = make_op<Assign>(bvX2, bvX1);
      ex.execute(assign_X2_20); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _gemv<20, SYCL>(ex, ROW_TEST, dimR - shftC, dimR - shftR, CONS_COL2, bmM0(shftR, shftC),
                  dimL, bvX0, 1, CONS_COL1, bvX2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t6_gmvR = t_stop - t_start;
      } else if (i > 0) {
        t6_gmvR += t_stop - t_start;
      } else {
        t6_gmvR = t_start - t_start;
      }
      v6_gmvR[i] = t_stop - t_start;
  #endif
      auto reducOpX_20 = make_addAssignReduction(bvR6, bvX2, 256, 256);
      ex.reduce(reducOpX_20); q.wait_and_throw();
/* */
      /*****************************************/
//      auto assign1_Seg = make_op<Assign>(bvX2, bvY2);
//      ex.execute(assign1_Seg); q.wait_and_throw();
/* */
      auto assign_Y2_1 = make_op<Assign>(bvY2, bvY1);
      ex.execute(assign_Y2_1); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _gemv<1, SYCL>(ex, COL_TEST, dimR - shftR, dimC - shftC, CONS_ROW2, bmM0(shftR, shftC),
                  dimL, bvY0, 1, CONS_ROW1, bvY2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t1_gmvC = t_stop - t_start;
      } else if (i > 0) {
        t1_gmvC += t_stop - t_start;
      } else {
        t1_gmvC = t_start - t_start;
      }
      v1_gmvC[i] = t_stop - t_start;
  #endif
      auto reducOpY_1 = make_addAssignReduction(bvS1, bvY2, 256, 256);
      ex.reduce(reducOpY_1); q.wait_and_throw();
/* */
//      auto assign1_Rec = make_op<Assign>(bvY2, bvX2);
//      ex.execute(assign1_Rec); q.wait_and_throw();
      /*****************************************/
/* */
      auto assign_Y2_2 = make_op<Assign>(bvY2, bvY1);
      ex.execute(assign_Y2_2); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _gemv<2, SYCL>(ex, COL_TEST, dimR - shftR, dimC - shftC, CONS_ROW2, bmM0(shftR, shftC),
                  dimL, bvY0, 1, CONS_ROW1, bvY2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t2_gmvC = t_stop - t_start;
      } else if (i > 0) {
        t2_gmvC += t_stop - t_start;
      } else {
        t2_gmvC = t_start - t_start;
      }
      v2_gmvC[i] = t_stop - t_start;
  #endif
      auto reducOpY_2 = make_addAssignReduction(bvS2, bvY2, 256, 256);
      ex.reduce(reducOpY_2); q.wait_and_throw();
/* */
//      auto assign2_Rec = make_op<Assign>(bvY2, bvX2);
//      ex.execute(assign2_Rec); q.wait_and_throw();
      /*****************************************/
/* */
      auto assign_Y2_3 = make_op<Assign>(bvY2, bvY1);
      ex.execute(assign_Y2_3); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _gemv<3, SYCL>(ex, COL_TEST, dimR - shftR, dimC - shftC, CONS_ROW2, bmM0(shftR, shftC),
                  dimL, bvY0, 1, CONS_ROW1, bvY2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t3_gmvC = t_stop - t_start;
      } else if (i > 0) {
        t3_gmvC += t_stop - t_start;
      } else {
        t3_gmvC = t_start - t_start;
      }
      v3_gmvC[i] = t_stop - t_start;
  #endif
      auto reducOpY_3 = make_addAssignReduction(bvS3, bvY2, 256, 256);
      ex.reduce(reducOpY_3); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_Y2_11 = make_op<Assign>(bvY2, bvY1);
      ex.execute(assign_Y2_11); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _gemv<11, SYCL>(ex, COL_TEST, dimR - shftR, dimC - shftC, CONS_ROW2, bmM0(shftR, shftC),
                  dimL, bvY0, 1, CONS_ROW1, bvY2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t4_gmvC = t_stop - t_start;
      } else if (i > 0) {
        t4_gmvC += t_stop - t_start;
      } else {
        t4_gmvC = t_start - t_start;
      }
      v4_gmvC[i] = t_stop - t_start;
  #endif
      auto reducOpY_11= make_addAssignReduction(bvS4, bvY2, 256, 256);
      ex.reduce(reducOpY_11); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_Y2_12 = make_op<Assign>(bvY2, bvY1);
      ex.execute(assign_Y2_12); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _gemv<12, SYCL>(ex, COL_TEST, dimR - shftR, dimC - shftC, CONS_ROW2, bmM0(shftR, shftC),
                  dimL, bvY0, 1, CONS_ROW1, bvY2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t5_gmvC = t_stop - t_start;
      } else if (i > 0) {
        t5_gmvC += t_stop - t_start;
      } else {
        t5_gmvC = t_start - t_start;
      }
      v5_gmvC[i] = t_stop - t_start;
  #endif
      auto reducOpY_12 = make_addAssignReduction(bvS5, bvY2, 256, 256);
      ex.reduce(reducOpY_12); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_Y2_13 = make_op<Assign>(bvY2, bvY1);
      ex.execute(assign_Y2_13); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _gemv<13, SYCL>(ex, COL_TEST, dimR - shftR, dimC - shftC, CONS_ROW2, bmM0(shftR, shftC),
                  dimL, bvY0, 1, CONS_ROW1, bvY2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t6_gmvC = t_stop - t_start;
      } else if (i > 0) {
        t6_gmvC += t_stop - t_start;
      } else {
        t6_gmvC = t_start - t_start;
      }
      v6_gmvC[i] = t_stop - t_start;
  #endif
      auto reducOpY_13 = make_addAssignReduction(bvS6, bvY2, 256, 256);
      ex.reduce(reducOpY_13); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_Y2_14 = make_op<Assign>(bvY2, bvY1);
      ex.execute(assign_Y2_14); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _gemv<14, SYCL>(ex, COL_TEST, dimR - shftR, dimC - shftC, CONS_ROW2, bmM0(shftR, shftC),
                  dimL, bvY0, 1, CONS_ROW1, bvY2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t7_gmvC = t_stop - t_start;
      } else if (i > 0) {
        t7_gmvC += t_stop - t_start;
      } else {
        t7_gmvC = t_start - t_start;
      }
      v7_gmvC[i] = t_stop - t_start;
  #endif
      auto reducOpY_14 = make_addAssignReduction(bvS7, bvY2, 256, 256);
      ex.reduce(reducOpY_14); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_Y2_15 = make_op<Assign>(bvY2, bvY1);
      ex.execute(assign_Y2_15); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _gemv<15, SYCL>(ex, COL_TEST, dimR - shftR, dimC - shftC, CONS_ROW2, bmM0(shftR, shftC),
                  dimL, bvY0, 1, CONS_ROW1, bvY2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t8_gmvC = t_stop - t_start;
      } else if (i > 0) {
        t8_gmvC += t_stop - t_start;
      } else {
        t8_gmvC = t_start - t_start;
      }
      v8_gmvC[i] = t_stop - t_start;
  #endif
      auto reducOpY_15 = make_addAssignReduction(bvS8, bvY2, 256, 256);
      ex.reduce(reducOpY_15); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_Y2_16 = make_op<Assign>(bvY2, bvY1);
      ex.execute(assign_Y2_16); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _gemv<16, SYCL>(ex, COL_TEST, dimR - shftR, dimC - shftC, CONS_ROW2, bmM0(shftR, shftC),
                  dimL, bvY0, 1, CONS_ROW1, bvY2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t9_gmvC = t_stop - t_start;
      } else if (i > 0) {
        t9_gmvC += t_stop - t_start;
      } else {
        t9_gmvC = t_start - t_start;
      }
      v9_gmvC[i] = t_stop - t_start;
  #endif
      auto reducOpY_16 = make_addAssignReduction(bvS9, bvY2, 256, 256);
      ex.reduce(reducOpY_16); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_Y2_17 = make_op<Assign>(bvY2, bvY1);
      ex.execute(assign_Y2_17); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _gemv<17, SYCL>(ex, COL_TEST, dimR - shftR, dimC - shftC, CONS_ROW2, bmM0(shftR, shftC),
                  dimL, bvY0, 1, CONS_ROW1, bvY2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t10_gmvC = t_stop - t_start;
      } else if (i > 0) {
        t10_gmvC += t_stop - t_start;
      } else {
        t10_gmvC = t_start - t_start;
      }
      v10_gmvC[i] = t_stop - t_start;
  #endif
      auto reducOpY_17 = make_addAssignReduction(bvS10, bvY2, 256, 256);
      ex.reduce(reducOpY_17); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_Y2_18 = make_op<Assign>(bvY2, bvY1);
      ex.execute(assign_Y2_18); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _gemv<18, SYCL>(ex, COL_TEST, dimR - shftR, dimC - shftC, CONS_ROW2, bmM0(shftR, shftC),
                  dimL, bvY0, 1, CONS_ROW1, bvY2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t11_gmvC = t_stop - t_start;
      } else if (i > 0) {
        t11_gmvC += t_stop - t_start;
      } else {
        t11_gmvC = t_start - t_start;
      }
      v11_gmvC[i] = t_stop - t_start;
  #endif
      auto reducOpY_18 = make_addAssignReduction(bvS11, bvY2, 256, 256);
      ex.reduce(reducOpY_18); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_Y2_19 = make_op<Assign>(bvY2, bvY1);
      ex.execute(assign_Y2_19); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _gemv<19, SYCL>(ex, COL_TEST, dimR - shftR, dimC - shftC, CONS_ROW2, bmM0(shftR, shftC),
                  dimL, bvY0, 1, CONS_ROW1, bvY2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t12_gmvC = t_stop - t_start;
      } else if (i > 0) {
        t12_gmvC += t_stop - t_start;
      } else {
        t12_gmvC = t_start - t_start;
      }
      v12_gmvC[i] = t_stop - t_start;
  #endif
      auto reducOpY_19 = make_addAssignReduction(bvS12, bvY2, 256, 256);
      ex.reduce(reducOpY_19); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_Y2_20 = make_op<Assign>(bvY2, bvY1);
      ex.execute(assign_Y2_20); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _gemv<20, SYCL>(ex, COL_TEST, dimR - shftR, dimC - shftC, CONS_ROW2, bmM0(shftR, shftC),
                  dimL, bvY0, 1, CONS_ROW1, bvY2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t13_gmvC = t_stop - t_start;
      } else if (i > 0) {
        t13_gmvC += t_stop - t_start;
      } else {
        t13_gmvC = t_start - t_start;
      }
      v13_gmvC[i] = t_stop - t_start;
  #endif
      auto reducOpY_20 = make_addAssignReduction(bvS13, bvY2, 256, 256);
      ex.reduce(reducOpY_20); q.wait_and_throw();
/* */
      /*****************************************/
/* */
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _ger<1, SYCL>(ex, dimR - shftR, dimC - shftC, CONS_GER, bvY0, 1, bvX0, 1,
                 bmM0(shftR, shftC), dimL);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t1_ger = t_stop - t_start;
      } else if (i > 0) {
        t1_ger += t_stop - t_start;
      } else {
        t1_ger = t_start - t_start;
      }
      v1_ger[i] = t_stop - t_start;
  #endif
      auto reducOpV_1 = make_addAssignReduction(bvT1, bvV0, 256, 256);
      ex.reduce(reducOpV_1); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_M0_11 = make_op<Assign>(bmM0, bmM1);
      ex.execute(assign_M0_11); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _ger<11, SYCL>(ex, dimR - shftR, dimC - shftC, CONS_GER, bvY0, 1, bvX0, 1,
                 bmM0(shftR, shftC), dimL);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t2_ger = t_stop - t_start;
      } else if (i > 0) {
        t2_ger += t_stop - t_start;
      } else {
        t2_ger = t_start - t_start;
      }
      v2_ger[i] = t_stop - t_start;
  #endif
      auto reducOpV_11 = make_addAssignReduction(bvT2, bvV0, 256, 256);
      ex.reduce(reducOpV_11); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_M0_12 = make_op<Assign>(bmM0, bmM1);
      ex.execute(assign_M0_12); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _ger<12, SYCL>(ex, dimR - shftR, dimC - shftC, CONS_GER, bvY0, 1, bvX0, 1,
                 bmM0(shftR, shftC), dimL);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t3_ger = t_stop - t_start;
      } else if (i > 0) {
        t3_ger += t_stop - t_start;
      } else {
        t3_ger = t_start - t_start;
      }
      v3_ger[i] = t_stop - t_start;
  #endif
      auto reducOpV_12 = make_addAssignReduction(bvT3, bvV0, 256, 256);
      ex.reduce(reducOpV_12); q.wait_and_throw();
/* */
    }
  }

#ifdef SHOW_TIMES
    int div = (NUMBER_REPEATS == 1)? 1: (NUMBER_REPEATS-1);
//    int div = 1;
    // COMPUTATIONAL TIMES
    std::cout << "t_gemvR , " << t1_gmvR.count()/div
              << ", " << t2_gmvR.count()/div
              << ", " << t3_gmvR.count()/div
              << ", " << t4_gmvR.count()/div
              << ", " << t5_gmvR.count()/div
              << ", " << t6_gmvR.count()/div
              << std::endl;
    std::cout << "t_gemvC , " << t1_gmvC.count()/div
              << ", " << t2_gmvC.count()/div
              << ", " << t3_gmvC.count()/div
              << ", " << t4_gmvC.count()/div
              << ", " << t5_gmvC.count()/div
              << ", " << t6_gmvC.count()/div
              << ", " << t7_gmvC.count()/div
              << ", " << t8_gmvC.count()/div
              << ", " << t9_gmvC.count()/div
              << ", " << t10_gmvC.count()/div
              << ", " << t11_gmvC.count()/div
              << ", " << t12_gmvC.count()/div
              << ", " << t13_gmvC.count()/div
              << std::endl;
    std::cout << "t_ger   , " << t1_ger.count()/div
              <<  ", "        << t2_ger.count()/div
              <<  ", "        << t3_ger.count()/div
              << std::endl;
    std::sort (v1_gmvR.begin()+1, v1_gmvR.end());
    std::sort (v2_gmvR.begin()+1, v2_gmvR.end());
    std::sort (v3_gmvR.begin()+1, v3_gmvR.end());
    std::sort (v4_gmvR.begin()+1, v4_gmvR.end());
    std::sort (v5_gmvR.begin()+1, v5_gmvR.end());
    std::sort (v6_gmvR.begin()+1, v6_gmvR.end());
    std::cout << "m_gemvR , " << v1_gmvR[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v2_gmvR[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v3_gmvR[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v4_gmvR[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v5_gmvR[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v6_gmvR[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
    std::sort (v1_gmvC.begin()+1, v1_gmvC.end());
    std::sort (v2_gmvC.begin()+1, v2_gmvC.end());
    std::sort (v3_gmvC.begin()+1, v3_gmvC.end());
    std::sort (v4_gmvC.begin()+1, v4_gmvC.end());
    std::sort (v5_gmvC.begin()+1, v5_gmvC.end());
    std::sort (v6_gmvC.begin()+1, v6_gmvC.end());
    std::sort (v7_gmvC.begin()+1, v7_gmvC.end());
    std::sort (v8_gmvC.begin()+1, v8_gmvC.end());
    std::sort (v9_gmvC.begin()+1, v9_gmvC.end());
    std::sort (v10_gmvC.begin()+1, v10_gmvC.end());
    std::sort (v11_gmvC.begin()+1, v11_gmvC.end());
    std::sort (v12_gmvC.begin()+1, v12_gmvC.end());
    std::sort (v13_gmvC.begin()+1, v13_gmvC.end());
    std::cout << "m_gemvC , " << v1_gmvC[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v2_gmvC[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v3_gmvC[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v4_gmvC[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v5_gmvC[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v6_gmvC[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v7_gmvC[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v8_gmvC[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v9_gmvC[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v10_gmvC[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v11_gmvC[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v12_gmvC[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v13_gmvC[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
    std::sort (v1_ger.begin()+1, v1_ger.end());
    std::sort (v2_ger.begin()+1, v2_ger.end());
    std::sort (v3_ger.begin()+1, v3_ger.end());
    std::cout << "m_ger   , " << v1_ger[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v2_ger[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v3_ger[(NUMBER_REPEATS+1)/2].count()
              << std::endl;

#endif

  std::cout << "GEMVR ANALYSYS!!" << std::endl;
  // ANALYSIS OF THE RESULTS
  for (int i=0; i<6; i++) {
    res = vR[i];
#ifdef SHOW_VALUES
    std::cout << "( " << i+((i>4)?15:((i>0)?10:1)) << ") ";
    std::cout << "VALUES!! --> res = " << res << " , addX = " << addX
              << " , err = " << addX - res << std::endl;
#endif  // VERBOSE
    if (std::abs((res - addX) / res) > ERROR_ALLOWED) {
      std::cout << "( " << i+((i>0)?10:1) << ") ";
      std::cout << "ERROR!! --> res = " << res << " , addX = " << addX
                << " , err = " << addX - res << std::endl;
      returnVal += 1;
    }
  }

  std::cout << "GEMVC ANALYSYS!!" << std::endl;
  for (int i=0; i<13; i++) {
    res = vS[i];
  #ifdef SHOW_VALUES
    std::cout << "( " << i+((i>2)?8:1) << ") ";
    std::cout << "VALUES!! --> res = " << res << " , addY = " << addY
              << " , err = " << addY - res << std::endl;
  #endif  // VERBOSE
    if (std::abs((res - addY) / res) > ERROR_ALLOWED) {
      std::cout << "( " << i+((i>2)?8:1) << ") ";
      std::cout << "ERROR!! --> res = " << res << " , addY = " << addY
                << " , err = " << addY - res << std::endl;
      returnVal += 2;
    }
  }

  std::cout << "GER ANALYSYS!!" << std::endl;
  for (int i=0; i<3; i++) {
    res = vT[i];
  #ifdef SHOW_VALUES
    std::cout << "( " << i+((i>0)?10:1) << ") ";
    std::cout << "VALUES!! --> res = " << res << " , addRng1 = " << addRng1
              << " , err = " << addRng1 - res << std::endl;
  #endif  // VERBOSE
    if (std::abs((res - addRng1) / res) > ERROR_ALLOWED) {
      std::cout << "( " << i+((i>0)?10:1) << ") ";
      std::cout << "ERROR!! --> res = " << res << " , addRng1 = " << addRng1
                << " , err = " << addRng1 - res << std::endl;
      returnVal += 2;
    }
  }

  return returnVal;
}

size_t TestingBLAS2_New(bool accessDev, size_t dim, size_t divSz, size_t shftR,
                        size_t shftC) {
  // CREATING DATA
  size_t dimR = dim / divSz;
  size_t dimC = dim * divSz;
  size_t dimL = ((accessDev) ? dimC : dimR);
  std::vector<BASETYPE> vM0(dimR * dimC);
  std::vector<BASETYPE> vM1(dimR * dimC);
/*
  std::vector<BASETYPE> vX0(dimC);
  std::vector<BASETYPE> vY0(dimR);
  std::vector<BASETYPE> vX1(dimC);
  std::vector<BASETYPE> vY1(dimR);
  std::vector<BASETYPE> vX2(dimC);
  std::vector<BASETYPE> vY2(dimR);
*/
  std::vector<BASETYPE> vX0(dimR);
  std::vector<BASETYPE> vY0(dimC);
  std::vector<BASETYPE> vX1(dimC);
  std::vector<BASETYPE> vY1(dimR);
  std::vector<BASETYPE> vX2(dimC);
  std::vector<BASETYPE> vY2(dimR);
/**/
  std::vector<BASETYPE> vR(10);
  std::vector<BASETYPE> vS(10);
  std::vector<BASETYPE> vT(10);
  std::vector<BASETYPE> vTX(10);
  std::vector<BASETYPE> vTY(10);
  std::vector<BASETYPE> vTU(10);
  std::vector<BASETYPE> vTL(10);
  std::vector<BASETYPE> vLX(10);
//  std::vector<BASETYPE> vDX(10);
  std::vector<BASETYPE> vUX(10);
  std::vector<BASETYPE> vLY(10);
//  std::vector<BASETYPE> vDY(10);
  std::vector<BASETYPE> vUY(10);
  std::vector<BASETYPE> vSX(10);
  std::vector<BASETYPE> vSY(10);
#ifdef SHOW_TIMES
  std::chrono::time_point<std::chrono::steady_clock> t_start, t_stop;
//#ifdef ROW_TRMV
  std::chrono::duration<BASETYPE> t0_gmvR, t0_gmvC, t0_ger, t0_syrX, t0_syrY, t0_sr2X, t0_sr2Y;
  std::chrono::duration<BASETYPE> t1_gmvR, t1_gmvC, t1_ger, t1_syrX, t1_syrY, t1_sr2X, t1_sr2Y;
  std::chrono::duration<BASETYPE> t3_gmvR, t3_gmvC, t3_ger, t3_syrX, t2_syrY, t2_sr2X, t2_sr2Y;
  std::chrono::duration<BASETYPE> t2_gmvR, t2_gmvC, t2_ger, t2_syrX, t3_syrY, t3_sr2X, t3_sr2Y;
  std::chrono::duration<BASETYPE> t4_gmvR, t4_gmvC, t4_ger, t4_syrX, t4_syrY, t4_sr2X, t4_sr2Y;
  std::chrono::duration<BASETYPE> t5_gmvR, t5_gmvC, t5_ger, t5_syrX, t5_syrY, t5_sr2X, t5_sr2Y;
  std::chrono::duration<BASETYPE> t7_gmvR, t7_gmvC, t7_ger, t7_syrX, t7_syrY, t6_sr2X, t6_sr2Y;
  std::chrono::duration<BASETYPE> t6_gmvR, t6_gmvC, t6_ger, t6_syrX, t6_syrY, t7_sr2X, t7_sr2Y;
  std::chrono::duration<BASETYPE> t8_gmvR, t8_gmvC, t8_ger, t8_syrX, t8_syrY, t8_sr2X, t8_sr2Y;
  std::chrono::duration<BASETYPE> t9_gmvR, t9_gmvC, t9_ger, t9_syrX, t9_syrY, t9_sr2X, t9_sr2Y;
//#else
  std::chrono::duration<BASETYPE> t0_lowX, t0_uppX, t0_symX, t0_lowY, t0_uppY, t0_symY;
  std::chrono::duration<BASETYPE> t1_lowX, t1_uppX, t1_symX, t1_lowY, t1_uppY, t1_symY;
  std::chrono::duration<BASETYPE> t2_lowX, t2_uppX, t2_symX, t2_lowY, t2_uppY, t2_symY;
  std::chrono::duration<BASETYPE> t3_lowX, t3_uppX, t3_symX, t3_lowY, t3_uppY, t3_symY;
  std::chrono::duration<BASETYPE> t4_lowX, t4_uppX, t4_symX, t4_lowY, t4_uppY, t4_symY;
  std::chrono::duration<BASETYPE> t5_lowX, t5_uppX, t5_symX, t5_lowY, t5_uppY, t5_symY;
  std::chrono::duration<BASETYPE> t6_lowX, t6_uppX, t6_symX, t6_lowY, t6_uppY, t6_symY;
  std::chrono::duration<BASETYPE> t7_lowX, t7_uppX, t7_symX, t7_lowY, t7_uppY, t7_symY;
  std::chrono::duration<BASETYPE> t8_lowX, t8_uppX, t8_symX, t8_lowY, t8_uppY, t8_symY;
  std::chrono::duration<BASETYPE> t9_lowX, t9_uppX, t9_symX, t9_lowY, t9_uppY, t9_symY;
//#endif
  std::vector<std::chrono::duration<BASETYPE>> v0_gmvR(NUMBER_REPEATS), v0_gmvC(NUMBER_REPEATS), v0_ger(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v1_gmvR(NUMBER_REPEATS), v1_gmvC(NUMBER_REPEATS), v1_ger(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v2_gmvR(NUMBER_REPEATS), v2_gmvC(NUMBER_REPEATS), v2_ger(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v3_gmvR(NUMBER_REPEATS), v3_gmvC(NUMBER_REPEATS), v3_ger(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v4_gmvR(NUMBER_REPEATS), v4_gmvC(NUMBER_REPEATS), v4_ger(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v5_gmvR(NUMBER_REPEATS), v5_gmvC(NUMBER_REPEATS), v5_ger(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v6_gmvR(NUMBER_REPEATS), v6_gmvC(NUMBER_REPEATS), v6_ger(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v7_gmvR(NUMBER_REPEATS), v7_gmvC(NUMBER_REPEATS), v7_ger(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v8_gmvR(NUMBER_REPEATS), v8_gmvC(NUMBER_REPEATS), v8_ger(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v9_gmvR(NUMBER_REPEATS), v9_gmvC(NUMBER_REPEATS), v9_ger(NUMBER_REPEATS);
//#ifdef ROW_TRMV
  std::vector<std::chrono::duration<BASETYPE>> v0_lowX(NUMBER_REPEATS), v0_uppX(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v1_lowX(NUMBER_REPEATS), v1_uppX(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v2_lowX(NUMBER_REPEATS), v2_uppX(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v3_lowX(NUMBER_REPEATS), v3_uppX(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v4_lowX(NUMBER_REPEATS), v4_uppX(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v5_lowX(NUMBER_REPEATS), v5_uppX(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v6_lowX(NUMBER_REPEATS), v6_uppX(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v7_lowX(NUMBER_REPEATS), v7_uppX(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v8_lowX(NUMBER_REPEATS), v8_uppX(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v9_lowX(NUMBER_REPEATS), v9_uppX(NUMBER_REPEATS);
//#else
  std::vector<std::chrono::duration<BASETYPE>> v0_lowY(NUMBER_REPEATS), v0_uppY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v1_lowY(NUMBER_REPEATS), v1_uppY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v2_lowY(NUMBER_REPEATS), v2_uppY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v3_lowY(NUMBER_REPEATS), v3_uppY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v4_lowY(NUMBER_REPEATS), v4_uppY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v5_lowY(NUMBER_REPEATS), v5_uppY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v6_lowY(NUMBER_REPEATS), v6_uppY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v7_lowY(NUMBER_REPEATS), v7_uppY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v8_lowY(NUMBER_REPEATS), v8_uppY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v9_lowY(NUMBER_REPEATS), v9_uppY(NUMBER_REPEATS);
//#endif
  std::vector<std::chrono::duration<BASETYPE>> v0_symX(NUMBER_REPEATS), v0_symY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v1_symX(NUMBER_REPEATS), v1_symY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v2_symX(NUMBER_REPEATS), v2_symY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v3_symX(NUMBER_REPEATS), v3_symY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v4_symX(NUMBER_REPEATS), v4_symY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v5_symX(NUMBER_REPEATS), v5_symY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v6_symX(NUMBER_REPEATS), v6_symY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v7_symX(NUMBER_REPEATS), v7_symY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v8_symX(NUMBER_REPEATS), v8_symY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v9_symX(NUMBER_REPEATS), v9_symY(NUMBER_REPEATS);

  std::vector<std::chrono::duration<BASETYPE>> v0_syrX(NUMBER_REPEATS), v0_syrY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v1_syrX(NUMBER_REPEATS), v1_syrY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v2_syrX(NUMBER_REPEATS), v2_syrY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v3_syrX(NUMBER_REPEATS), v3_syrY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v4_syrX(NUMBER_REPEATS), v4_syrY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v5_syrX(NUMBER_REPEATS), v5_syrY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v6_syrX(NUMBER_REPEATS), v6_syrY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v7_syrX(NUMBER_REPEATS), v7_syrY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v8_syrX(NUMBER_REPEATS), v8_syrY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v9_syrX(NUMBER_REPEATS), v9_syrY(NUMBER_REPEATS);

  std::vector<std::chrono::duration<BASETYPE>> v0_sr2X(NUMBER_REPEATS), v0_sr2Y(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v1_sr2X(NUMBER_REPEATS), v1_sr2Y(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v2_sr2X(NUMBER_REPEATS), v2_sr2Y(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v3_sr2X(NUMBER_REPEATS), v3_sr2Y(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v4_sr2X(NUMBER_REPEATS), v4_sr2Y(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v5_sr2X(NUMBER_REPEATS), v5_sr2Y(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v6_sr2X(NUMBER_REPEATS), v6_sr2Y(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v7_sr2X(NUMBER_REPEATS), v7_sr2Y(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v8_sr2X(NUMBER_REPEATS), v8_sr2Y(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v9_sr2X(NUMBER_REPEATS), v9_sr2Y(NUMBER_REPEATS);
#endif

  // INITIALIZING DATA
  size_t vSeed, gap;
  BASETYPE minV, maxV;
#ifdef RANDOM_DATA
  vSeed = 1;
  minV = -10.0;
  maxV = 10.0;
  gap = (size_t)(maxV - minV + 1);
  srand(vSeed);
#else   // RANDOM_DATA
  minV = 1.00;
  maxV = 1.00;
#endif  // RANDOM_DATA
  std::for_each(std::begin(vM1), std::end(vM1), [&](BASETYPE &elem) {
#ifdef RANDOM_DATA
    elem = minV + (BASETYPE)(rand() % gap);
#else   // RANDOM_DATA
    elem = minV; minV += maxV;
#endif  // RANDOM_DATA

  });

#ifdef RANDOM_DATA
  vSeed = 1;
  minV = -10.0;
  maxV = 10.0;
  gap = (size_t)(maxV - minV + 1);
  srand(vSeed);
#else   // RANDOM_DATA
  minV = 1.00;
  maxV = 1.00;
#endif  // RANDOM_DATA
  std::for_each(std::begin(vX0), std::end(vX0), [&](BASETYPE &elem) {
#ifdef RANDOM_DATA
    elem = minV + (BASETYPE)(rand() % gap);
#else   // RANDOM_DATA
    elem = minV; minV += maxV;
#endif  // RANDOM_DATA
  });

#ifdef RANDOM_DATA
  vSeed = 2;
  minV = -10.0;
  maxV = 10.0;
  gap = (size_t)(maxV - minV + 1);
  srand(vSeed);
#else   // RANDOM_DATA
  minV = 1.00;
  maxV = 1.00;
#endif  // RANDOM_DATA
  std::for_each(std::begin(vX1), std::end(vX1), [&](BASETYPE &elem) {
#ifdef RANDOM_DATA
    elem = minV + (BASETYPE)(rand() % gap);
#else   // RANDOM_DATA
    elem = minV; minV += maxV;
#endif  // RANDOM_DATA
  });

#ifdef RANDOM_DATA
  vSeed = 3;
  minV = -10.0;
  maxV = 10.0;
  gap = (size_t)(maxV - minV + 1);
  srand(vSeed);
#else   // RANDOM_DATA
  minV = 1.00;
  maxV = 1.00;
#endif  // RANDOM_DATA
  std::for_each(std::begin(vY0), std::end(vY0), [&](BASETYPE &elem) {
#ifdef RANDOM_DATA
    elem = minV + (BASETYPE)(rand() % gap);
#else   // RANDOM_DATA
    elem = minV; minV += maxV;
#endif  // RANDOM_DATA
  });

#ifdef RANDOM_DATA
  vSeed = 4;
  minV = -10.0;
  maxV = 10.0;
  gap = (size_t)(maxV - minV + 1);
  srand(vSeed);
#else   // RANDOM_DATA
  minV = 1.00;
  maxV = 1.00;
#endif  // RANDOM_DATA
  std::for_each(std::begin(vY1), std::end(vY1), [&](BASETYPE &elem) {
#ifdef RANDOM_DATA
    elem = minV + (BASETYPE)(rand() % gap);
#else   // RANDOM_DATA
    elem = minV; minV += maxV;
#endif  // RANDOM_DATA
  });

  for (int i=0; i<10; i++) {
    vR [i] = 0.0;
    vS [i] = 0.0;
    vT [i] = 0.0;
    vTX[i] = 0.0;
    vTY[i] = 0.0;
    vTU[i] = 0.0;
    vTL[i] = 0.0;
    vLX[i] = 0.0;
//    vDX[i] = 0.0;
    vUX[i] = 0.0;
    vLY[i] = 0.0;
//    vDY[i] = 0.0;
    vUY[i] = 0.0;
    vSX[i] = 0.0;
    vSY[i] = 0.0;
  }

  // COMPUTING THE RESULTS
  size_t returnVal = 0;
  BASETYPE res;

  BASETYPE addY = 0.0, auxY, auxSY, symY = 0.0;
  BASETYPE lowY = 0.0, diaY = 0.0, uppY = 0.0, untY = 0.0;
  for (size_t i = shftR; i < dimR; i++) {
//    vY2[i - shftR] = 1.5 * vY[i - shftR];
    auxY  = CONS_ROW1 * vY1[i - shftR];
    auxSY = CONS_SYM1 * vY1[i - shftR];
    for (size_t j = shftC; j < dimC; j++) {
      if (accessDev) {
//        vY2[i - shftR] += 2.0 * vM[dimC * i + j] * vY[j - shftC];
        auxY += CONS_ROW2 * vM1[dimC * i + j] * vY0[j - shftC];
      } else {
//        vY2[i - shftR] += 2.0 * vM[dimR * j + i] * vY[j - shftC];
        auxY += CONS_ROW2 * vM1[dimR * j + i] * vY0[j - shftC];
      }
      if (i-shftR > j-shftC) {
        lowY  += vM1[(accessDev?(dimC*i+j):(dimR*j+i))] * vY0[j - shftC];
        auxSY += CONS_SYM2 * vM1[(accessDev?(dimC*i+j):(dimR*j+i))] *
                  (vY0[j - shftC] + vY0[i - shftR]);
      } else if (i-shftR < j-shftC) {
        uppY  += vM1[(accessDev?(dimC*i+j):(dimR*j+i))] * vY0[j - shftC];
      } else {
        diaY  += vM1[(accessDev?(dimC*i+j):(dimR*j+i))] * vY0[j - shftC];
        untY  += vM1[(accessDev?(dimC*i+j):(dimR*j+i))];
        auxSY += CONS_SYM2 * vM1[(accessDev?(dimC*i+j):(dimR*j+i))] * vY0[j - shftC];
      }
    }
//    addY += vY2[i - shftR];
    addY += auxY;
    symY += auxSY;
  }
  for (size_t i = dimR - shftR; i < dimR; i++) {
    addY += vY1[i];
    symY += vY1[i];
//    lowY += vY0[i];
    diaY += vY0[i];
    untY += vY0[i];
//    uppY += vY0[i];
  }

  BASETYPE addX = 0.0, auxX, auxSX, symX = 0.0;
  BASETYPE lowX = 0.0, diaX = 0.0, uppX = 0.0, untX = 0.0;
//#ifdef ROW_TRMV
//  bool accessTRMV = !accessDev;
//#else
//  bool accessTRMV = accessDev;
//#endif
  for (size_t j = shftC; j < dimC; j++) {
//    vX2[j - shftC] = 0.5 * vX2[j - shftC];
    auxX  = CONS_COL1 * vX1[j - shftC];
    auxSX = CONS_SYM1 * vX1[j - shftC];
    for (size_t i = shftR; i < dimR; i++) {
      if (accessDev) {
//        vX2[j - shftC] += 2.5 * vM[dimC * i + j] * vX[i - shftR];
        auxX += CONS_COL2 * vM1[dimC * i + j] * vX0[i - shftR];
      } else {
//        vX2[j - shftC] += 2.5 * vM[dimR * j + i] * vX[i - shftR];
        auxX += CONS_COL2 * vM1[dimR * j + i] * vX0[i - shftR];
      }
      if (i-shftR > j-shftC) {
//        lowX += vM1[(accessTRMV?(dimC*i+j):(dimR*j+i))] * vX0[i - shftR];
        lowX += vM1[(accessDev?(dimC*i+j):(dimR*j+i))] * vX0[i - shftR];
      } else if (i-shftR < j-shftC) {
//        uppX += vM1[(accessTRMV?(dimC*i+j):(dimR*j+i))] * vX0[i - shftR];
        uppX  += vM1[(accessDev?(dimC*i+j):(dimR*j+i))] * vX0[i - shftR];
        auxSX += CONS_SYM2 * vM1[(accessDev?(dimC*i+j):(dimR*j+i))] * vX0[i - shftR] +
                  CONS_SYM2 * vM1[(accessDev?(dimC*i+j):(dimR*j+i))] * vX0[j - shftC];
//        auxSX += CONS_SYM2 * vM1[(accessDev?(dimC*i+j):(dimR*j+i))] *
//                  (vX0[i - shftR] + vX0[j - shftC]);
      } else {
//        diaX += vM1[(accessTRMV?(dimC*i+j):(dimR*j+i))] * vX0[i - shftR];
        diaX  += vM1[(accessDev?(dimC*i+j):(dimR*j+i))] * vX0[i - shftR];
//        untX += vM1[(accessTRMV?(dimC*i+j):(dimR*j+i))];
        untX  += vM1[(accessDev?(dimC*i+j):(dimR*j+i))];
        auxSX += CONS_SYM2 * vM1[(accessDev?(dimC*i+j):(dimR*j+i))] * vX0[i - shftR];
      }
    }
//    addX += vX2[j - shftC];
//    printf("VX2(%lu) = %f\n", j, auxX);
    addX += auxX;
    symX += auxSX;
  }
  for (size_t j = dimC - shftC; j < dimC; j++) {
    addX += vX1[j];
    symX += vX1[j];
//    lowX += vX0[j];
    diaX += vX0[j];
    untX += vX0[j];
//    uppX += vX0[j];
  }

  BASETYPE addRng1 = 0.0, addRng1U = 0.0, addRng1L = 0.0;
  BASETYPE addRng2U = 0.0, addRng2L = 0.0;
  for (size_t i = 0; i < dimR; i++) {
    for (size_t j = 0; j < dimC; j++) {
      BASETYPE aux = 0.0, auxU = 0.0, auxL = 0.0;
      BASETYPE aux2U = 0.0, aux2L = 0.0;
//      addRng1 += (accessDev) ? vM1[dimC * i + j] : vM1[dimR * j + i];
      aux   += (accessDev) ? vM1[dimC * i + j] : vM1[dimR * j + i];
      auxU  += (accessDev) ? vM1[dimC * i + j] : vM1[dimR * j + i];
      auxL  += (accessDev) ? vM1[dimC * i + j] : vM1[dimR * j + i];
      aux2U += (accessDev) ? vM1[dimC * i + j] : vM1[dimR * j + i];
      aux2L += (accessDev) ? vM1[dimC * i + j] : vM1[dimR * j + i];
      if ((i >= shftR) && (j >= shftC)) {
//        addRng1 += (3.0 * vY0[i - shftR] * vX0[j - shftC]);
        aux += (CONS_GER * vX0[i - shftR] * vY0[j - shftC]);
        if ((i-shftR) <= (j-shftC)) {
          auxU  += (CONS_SYR  * vX0[i - shftR] * vX0[j - shftC]);
          aux2U += (CONS_SYR2 * vX0[i - shftR] * vY0[j - shftC]) +
                   (CONS_SYR2 * vY0[i - shftR] * vX0[j - shftC]);
        }
        if ((i-shftR) >= (j-shftC)) {
          auxL  += (CONS_SYR  * vY0[i - shftR] * vY0[j - shftC]);
          aux2L += (CONS_SYR2 * vX0[i - shftR] * vY0[j - shftC])+
                   (CONS_SYR2 * vY0[i - shftR] * vX0[j - shftC]);
        }
      }
      addRng1  += aux ;
      addRng1U += auxU ; addRng1L += auxL ;
      addRng2U += aux2U; addRng2L += aux2L;
    }
  }

  // CREATING THE SYCL QUEUE AND EXECUTOR
  // cl::sycl::cpu_selector s; NOOOOO
#ifdef EXECUTED_ON_GPU
  cl::sycl::gpu_selector   s;
#else
  cl::sycl::intel_selector s;
#endif
  cl::sycl::queue q([=](cl::sycl::exception_list eL) {
    try {
      for (auto &e : eL) {
        std::rethrow_exception(e);
      }
    } catch (cl::sycl::exception &e) {
      std::cout << " E " << e.what() << std::endl;
    } catch (...) {
      std::cout << " An exception " << std::endl;
    }
  });
  Executor<SYCL> ex(q);

  {
    // CREATION OF THE BUFFERS
    buffer<BASETYPE, 1> bM0(vM0.data(), range<1>{vM0.size()});
    buffer<BASETYPE, 1> bM1(vM1.data(), range<1>{vM1.size()});
    buffer<BASETYPE, 1> bX0(vX0.data(), range<1>{vX0.size()});
    buffer<BASETYPE, 1> bY0(vY0.data(), range<1>{vY0.size()});
    buffer<BASETYPE, 1> bX1(vX1.data(), range<1>{vX1.size()});
    buffer<BASETYPE, 1> bY1(vY1.data(), range<1>{vY1.size()});
    buffer<BASETYPE, 1> bX2(vX2.data(), range<1>{vX2.size()});
    buffer<BASETYPE, 1> bY2(vY2.data(), range<1>{vY2.size()});
    buffer<BASETYPE, 1> bR (vR.data() , range<1>{vR.size ()});
    buffer<BASETYPE, 1> bS (vS.data() , range<1>{vS.size ()});
    buffer<BASETYPE, 1> bT (vT.data() , range<1>{vT.size ()});
    buffer<BASETYPE, 1> bTX(vTX.data(), range<1>{vTX.size()});
    buffer<BASETYPE, 1> bTY(vTY.data(), range<1>{vTY.size()});
    buffer<BASETYPE, 1> bTU(vTU.data(), range<1>{vTU.size()});
    buffer<BASETYPE, 1> bTL(vTL.data(), range<1>{vTL.size()});
    buffer<BASETYPE, 1> bLX(vLX.data(), range<1>{vLX.size()});
//    buffer<BASETYPE, 1> bDX(vDX.data(), range<1>{vDX.size()});
    buffer<BASETYPE, 1> bUX(vUX.data(), range<1>{vUX.size()});
    buffer<BASETYPE, 1> bSX(vSX.data(), range<1>{vSX.size()});
    buffer<BASETYPE, 1> bLY(vLY.data(), range<1>{vLY.size()});
//    buffer<BASETYPE, 1> bDY(vDY.data(), range<1>{vDY.size()});
    buffer<BASETYPE, 1> bUY(vUY.data(), range<1>{vUY.size()});
    buffer<BASETYPE, 1> bSY(vSY.data(), range<1>{vSY.size()});

    // BUILDING A SYCL VIEW OF THE BUFFERS
    BufferMatrixView<BASETYPE> bmM0(bM0, accessDev, dimR, dimC);
    BufferMatrixView<BASETYPE> bmM1(bM1, accessDev, dimR, dimC);
    BufferVectorView<BASETYPE> bvV0(bM0);
    BufferVectorView<BASETYPE> bvX0(bX0);
    BufferVectorView<BASETYPE> bvY0(bY0);
    BufferVectorView<BASETYPE> bvX1(bX1);
    BufferVectorView<BASETYPE> bvY1(bY1);
    BufferVectorView<BASETYPE> bvX2(bX2);
    BufferVectorView<BASETYPE> bvY2(bY2);
    BufferVectorView<BASETYPE> bvR0 (bR ,0), bvR1 (bR ,1), bvR2 (bR ,2), bvR3 (bR ,3), bvR4 (bR ,4);
    BufferVectorView<BASETYPE> bvR5 (bR ,5), bvR6 (bR ,6), bvR7 (bR ,7), bvR8 (bR ,8), bvR9 (bR ,9);
    BufferVectorView<BASETYPE> bvS0 (bS ,0), bvS1 (bS ,1), bvS2 (bS ,2), bvS3 (bS ,3), bvS4 (bS ,4);
    BufferVectorView<BASETYPE> bvS5 (bS ,5), bvS6 (bS ,6), bvS7 (bS ,7), bvS8 (bS ,8), bvS9 (bS ,9);
    BufferVectorView<BASETYPE> bvT0 (bT ,0), bvT1 (bT ,1), bvT2 (bT ,2), bvT3 (bT ,3), bvT4 (bT ,4);
    BufferVectorView<BASETYPE> bvT5 (bT ,5), bvT6 (bT ,6), bvT7 (bT ,7), bvT8 (bT ,8), bvT9 (bT ,9);
    BufferVectorView<BASETYPE> bvTX0(bTX,0), bvTX1(bTX,1), bvTX2(bTX,2), bvTX3(bTX,3), bvTX4(bTX,4);
    BufferVectorView<BASETYPE> bvTX5(bTX,5), bvTX6(bTX,6), bvTX7(bTX,7), bvTX8(bTX,8), bvTX9(bTX,9);
    BufferVectorView<BASETYPE> bvTY0(bTY,0), bvTY1(bTY,1), bvTY2(bTY,2), bvTY3(bTY,3), bvTY4(bTY,4);
    BufferVectorView<BASETYPE> bvTY5(bTY,5), bvTY6(bTY,6), bvTY7(bTY,7), bvTY8(bTY,8), bvTY9(bTY,9);
    BufferVectorView<BASETYPE> bvTU0(bTU,0), bvTU1(bTU,1), bvTU2(bTU,2), bvTU3(bTU,3), bvTU4(bTU,4);
    BufferVectorView<BASETYPE> bvTU5(bTU,5), bvTU6(bTU,6), bvTU7(bTU,7), bvTU8(bTU,8), bvTU9(bTU,9);
    BufferVectorView<BASETYPE> bvTL0(bTL,0), bvTL1(bTL,1), bvTL2(bTL,2), bvTL3(bTL,3), bvTL4(bTL,4);
    BufferVectorView<BASETYPE> bvTL5(bTL,5), bvTL6(bTL,6), bvTL7(bTL,7), bvTL8(bTL,8), bvTL9(bTL,9);
    BufferVectorView<BASETYPE> bvLX0(bLX,0), bvLX1(bLX,1), bvLX2(bLX,2), bvLX3(bLX,3), bvLX4(bLX,4);
    BufferVectorView<BASETYPE> bvLX5(bLX,5), bvLX6(bLX,6), bvLX7(bLX,7), bvLX8(bLX,8), bvLX9(bLX,9);
//    BufferVectorView<BASETYPE> bvDX0(bDX,0), bvDX1(bDX,1), bvDX2(bDX,2), bvDX3(bDX,3), bvDX4(bDX,4);
//    BufferVectorView<BASETYPE> bvDX5(bDX,5), bvDX6(bDX,6), bvDX7(bDX,7), bvDX8(bDX,8), bvDX9(bDX,9);
    BufferVectorView<BASETYPE> bvUX0(bUX,0), bvUX1(bUX,1), bvUX2(bUX,2), bvUX3(bUX,3), bvUX4(bUX,4);
    BufferVectorView<BASETYPE> bvUX5(bUX,5), bvUX6(bUX,6), bvUX7(bUX,7), bvUX8(bUX,8), bvUX9(bUX,9);
    BufferVectorView<BASETYPE> bvSX0(bSX,0), bvSX1(bSX,1), bvSX2(bSX,2), bvSX3(bSX,3), bvSX4(bSX,4);
    BufferVectorView<BASETYPE> bvSX5(bSX,5), bvSX6(bSX,6), bvSX7(bSX,7), bvSX8(bSX,8), bvSX9(bSX,9);
    BufferVectorView<BASETYPE> bvLY0(bLY,0), bvLY1(bLY,1), bvLY2(bLY,2), bvLY3(bLY,3), bvLY4(bLY,4);
    BufferVectorView<BASETYPE> bvLY5(bLY,5), bvLY6(bLY,6), bvLY7(bLY,7), bvLY8(bLY,8), bvLY9(bLY,9);
//    BufferVectorView<BASETYPE> bvDY0(bDY,0), bvDY1(bDY,1), bvDY2(bDY,2), bvDY3(bDY,3), bvDY4(bDY,4);
//    BufferVectorView<BASETYPE> bvDY5(bDY,5), bvDY6(bDY,6), bvDY7(bDY,7), bvDY8(bDY,8), bvDY9(bDY,9);
    BufferVectorView<BASETYPE> bvUY0(bUY,0), bvUY1(bUY,1), bvUY2(bUY,2), bvUY3(bUY,3), bvUY4(bUY,4);
    BufferVectorView<BASETYPE> bvUY5(bUY,5), bvUY6(bUY,6), bvUY7(bUY,7), bvUY8(bUY,8), bvUY9(bUY,9);
    BufferVectorView<BASETYPE> bvSY0(bSY,0), bvSY1(bSY,1), bvSY2(bSY,2), bvSY3(bSY,3), bvSY4(bSY,4);
    BufferVectorView<BASETYPE> bvSY5(bSY,5), bvSY6(bSY,6), bvSY7(bSY,7), bvSY8(bSY,8), bvSY9(bSY,9);

    // EXECUTION OF THE ROUTINES
    for (int i = 0; i < NUMBER_REPEATS; i++) {
      /*****************************************/
      auto assign_M0 = make_op<Assign>(bmM0, bmM1);
      ex.execute(assign_M0); q.wait_and_throw();

#ifdef MATRIX_VECTOR_PRODUCT
      /*****************************************/
/**/
      auto assign_X2_0 = make_op<Assign>(bvX2, bvX1);
      ex.execute(assign_X2_0); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _GEMV(ex, ROW_TEST, dimC - shftC, dimR - shftR, CONS_COL2, bmM0(shftR, shftC),
                  dimL, bvX0, 1, CONS_COL1, bvX2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t0_gmvR = t_stop - t_start;
      } else if (i > 0) {
        t0_gmvR += t_stop - t_start;
      } else {
        t0_gmvR = t_start - t_start;
      }
      v0_gmvR[i] = t_stop - t_start;
  #endif
      auto reducOpX_0 = make_addAssignReduction(bvR0, bvX2, 256, 256);
      ex.reduce(reducOpX_0); q.wait_and_throw();
/**/
      /*****************************************/
/**/
      auto assign_X2_1 = make_op<Assign>(bvX2, bvX1);
      ex.execute(assign_X2_1); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _GEMV<256>(ex, ROW_TEST, dimC - shftC, dimR - shftR, CONS_COL2, bmM0(shftR, shftC),
                  dimL, bvX0, 1, CONS_COL1, bvX2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t1_gmvR = t_stop - t_start;
      } else if (i > 0) {
        t1_gmvR += t_stop - t_start;
      } else {
        t1_gmvR = t_start - t_start;
      }
      v1_gmvR[i] = t_stop - t_start;
  #endif
      auto reducOpX_1 = make_addAssignReduction(bvR1, bvX2, 256, 256);
      ex.reduce(reducOpX_1); q.wait_and_throw();
/**/
      /*****************************************/
/**/
      auto assign_X2_2 = make_op<Assign>(bvX2, bvX1);
      ex.execute(assign_X2_2); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _GEMV<256,256,256,256>(ex, ROW_TEST, dimC - shftC, dimR - shftR, CONS_COL2, bmM0(shftR, shftC),
//      _GEMV<256,0,1,4096>(ex, ROW_TEST, dimC - shftC, dimR - shftR, CONS_COL2, bmM0(shftR, shftC),
                  dimL, bvX0, 1, CONS_COL1, bvX2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t2_gmvR = t_stop - t_start;
      } else if (i > 0) {
        t2_gmvR += t_stop - t_start;
      } else {
        t2_gmvR = t_start - t_start;
      }
      v2_gmvR[i] = t_stop - t_start;
  #endif
      auto reducOpX_2 = make_addAssignReduction(bvR2, bvX2, 256, 256);
      ex.reduce(reducOpX_2); q.wait_and_throw();
/**/
      /*****************************************/
/**/
      auto assign_Y2_0 = make_op<Assign>(bvY2, bvY1);
      ex.execute(assign_Y2_0); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _GEMV(ex, COL_TEST, dimR - shftR, dimC - shftC, CONS_ROW2, bmM0(shftR, shftC),
                  dimL, bvY0, 1, CONS_ROW1, bvY2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t0_gmvC = t_stop - t_start;
      } else if (i > 0) {
        t0_gmvC += t_stop - t_start;
      } else {
        t0_gmvC = t_start - t_start;
      }
      v0_gmvC[i] = t_stop - t_start;
  #endif
      auto reducOpY_0 = make_addAssignReduction(bvS0, bvY2, 256, 256);
      ex.reduce(reducOpY_0); q.wait_and_throw();
/**/
      /*****************************************/
/**/
      auto assign_Y2_1 = make_op<Assign>(bvY2, bvY1);
      ex.execute(assign_Y2_1); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _GEMV<256>(ex, COL_TEST, dimR - shftR, dimC - shftC, CONS_ROW2, bmM0(shftR, shftC),
                  dimL, bvY0, 1, CONS_ROW1, bvY2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t1_gmvC = t_stop - t_start;
      } else if (i > 0) {
        t1_gmvC += t_stop - t_start;
      } else {
        t1_gmvC = t_start - t_start;
      }
      v1_gmvC[i] = t_stop - t_start;
  #endif
      auto reducOpY_1 = make_addAssignReduction(bvS1, bvY2, 256, 256);
      ex.reduce(reducOpY_1); q.wait_and_throw();
/**/
      /*****************************************/
/**/
      auto assign_Y2_2 = make_op<Assign>(bvY2, bvY1);
      ex.execute(assign_Y2_2); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
/* */
  #ifdef EXECUTED_ON_GPU
      if ((dimR - shftR) == 4096) {
        _GEMV<256,256,1,4096>(ex, COL_TEST, dimR - shftR, dimC - shftC, CONS_ROW2, bmM0(shftR, shftC),
                    dimL, bvY0, 1, CONS_ROW1, bvY2, 1);
      } else if ((dimR - shftR) == 8192) {
        _GEMV<256,256,1,8192>(ex, COL_TEST, dimR - shftR, dimC - shftC, CONS_ROW2, bmM0(shftR, shftC),
                    dimL, bvY0, 1, CONS_ROW1, bvY2, 1);
      } else if ((dimR - shftR) == 16384) {
        _GEMV<256,256,1,16384>(ex, COL_TEST, dimR - shftR, dimC - shftC, CONS_ROW2, bmM0(shftR, shftC),
                    dimL, bvY0, 1, CONS_ROW1, bvY2, 1);
      } else if ((dimR - shftR) == 4000) {
        _GEMV<256,256,1,4000>(ex, COL_TEST, dimR - shftR, dimC - shftC, CONS_ROW2, bmM0(shftR, shftC),
                    dimL, bvY0, 1, CONS_ROW1, bvY2, 1);
      } else if ((dimR - shftR) == 1990) {
        _GEMV<256,256,1,1990>(ex, COL_TEST, dimR - shftR, dimC - shftC, CONS_ROW2, bmM0(shftR, shftC),
                    dimL, bvY0, 1, CONS_ROW1, bvY2, 1);
      } else {
        std::cout << "The value of " << dimR - shftR << " is not considered " << std::endl;
      }
  #else
      _GEMV<256,256,256,256>(ex, COL_TEST, dimR - shftR, dimC - shftC, CONS_ROW2, bmM0(shftR, shftC),
                dimL, bvY0, 1, CONS_ROW1, bvY2, 1);
  #endif
/* */
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t2_gmvC = t_stop - t_start;
      } else if (i > 0) {
        t2_gmvC += t_stop - t_start;
      } else {
        t2_gmvC = t_start - t_start;
      }
      v2_gmvC[i] = t_stop - t_start;
  #endif
      auto reducOpY_2 = make_addAssignReduction(bvS2, bvY2, 256, 256);
      ex.reduce(reducOpY_2); q.wait_and_throw();
/**/
      /*****************************************/
/**/
      auto assign_UX_0 = make_op<Assign>(bvX2, bvX0);
      ex.execute(assign_UX_0); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _TRMV(ex, "U", ROW_TEST, UNT_TEST, dimR - shftR, bmM0(shftR, shftC), dimL, bvX2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t0_uppX = t_stop - t_start;
      } else if (i > 0) {
        t0_uppX += t_stop - t_start;
      } else {
        t0_uppX = t_start - t_start;
      }
      v0_uppX[i] = t_stop - t_start;
  #endif
      auto reducOpUX_0 = make_addAssignReduction(bvUX0, bvX2, 256, 256);
      ex.reduce(reducOpUX_0); q.wait_and_throw();
/**/
      /*****************************************/
/**/
      auto assign_UX_1 = make_op<Assign>(bvX2, bvX0);
      ex.execute(assign_UX_1); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _TRMV<256>(ex, "U", ROW_TEST, UNT_TEST, dimR - shftR, bmM0(shftR, shftC), dimL, bvX2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t1_uppX = t_stop - t_start;
      } else if (i > 0) {
        t1_uppX += t_stop - t_start;
      } else {
        t1_uppX = t_start - t_start;
      }
      v1_uppX[i] = t_stop - t_start;
  #endif
      auto reducOpUX_1 = make_addAssignReduction(bvUX1, bvX2, 256, 256);
      ex.reduce(reducOpUX_1); q.wait_and_throw();
/**/
      /*****************************************/
/**/
      auto assign_UX_2 = make_op<Assign>(bvX2, bvX0);
      ex.execute(assign_UX_2); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _TRMV<256,256,256,256>(ex, "U", ROW_TEST, UNT_TEST, dimR - shftR, bmM0(shftR, shftC), dimL, bvX2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t2_uppX = t_stop - t_start;
      } else if (i > 0) {
        t2_uppX += t_stop - t_start;
      } else {
        t2_uppX = t_start - t_start;
      }
      v2_uppX[i] = t_stop - t_start;
  #endif
      auto reducOpUX_2 = make_addAssignReduction(bvUX2, bvX2, 256, 256);
      ex.reduce(reducOpUX_2); q.wait_and_throw();
/**/
      /*****************************************/
/**/
      auto assign_LX_0 = make_op<Assign>(bvX2, bvX0);
      ex.execute(assign_LX_0); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
//    _TRMV<256,0>(ex, "L", COL_TEST, UNT_TEST, dimR - shftR, bmM0(shftR, shftC), dimL, bvX2, 1);
      _TRMV(ex, "L", ROW_TEST, UNT_TEST, dimR - shftR, bmM0(shftR, shftC), dimL, bvX2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t0_lowX = t_stop - t_start;
      } else if (i > 0) {
        t0_lowX += t_stop - t_start;
      } else {
        t0_lowX = t_start - t_start;
      }
      v0_lowX[i] = t_stop - t_start;
  #endif
      auto reducOpLX_0 = make_addAssignReduction(bvLX0, bvX2, 256, 256);
      ex.reduce(reducOpLX_0); q.wait_and_throw();
/**/
      /*****************************************/
/**/
      auto assign_LX_1 = make_op<Assign>(bvX2, bvX0);
      ex.execute(assign_LX_1); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
//    _TRMV<256,0>(ex, "L", COL_TEST, UNT_TEST, dimR - shftR, bmM0(shftR, shftC), dimL, bvX2, 1);
      _TRMV<256>(ex, "L", ROW_TEST, UNT_TEST, dimR - shftR, bmM0(shftR, shftC), dimL, bvX2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t1_lowX = t_stop - t_start;
      } else if (i > 0) {
        t1_lowX += t_stop - t_start;
      } else {
        t1_lowX = t_start - t_start;
      }
      v1_lowX[i] = t_stop - t_start;
  #endif
      auto reducOpLX_1 = make_addAssignReduction(bvLX1, bvX2, 256, 256);
      ex.reduce(reducOpLX_1); q.wait_and_throw();
/**/
      /*****************************************/
/**/
      auto assign_LX_2 = make_op<Assign>(bvX2, bvX0);
      ex.execute(assign_LX_2); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
//    _TRMV<256,0>(ex, "L", COL_TEST, UNT_TEST, dimR - shftR, bmM0(shftR, shftC), dimL, bvX2, 1);
      _TRMV<256,256,256,256>(ex, "L", ROW_TEST, UNT_TEST, dimR - shftR, bmM0(shftR, shftC), dimL, bvX2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t2_lowX = t_stop - t_start;
      } else if (i > 0) {
        t2_lowX += t_stop - t_start;
      } else {
        t2_lowX = t_start - t_start;
      }
      v2_lowX[i] = t_stop - t_start;
  #endif
      auto reducOpLX_2 = make_addAssignReduction(bvLX2, bvX2, 256, 256);
      ex.reduce(reducOpLX_2); q.wait_and_throw();
/**/
      /*****************************************/
/**/
      auto assign_UY_0 = make_op<Assign>(bvY2, bvY0);
      ex.execute(assign_UY_0); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _TRMV(ex, "U", COL_TEST, UNT_TEST, dimR - shftR, bmM0(shftR, shftC), dimL, bvY2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t0_uppY = t_stop - t_start;
      } else if (i > 0) {
        t0_uppY += t_stop - t_start;
      } else {
        t0_uppY = t_start - t_start;
      }
      v0_uppY[i] = t_stop - t_start;
  #endif
      auto reducOpUY_0 = make_addAssignReduction(bvUY0, bvY2, 256, 256);
      ex.reduce(reducOpUY_0); q.wait_and_throw();
/**/
      /*****************************************/
/**/
      auto assign_UY_1 = make_op<Assign>(bvY2, bvY0);
      ex.execute(assign_UY_1); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
//      _TRMV<256,0>(ex, "U", COL_TEST, UNT_TEST, dimR - shftR, bmM0(shftR, shftC), dimL, bvY2, 1);
      _TRMV<256>(ex, "U", COL_TEST, UNT_TEST, dimR - shftR, bmM0(shftR, shftC), dimL, bvY2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t1_uppY = t_stop - t_start;
      } else if (i > 0) {
        t1_uppY += t_stop - t_start;
      } else {
        t1_uppY = t_start - t_start;
      }
      v1_uppY[i] = t_stop - t_start;
  #endif
      auto reducOpUY_1 = make_addAssignReduction(bvUY1, bvY2, 256, 256);
      ex.reduce(reducOpUY_1); q.wait_and_throw();
/**/
      /*****************************************/
/**/
      auto assign_UY_2 = make_op<Assign>(bvY2, bvY0);
      ex.execute(assign_UY_2); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
//      _TRMV<256,0>(ex, "U", COL_TEST, UNT_TEST, dimR - shftR, bmM0(shftR, shftC), dimL, bvY2, 1);
      _TRMV<256,256,256,256>(ex, "U", COL_TEST, UNT_TEST, dimR - shftR, bmM0(shftR, shftC), dimL, bvY2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t2_uppY = t_stop - t_start;
      } else if (i > 0) {
        t2_uppY += t_stop - t_start;
      } else {
        t2_uppY = t_start - t_start;
      }
      v2_uppY[i] = t_stop - t_start;
  #endif
      auto reducOpUY_2 = make_addAssignReduction(bvUY2, bvY2, 256, 256);
      ex.reduce(reducOpUY_2); q.wait_and_throw();
/**/
      /*****************************************/
/**/
      auto assign_LY_0 = make_op<Assign>(bvY2, bvY0);
      ex.execute(assign_LY_0); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
//    _TRMV<256,0>(ex, "L", COL_TEST, UNT_TEST, dimR - shftR, bmM0(shftR, shftC), dimL, bvY2, 1);
      _TRMV(ex, "L", COL_TEST, UNT_TEST, dimR - shftR, bmM0(shftR, shftC), dimL, bvY2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t0_lowY = t_stop - t_start;
      } else if (i > 0) {
        t0_lowY += t_stop - t_start;
      } else {
        t0_lowY = t_start - t_start;
      }
      v0_lowY[i] = t_stop - t_start;
  #endif
      auto reducOpLY_0 = make_addAssignReduction(bvLY0, bvY2, 256, 256);
      ex.reduce(reducOpLY_0); q.wait_and_throw();
/**/
      /*****************************************/
/**/
      auto assign_LY_1 = make_op<Assign>(bvY2, bvY0);
      ex.execute(assign_LY_1); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      //    _TRMV<256,0>(ex, "L", COL_TEST, UNT_TEST, dimR - shftR, bmM0(shftR, shftC), dimL, bvY2, 1);
      _TRMV<256>(ex, "L", COL_TEST, UNT_TEST, dimR - shftR, bmM0(shftR, shftC), dimL, bvY2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t1_lowY = t_stop - t_start;
      } else if (i > 0) {
        t1_lowY += t_stop - t_start;
      } else {
        t1_lowY = t_start - t_start;
      }
      v1_lowY[i] = t_stop - t_start;
  #endif
      auto reducOpLY_1 = make_addAssignReduction(bvLY1, bvY2, 256, 256);
      ex.reduce(reducOpLY_1); q.wait_and_throw();
/**/
      /*****************************************/
/**/
      auto assign_LY_2 = make_op<Assign>(bvY2, bvY0);
      ex.execute(assign_LY_2); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      //    _TRMV<256,0>(ex, "L", COL_TEST, UNT_TEST, dimR - shftR, bmM0(shftR, shftC), dimL, bvY2, 1);
      _TRMV<256,256,256,256>(ex, "L", COL_TEST, UNT_TEST, dimR - shftR, bmM0(shftR, shftC), dimL, bvY2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t2_lowY = t_stop - t_start;
      } else if (i > 0) {
        t2_lowY += t_stop - t_start;
      } else {
        t2_lowY = t_start - t_start;
      }
      v2_lowY[i] = t_stop - t_start;
  #endif
      auto reducOpLY_2 = make_addAssignReduction(bvLY2, bvY2, 256, 256);
      ex.reduce(reducOpLY_2); q.wait_and_throw();
/**/
      /*****************************************/
/* */
      auto assign_SX2_0 = make_op<Assign>(bvX2, bvX1);
      ex.execute(assign_SX2_0); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _SYMV(ex, "U", dimR - shftR, CONS_SYM2, bmM0(shftR, shftC),
                  dimL, bvX0, 1, CONS_SYM1, bvX2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t0_symX = t_stop - t_start;
      } else if (i > 0) {
        t0_symX += t_stop - t_start;
      } else {
        t0_symX = t_start - t_start;
      }
      v0_symX[i] = t_stop - t_start;
  #endif
      auto reducOpSX_0 = make_addAssignReduction(bvSX0, bvX2, 256, 256);
      ex.reduce(reducOpSX_0); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_SX2_1 = make_op<Assign>(bvX2, bvX1);
      ex.execute(assign_SX2_1); q.wait_and_throw();
      #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
      #endif
      _SYMV<256>(ex, "U", dimR - shftR, CONS_SYM2, bmM0(shftR, shftC),
                  dimL, bvX0, 1, CONS_SYM1, bvX2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t1_symX = t_stop - t_start;
      } else if (i > 0) {
        t1_symX += t_stop - t_start;
      } else {
        t1_symX = t_start - t_start;
      }
      v1_symX[i] = t_stop - t_start;
  #endif
      auto reducOpSX_1 = make_addAssignReduction(bvSX1, bvX2, 256, 256);
      ex.reduce(reducOpSX_1); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_SX2_2 = make_op<Assign>(bvX2, bvX1);
      ex.execute(assign_SX2_2); q.wait_and_throw();
      #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
      #endif
      _SYMV<256,256,256,256>(ex, "U", dimR - shftR, CONS_SYM2, bmM0(shftR, shftC),
                  dimL, bvX0, 1, CONS_SYM1, bvX2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t2_symX = t_stop - t_start;
      } else if (i > 0) {
        t2_symX += t_stop - t_start;
      } else {
        t2_symX = t_start - t_start;
      }
      v2_symX[i] = t_stop - t_start;
  #endif
      auto reducOpSX_2 = make_addAssignReduction(bvSX2, bvX2, 256, 256);
      ex.reduce(reducOpSX_2); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_SY2_0 = make_op<Assign>(bvY2, bvY1);
      ex.execute(assign_SY2_0); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _SYMV(ex, "L", dimR - shftR, CONS_SYM2, bmM0(shftR, shftC),
                  dimL, bvY0, 1, CONS_SYM1, bvY2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t0_symY = t_stop - t_start;
      } else if (i > 0) {
        t0_symY += t_stop - t_start;
      } else {
        t0_symY = t_start - t_start;
      }
      v0_symY[i] = t_stop - t_start;
  #endif
      auto reducOpSY_0 = make_addAssignReduction(bvSY0, bvY2, 256, 256);
      ex.reduce(reducOpSY_0); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_SY2_1 = make_op<Assign>(bvY2, bvY1);
      ex.execute(assign_SY2_1); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _SYMV<256>(ex, "L", dimR - shftR, CONS_SYM2, bmM0(shftR, shftC),
                  dimL, bvY0, 1, CONS_SYM1, bvY2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t1_symY = t_stop - t_start;
      } else if (i > 0) {
        t1_symY += t_stop - t_start;
      } else {
        t1_symY = t_start - t_start;
      }
      v1_symY[i] = t_stop - t_start;
  #endif
      auto reducOpSY_1 = make_addAssignReduction(bvSY1, bvY2, 256, 256);
      ex.reduce(reducOpSY_1); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_SY2_2 = make_op<Assign>(bvY2, bvY1);
      ex.execute(assign_SY2_2); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _SYMV<256,256,256,256>(ex, "L", dimR - shftR, CONS_SYM2, bmM0(shftR, shftC),
                  dimL, bvY0, 1, CONS_SYM1, bvY2, 1);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t2_symY = t_stop - t_start;
      } else if (i > 0) {
        t2_symY += t_stop - t_start;
      } else {
        t2_symY = t_start - t_start;
      }
      v2_symY[i] = t_stop - t_start;
  #endif
      auto reducOpSY_2 = make_addAssignReduction(bvSY2, bvY2, 256, 256);
      ex.reduce(reducOpSY_2); q.wait_and_throw();
/* */
    /*****************************************/
#else // MATRIX_VECTOR_PRODUCT
    /*****************************************/
/* */
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _GER(ex, dimR - shftR, dimC - shftC, CONS_GER, bvX0, 1, bvY0, 1,
                 bmM0(shftR, shftC), dimL);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t0_ger = t_stop - t_start;
      } else if (i > 0) {
        t0_ger += t_stop - t_start;
      } else {
        t0_ger = t_start - t_start;
      }
      v0_ger[i] = t_stop - t_start;
  #endif
      auto reducOpV_0 = make_addAssignReduction(bvT0, bvV0, 256, 256);
      ex.reduce(reducOpV_0); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_M0_1 = make_op<Assign>(bmM0, bmM1);
      ex.execute(assign_M0_1); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _GER<256>(ex, dimR - shftR, dimC - shftC, CONS_GER, bvX0, 1, bvY0, 1,
                 bmM0(shftR, shftC), dimL);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t1_ger = t_stop - t_start;
      } else if (i > 0) {
        t1_ger += t_stop - t_start;
      } else {
        t1_ger = t_start - t_start;
      }
      v1_ger[i] = t_stop - t_start;
  #endif
      auto reducOpV_1 = make_addAssignReduction(bvT1, bvV0, 256, 256);
      ex.reduce(reducOpV_1); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_M0_2 = make_op<Assign>(bmM0, bmM1);
      ex.execute(assign_M0_2); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
  #ifdef EXECUTED_ON_GPU
      _GER<256,256,1,256>(ex, dimR - shftR, dimC - shftC, CONS_GER, bvX0, 1, bvY0, 1,
                 bmM0(shftR, shftC), dimL);
  #else
      _GER<256,256,256,256>(ex, dimR - shftR, dimC - shftC, CONS_GER, bvX0, 1, bvY0, 1,
                 bmM0(shftR, shftC), dimL);
  #endif
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t2_ger = t_stop - t_start;
      } else if (i > 0) {
        t2_ger += t_stop - t_start;
      } else {
        t2_ger = t_start - t_start;
      }
      v2_ger[i] = t_stop - t_start;
  #endif
      auto reducOpV_2 = make_addAssignReduction(bvT2, bvV0, 256, 256);
      ex.reduce(reducOpV_2); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_MX0_0 = make_op<Assign>(bmM0, bmM1);
      ex.execute(assign_MX0_0); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _SYR(ex, "U", dimR - shftR, CONS_SYR, bvX0, 1,
                  bmM0(shftR, shftC), dimL);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t0_syrX = t_stop - t_start;
      } else if (i > 0) {
        t0_syrX += t_stop - t_start;
      } else {
        t0_syrX = t_start - t_start;
      }
      v0_syrX[i] = t_stop - t_start;
  #endif
      auto reducOpVX_0 = make_addAssignReduction(bvTX0, bvV0, 256, 256);
      ex.reduce(reducOpVX_0); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_MX0_1 = make_op<Assign>(bmM0, bmM1);
      ex.execute(assign_MX0_1); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _SYR<256>(ex, "U", dimR - shftR, CONS_SYR, bvX0, 1,
                  bmM0(shftR, shftC), dimL);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t1_syrX = t_stop - t_start;
      } else if (i > 0) {
        t1_syrX += t_stop - t_start;
      } else {
        t1_syrX = t_start - t_start;
      }
      v1_syrX[i] = t_stop - t_start;
  #endif
      auto reducOpVX_1 = make_addAssignReduction(bvTX1, bvV0, 256, 256);
      ex.reduce(reducOpVX_1); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_MX0_2 = make_op<Assign>(bmM0, bmM1);
      ex.execute(assign_MX0_2); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _SYR(ex, "U", dimR - shftR, CONS_SYR, bvX0, 1,
                  bmM0(shftR, shftC), dimL);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t2_syrX = t_stop - t_start;
      } else if (i > 0) {
        t2_syrX += t_stop - t_start;
      } else {
        t2_syrX = t_start - t_start;
      }
      v2_syrX[i] = t_stop - t_start;
  #endif
      auto reducOpVX_2 = make_addAssignReduction(bvTX2, bvV0, 256, 256);
      ex.reduce(reducOpVX_2); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_MY0_0 = make_op<Assign>(bmM0, bmM1);
      ex.execute(assign_MY0_0); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _SYR(ex, "L", dimR - shftR, CONS_SYR, bvY0, 1,
                  bmM0(shftR, shftC), dimL);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t0_syrY = t_stop - t_start;
      } else if (i > 0) {
        t0_syrY += t_stop - t_start;
      } else {
        t0_syrY = t_start - t_start;
      }
      v0_syrY[i] = t_stop - t_start;
  #endif
      auto reducOpVY_0 = make_addAssignReduction(bvTY0, bvV0, 256, 256);
      ex.reduce(reducOpVY_0); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_MY0_1 = make_op<Assign>(bmM0, bmM1);
      ex.execute(assign_MY0_1); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _SYR(ex, "L", dimR - shftR, CONS_SYR, bvY0, 1,
                  bmM0(shftR, shftC), dimL);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t1_syrY = t_stop - t_start;
      } else if (i > 0) {
        t1_syrY += t_stop - t_start;
      } else {
        t1_syrY = t_start - t_start;
      }
      v1_syrY[i] = t_stop - t_start;
  #endif
      auto reducOpVY_1 = make_addAssignReduction(bvTY1, bvV0, 256, 256);
      ex.reduce(reducOpVY_1); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_MY0_2 = make_op<Assign>(bmM0, bmM1);
      ex.execute(assign_MY0_2); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _SYR(ex, "L", dimR - shftR, CONS_SYR, bvY0, 1,
                  bmM0(shftR, shftC), dimL);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t2_syrY = t_stop - t_start;
      } else if (i > 0) {
        t2_syrY += t_stop - t_start;
      } else {
        t2_syrY = t_start - t_start;
      }
      v2_syrY[i] = t_stop - t_start;
  #endif
      auto reducOpVY_2 = make_addAssignReduction(bvTY2, bvV0, 256, 256);
      ex.reduce(reducOpVY_2); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_MSX_0 = make_op<Assign>(bmM0, bmM1);
      ex.execute(assign_MSX_0); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _SYR2(ex, "U", dimR - shftR, CONS_SYR2, bvX0, 1, bvY0, 1,
                 bmM0(shftR, shftC), dimL);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t0_sr2X = t_stop - t_start;
      } else if (i > 0) {
        t0_sr2X += t_stop - t_start;
      } else {
        t0_sr2X = t_start - t_start;
      }
      v0_sr2X[i] = t_stop - t_start;
  #endif
      auto reducOp2SX_0 = make_addAssignReduction(bvTU0, bvV0, 256, 256);
      ex.reduce(reducOp2SX_0); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_MSX_1 = make_op<Assign>(bmM0, bmM1);
      ex.execute(assign_MSX_1); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
      _SYR2<256>(ex, "U", dimR - shftR, CONS_SYR2, bvX0, 1, bvY0, 1,
                 bmM0(shftR, shftC), dimL);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t1_sr2X = t_stop - t_start;
      } else if (i > 0) {
        t1_sr2X += t_stop - t_start;
      } else {
        t1_sr2X = t_start - t_start;
      }
      v1_sr2X[i] = t_stop - t_start;
  #endif
      auto reducOp2SX_1 = make_addAssignReduction(bvTU1, bvV0, 256, 256);
      ex.reduce(reducOp2SX_1); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_MSX_2 = make_op<Assign>(bmM0, bmM1);
      ex.execute(assign_MSX_2); q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
//      _SYR2<256,256,256,256>(ex, "U", dimR - shftR, CONS_SYR2, bvX0, 1, bvY0, 1,
      _SYR2<256,512,256,256>(ex, "U", dimR - shftR, CONS_SYR2, bvX0, 1, bvY0, 1,
                 bmM0(shftR, shftC), dimL);
      q.wait_and_throw();
  #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t2_sr2X = t_stop - t_start;
      } else if (i > 0) {
        t2_sr2X += t_stop - t_start;
      } else {
        t2_sr2X = t_start - t_start;
      }
      v2_sr2X[i] = t_stop - t_start;
  #endif
      auto reducOp2SX_2 = make_addAssignReduction(bvTU2, bvV0, 256, 256);
      ex.reduce(reducOp2SX_2); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_MSY_0 = make_op<Assign>(bmM0, bmM1);
      ex.execute(assign_MSY_0); q.wait_and_throw();
    #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
    #endif
      _SYR2(ex, "L", dimR - shftR, CONS_SYR2, bvX0, 1, bvY0, 1,
                 bmM0(shftR, shftC), dimL);
      q.wait_and_throw();
    #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t0_sr2Y = t_stop - t_start;
      } else if (i > 0) {
        t0_sr2Y += t_stop - t_start;
      } else {
        t0_sr2Y = t_start - t_start;
      }
      v0_sr2Y[i] = t_stop - t_start;
    #endif
      auto reducOp2SY_0 = make_addAssignReduction(bvTL0, bvV0, 256, 256);
      ex.reduce(reducOp2SY_0); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_MSY_1 = make_op<Assign>(bmM0, bmM1);
      ex.execute(assign_MSY_1); q.wait_and_throw();
    #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
    #endif
      _SYR2<256>(ex, "L", dimR - shftR, CONS_SYR2, bvX0, 1, bvY0, 1,
                 bmM0(shftR, shftC), dimL);
      q.wait_and_throw();
    #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t1_sr2Y = t_stop - t_start;
      } else if (i > 0) {
        t1_sr2Y += t_stop - t_start;
      } else {
        t1_sr2Y = t_start - t_start;
      }
      v1_sr2Y[i] = t_stop - t_start;
    #endif
      auto reducOp2SY_1 = make_addAssignReduction(bvTL1, bvV0, 256, 256);
      ex.reduce(reducOp2SY_1); q.wait_and_throw();
/* */
      /*****************************************/
/* */
      auto assign_MSY_2 = make_op<Assign>(bmM0, bmM1);
      ex.execute(assign_MSY_2); q.wait_and_throw();
    #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
    #endif
//    _SYR2<256,256,256,256>(ex, "L", dimR - shftR, CONS_SYR2, bvX0, 1, bvY0, 1,
    _SYR2<256,512,256,256>(ex, "L", dimR - shftR, CONS_SYR2, bvX0, 1, bvY0, 1,
                 bmM0(shftR, shftC), dimL);
      q.wait_and_throw();
    #ifdef SHOW_TIMES
      t_stop = std::chrono::steady_clock::now();
      if (NUMBER_REPEATS == 1) {
        t2_sr2Y = t_stop - t_start;
      } else if (i > 0) {
        t2_sr2Y += t_stop - t_start;
      } else {
        t2_sr2Y = t_start - t_start;
      }
      v2_sr2Y[i] = t_stop - t_start;
    #endif
      auto reducOp2SY_2 = make_addAssignReduction(bvTL2, bvV0, 256, 256);
      ex.reduce(reducOp2SY_2); q.wait_and_throw();
/* */
      /*****************************************/
#endif // MATRIX_VECTOR_PRODUCT
    }
  }

#ifdef SHOW_TIMES
    int div = (NUMBER_REPEATS == 1)? 1: (NUMBER_REPEATS-1);
//    int div = 1;
#ifdef MATRIX_VECTOR_PRODUCT
    // COMPUTATIONAL TIMES
    std::cout << "t_gemvR , " << t0_gmvR.count()/div
              << ", "         << t1_gmvR.count()/div
              << ", "         << t2_gmvR.count()/div
//              << ", " << t3_gmvR.count()/div
//              << ", " << t4_gmvR.count()/div
//              << ", " << t5_gmvR.count()/div
//              << ", " << t6_gmvR.count()/div
              << std::endl;
    std::cout << "t_gemvC , " << t0_gmvC.count()/div
              << ", "         << t1_gmvC.count()/div
              << ", "         << t2_gmvC.count()/div
//              << ", " << t3_gmvC.count()/div
//              << ", " << t4_gmvC.count()/div
//              << ", " << t5_gmvC.count()/div
//              << ", " << t6_gmvC.count()/div
//              << ", " << t7_gmvC.count()/div
//              << ", " << t8_gmvC.count()/div
//              << ", " << t9_gmvC.count()/div
              << std::endl;
//#ifdef ROW_TRMV
    std::cout << "t_uppX  , " << t0_uppX.count()/div
              << ", "         << t1_uppX.count()/div
              << ", "         << t2_uppX.count()/div
              << std::endl;
    std::cout << "t_lowX  , " << t0_lowX.count()/div
              << ", "         << t1_lowX.count()/div
              << ", "         << t2_lowX.count()/div
              << std::endl;
//#else
    std::cout << "t_uppY  , " << t0_uppY.count()/div
              << ", "         << t1_uppY.count()/div
              << ", "         << t2_uppY.count()/div
              << std::endl;
    std::cout << "t_lowY  , " << t0_lowY.count()/div
              << ", "         << t1_lowY.count()/div
              << ", "         << t2_lowY.count()/div
              << std::endl;
//#endif
    std::cout << "t_symX  , " << t0_symX.count()/div
              << ", "         << t1_uppX.count()/div
              << ", "         << t2_uppX.count()/div
              << std::endl;
    std::cout << "t_symY  , " << t0_symY.count()/div
              << ", "         << t1_uppY.count()/div
              << ", "         << t2_uppY.count()/div
              << std::endl;
#else // MATRIX_VECTOR_PRODUCT
    std::cout << "t_ger   , " << t0_ger.count()/div
              <<  ", "        << t1_ger.count()/div
              <<  ", "        << t2_ger.count()/div
              << std::endl;
    std::cout << "t_syrX  , " << t0_syrX.count()/div
              <<  ", "        << t1_syrX.count()/div
              <<  ", "        << t2_syrX.count()/div
              << std::endl;
    std::cout << "t_syrY  , " << t0_syrY.count()/div
              <<  ", "        << t1_syrY.count()/div
              <<  ", "        << t2_syrY.count()/div
              << std::endl;
    std::cout << "t_sr2X  , " << t0_sr2X.count()/div
              <<  ", "        << t1_sr2X.count()/div
              <<  ", "        << t2_sr2X.count()/div
              << std::endl;
    std::cout << "t_sr2Y  , " << t0_sr2Y.count()/div
              <<  ", "        << t1_sr2Y.count()/div
              <<  ", "        << t2_sr2Y.count()/div
              << std::endl;
#endif // MATRIX_VECTOR_PRODUCT
#ifdef MATRIX_VECTOR_PRODUCT
    std::sort (v0_gmvR.begin()+1, v0_gmvR.end());
    std::sort (v1_gmvR.begin()+1, v1_gmvR.end());
    std::sort (v2_gmvR.begin()+1, v2_gmvR.end());
//    std::sort (v3_gmvR.begin()+1, v3_gmvR.end());
//    std::sort (v4_gmvR.begin()+1, v4_gmvR.end());
//    std::sort (v5_gmvR.begin()+1, v5_gmvR.end());
//    std::sort (v6_gmvR.begin()+1, v6_gmvR.end());
    std::cout << "m_gemvR , " << v0_gmvR[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v1_gmvR[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v2_gmvR[(NUMBER_REPEATS+1)/2].count()
//              << ", "         << v3_gmvR[(NUMBER_REPEATS+1)/2].count()
//              << ", "         << v4_gmvR[(NUMBER_REPEATS+1)/2].count()
//              << ", "         << v5_gmvR[(NUMBER_REPEATS+1)/2].count()
//              << ", "         << v6_gmvR[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
    std::sort (v0_gmvC.begin()+1, v0_gmvC.end());
    std::sort (v1_gmvC.begin()+1, v1_gmvC.end());
    std::sort (v2_gmvC.begin()+1, v2_gmvC.end());
//    std::sort (v3_gmvC.begin()+1, v3_gmvC.end());
//    std::sort (v4_gmvC.begin()+1, v4_gmvC.end());
//    std::sort (v5_gmvC.begin()+1, v5_gmvC.end());
//    std::sort (v6_gmvC.begin()+1, v6_gmvC.end());
//    std::sort (v7_gmvC.begin()+1, v7_gmvC.end());
//    std::sort (v8_gmvC.begin()+1, v8_gmvC.end());
//    std::sort (v9_gmvC.begin()+1, v9_gmvC.end());
    std::cout << "m_gemvC , " << v0_gmvC[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v1_gmvC[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v2_gmvC[(NUMBER_REPEATS+1)/2].count()
//              << ", "         << v3_gmvC[(NUMBER_REPEATS+1)/2].count()
//              << ", "         << v4_gmvC[(NUMBER_REPEATS+1)/2].count()
//              << ", "         << v5_gmvC[(NUMBER_REPEATS+1)/2].count()
//              << ", "         << v6_gmvC[(NUMBER_REPEATS+1)/2].count()
//              << ", "         << v7_gmvC[(NUMBER_REPEATS+1)/2].count()
//              << ", "         << v8_gmvC[(NUMBER_REPEATS+1)/2].count()
//              << ", "         << v9_gmvC[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
//#ifdef ROW_TRMV
    std::sort (v0_uppX.begin()+1, v0_uppX.end());
    std::sort (v1_uppX.begin()+1, v1_uppX.end());
    std::sort (v2_uppX.begin()+1, v2_uppX.end());
    std::cout << "m_uppX  , " << v0_uppX[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v1_uppX[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v2_uppX[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
    std::sort (v0_lowX.begin()+1, v0_lowX.end());
    std::sort (v1_lowX.begin()+1, v1_lowX.end());
    std::sort (v2_lowX.begin()+1, v2_lowX.end());
    std::cout << "m_lowX  , " << v0_lowX[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v1_lowX[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v2_lowX[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
//#else
    std::sort (v0_uppY.begin()+1, v0_uppY.end());
    std::sort (v1_uppY.begin()+1, v1_uppY.end());
    std::sort (v2_uppY.begin()+1, v2_uppY.end());
    std::cout << "m_uppY  , " << v0_uppY[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v1_uppY[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v2_uppY[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
    std::sort (v0_lowY.begin()+1, v0_lowY.end());
    std::sort (v1_lowY.begin()+1, v1_lowY.end());
    std::sort (v2_lowY.begin()+1, v2_lowY.end());
    std::cout << "m_lowY  , " << v0_lowY[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v1_lowY[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v2_lowY[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
//#endif
    std::sort (v0_symX.begin()+1, v0_symX.end());
    std::sort (v1_symX.begin()+1, v1_symX.end());
    std::sort (v2_symX.begin()+1, v2_symX.end());
    std::cout << "m_symX  , " << v0_symX[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v1_symX[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v2_symX[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
    std::sort (v0_symY.begin()+1, v0_symY.end());
    std::sort (v1_symY.begin()+1, v1_symY.end());
    std::sort (v2_symY.begin()+1, v2_symY.end());
    std::cout << "m_symY  , " << v0_symY[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v1_symY[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v2_symY[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
#else // MATRIX_VECTOR_PRODUCT
    std::sort (v0_ger.begin()+1, v0_ger.end());
    std::sort (v1_ger.begin()+1, v1_ger.end());
    std::sort (v2_ger.begin()+1, v2_ger.end());
    std::cout << "m_ger   , " << v0_ger[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v1_ger[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v2_ger[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
    std::sort (v0_syrX.begin()+1, v0_syrX.end());
    std::sort (v1_syrX.begin()+1, v1_syrX.end());
    std::sort (v2_syrX.begin()+1, v2_syrX.end());
    std::cout << "m_syrX  , " << v0_syrX[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v1_syrX[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v2_syrX[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
    std::sort (v0_syrY.begin()+1, v0_syrY.end());
    std::sort (v1_syrY.begin()+1, v1_syrY.end());
    std::sort (v2_syrY.begin()+1, v2_syrY.end());
    std::cout << "m_syrY  , " << v0_syrY[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v1_syrY[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v2_syrY[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
    std::sort (v0_sr2X.begin()+1, v0_sr2X.end());
    std::sort (v1_sr2X.begin()+1, v1_sr2X.end());
    std::sort (v2_sr2X.begin()+1, v2_sr2X.end());
    std::cout << "m_sr2X  , " << v0_sr2X[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v1_sr2X[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v2_sr2X[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
    std::sort (v0_sr2Y.begin()+1, v0_sr2Y.end());
    std::sort (v1_sr2Y.begin()+1, v1_sr2Y.end());
    std::sort (v2_sr2Y.begin()+1, v2_sr2Y.end());
    std::cout << "m_sr2Y  , " << v0_sr2Y[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v1_sr2Y[(NUMBER_REPEATS+1)/2].count()
              << ", "         << v2_sr2Y[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
#endif // MATRIX_VECTOR_PRODUCT

#endif

#ifdef MATRIX_VECTOR_PRODUCT
  std::cout << "GEMVR ANALYSYS!!" << std::endl;
  // ANALYSIS OF THE RESULTS
  for (int i=0; i<3; i++) {
    res = vR[i];
#ifdef SHOW_VALUES
//    std::cout << "( " << i+((i>4)?15:((i>0)?10:1)) << ") ";
    std::cout << "( " << i << ") ";
    std::cout << "VALUES!! --> res = " << res << " , addX = " << addX
              << " , err = " << addX - res << std::endl;
#endif  // VERBOSE
    if (std::abs((res - addX) / res) > ERROR_ALLOWED) {
      std::cout << "( " << i << ") ";
      std::cout << "ERROR!! --> res = " << res << " , addX = " << addX
                << " , err = " << addX - res << std::endl;
      returnVal += 1;
    }
  }

  std::cout << "GEMVC ANALYSYS!!" << std::endl;
  for (int i=0; i<3; i++) {
    res = vS[i];
  #ifdef SHOW_VALUES
//    std::cout << "( " << i+((i>2)?8:1) << ") ";
    std::cout << "( " << i << ") ";
    std::cout << "VALUES!! --> res = " << res << " , addY = " << addY
              << " , err = " << addY - res << std::endl;
  #endif  // VERBOSE
    if (std::abs((res - addY) / res) > ERROR_ALLOWED) {
      std::cout << "( " << i << ") ";
      std::cout << "ERROR!! --> res = " << res << " , addY = " << addY
                << " , err = " << addY - res << std::endl;
      returnVal += 2;
    }
  }

//#ifdef ROW_TRMV
  std::cout << "UPPX ANALYSYS!!" << std::endl;
  for (int i=0; i<3; i++) {
    res = vUX[i];
  #ifdef SHOW_VALUES
//    std::cout << "( " << i+((i>2)?8:1) << ") ";
    std::cout << "( " << i << ") ";
    std::cout << "VALUES!! --> res = " << res << " , uppX = " << uppX
              << " , diaX = " << diaX << " , untX = " << untX
#ifdef TRM_UNIT
              << " , err = " << (uppX+untX) - res << std::endl;
#else
              << " , err = " << (uppX+diaX) - res << std::endl;
#endif
  #endif  // VERBOSE
#ifdef TRM_UNIT
    if (std::abs((res - (uppX+untX)) / res) > ERROR_ALLOWED) {
#else
    if (std::abs((res - (uppX+diaX)) / res) > ERROR_ALLOWED) {
#endif
      std::cout << "( " << i << ") ";
      std::cout << "ERROR!! --> res = " << res << " , uppX = " << uppX
                << " , diaX = " << diaX << " , untX = " << untX
#ifdef TRM_UNIT
                << " , err = " << (uppX+untX) - res << std::endl;
#else
                << " , err = " << (uppX+diaX) - res << std::endl;
#endif
      returnVal += 2;
    }
  }

  std::cout << "LOWX ANALYSYS!!" << std::endl;
  for (int i=0; i<3; i++) {
    res = vLX[i];
  #ifdef SHOW_VALUES
//    std::cout << "( " << i+((i>2)?8:1) << ") ";
    std::cout << "( " << i << ") ";
    std::cout << "VALUES!! --> res = " << res << " , lowX = " << lowX
              << " , diaX = " << diaX << " , untX = " << untX
#ifdef TRM_UNIT
              << " , err = " << (lowX+untX) - res << std::endl;
#else
              << " , err = " << (lowX+diaX) - res << std::endl;
#endif
  #endif  // VERBOSE
#ifdef TRM_UNIT
    if (std::abs((res - (lowX+untX)) / res) > ERROR_ALLOWED) {
#else
    if (std::abs((res - (lowX+diaX)) / res) > ERROR_ALLOWED) {
#endif
      std::cout << "( " << i << ") ";
      std::cout << "ERROR!! --> res = " << res << " , lowX = " << lowX
                << " , diaX = " << diaX << " , untX = " << untX
#ifdef TRM_UNIT
                << " , err = " << (lowX+untX) - res << std::endl;
#else
                << " , err = " << (lowX+diaX) - res << std::endl;
#endif
      returnVal += 2;
    }
  }
//#else
  std::cout << "UPPY ANALYSYS!!" << std::endl;
  for (int i=0; i<3; i++) {
    res = vUY[i];
  #ifdef SHOW_VALUES
//    std::cout << "( " << i+((i>2)?8:1) << ") ";
    std::cout << "( " << i << ") ";
    std::cout << "VALUES!! --> res = " << res << " , uppY = " << uppY
              << " , diaY = " << diaY << " , untY = " << untY
#ifdef TRM_UNIT
              << " , err = " << (uppY+untY) - res << std::endl;
#else
              << " , err = " << (uppY+diaY) - res << std::endl;
#endif
  #endif  // VERBOSE
#ifdef TRM_UNIT
    if (std::abs((res - (uppY+untY)) / res) > ERROR_ALLOWED) {
#else
    if (std::abs((res - (uppY+diaY)) / res) > ERROR_ALLOWED) {
#endif
      std::cout << "( " << i << ") ";
      std::cout << "ERROR!! --> res = " << res << " , uppY = " << uppY
                << " , diaY = " << diaY << " , untY = " << untY
#ifdef TRM_UNIT
                << " , err = " << (uppY+untY) - res << std::endl;
#else
                << " , err = " << (uppY+diaY) - res << std::endl;
#endif
      returnVal += 2;
    }
  }

  std::cout << "LOWY ANALYSYS!!" << std::endl;
  for (int i=0; i<3; i++) {
    res = vLY[i];
  #ifdef SHOW_VALUES
//    std::cout << "( " << i+((i>2)?8:1) << ") ";
    std::cout << "( " << i << ") ";
    std::cout << "VALUES!! --> res = " << res << " , lowY = " << lowY
              << " , diaY = " << diaY << " , untY = " << untY
#ifdef TRM_UNIT
              << " , err = " << (lowY+untY) - res << std::endl;
#else
              << " , err = " << (lowY+diaY) - res << std::endl;
#endif
  #endif  // VERBOSE
#ifdef TRM_UNIT
    if (std::abs((res - (lowY+untY)) / res) > ERROR_ALLOWED) {
#else
    if (std::abs((res - (lowY+diaY)) / res) > ERROR_ALLOWED) {
#endif
      std::cout << "( " << i << ") ";
      std::cout << "ERROR!! --> res = " << res << " , lowY = " << lowY
                << " , diaY = " << diaY << " , untY = " << untY
#ifdef TRM_UNIT
                << " , err = " << (lowY+untY) - res << std::endl;
#else
                << " , err = " << (lowY+diaY) - res << std::endl;
#endif
      returnVal += 2;
    }
  }
//#endif

  std::cout << "SYMX ANALYSYS!!" << std::endl;
  for (int i=0; i<3; i++) {
    res = vSX[i];
  #ifdef SHOW_VALUES
  //    std::cout << "( " << i+((i>2)?8:1) << ") ";
    std::cout << "( " << i << ") ";
    std::cout << "VALUES!! --> res = " << res
              << " , symX = " << symX
              << " , err = " << (symX - res) << std::endl;
  #endif  // VERBOSE
    if (std::abs((res - symX) / res) > ERROR_ALLOWED) {
      std::cout << "( " << i << ") ";
      std::cout << "ERROR!! --> res = " << res
                << " , symX = " << symX
                << " , err = " << (symX - res) << std::endl;
      returnVal += 2;
    }
  }

  std::cout << "SYMY ANALYSYS!!" << std::endl;
  for (int i=0; i<3; i++) {
    res = vSY[i];
  #ifdef SHOW_VALUES
  //    std::cout << "( " << i+((i>2)?8:1) << ") ";
    std::cout << "( " << i << ") ";
    std::cout << "VALUES!! --> res = " << res
              << " , symY = " << symY
              << " , err = " << (symY - res) << std::endl;
  #endif  // VERBOSE
    if (std::abs((res - symY) / res) > ERROR_ALLOWED) {
      std::cout << "( " << i << ") ";
      std::cout << "ERROR!! --> res = " << res
                << " , symY = " << symY
                << " , err = " << (symY - res) << std::endl;
      returnVal += 2;
    }
  }
#else // MATRIX_VECTOR_PRODUCT

  std::cout << "GER ANALYSYS!!" << std::endl;
  for (int i=0; i<3; i++) {
    res = vT[i];
  #ifdef SHOW_VALUES
//    std::cout << "( " << i+((i>0)?10:1) << ") ";
    std::cout << "( " << i << ") ";
    std::cout << "VALUES!! --> res = " << res << " , addRng1 = " << addRng1
              << " , err = " << addRng1 - res << std::endl;
  #endif  // VERBOSE
    if (std::abs((res - addRng1) / res) > ERROR_ALLOWED) {
      std::cout << "( " << i << ") ";
      std::cout << "ERROR!! --> res = " << res << " , addRng1 = " << addRng1
                << " , err = " << addRng1 - res << std::endl;
      returnVal += 2;
    }
  }

  std::cout << "SYRX ANALYSYS!!" << std::endl;
  for (int i=0; i<3; i++) {
    res = vTX[i];
  #ifdef SHOW_VALUES
//    std::cout << "( " << i+((i>0)?10:1) << ") ";
    std::cout << "( " << i << ") ";
    std::cout << "VALUES!! --> res = " << res << " , addRng1U = " << addRng1U
              << " , err = " << addRng1U - res << std::endl;
  #endif  // VERBOSE
    if (std::abs((res - addRng1U) / res) > ERROR_ALLOWED) {
      std::cout << "( " << i << ") ";
      std::cout << "ERROR!! --> res = " << res << " , addRng1U = " << addRng1U
                << " , err = " << addRng1U - res << std::endl;
      returnVal += 2;
    }
  }

  std::cout << "SYRY ANALYSYS!!" << std::endl;
  for (int i=0; i<3; i++) {
    res = vTY[i];
  #ifdef SHOW_VALUES
//    std::cout << "( " << i+((i>0)?10:1) << ") ";
    std::cout << "( " << i << ") ";
    std::cout << "VALUES!! --> res = " << res << " , addRng1L = " << addRng1L
              << " , err = " << addRng1L - res << std::endl;
  #endif  // VERBOSE
    if (std::abs((res - addRng1L) / res) > ERROR_ALLOWED) {
      std::cout << "( " << i << ") ";
      std::cout << "ERROR!! --> res = " << res << " , addRng1L = " << addRng1L
                << " , err = " << addRng1L - res << std::endl;
      returnVal += 2;
    }
  }

  std::cout << "SYR2X ANALYSYS!!" << std::endl;
  for (int i=0; i<3; i++) {
    res = vTU[i];
  #ifdef SHOW_VALUES
//    std::cout << "( " << i+((i>0)?10:1) << ") ";
    std::cout << "( " << i << ") ";
    std::cout << "VALUES!! --> res = " << res << " , addRng2U = " << addRng2U
              << " , err = " << addRng2U - res << std::endl;
  #endif  // VERBOSE
    if (std::abs((res - addRng2U) / res) > ERROR_ALLOWED) {
      std::cout << "( " << i << ") ";
      std::cout << "ERROR!! --> res = " << res << " , addRng2U = " << addRng2U
                << " , err = " << addRng2U - res << std::endl;
      returnVal += 2;
    }
  }

  std::cout << "SYR2Y ANALYSYS!!" << std::endl;
  for (int i=0; i<3; i++) {
    res = vTL[i];
  #ifdef SHOW_VALUES
//    std::cout << "( " << i+((i>0)?10:1) << ") ";
    std::cout << "( " << i << ") ";
    std::cout << "VALUES!! --> res = " << res << " , addRng2L = " << addRng2L
              << " , err = " << addRng2L - res << std::endl;
  #endif  // VERBOSE
    if (std::abs((res - addRng2L) / res) > ERROR_ALLOWED) {
      std::cout << "( " << i << ") ";
      std::cout << "ERROR!! --> res = " << res << " , addRng2L = " << addRng2L
                << " , err = " << addRng2L - res << std::endl;
      returnVal += 2;
    }
  }

#endif // MATRIX_VECTOR_PRODUCT

  return returnVal;
}


int main(int argc, char *argv[]) {
  //  using namespace SyclBlas;
  bool accessDev = DEFAULT_ACCESS;
  size_t sizeV = 0, divSz = 1, shftR = 0, shftC = 0;
  size_t returnVal = 0;

  if (argc == 1) {
    sizeV = DEF_SIZE_VECT;
  } else if (argc == 2) {
    if (atoi(argv[1]) < 0) {
      sizeV = -atoi(argv[1]);
      accessDev = !DEFAULT_ACCESS;
    } else
      sizeV = atoi(argv[1]);
  } else if (argc == 3) {
    if (atoi(argv[1]) < 0) {
      sizeV = -atoi(argv[1]);
      accessDev = !DEFAULT_ACCESS;
    } else
      sizeV = atoi(argv[1]);
    divSz = atoi(argv[2]);
    ;
  } else if (argc == 4) {
    if (atoi(argv[1]) < 0) {
      sizeV = -atoi(argv[1]);
      //      accessDev = false;
      accessDev = !DEFAULT_ACCESS;
    } else
      sizeV = atoi(argv[1]);
    shftR = atoi(argv[2]);
    shftC = atoi(argv[3]);
  } else if (argc == 5) {
    if (atoi(argv[1]) < 0) {
      sizeV = -atoi(argv[1]);
      //      accessDev = false;
      accessDev = !DEFAULT_ACCESS;
    } else
      sizeV = atoi(argv[1]);
    divSz = atoi(argv[2]);
    ;
    shftR = atoi(argv[3]);
    shftC = atoi(argv[4]);
  } else {
    std::cout << "ERROR!! --> Incorrect number of input parameters"
              << std::endl;
    returnVal = 1;
  }
/* */
  if (returnVal == 0) {
//    returnVal = 2 * TestingBLAS2(accessDev, sizeV, divSz, shftR, shftC);
//    returnVal  = 2 * TestingBLAS2(DEFAULT_ACCESS, sizeV, divSz, shftR, shftC);
    returnVal += 4 * TestingBLAS2_New(accessDev, sizeV, divSz, shftR, shftC);
  }

  return returnVal;
}
