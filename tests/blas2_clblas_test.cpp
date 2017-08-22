/*
#include <algorithm>
#include <cstdlib>
#include <interface/blas1_interface_sycl.hpp>
#include <iostream>
#include <operations/blas1_trees.hpp>
#include <stdexcept>
#include <vector>

#include <clBLAS.h>

using namespace cl::sycl;
using namespace blas;

*/
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <CL/sycl.hpp>
#include <clBLAS.h>

using namespace cl::sycl;
//using namespace blas;
/* */

// #define GPU 1

#define DEF_NUM_ELEM 1200
#define ERROR_ALLOWED 1.0E-6
// #define RANDOM_DATA 1 // IF REMOVE AN ERROR APPEARS, BECAUSE CLBLAS USES THE ABSOLUTE VALUE ADDITION
// #define SHOW_VALUES   1

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


#ifdef EXECUTING_FLOAT
  #define BASETYPE float
  #define CONS_ROW1 1.5F
  #define CONS_ROW2 2.0F
  #define CONS_COL1 0.5F
  #define CONS_COL2 2.5F
  #define CONS_SYM1 0.4F
  #define CONS_SYM2 3.5F
  #define CONS_GER  3.0F
#else
  #define BASETYPE double
  #define CONS_ROW1 1.5
  #define CONS_ROW2 2.0
  #define CONS_COL1 0.5
  #define CONS_COL2 2.5
  #define CONS_SYM1 0.4
  #define CONS_SYM2 3.5
  #define CONS_GER  3.0
#endif

// TESTING ROUTINE

size_t TestingBLAS2(bool accessDev, size_t dim, size_t divSz, size_t shftR,
                    size_t shftC) {
  // CREATING DATA
  clblasOrder clOrder=(accessDev)? clblasRowMajor: clblasColumnMajor;
#ifdef TRM_UNIT
  clblasDiag clUnit=clblasUnit;
#else
  clblasDiag clUnit=clblasNonUnit;
#endif
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
  std::vector<BASETYPE> vR(3);
  std::vector<BASETYPE> vS(3);
  std::vector<BASETYPE> vT(3);
  std::vector<BASETYPE> vLX(3);
  std::vector<BASETYPE> vDX(3);
  std::vector<BASETYPE> vUX(3);
  std::vector<BASETYPE> vLY(3);
  std::vector<BASETYPE> vDY(3);
  std::vector<BASETYPE> vUY(3);
  std::vector<BASETYPE> vSX(3);
  std::vector<BASETYPE> vSY(3);
#ifdef SHOW_TIMES
  std::chrono::time_point<std::chrono::steady_clock> t_start, t_stop;
  std::chrono::duration<BASETYPE> t0_gmvR, t0_gmvC, t0_ger;
  std::chrono::duration<BASETYPE> t0_lowX, t0_uppX, t0_lowY, t0_uppY;
  std::chrono::duration<BASETYPE> t0_symX, t0_symY;
  std::vector<std::chrono::duration<BASETYPE>> v0_gmvR(NUMBER_REPEATS), v0_gmvC(NUMBER_REPEATS), v0_ger(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v0_lowX(NUMBER_REPEATS), v0_uppX(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v0_lowY(NUMBER_REPEATS), v0_uppY(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v0_symX(NUMBER_REPEATS), v0_symY(NUMBER_REPEATS);
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
    vLX[i] = 0.0;
    vDX[i] = 0.0;
    vUX[i] = 0.0;
    vLY[i] = 0.0;
    vDY[i] = 0.0;
    vUY[i] = 0.0;
    vSX[i] = 0.0;
    vSY[i] = 0.0;
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

  BASETYPE addY = 0.0, auxY, auxSY, symY = 0.0;
  BASETYPE lowY = 0.0, diaY = 0.0, uppY = 0.0, untY = 0.0;
  for (size_t i = shftR; i < dimR; i++) {
//    vY2[i - shftR] = CONS_ROW1 * vY[i - shftR];
    auxY = CONS_ROW1 * vY1[i - shftR];
    auxSY = CONS_SYM1 * vY1[i - shftR];
//    printf ("(AB) %f = CONS_ROW1 * %f\n", auxY, vY2[i - shftC]);
    for (size_t j = shftC; j < dimC; j++) {
      if (accessDev) {
//        vY2[i - shftR] += CONS_ROW2 * vM[dimC * i + j] * vY[j - shftC];
//        printf ("(A) %f += CONS_ROW2 * %f * %f\n", auxY, vM[dimC * i + j], vY[j - shftC]);
        auxY += CONS_ROW2 * vM1[dimC * i + j] * vY0[j - shftC];
      } else {
//        vY2[i - shftR] += CONS_ROW2 * vM[dimR * j + i] * vY[j - shftC];
//        printf ("(B) %f += CONS_ROW2 * %f * %f\n", auxY, vM[dimR * j + i], vY[j - shftC]);
        auxY += CONS_ROW2 * vM1[dimR * j + i] * vY0[j - shftC];
      }
      if (i-shftR > j-shftC) {
        lowY += vM1[(accessDev?(dimC*i+j):(dimR*j+i))] * vY0[j - shftC];
        auxSY += CONS_SYM2 * vM1[(accessDev?(dimC*i+j):(dimR*j+i))] *
                  (vY0[j - shftC] + vY0[i - shftR]);
      } else if (i-shftR < j-shftC) {
        uppY += vM1[(accessDev?(dimC*i+j):(dimR*j+i))] * vY0[j - shftC];
      } else {
        diaY += vM1[(accessDev?(dimC*i+j):(dimR*j+i))] * vY0[j - shftC];
        untY += vM1[(accessDev?(dimC*i+j):(dimR*j+i))];
        auxSY += CONS_SYM2 * vM1[(accessDev?(dimC*i+j):(dimR*j+i))] * vY0[j - shftC];
      }
    }
//    addY += vY2[i - shftR];
    addY += auxY;
    symY += auxSY;
//    printf("VY2(%lu) = %f\n", i, auxY);
  }
  for (size_t i = dimR - shftR; i < dimR; i++) {
#ifdef VERBOSE
    std::cout << "+" << vY[i] << std::endl;
#endif  // VERBOSE
    addY += vY1[i];
    symY += vY1[i];
//    lowY += vY1[i];
    diaY += vY0[i];
    untY += vY0[i];
//    uppY += vY1[i];
  }

  BASETYPE addX = 0.0, auxX, auxSX, symX = 0.0;
  BASETYPE lowX = 0.0, diaX = 0.0, uppX = 0.0, untX = 0.0;
  for (size_t j = shftC; j < dimC; j++) {
//    vX2[j - shftC] = CONS_COL1 * vX2[j - shftC];
    auxX = CONS_COL1 * vX1[j - shftC];
    auxSX = CONS_SYM1 * vX1[j - shftC];
    for (size_t i = shftR; i < dimR; i++) {
      if (accessDev) {
//        vX2[j - shftC] += CONS_COL2 * vM[dimC * i + j] * vX[i - shftR];
        auxX += CONS_COL2 * vM1[dimC * i + j] * vX0[i - shftR];
      } else {
//        vX2[j - shftC] += CONS_COL2 * vM[dimR * j + i] * vX[i - shftR];
        auxX += CONS_COL2 * vM1[dimR * j + i] * vX0[i - shftR];
      }
      if (i-shftR > j-shftC) {
        lowX += vM1[(accessDev?(dimC*i+j):(dimR*j+i))] * vX0[i - shftR];
      } else if (i-shftR < j-shftC) {
        uppX += vM1[(accessDev?(dimC*i+j):(dimR*j+i))] * vX0[i - shftR];
        auxSX += CONS_SYM2 * vM1[(accessDev?(dimC*i+j):(dimR*j+i))] * vX0[i - shftR] +
                  CONS_SYM2 * vM1[(accessDev?(dimC*i+j):(dimR*j+i))] * vX0[j - shftC];
      } else {
        diaX += vM1[(accessDev?(dimC*i+j):(dimR*j+i))] * vX0[i - shftR];
        untX += vM1[(accessDev?(dimC*i+j):(dimR*j+i))];
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

  BASETYPE addRng1 = 0.0;
  for (size_t i = 0; i < dimR; i++) {
    for (size_t j = 0; j < dimC; j++) {
      addRng1 += (accessDev) ? vM1[dimC * i + j] : vM1[dimR * j + i];
      if ((i >= shftR) && (j >= shftC)) {
        addRng1 += CONS_GER * vY0[i - shftR] * vX0[j - shftC];
      }
    }
  }

  // CREATING THE SYCL QUEUE AND EXECUTOR
  #ifdef EXECUTED_ON_GPU
      cl::sycl::gpu_selector   s;
  #else
      cl::sycl::intel_selector s;
      // cl::sycl::cpu_selector s; NOOOOO
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

  {
    cl_context clContext = q.get_context().get();
    cl_command_queue clQueue = q.get();

    cl_int err = CL_SUCCESS;

    err = clblasSetup();

    if (err != CL_SUCCESS) {
      std::cout << "Error during initialization of clBlas" << std::endl;
    }


    // CREATION OF THE BUFFERS
/*
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
*/
    cl_mem bM0_cl =
        clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                        vM0.size() * sizeof(BASETYPE), nullptr, &err);
    cl_mem bM1_cl =
        clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                        vM1.size() * sizeof(BASETYPE), nullptr, &err);
    cl_mem bX0_cl =
        clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                        vX0.size() * sizeof(BASETYPE), nullptr, &err);
    cl_mem bY0_cl =
        clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                        vY0.size() * sizeof(BASETYPE), nullptr, &err);
    cl_mem bX1_cl =
        clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                        vX1.size() * sizeof(BASETYPE), nullptr, &err);
    cl_mem bY1_cl =
        clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                        vY1.size() * sizeof(BASETYPE), nullptr, &err);
    cl_mem bX2_cl =
        clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                        vX2.size() * sizeof(BASETYPE), nullptr, &err);
    cl_mem bY2_cl =
        clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                        vY2.size() * sizeof(BASETYPE), nullptr, &err);
    cl_mem bR_cl =
        clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                        vR.size() * sizeof(BASETYPE), nullptr, &err);
    cl_mem bS_cl =
        clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                        vS.size() * sizeof(BASETYPE), nullptr, &err);
    cl_mem bT_cl =
        clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                        vT.size() * sizeof(BASETYPE), nullptr, &err);
/* */
    cl_mem bLX_cl =
        clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                        vLX.size() * sizeof(BASETYPE), nullptr, &err);
    cl_mem bDX_cl =
        clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                        vDX.size() * sizeof(BASETYPE), nullptr, &err);
    cl_mem bUX_cl =
        clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                        vUX.size() * sizeof(BASETYPE), nullptr, &err);
    cl_mem bLY_cl =
        clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                        vLY.size() * sizeof(BASETYPE), nullptr, &err);
    cl_mem bDY_cl =
        clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                        vDY.size() * sizeof(BASETYPE), nullptr, &err);
    cl_mem bUY_cl =
        clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                        vUY.size() * sizeof(BASETYPE), nullptr, &err);
/* */
    cl_mem bSX_cl =
        clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                        vSX.size() * sizeof(BASETYPE), nullptr, &err);
    cl_mem bSY_cl =
        clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                        vSY.size() * sizeof(BASETYPE), nullptr, &err);
    cl_mem scratchM_cl =
        clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                       vM0.size() * sizeof(BASETYPE), nullptr, &err);
    cl_mem scratchX_cl =
        clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                       vX0.size() * sizeof(BASETYPE), nullptr, &err);
    cl_mem scratchY_cl =
        clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                       vY0.size() * sizeof(BASETYPE), nullptr, &err);

    // BUILDING A SYCL VIEW OF THE BUFFERS
/*
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
    BufferVectorView<BASETYPE> bvS1(bS,0);
    BufferVectorView<BASETYPE> bvS2(bS,1);
    BufferVectorView<BASETYPE> bvS3(bS,2);
    BufferVectorView<BASETYPE> bvT(bT);
*/
    {
      err = clEnqueueWriteBuffer(clQueue, bM0_cl, CL_FALSE, 0,
                                 (vM0.size() * sizeof(BASETYPE)), vM0.data(), 0,
                                 NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueWriteBuffer(clQueue, bM1_cl, CL_FALSE, 0,
                                 (vM1.size() * sizeof(BASETYPE)), vM1.data(), 0,
                                 NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueWriteBuffer(clQueue, bX0_cl, CL_FALSE, 0,
                                 (vX0.size() * sizeof(BASETYPE)), vX0.data(), 0,
                                 NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueWriteBuffer(clQueue, bY0_cl, CL_FALSE, 0,
                                 (vY0.size() * sizeof(BASETYPE)), vY0.data(), 0,
                                 NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueWriteBuffer(clQueue, bX1_cl, CL_FALSE, 0,
                                 (vX1.size() * sizeof(BASETYPE)), vX1.data(), 0,
                                 NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueWriteBuffer(clQueue, bY1_cl, CL_FALSE, 0,
                                 (vY1.size() * sizeof(BASETYPE)), vY1.data(), 0,
                                 NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueWriteBuffer(clQueue, bX2_cl, CL_FALSE, 0,
                                 (vX2.size() * sizeof(BASETYPE)), vX2.data(), 0,
                                 NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueWriteBuffer(clQueue, bY2_cl, CL_FALSE, 0,
                                 (vY2.size() * sizeof(BASETYPE)), vY2.data(), 0,
                                 NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueWriteBuffer(clQueue, bR_cl, CL_FALSE, 0,
                                 (vR.size() * sizeof(BASETYPE)), vR.data(), 0,
                                 NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueWriteBuffer(clQueue, bS_cl, CL_FALSE, 0,
                                 (vS.size() * sizeof(BASETYPE)), vS.data(), 0,
                                 NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueWriteBuffer(clQueue, bT_cl, CL_FALSE, 0,
                                 (vT.size() * sizeof(BASETYPE)), vT.data(), 0,
                                 NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
/* */
      err = clEnqueueWriteBuffer(clQueue, bLX_cl, CL_FALSE, 0,
                                 (vLX.size() * sizeof(BASETYPE)), vLX.data(), 0,
                                 NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueWriteBuffer(clQueue, bDX_cl, CL_FALSE, 0,
                                 (vDX.size() * sizeof(BASETYPE)), vDX.data(), 0,
                                 NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueWriteBuffer(clQueue, bUX_cl, CL_FALSE, 0,
                                 (vUX.size() * sizeof(BASETYPE)), vUX.data(), 0,
                                 NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueWriteBuffer(clQueue, bLY_cl, CL_FALSE, 0,
                                 (vLY.size() * sizeof(BASETYPE)), vLY.data(), 0,
                                 NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueWriteBuffer(clQueue, bDY_cl, CL_FALSE, 0,
                                 (vDY.size() * sizeof(BASETYPE)), vDY.data(), 0,
                                 NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueWriteBuffer(clQueue, bUY_cl, CL_FALSE, 0,
                                 (vUY.size() * sizeof(BASETYPE)), vUY.data(), 0,
                                 NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueWriteBuffer(clQueue, bSX_cl, CL_FALSE, 0,
                                 (vSX.size() * sizeof(BASETYPE)), vSX.data(), 0,
                                 NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueWriteBuffer(clQueue, bSY_cl, CL_FALSE, 0,
                                 (vSY.size() * sizeof(BASETYPE)), vSY.data(), 0,
                                 NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
/* */
    }

    // EXECUTION OF THE ROUTINES
    for (int i = 0; i < NUMBER_REPEATS; i++) {
      /*****************************************/
//      auto assign_M0 = make_op<Assign>(bmM0, bmM1);
//      ex.execute(assign_M0); q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDcopy(dimR*dimC, bM1_cl, 0, 1, bM0_cl, 0, 1, 1, &clQueue,
                          0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);
        if (err != CL_SUCCESS) {
          std::cout << " ERROR " << err << std::endl;
        }
      }

      /*****************************************/
//      auto assign_X2_1 = make_op<Assign>(bvX2, bvX1);
//      ex.execute(assign_X2_1); q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDcopy(dimC, bX1_cl, 0, 1, bX2_cl, 0, 1, 1, &clQueue,
                          0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);
        if (err != CL_SUCCESS) {
          std::cout << " ERROR " << err << std::endl;
        }
      }
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
//      _gemv<1, SYCL>(ex, "Tr", dimC - shftC, dimR - shftR, CONS_COL2, bmM0(shftR, shftC),
//                      dimL, bvX0, 1, CONS_COL1, bvX2, 1);
//      q.wait_and_throw();
      {
        cl_event events[1];

//        err = clblasDgemv (clOrder, clblasTrans, dimC, dimR, 2.5, bM0_cl, 0, dimL,
//                          bX0_cl, 0, 1, 0.5, bX2_cl, 0, 1, 1, &clQueue, 0, NULL, &events[0]);
        err = clblasDgemv (clOrder, clblasTrans, dimC, dimR, CONS_COL2, bM0_cl, 0, dimL,
                          bX0_cl, 0, 1, CONS_COL1, bX2_cl, 0, 1, 1, &clQueue, 0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);
        if (err != CL_SUCCESS) {
          std::cout << " ERROR " << err << std::endl;
        }
      }
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
//      auto reducOpX1 = make_addAssignReduction(bvR1, bvX2, 256, 256);
//      ex.reduce(reducOpX1); q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDasum(dimC,
                          bR_cl, 0, bX2_cl, 0, 1, scratchX_cl, 1, &clQueue,
                          0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);

        if (err != CL_SUCCESS) {
          std::cout << __LINE__ << ": ERROR " << err << std::endl;
        }
      }  // End of copy

      /*****************************************/
//      auto assign_Y2_1 = make_op<Assign>(bvY2, bvY1);
//      ex.execute(assign_Y2_1); q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDcopy(dimC, bY1_cl, 0, 1, bY2_cl, 0, 1, 1, &clQueue,
                          0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);
        if (err != CL_SUCCESS) {
          std::cout << " ERROR " << err << std::endl;
        }
      }
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
//      _gemv<1, SYCL>(ex, "No", dimR - shftR, dimC - shftC, CONS_ROW2, bmM0(shftR, shftC),
//                  dimL, bvY0, 1, CONS_ROW1, bvY2, 1);
//      q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDgemv (clOrder, clblasNoTrans, dimR, dimC, CONS_ROW2, bM0_cl, 0, dimL,
                          bY0_cl, 0, 1, CONS_ROW1, bY2_cl, 0, 1, 1, &clQueue, 0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);
        if (err != CL_SUCCESS) {
          std::cout << " ERROR " << err << std::endl;
        }
      }
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
//      auto reducOpY1 = make_addAssignReduction(bvS1, bvY2, 256, 256);
//      ex.reduce(reducOpY1); q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDasum(dimR,
                          bS_cl, 0, bY2_cl, 0, 1, scratchY_cl, 1, &clQueue,
                          0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);

        if (err != CL_SUCCESS) {
          std::cout << __LINE__ << ": ERROR " << err << std::endl;
        }
      }  // End of copy
      /*****************************************/
/* */
//      auto assign_UX_0 = make_op<Assign>(bvX2, bvX0);
//      ex.execute(assign_UX_0); q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDcopy(dimC, bX0_cl, 0, 1, bX2_cl, 0, 1, 1, &clQueue,
                          0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);
        if (err != CL_SUCCESS) {
          std::cout << " ERROR " << err << std::endl;
        }
      }
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
//      _TRMV(ex, "U", ROW_TEST, UNT_TEST, dimR - shftR, bmM0(shftR, shftC), dimL, bvX2, 1);
//      q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDtrmv(clOrder, clblasUpper, clblasTrans, clUnit,
                          dimR - shftR, bM0_cl, 0, dimL, bX2_cl, 0, 1,
                          scratchX_cl, 1, &clQueue, 0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);
        if (err != CL_SUCCESS) {
          std::cout << " ERROR " << err << std::endl;
        }
      }
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
//      auto reducOpX1 = make_addAssignReduction(bvR1, bvX2, 256, 256);
//      ex.reduce(reducOpX1); q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDasum(dimC,
                          bUX_cl, 0, bX2_cl, 0, 1, scratchX_cl, 1, &clQueue,
                          0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);

        if (err != CL_SUCCESS) {
          std::cout << __LINE__ << ": ERROR " << err << std::endl;
        }
      }  // End of copy
/* */
      /*****************************************/
/* */
//      auto assign_UX_0 = make_op<Assign>(bvX2, bvX0);
//      ex.execute(assign_UX_0); q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDcopy(dimC, bX0_cl, 0, 1, bX2_cl, 0, 1, 1, &clQueue,
                          0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);
        if (err != CL_SUCCESS) {
          std::cout << " ERROR " << err << std::endl;
        }
      }
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
//      _TRMV(ex, "U", ROW_TEST, UNT_TEST, dimR - shftR, bmM0(shftR, shftC), dimL, bvX2, 1);
//      q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDtrmv(clOrder, clblasLower, clblasTrans, clUnit,
                          dimR - shftR, bM0_cl, 0, dimL, bX2_cl, 0, 1,
                          scratchX_cl, 1, &clQueue, 0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);
        if (err != CL_SUCCESS) {
          std::cout << " ERROR " << err << std::endl;
        }
      }
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
//      auto reducOpX1 = make_addAssignReduction(bvR1, bvX2, 256, 256);
//      ex.reduce(reducOpX1); q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDasum(dimC,
                          bLX_cl, 0, bX2_cl, 0, 1, scratchX_cl, 1, &clQueue,
                          0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);

        if (err != CL_SUCCESS) {
          std::cout << __LINE__ << ": ERROR " << err << std::endl;
        }
      }  // End of copy
/* */
      /*****************************************/
/* */
//      auto assign_UY_0 = make_op<Assign>(bvY2, bvY0);
//      ex.execute(assign_UY_0); q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDcopy(dimC, bY0_cl, 0, 1, bY2_cl, 0, 1, 1, &clQueue,
                          0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);
        if (err != CL_SUCCESS) {
          std::cout << " ERROR " << err << std::endl;
        }
      }
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
//      _TRMV(ex, "U", ROW_TEST, UNT_TEST, dimR - shftR, bmM0(shftR, shftC), dimL, bvY2, 1);
//      q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDtrmv(clOrder, clblasUpper, clblasNoTrans, clUnit,
                          dimR - shftR, bM0_cl, 0, dimL, bY2_cl, 0, 1,
                          scratchY_cl, 1, &clQueue, 0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);
        if (err != CL_SUCCESS) {
          std::cout << " ERROR " << err << std::endl;
        }
      }
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
//      auto reducOpY1 = make_addAssignReduction(bvR1, bvY2, 256, 256);
//      ex.reduce(reducOpY1); q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDasum(dimC,
                          bUY_cl, 0, bY2_cl, 0, 1, scratchY_cl, 1, &clQueue,
                          0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);

        if (err != CL_SUCCESS) {
          std::cout << __LINE__ << ": ERROR " << err << std::endl;
        }
      }  // End of copy
/* */
      /*****************************************/
/* */
//      auto assign_UY_0 = make_op<Assign>(bvY2, bvY0);
//      ex.execute(assign_UY_0); q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDcopy(dimC, bY0_cl, 0, 1, bY2_cl, 0, 1, 1, &clQueue,
                          0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);
        if (err != CL_SUCCESS) {
          std::cout << " ERROR " << err << std::endl;
        }
      }
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
//      _TRMV(ex, "U", ROW_TEST, UNT_TEST, dimR - shftR, bmM0(shftR, shftC), dimL, bvY2, 1);
//      q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDtrmv(clOrder, clblasLower, clblasNoTrans, clUnit,
                          dimR - shftR, bM0_cl, 0, dimL, bY2_cl, 0, 1,
                          scratchY_cl, 1, &clQueue, 0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);
        if (err != CL_SUCCESS) {
          std::cout << " ERROR " << err << std::endl;
        }
      }
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
//      auto reducOpY1 = make_addAssignReduction(bvR1, bvY2, 256, 256);
//      ex.reduce(reducOpY1); q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDasum(dimC,
                          bLY_cl, 0, bY2_cl, 0, 1, scratchY_cl, 1, &clQueue,
                          0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);

        if (err != CL_SUCCESS) {
          std::cout << __LINE__ << ": ERROR " << err << std::endl;
        }
      }  // End of copy
/* */
      /*****************************************/
//      auto assign_X2_1 = make_op<Assign>(bvX2, bvX1);
//      ex.execute(assign_X2_1); q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDcopy(dimC, bX1_cl, 0, 1, bX2_cl, 0, 1, 1, &clQueue,
                          0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);
        if (err != CL_SUCCESS) {
          std::cout << " ERROR " << err << std::endl;
        }
      }
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
//      _SYMV(ex, "U", dimR - shftR, CONS_SYM2, bmM0(shftR, shftC),
//                  dimL, bvX0, 1, CONS_SYM1, bvX2, 1);
//      q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDsymv (clOrder, clblasUpper, dimR, CONS_SYM2, bM0_cl, 0, dimL,
                          bX0_cl, 0, 1, CONS_SYM1, bX2_cl, 0, 1, 1, &clQueue, 0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);
        if (err != CL_SUCCESS) {
          std::cout << " ERROR " << err << std::endl;
        }
      }
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
//      auto reducOpX1 = make_addAssignReduction(bvR1, bvX2, 256, 256);
//      ex.reduce(reducOpX1); q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDasum(dimC,
                          bSX_cl, 0, bX2_cl, 0, 1, scratchX_cl, 1, &clQueue,
                          0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);

        if (err != CL_SUCCESS) {
          std::cout << __LINE__ << ": ERROR " << err << std::endl;
        }
      }  // End of copy

      /*****************************************/
//      auto assign_Y2_1 = make_op<Assign>(bvY2, bvY1);
//      ex.execute(assign_Y2_1); q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDcopy(dimC, bY1_cl, 0, 1, bY2_cl, 0, 1, 1, &clQueue,
                          0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);
        if (err != CL_SUCCESS) {
          std::cout << " ERROR " << err << std::endl;
        }
      }
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
//      _SYMV(ex, "L", dimR - shftR, CONS_SYM2, bmM0(shftR, shftC),
//                  dimL, bvY0, 1, CONS_SYM1, bvY2, 1);
//      q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDsymv (clOrder, clblasLower, dimR, CONS_SYM2, bM0_cl, 0, dimL,
                          bY0_cl, 0, 1, CONS_SYM1, bY2_cl, 0, 1, 1, &clQueue, 0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);
        if (err != CL_SUCCESS) {
          std::cout << " ERROR " << err << std::endl;
        }
      }
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
//      auto reducOpY1 = make_addAssignReduction(bvR1, bvY2, 256, 256);
//      ex.reduce(reducOpY1); q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDasum(dimC,
                          bSY_cl, 0, bY2_cl, 0, 1, scratchY_cl, 1, &clQueue,
                          0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);

        if (err != CL_SUCCESS) {
          std::cout << __LINE__ << ": ERROR " << err << std::endl;
        }
      }  // End of copy

      /*****************************************/
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
//      _ger<SYCL>(ex, dimR - shftR, dimC - shftC, CONS_GER, bvY0, 1, bvX0, 1,
//                 bmM0(shftR, shftC), dimL);
//      q.wait_and_throw();
      {
        cl_event events[1];
        err = clblasDger (clOrder, dimR, dimC, CONS_GER, bY0_cl, 0, 1, bX0_cl, 0, 1,
                          bM0_cl, 0, dimL, 1, &clQueue, 0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);
        if (err != CL_SUCCESS) {
          std::cout << " ERROR " << err << std::endl;
        }
      }
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
//      auto reducOpV = make_addAssignReduction(bvT, bvV0, 256, 256);
//      ex.reduce(reducOpV); q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDasum(dimR*dimC,
                          bT_cl, 0, bM0_cl, 0, 1, scratchM_cl, 1, &clQueue,
                          0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);

        if (err != CL_SUCCESS) {
          std::cout << __LINE__ << ": ERROR " << err << std::endl;
        }
      }  // End of copy
    }

    {
      err = clEnqueueReadBuffer(clQueue, bM0_cl, CL_FALSE, 0,
                                (vM0.size() * sizeof(BASETYPE)), vM0.data(), 0,
                                NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueReadBuffer(clQueue, bM1_cl, CL_FALSE, 0,
                                (vM1.size() * sizeof(BASETYPE)), vM1.data(), 0,
                                NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueReadBuffer(clQueue, bX0_cl, CL_FALSE, 0,
                                (vX0.size() * sizeof(BASETYPE)), vX0.data(), 0,
                                NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueReadBuffer(clQueue, bY0_cl, CL_FALSE, 0,
                                (vY0.size() * sizeof(BASETYPE)), vY0.data(), 0,
                                NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueReadBuffer(clQueue, bX1_cl, CL_FALSE, 0,
                                (vX1.size() * sizeof(BASETYPE)), vX1.data(), 0,
                                NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueReadBuffer(clQueue, bY1_cl, CL_FALSE, 0,
                                (vY1.size() * sizeof(BASETYPE)), vY1.data(), 0,
                                NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueReadBuffer(clQueue, bX2_cl, CL_FALSE, 0,
                                (vX2.size() * sizeof(BASETYPE)), vX2.data(), 0,
                                NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueReadBuffer(clQueue, bY2_cl, CL_FALSE, 0,
                                (vY2.size() * sizeof(BASETYPE)), vY2.data(), 0,
                                NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueReadBuffer(clQueue, bR_cl, CL_FALSE, 0,
                                (vR.size() * sizeof(BASETYPE)), vR.data(), 0,
                                NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueReadBuffer(clQueue, bS_cl, CL_FALSE, 0,
                                (vS.size() * sizeof(BASETYPE)), vS.data(), 0,
                                NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueReadBuffer(clQueue, bT_cl, CL_FALSE, 0,
                                (vT.size() * sizeof(BASETYPE)), vT.data(), 0,
                                NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
/* */
      err = clEnqueueReadBuffer(clQueue, bLX_cl, CL_FALSE, 0,
                                (vLX.size() * sizeof(BASETYPE)), vLX.data(), 0,
                                NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueReadBuffer(clQueue, bDX_cl, CL_FALSE, 0,
                                (vDX.size() * sizeof(BASETYPE)), vDX.data(), 0,
                                NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueReadBuffer(clQueue, bUX_cl, CL_FALSE, 0,
                                (vUX.size() * sizeof(BASETYPE)), vUX.data(), 0,
                                NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueReadBuffer(clQueue, bLY_cl, CL_FALSE, 0,
                                (vLY.size() * sizeof(BASETYPE)), vLY.data(), 0,
                                NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueReadBuffer(clQueue, bDY_cl, CL_FALSE, 0,
                                (vDY.size() * sizeof(BASETYPE)), vDY.data(), 0,
                                NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueReadBuffer(clQueue, bUY_cl, CL_FALSE, 0,
                                (vUY.size() * sizeof(BASETYPE)), vUY.data(), 0,
                                NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
/* */
      err = clEnqueueReadBuffer(clQueue, bSX_cl, CL_FALSE, 0,
                                (vSX.size() * sizeof(BASETYPE)), vSX.data(), 0,
                                NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
      err = clEnqueueReadBuffer(clQueue, bSY_cl, CL_FALSE, 0,
                                (vSY.size() * sizeof(BASETYPE)), vSY.data(), 0,
                                NULL, NULL);
      if (err != CL_SUCCESS) {
        std::cout << " Error copying to device " << err << std::endl;
      }
    }
    clFinish(clQueue);

    {
      clReleaseMemObject(bM0_cl);
      clReleaseMemObject(bM1_cl);
      clReleaseMemObject(bX0_cl);
      clReleaseMemObject(bY0_cl);
      clReleaseMemObject(bX1_cl);
      clReleaseMemObject(bY1_cl);
      clReleaseMemObject(bX2_cl);
      clReleaseMemObject(bY2_cl);
      clReleaseMemObject(bR_cl);
      clReleaseMemObject(bS_cl);
      clReleaseMemObject(bT_cl);
/* */
      clReleaseMemObject(bLX_cl);
      clReleaseMemObject(bDX_cl);
      clReleaseMemObject(bUX_cl);
      clReleaseMemObject(bLY_cl);
      clReleaseMemObject(bDY_cl);
      clReleaseMemObject(bUY_cl);
/* */
      clReleaseMemObject(bSX_cl);
      clReleaseMemObject(bSY_cl);
    }

    clblasTeardown();
  }

#ifdef SHOW_TIMES
    int div = (NUMBER_REPEATS == 1)? 1: (NUMBER_REPEATS-1);
//    int div = 1;
    // COMPUTATIONAL TIMES
    std::cout << "t_gemvR , " << t0_gmvR.count()/div
//              << ", " << t2_gmvR.count()/div
//              << ", " << t3_gmvR.count()/div
              << std::endl;
    std::cout << "t_gemvC , " << t0_gmvC.count()/div
//              << ", " << t2_gmvC.count()/div
//              << ", " << t3_gmvC.count()/div
              << std::endl;
    std::cout << "t_uppX  , " << t0_uppX.count()/div
//              << ", "         << t0_uppX.count()/div
//              << ", "         << t2_uppX.count()/div
              << std::endl;
    std::cout << "t_lowX  , " << t0_lowX.count()/div
//              << ", "         << t0_lowX.count()/div
//              << ", "         << t2_lowX.count()/div
              << std::endl;
    std::cout << "t_uppY  , " << t0_uppY.count()/div
//              << ", "         << t0_uppY.count()/div
//              << ", "         << t2_uppY.count()/div
              << std::endl;
    std::cout << "t_lowY  , " << t0_lowY.count()/div
//              << ", "         << t0_lowY.count()/div
//              << ", "         << t2_lowY.count()/div
              << std::endl;
    std::cout << "t_symX  , " << t0_symX.count()/div
//              << ", "         << t1_uppX.count()/div
//              << ", "         << t2_uppX.count()/div
              << std::endl;
    std::cout << "t_symY  , " << t0_symY.count()/div
//              << ", "         << t1_uppY.count()/div
//              << ", "         << t2_uppY.count()/div
              << std::endl;
    std::cout << "t_ger   , " << t0_ger.count()/div << std::endl;

    std::sort (v0_gmvR.begin()+1, v0_gmvR.end());
//    std::sort (v2_gmvR.begin()+1, v2_gmvR.end());
//    std::sort (v3_gmvR.begin()+1, v3_gmvR.end());
    std::cout << "m_gmvR , " << v0_gmvR[(NUMBER_REPEATS+1)/2].count()
//              << ", "        << v2_gmvR[(NUMBER_REPEATS+1)/2].count()
//              << ", "        << v3_gmvR[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
    std::sort (v0_gmvC.begin()+1, v0_gmvC.end());
//    std::sort (v2_gmvC.begin()+1, v2_gmvC.end());

    std::cout << "m_gmvC , " << v0_gmvC[(NUMBER_REPEATS+1)/2].count()
//              << ", "        << v2_gmvC[(NUMBER_REPEATS+1)/2].count()
//              << ", "        << v3_gmvC[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
    std::sort (v0_uppX.begin()+1, v0_uppX.end());
//    std::sort (v0_uppX.begin()+1, v0_uppX.end());
//    std::sort (v2_uppX.begin()+1, v2_uppX.end());
    std::cout << "m_uppX  , " << v0_uppX[(NUMBER_REPEATS+1)/2].count()
//              << ", "         << v0_uppX[(NUMBER_REPEATS+1)/2].count()
//              << ", "         << v2_uppX[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
    std::sort (v0_lowX.begin()+1, v0_lowX.end());
//    std::sort (v0_lowX.begin()+1, v0_lowX.end());
//    std::sort (v2_lowX.begin()+1, v2_lowX.end());
    std::cout << "m_lowX  , " << v0_lowX[(NUMBER_REPEATS+1)/2].count()
//              << ", "         << v0_lowX[(NUMBER_REPEATS+1)/2].count()
//              << ", "         << v2_lowX[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
    std::sort (v0_uppY.begin()+1, v0_uppY.end());
//    std::sort (v0_uppY.begin()+1, v0_uppY.end());
//    std::sort (v2_uppY.begin()+1, v2_uppY.end());
    std::cout << "m_uppY  , " << v0_uppY[(NUMBER_REPEATS+1)/2].count()
//              << ", "         << v0_uppY[(NUMBER_REPEATS+1)/2].count()
//              << ", "         << v2_uppY[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
    std::sort (v0_lowY.begin()+1, v0_lowY.end());
//    std::sort (v0_lowY.begin()+1, v0_lowY.end());
//    std::sort (v2_lowY.begin()+1, v2_lowY.end());
    std::cout << "m_lowY  , " << v0_lowY[(NUMBER_REPEATS+1)/2].count()
//              << ", "         << v0_lowY[(NUMBER_REPEATS+1)/2].count()
//              << ", "         << v2_lowY[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
    std::sort (v0_symX.begin()+1, v0_symX.end());
//    std::sort (v1_symX.begin()+1, v1_symX.end());
//    std::sort (v2_symX.begin()+1, v2_symX.end());
    std::cout << "m_symX  , " << v0_symX[(NUMBER_REPEATS+1)/2].count()
//              << ", "         << v1_symX[(NUMBER_REPEATS+1)/2].count()
//              << ", "         << v2_symX[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
    std::sort (v0_symY.begin()+1, v0_symY.end());
//    std::sort (v1_symY.begin()+1, v1_symY.end());
//    std::sort (v2_symY.begin()+1, v2_symY.end());
    std::cout << "m_symY  , " << v0_symY[(NUMBER_REPEATS+1)/2].count()
//              << ", "         << v1_symY[(NUMBER_REPEATS+1)/2].count()
//              << ", "         << v2_symY[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
    std::sort (v0_ger.begin()+1, v0_ger.end());
    std::cout << "m_ger  , " << v0_ger[(NUMBER_REPEATS+1)/2].count()
              << std::endl;

#endif

  // ANALYSIS OF THE RESULTS
  for (int i=0; i<1; i++) {
    res = vR[i];
#ifdef SHOW_VALUES
    std::cout << "VALUES!! --> res = " << res << " , addX = " << addX
              << " , err = " << addX - res << std::endl;
#endif  // VERBOSE
    if (std::abs((res - addX) / res) > ERROR_ALLOWED) {
      std::cout << "ERROR!! --> res = " << res << " , addX = " << addX
                << " , err = " << addX - res << std::endl;
      returnVal += 1;
    }
  }

  for (int i=0; i<1; i++) {
    res = vS[i];
  #ifdef SHOW_VALUES
    std::cout << "VALUES!! --> res = " << res << " , addY = " << addY
              << " , err = " << addY - res << std::endl;
  #endif  // VERBOSE
    if (std::abs((res - addY) / res) > ERROR_ALLOWED) {
      std::cout << "ERROR!! --> res = " << res << " , addY = " << addY
                << " , err = " << addY - res << std::endl;
      returnVal += 2;
    }
  }

  std::cout << "UPPX ANALYSYS!!" << std::endl;
  for (int i=0; i<1; i++) {
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
  for (int i=0; i<1; i++) {
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

  std::cout << "UPPY ANALYSYS!!" << std::endl;
  for (int i=0; i<1; i++) {
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
  for (int i=0; i<1; i++) {
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

  std::cout << "SYMX ANALYSYS!!" << std::endl;
  for (int i=0; i<1; i++) {
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
  for (int i=0; i<1; i++) {
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

  res = vT[0];
#ifdef SHOW_VALUES
  std::cout << "VALUES!! --> res = " << res << " , addRng1 = " << addRng1
            << " , err = " << addRng1 - res << std::endl;
#endif  // VERBOSE
  if (std::abs((res - addRng1) / res) > ERROR_ALLOWED) {
    std::cout << "ERROR!! --> res = " << res << " , addRng1 = " << addRng1
              << " , err = " << addRng1 - res << std::endl;
    returnVal += 2;
  }

  return returnVal;
}



int main(int argc, char *argv[]) {
  //  using namespace SyclBlas;
  bool accessDev = DEFAULT_ACCESS;
  size_t sizeV = 0, divSz = 1, shftR = 0, shftC = 0;
  size_t returnVal = 0;

  if (argc == 1) {
    sizeV = DEF_NUM_ELEM;
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

  if (returnVal == 0)
    returnVal = 2 * TestingBLAS2(accessDev, sizeV, divSz, shftR, shftC);

  return returnVal;
}
