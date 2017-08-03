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
#define DEF_STRIDE 1
#define ERROR_ALLOWED 1.0E-6
// #define SHOW_VALUES   1

#define EXECUTED_ON_GPU 1
#define BASETYPE double

#define SHOW_TIMES 1  // If it exists, the code prints the execution time
// The ... should be changed by the corresponding routine
#define NUMBER_REPEATS 6  // Number of times the computations are made
// If it is greater than 1, the compile time is not considered

#ifdef EXECUTED_ON_GPU
#define DEFAULT_ACCESS false
#else
#define DEFAULT_ACCESS true
#endif

// #########################
/*
int main(int argc, char *argv[]) {
  size_t numE, strd, sizeV, returnVal = 0;
  if (argc == 1) {
    numE = DEF_NUM_ELEM;
    strd = DEF_STRIDE;
  } else if (argc == 2) {
    numE = atoi(argv[1]);
    strd = DEF_STRIDE;
  } else if (argc == 3) {
    numE = atoi(argv[1]);
    strd = atoi(argv[2]);
  } else {
    std::cout << "ERROR!! --> Incorrect number of input parameters"
              << std::endl;
    returnVal = 1;
  }
  if (returnVal == 0) {
    sizeV = numE * strd;
#ifdef SHOW_TIMES
    // VARIABLES FOR TIMING
    std::chrono::time_point<std::chrono::steady_clock> t_start, t_stop;
    std::chrono::duration<BASETYPE> t0_copy, t0_axpy, t0_add;
    std::chrono::duration<BASETYPE> t1_copy, t1_axpy, t1_add;
    std::chrono::duration<BASETYPE> t2_copy, t2_axpy, t2_add;
    std::chrono::duration<BASETYPE> t3_copy, t3_axpy, t3_add;
    std::vector<std::chrono::duration<BASETYPE>> v0_copy(NUMBER_REPEATS), v0_axpy(NUMBER_REPEATS), v0_add(NUMBER_REPEATS);
    std::vector<std::chrono::duration<BASETYPE>> v1_copy(NUMBER_REPEATS), v1_axpy(NUMBER_REPEATS), v1_add(NUMBER_REPEATS);
    std::vector<std::chrono::duration<BASETYPE>> v2_copy(NUMBER_REPEATS), v2_axpy(NUMBER_REPEATS), v2_add(NUMBER_REPEATS);
    std::vector<std::chrono::duration<BASETYPE>> v3_copy(NUMBER_REPEATS), v3_axpy(NUMBER_REPEATS), v3_add(NUMBER_REPEATS);
#endif

    // CREATING DATA
    std::vector<BASETYPE> vX1(sizeV);
    std::vector<BASETYPE> vY1(sizeV);
    std::vector<BASETYPE> vZ1(sizeV);
    std::vector<BASETYPE> vS1(sizeV);
    std::vector<BASETYPE> vX2(sizeV);
    std::vector<BASETYPE> vY2(sizeV);
    std::vector<BASETYPE> vZ2(sizeV);
    std::vector<BASETYPE> vS2(sizeV);
    std::vector<BASETYPE> vX3(sizeV);
    std::vector<BASETYPE> vY3(sizeV);
    std::vector<BASETYPE> vZ3(sizeV);
    std::vector<BASETYPE> vS3(sizeV);
    std::vector<BASETYPE> vX4(sizeV);
    std::vector<BASETYPE> vY4(sizeV);
    std::vector<BASETYPE> vZ4(sizeV);
    std::vector<BASETYPE> vS4(sizeV);

    // INITIALIZING DATA
    size_t vSeed, gap;
    BASETYPE minV, maxV;

    vSeed = 1;
    minV = -10.0;
    maxV = 10.0;
    gap = (size_t)(maxV - minV + 1);
    srand(vSeed);
    std::for_each(std::begin(vX1), std::end(vX1),
                  [&](BASETYPE &elem) { elem = minV + (BASETYPE)(rand() % gap); });
    std::for_each(std::begin(vX2), std::end(vX2),
                  [&](BASETYPE &elem) { elem = minV + (BASETYPE)(rand() % gap); });
    std::for_each(std::begin(vX3), std::end(vX3),
                  [&](BASETYPE &elem) { elem = minV + (BASETYPE)(rand() % gap); });
    std::for_each(std::begin(vX4), std::end(vX4),
                  [&](BASETYPE &elem) { elem = minV + (BASETYPE)(rand() % gap); });

    vSeed = 1;
    minV = -30.0;
    maxV = 10.0;
    gap = (size_t)(maxV - minV + 1);
    srand(vSeed);
    std::for_each(std::begin(vZ1), std::end(vZ1),
                  [&](BASETYPE &elem) { elem = minV + (BASETYPE)(rand() % gap); });
    std::for_each(std::begin(vZ2), std::end(vZ2),
                  [&](BASETYPE &elem) { elem = minV + (BASETYPE)(rand() % gap); });
    std::for_each(std::begin(vZ3), std::end(vZ3),
                  [&](BASETYPE &elem) { elem = minV + (BASETYPE)(rand() % gap); });
    std::for_each(std::begin(vZ4), std::end(vZ4),
                  [&](BASETYPE &elem) { elem = minV + (BASETYPE)(rand() % gap); });

    std::for_each(std::begin(vS1), std::end(vS1),
                  [&](BASETYPE &elem) { elem = 0.0; });
    std::for_each(std::begin(vS2), std::end(vS2),
                  [&](BASETYPE &elem) { elem = 1.0; });
    std::for_each(std::begin(vS3), std::end(vS3),
                  [&](BASETYPE &elem) { elem = 2.0; });
    std::for_each(std::begin(vS4), std::end(vS4),
                  [&](BASETYPE &elem) { elem = 3.0; });

    // COMPUTING THE RESULTS
    int i;
    BASETYPE sum1 = 0.0, alpha1 = 1.1;
    BASETYPE sum2 = 0.0, alpha2 = 2.2;
    BASETYPE sum3 = 0.0, alpha3 = 3.3;
    BASETYPE sum4 = 0.0, alpha4 = 4.4;
    BASETYPE ONE = 1.0f;

    i = 0;
    std::for_each(std::begin(vY1), std::end(vY1), [&](BASETYPE &elem) {
      elem = vZ1[i] + alpha1 * vX1[i];
      if ((i % strd) == 0) sum1 += std::abs(elem);
      i++;
    });
    //    vS1[0] = sum1;
    i = 0;
    std::for_each(std::begin(vY2), std::end(vY2), [&](BASETYPE &elem) {
      elem = vZ2[i] + alpha2 * vX2[i];
      if ((i % strd) == 0) sum2 += std::abs(elem);
      i++;
    });
    //    vS2[0] = sum2;
    i = 0;
    std::for_each(std::begin(vY3), std::end(vY3), [&](BASETYPE &elem) {
      elem = vZ3[i] + alpha3 * vX3[i];
      if ((i % strd) == 0) sum3 += std::abs(elem);
      i++;
    });
    //    vS3[0] = sum3;
    i = 0;
    std::for_each(std::begin(vY4), std::end(vY4), [&](BASETYPE &elem) {
      elem = vZ4[i] + alpha4 * vX4[i];
      if ((i % strd) == 0) sum4 += std::abs(elem);
      i++;
    });
//    vS4[0] = sum4;

#ifdef SHOW_VALUES
    std::cout << "VECTORS BEFORE COMPUTATION" << std::endl;
    for (int i = 0; i < sizeV; i++) {
      std::cout << "Component = " << i << std::endl;
      std::cout << "vX = (" << vX1[i] << ", " << vX2[i] << ", " << vX3[i]
                << ", " << vX4[i] << ")" << std::endl;
      std::cout << "vY = (" << vY1[i] << ", " << vY2[i] << ", " << vY3[i]
                << ", " << vY4[i] << ")" << std::endl;
      std::cout << "vZ = (" << vZ1[i] << ", " << vZ2[i] << ", " << vZ3[i]
                << ", " << vZ4[i] << ")" << std::endl;
    }
#endif

    // CREATING THE SYCL QUEUE AND EXECUTOR
#ifdef EXECUTED_ON_GPU
    cl::sycl::gpu_selector   s;
#else
    cl::sycl::intel_selector s;
    // cl::sycl::cpu_selector s; NOOOOO
#endif
    cl::sycl::queue q(s,[=](cl::sycl::exception_list eL) {
//    cl::sycl::queue q([=](cl::sycl::exception_list eL) {
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

      cl_mem bX3_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vX3.size() * sizeof(BASETYPE), nullptr, &err);
      cl_mem bY3_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vY3.size() * sizeof(BASETYPE), nullptr, &err);

      cl_mem bX4_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vX4.size() * sizeof(BASETYPE), nullptr, &err);
      cl_mem bY4_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vY4.size() * sizeof(BASETYPE), nullptr, &err);

      cl_mem bZ1_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vZ1.size() * sizeof(BASETYPE), nullptr, &err);
      cl_mem bZ2_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vZ2.size() * sizeof(BASETYPE), nullptr, &err);
      cl_mem bZ3_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vZ3.size() * sizeof(BASETYPE), nullptr, &err);
      cl_mem bZ4_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vZ4.size() * sizeof(BASETYPE), nullptr, &err);

      cl_mem bS1_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vS1.size() * sizeof(BASETYPE), nullptr, &err);
      cl_mem bS2_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vS2.size() * sizeof(BASETYPE), nullptr, &err);
      cl_mem bS3_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vS3.size() * sizeof(BASETYPE), nullptr, &err);
      cl_mem bS4_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vS4.size() * sizeof(BASETYPE), nullptr, &err);

      cl_mem scratch_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vY1.size() * sizeof(BASETYPE), nullptr, &err);

      {
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

        err = clEnqueueWriteBuffer(clQueue, bX3_cl, CL_FALSE, 0,
                                   (vX3.size() * sizeof(BASETYPE)), vX3.data(), 0,
                                   NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueWriteBuffer(clQueue, bY3_cl, CL_FALSE, 0,
                                   (vY3.size() * sizeof(BASETYPE)), vY3.data(), 0,
                                   NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueWriteBuffer(clQueue, bX4_cl, CL_FALSE, 0,
                                   (vX4.size() * sizeof(BASETYPE)), vX4.data(), 0,
                                   NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueWriteBuffer(clQueue, bY4_cl, CL_FALSE, 0,
                                   (vY4.size() * sizeof(BASETYPE)), vY4.data(), 0,
                                   NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueWriteBuffer(clQueue, bZ1_cl, CL_FALSE, 0,
                                   (vZ1.size() * sizeof(BASETYPE)), vZ1.data(), 0,
                                   NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueWriteBuffer(clQueue, bZ2_cl, CL_FALSE, 0,
                                   (vZ2.size() * sizeof(BASETYPE)), vZ2.data(), 0,
                                   NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueWriteBuffer(clQueue, bZ3_cl, CL_FALSE, 0,
                                   (vZ3.size() * sizeof(BASETYPE)), vZ3.data(), 0,
                                   NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueWriteBuffer(clQueue, bZ4_cl, CL_FALSE, 0,
                                   (vZ4.size() * sizeof(BASETYPE)), vZ4.data(), 0,
                                   NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }
      }  // End of copy

      for (int i = 0; i < NUMBER_REPEATS; i++) {
#ifdef SHOW_TIMES
        t_start = std::chrono::steady_clock::now();
#endif
        // One copy
        {
          cl_event events[4];

          err = clblasDcopy(numE, bZ1_cl, 0, strd, bY1_cl, 0, strd, 1, &clQueue,
                            0, NULL, &events[0]);
          err = clblasDcopy(numE, bZ2_cl, 0, strd, bY2_cl, 0, strd, 1, &clQueue,
                            0, NULL, &events[1]);
          err = clblasDcopy(numE, bZ3_cl, 0, strd, bY3_cl, 0, strd, 1, &clQueue,
                            0, NULL, &events[2]);
          err = clblasDcopy(numE, bZ4_cl, 0, strd, bY4_cl, 0, strd, 1, &clQueue,
                            0, NULL, &events[3]);

          err |= clWaitForEvents(4, events);

          if (err != CL_SUCCESS) {
            std::cout << " ERROR " << err << std::endl;
          }
        }
#ifdef SHOW_TIMES
        t_stop = std::chrono::steady_clock::now();
        if (NUMBER_REPEATS == 1) {
          t0_copy = t_stop - t_start;
        } else if (i > 0) {
          t0_copy += t_stop - t_start;
        } else {
          t0_copy = t_start - t_start;
        }
        v0_copy[i] = t_stop - t_start;
#endif

#ifdef SHOW_TIMES
        t_start = std::chrono::steady_clock::now();
#endif
        /* */
        // One axpy
        {
          cl_event events[4];

          err = clblasDaxpy(numE, alpha1, bX1_cl, 0, strd, bY1_cl, 0, strd, 1,
                            &clQueue, 0, NULL, &events[0]);
          err |= clblasDaxpy(numE, alpha2, bX2_cl, 0, strd, bY2_cl, 0, strd, 1,
                             &clQueue, 0, NULL, &events[1]);
          err |= clblasDaxpy(numE, alpha3, bX3_cl, 0, strd, bY3_cl, 0, strd, 1,
                             &clQueue, 0, NULL, &events[2]);
          err |= clblasDaxpy(numE, alpha4, bX4_cl, 0, strd, bY4_cl, 0, strd, 1,
                             &clQueue, 0, NULL, &events[3]);

          err |= clWaitForEvents(4, events);

          if (err != CL_SUCCESS) {
            std::cout << " ERROR " << err << std::endl;
          }
        }
/* */
#ifdef SHOW_TIMES
        t_stop = std::chrono::steady_clock::now();
        if (NUMBER_REPEATS == 1) {
          t0_axpy = t_stop - t_start;
        } else if (i > 0) {
          t0_axpy += t_stop - t_start;
        } else {
          t0_axpy = t_start - t_start;
        }
        v0_axpy[i] = t_stop - t_start;
#endif

#ifdef SHOW_TIMES
        t_start = std::chrono::steady_clock::now();
#endif
        // One add
        {
          cl_event events[4];
          err = clblasDasum(numE,
                            //              bY1_cl, 0,
                            //              bS1_cl, 0, 1,
                            bS1_cl, 0, bY1_cl, 0, strd, scratch_cl, 1, &clQueue,
                            0, NULL, &events[0]);
          err = clblasDasum(numE,
                            //              bY2_cl, 0,
                            //              bS2_cl, 0, 1,
                            bS2_cl, 0, bY2_cl, 0, strd, scratch_cl, 1, &clQueue,
                            0, NULL, &events[1]);
          err = clblasDasum(numE,
                            //              bY3_cl, 0,
                            //              bS3_cl, 0, 1,
                            bS3_cl, 0, bY3_cl, 0, strd, scratch_cl, 1, &clQueue,
                            0, NULL, &events[2]);
          err = clblasDasum(numE,
                            //              bY4_cl, 0,
                            //              bS4_cl, 0, 1,
                            bS4_cl, 0, bY4_cl, 0, strd, scratch_cl, 1, &clQueue,
                            0, NULL, &events[3]);

          err |= clWaitForEvents(4, events);

          if (err != CL_SUCCESS) {
            std::cout << __LINE__ << ": ERROR " << err << std::endl;
          }
        }  // End of copy
#ifdef SHOW_TIMES
        t_stop = std::chrono::steady_clock::now();
        if (NUMBER_REPEATS == 1) {
          t0_add = t_stop - t_start;
        } else if (i > 0) {
          t0_add += t_stop - t_start;
        } else {
          t0_add = t_start - t_start;
        }
        v0_add[i] = t_stop - t_start;
#endif
      }

      {
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

        err = clEnqueueReadBuffer(clQueue, bX3_cl, CL_FALSE, 0,
                                  (vX3.size() * sizeof(BASETYPE)), vX3.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bY3_cl, CL_FALSE, 0,
                                  (vY3.size() * sizeof(BASETYPE)), vY3.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bX4_cl, CL_FALSE, 0,
                                  (vX4.size() * sizeof(BASETYPE)), vX4.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bY4_cl, CL_FALSE, 0,
                                  (vY4.size() * sizeof(BASETYPE)), vY4.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bZ1_cl, CL_FALSE, 0,
                                  (vZ1.size() * sizeof(BASETYPE)), vZ1.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bZ2_cl, CL_FALSE, 0,
                                  (vZ2.size() * sizeof(BASETYPE)), vZ2.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bZ3_cl, CL_FALSE, 0,
                                  (vZ3.size() * sizeof(BASETYPE)), vZ3.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bZ4_cl, CL_FALSE, 0,
                                  (vZ4.size() * sizeof(BASETYPE)), vZ4.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bS1_cl, CL_FALSE, 0,
                                  (vS1.size() * sizeof(BASETYPE)), vS1.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bS2_cl, CL_FALSE, 0,
                                  (vS2.size() * sizeof(BASETYPE)), vS2.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bS3_cl, CL_FALSE, 0,
                                  (vS3.size() * sizeof(BASETYPE)), vS3.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bS4_cl, CL_FALSE, 0,
                                  (vS4.size() * sizeof(BASETYPE)), vS4.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

      }  // End of enqueue
      clFinish(clQueue);

      clReleaseMemObject(bX1_cl);
      clReleaseMemObject(bY1_cl);
      clReleaseMemObject(bX2_cl);
      clReleaseMemObject(bY2_cl);
      clReleaseMemObject(bX3_cl);
      clReleaseMemObject(bY3_cl);
      clReleaseMemObject(bX4_cl);
      clReleaseMemObject(bY4_cl);
      clReleaseMemObject(bZ1_cl);
      clReleaseMemObject(bZ2_cl);
      clReleaseMemObject(bZ3_cl);
      clReleaseMemObject(bZ4_cl);
      clReleaseMemObject(bS1_cl);
      clReleaseMemObject(bS2_cl);
      clReleaseMemObject(bS3_cl);
      clReleaseMemObject(bS4_cl);

      clblasTeardown();
    }

#ifdef SHOW_VALUES
    std::cout << "VECTORS AFTER  COMPUTATION" << std::endl;
    for (int i = 0; i < sizeV; i++) {
      std::cout << "Component = " << i << std::endl;
      std::cout << "vX = (" << vX1[i] << ", " << vX2[i] << ", " << vX3[i]
                << ", " << vX4[i] << ")" << std::endl;
      std::cout << "vY = (" << vY1[i] << ", " << vY2[i] << ", " << vY3[i]
                << ", " << vY4[i] << ")" << std::endl;
      std::cout << "vZ = (" << vZ1[i] << ", " << vZ2[i] << ", " << vZ3[i]
                << ", " << vZ4[i] << ")" << std::endl;
    }
#endif

#ifdef SHOW_TIMES
    int div = (NUMBER_REPEATS == 1)? 1: (NUMBER_REPEATS-1);
    // COMPUTATIONAL TIMES
    std::cout << "t_copy , " << t0_copy.count()/div << std::endl;
    //    std::cout <<   "t_copy --> (" << t0_copy.count()/div << ", " <<
    //    t1_copy.count()/div
    //                          << ", " << t2_copy.count()/div << ", " <<
    //                          t3_copy.count()/div << ")" << std::endl;
    std::cout << "t_axpy , " << t0_axpy.count()/div << std::endl;
    //    std::cout <<   "t_axpy --> (" << t0_axpy.count()/div << ", " <<
    //    t1_axpy.count()/div
    //                          << ", " << t2_axpy.count()/div << ", " <<
    //                          t3_axpy.count()/div << ")" << std::endl;
    std::cout << "t_add  , " << t0_add.count()/div << std::endl;
//    std::cout <<   "t_add  --> (" << t0_add.count()/div  << ", " << t1_add.count()/div
//                          << ", " << t2_add.count()/div  << ", " << t3_add.count()/div
//                          << ")" << std::endl;
//

    std::sort (v0_copy.begin()+1, v0_copy.end());
//    std::sort (v1_copy.begin()+1, v1_copy.end());
//    std::sort (v2_copy.begin()+1, v2_copy.end());
//    std::sort (v3_copy.begin()+1, v3_copy.end());
    std::cout << "m_copy , " << v0_copy[(NUMBER_REPEATS+1)/2].count()
//              << ", "        << v1_copy[(NUMBER_REPEATS+1)/2].count()
//              << ", "        << v2_copy[(NUMBER_REPEATS+1)/2].count()
//              << ", "        << v3_copy[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
    std::sort (v0_axpy.begin()+1, v0_axpy.end());
//    std::sort (v1_axpy.begin()+1, v1_axpy.end());
//    std::sort (v2_axpy.begin()+1, v2_axpy.end());
//    std::sort (v3_axpy.begin()+1, v3_axpy.end());
    std::cout << "m_axpy , " << v0_axpy[(NUMBER_REPEATS+1)/2].count()
//              << ", "        << v1_axpy[(NUMBER_REPEATS+1)/2].count()
//              << ", "        << v2_axpy[(NUMBER_REPEATS+1)/2].count()
//              << ", "        << v3_axpy[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
    std::sort (v0_add.begin()+1, v0_add.end());
//    std::sort (v1_add.begin()+1, v1_add.end());
//    std::sort (v2_add.begin()+1, v2_add.end());
//    std::sort (v3_add.begin()+1, v3_add.end());
    std::cout << "m_add  , " << v0_add[(NUMBER_REPEATS+1)/2].count()
//              << ", "        << v1_add[(NUMBER_REPEATS+1)/2].count()
//              << ", "        << v2_add[(NUMBER_REPEATS+1)/2].count()
//              << ", "        << v3_add[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
#endif

    // ANALYSIS OF THE RESULTS
    BASETYPE res;

    for (i = 0; i < 0; i++) {
      res = vS1[i];
#ifdef SHOW_VALUES
      std::cout << "VALUES!! --> res = " << res << " , i = " << i
                << " , sum = " << sum1 << " , err = " << res - sum1
                << std::endl;
#endif  //  SHOW_VALUES
      if (std::abs((res - sum1) / res) > ERROR_ALLOWED) {
        std::cout << "VALUES!! --> res = " << res << " , i = " << i
                  << " , sum = " << sum1 << " , err = " << res - sum1
                  << std::endl;
        returnVal += 2 * i;
      }
    }

    for (i = 0; i < 0; i++) {
      res = vS2[i];
#ifdef SHOW_VALUES
      std::cout << "VALUES!! --> res = " << res << " , i = " << i
                << " , sum = " << sum2 << " , err = " << res - sum2
                << std::endl;
#endif  //  SHOW_VALUES
      if (std::abs((res - sum2) / res) > ERROR_ALLOWED) {
        std::cout << "VALUES!! --> res = " << res << " , i = " << i
                  << " , sum = " << sum2 << " , err = " << res - sum2
                  << std::endl;
        returnVal += 20 * i;
      }
    }

    for (i = 0; i < 0; i++) {
      res = vS3[i];
#ifdef SHOW_VALUES
      std::cout << "VALUES!! --> res = " << res << " , i = " << i
                << " , sum = " << sum3 << " , err = " << res - sum3
                << std::endl;
#endif  //  SHOW_VALUES
      if (std::abs((res - sum3) / res) > ERROR_ALLOWED) {
        std::cout << "VALUES!! --> res = " << res << " , i = " << i
                  << " , sum = " << sum3 << " , err = " << res - sum3
                  << std::endl;
        returnVal += 200 * i;
      }
    }

    for (i = 0; i < 0; i++) {
      res = vS4[i];
#ifdef SHOW_VALUES
      std::cout << "VALUES!! --> res = " << res << " , i = " << i
                << " , sum = " << sum4 << " , err = " << res - sum4
                << std::endl;
#endif  //  SHOW_VALUES
      if (std::abs((res - sum4) / res) > ERROR_ALLOWED) {
        std::cout << "VALUES!! --> res = " << res << " , i = " << i
                  << " , sum = " << sum4 << " , err = " << res - sum4
                  << std::endl;
        returnVal += 2000 * i;
      }
    }
  }

  return returnVal;
}
*/
// TESTING ROUTINE

size_t TestingBLAS2(bool accessDev, size_t dim, size_t divSz, size_t shftR,
                    size_t shftC) {
  // CREATING DATA
  clblasOrder clOrder=(accessDev)? clblasRowMajor: clblasColumnMajor;
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
#ifdef SHOW_TIMES
  std::chrono::time_point<std::chrono::steady_clock> t_start, t_stop;
  std::chrono::duration<BASETYPE> t1_gmvR, t1_gmvC, t1_ger;
  std::chrono::duration<BASETYPE> t2_gmvR, t2_gmvC, t2_ger;
  std::chrono::duration<BASETYPE> t3_gmvR, t3_gmvC, t3_ger;
  std::vector<std::chrono::duration<BASETYPE>> v1_gmvR(NUMBER_REPEATS), v1_gmvC(NUMBER_REPEATS), v1_ger(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v2_gmvR(NUMBER_REPEATS), v2_gmvC(NUMBER_REPEATS), v2_ger(NUMBER_REPEATS);
  std::vector<std::chrono::duration<BASETYPE>> v3_gmvR(NUMBER_REPEATS), v3_gmvC(NUMBER_REPEATS), v3_ger(NUMBER_REPEATS);
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
    auxY = 1.5 * vY1[i - shftR];
//    printf ("(AB) %f = 1.5 * %f\n", auxY, vY2[i - shftC]);
    for (size_t j = shftC; j < dimC; j++) {
      if (accessDev) {
//        vY2[i - shftR] += 2.0 * vM[dimC * i + j] * vY[j - shftC];
//        printf ("(A) %f += 2.0 * %f * %f\n", auxY, vM[dimC * i + j], vY[j - shftC]);
        auxY += 2.0 * vM1[dimC * i + j] * vY0[j - shftC];
      } else {
//        vY2[i - shftR] += 2.0 * vM[dimR * j + i] * vY[j - shftC];
//        printf ("(B) %f += 2.0 * %f * %f\n", auxY, vM[dimR * j + i], vY[j - shftC]);
        auxY += 2.0 * vM1[dimR * j + i] * vY0[j - shftC];
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
    auxX = 0.5 * vX1[j - shftC];
    for (size_t i = shftR; i < dimR; i++) {
      if (accessDev) {
//        vX2[j - shftC] += 2.5 * vM[dimC * i + j] * vX[i - shftR];
        auxX += 2.5 * vM1[dimC * i + j] * vX0[i - shftR];
      } else {
//        vX2[j - shftC] += 2.5 * vM[dimR * j + i] * vX[i - shftR];
        auxX += 2.5 * vM1[dimR * j + i] * vX0[i - shftR];
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
      addRng1 += (accessDev) ? vM1[dimC * i + j] : vM1[dimR * j + i];
      if ((i >= shftR) && (j >= shftC)) {
        addRng1 += 3.0 * vY0[i - shftR] * vX0[j - shftC];
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
//      _gemv<1, SYCL>(ex, "Tr", dimC - shftC, dimR - shftR, 2.5, bmM0(shftR, shftC),
//                      dimL, bvX0, 1, 0.5, bvX2, 1);
//      q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDgemv (clOrder, clblasTrans, dimC, dimR, 2.5, bM0_cl, 0, dimL,
                          bX0_cl, 0, 1, 0.5, bX2_cl, 0, 1, 1, &clQueue, 0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);
        if (err != CL_SUCCESS) {
          std::cout << " ERROR " << err << std::endl;
        }
      }
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
//      auto reducOpX1 = make_addAssignReduction(bvR1, bvX2, 256, 256);
//      ex.reduce(reducOpX1); q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDasum(dimC,
                          bR_cl, 0, bX2_cl, 0, strd, scratchX_cl, 1, &clQueue,
                          0, NULL, &events[0]);
        err |= clWaitForEvents(4, events);

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
//      _gemv<1, SYCL>(ex, "No", dimR - shftR, dimC - shftC, 2.0, bmM0(shftR, shftC),
//                  dimL, bvY0, 1, 1.5, bvY2, 1);
//      q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDgemv (clOrder, clblasNoTrans, dimR, dimC, 2.0, bM0_cl, 0, dimL,
                          bY0_cl, 0, 1, 1.5, bY2_cl, 0, 1, 1, &clQueue, 0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);
        if (err != CL_SUCCESS) {
          std::cout << " ERROR " << err << std::endl;
        }
      }
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
//      auto reducOpY1 = make_addAssignReduction(bvS1, bvY2, 256, 256);
//      ex.reduce(reducOpY1); q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDasum(dimR,
                          bS_cl, 0, bY2_cl, 0, strd, scratchY_cl, 1, &clQueue,
                          0, NULL, &events[0]);
        err |= clWaitForEvents(4, events);

        if (err != CL_SUCCESS) {
          std::cout << __LINE__ << ": ERROR " << err << std::endl;
        }
      }  // End of copy
      /*****************************************/
  #ifdef SHOW_TIMES
      t_start = std::chrono::steady_clock::now();
  #endif
//      _ger<SYCL>(ex, dimR - shftR, dimC - shftC, 3.0, bvY0, 1, bvX0, 1,
//                 bmM0(shftR, shftC), dimL);
//      q.wait_and_throw();
      {
        cl_event events[1];
        err = clblasDger (clOrder, dimR, dimC, 3.0, bY0_cl, 0, 1, bX0_cl, 0, 1,
                          bM0_cl, 0, dimL, 1, &clQueue, 0, NULL, &events[0]);
        err |= clWaitForEvents(1, events);
        if (err != CL_SUCCESS) {
          std::cout << " ERROR " << err << std::endl;
        }
      }
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
//      auto reducOpV = make_addAssignReduction(bvT, bvV0, 256, 256);
//      ex.reduce(reducOpV); q.wait_and_throw();
      {
        cl_event events[1];

        err = clblasDasum(dimR*dimC,
                          bT_cl, 0, bM0_cl, 0, strd, scratchY_cl, 1, &clQueue,
                          0, NULL, &events[0]);
        err |= clWaitForEvents(4, events);

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
    }
    clFinish(clQueue);

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

    clblasTeardown();
  }

#ifdef SHOW_TIMES
    int div = (NUMBER_REPEATS == 1)? 1: (NUMBER_REPEATS-1);
//    int div = 1;
    // COMPUTATIONAL TIMES
    std::cout << "t_gemvR , " << t1_gmvR.count()/div
//              << ", " << t2_gmvR.count()/div
//              << ", " << t3_gmvR.count()/div
              << std::endl;
    std::cout << "t_gemvC , " << t1_gmvC.count()/div
//              << ", " << t2_gmvC.count()/div
//              << ", " << t3_gmvC.count()/div
              << std::endl;
    std::cout << "t_ger   , " << t1_ger.count()/div << std::endl;
    std::sort (v1_gmvR.begin()+1, v1_gmvR.end());
//    std::sort (v2_gmvR.begin()+1, v2_gmvR.end());
//    std::sort (v3_gmvR.begin()+1, v3_gmvR.end());
    std::cout << "m_gmvC , " << v1_gmvR[(NUMBER_REPEATS+1)/2].count()
//              << ", "        << v2_gmvR[(NUMBER_REPEATS+1)/2].count()
//              << ", "        << v3_gmvR[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
    std::sort (v1_gmvC.begin()+1, v1_gmvC.end());
//    std::sort (v2_gmvC.begin()+1, v2_gmvC.end());
//    std::sort (v3_gmvC.begin()+1, v3_gmvC.end());
    std::cout << "m_gmvC , " << v1_gmvC[(NUMBER_REPEATS+1)/2].count()
//              << ", "        << v2_gmvC[(NUMBER_REPEATS+1)/2].count()
//              << ", "        << v3_gmvC[(NUMBER_REPEATS+1)/2].count()
              << std::endl;
    std::sort (v1_ger.begin()+1, v1_ger.end());
    std::cout << "m_gmvC , " << v1_ger[(NUMBER_REPEATS+1)/2].count()
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

  if (returnVal == 0)
    returnVal = 2 * TestingBLAS2(accessDev, sizeV, divSz, shftR, shftC);

  return returnVal;
}
