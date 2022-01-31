#include "sycl_blas.hpp"
#include <CL/sycl.hpp>
#include "tuner_types.hpp"

#include <ctime>
#include <chrono>

#include "util.hpp"

template <int VecSize, int Cls, typename Tile, bool DoubleBuffer, bool Nbca,
          bool Nbcb, typename Config, typename T, typename Executor_T>
void tune(int r, GemmArgs<T> a, Executor_T &ex) {
  using Gemm =
      ::blas::Gemm<MatrixContainer<T>, MatrixContainer<T>, DoubleBuffer, Nbca,
                   Nbcb, Cls, Tile, Config::TransA, Config::TransB, T, false,
                   static_cast<int>(Config::MemoryMode),
                   static_cast<int>(Config::ShapeMode),
                   static_cast<int>(Config::VecType), VecSize,
                   static_cast<int>(Config::BatchType)>;
  {
    using MilliSeconds = std::chrono::duration<double, std::milli>;
    {
      auto event_list = ex.get_policy_handler().copy_to_device(
          a.init_c.data(), a.c, a.init_c.size());
      event_list.back().wait_and_throw();
    }

    auto accA =
        ::blas::make_matrix_view<::blas::col_major>(ex, a.a, a.m, a.k, a.lda);
    auto accB =
        ::blas::make_matrix_view<::blas::col_major>(ex, a.b, a.k, a.n, a.ldb);
    auto accC =
        ::blas::make_matrix_view<::blas::col_major>(ex, a.c, a.m, a.n, a.ldc);
    auto gemm = Gemm(accA, accB, accC, a.alpha, a.beta, a.batch_size);
    const double flop_count = 2.0 * a.m * a.n * a.k * a.batch_size;


    // warmup
    for (int i=0; i<3; i++) {
      auto event_list = ex.execute(gemm);
      for (auto &event : event_list) {
        event.wait_and_throw();
      }
    }
    // time trial
    auto start = std::chrono::steady_clock::now();
    for (int i=0; i<r; i++) {
      auto event_list = ex.execute(gemm);
      for (auto &event : event_list) {
        event.wait_and_throw();
      }
    }
    auto end = std::chrono::steady_clock::now();
    auto runtime_secs = end - start;

    {
      auto event_list = ex.get_policy_handler().copy_to_host(
          a.c, a.output_c.data(), a.output_c.size());
      event_list.back().wait_and_throw();
    }

    auto seconds_per_iter = runtime_secs / r;
    auto milliseconds =
        std::chrono::duration_cast<MilliSeconds>(seconds_per_iter);
    std::cout << "," << milliseconds.count();
  }
}

#define BENCH_PARAMS(MEM, ALG, BATCH, VEC, ...)                             \
  do {                                                                      \
        tune<__VA_ARGS__, GemmConfig<0, 0, MEM, ALG, BATCH, VEC>,           \
             float>(500, args, executor);                                      \
  } while (0);

int main(int argc, char** argv) {
  /* Create a SYCL queue with the default device selector */
  cl::sycl::queue q = cl::sycl::queue(cl::sycl::default_selector());

  /* Create a SYCL-BLAS executor and get the policy handler */
  blas::Executor<blas::PolicyHandler<blas::codeplay_policy>> executor(q);
  auto policy_handler = executor.get_policy_handler();

  /* Arguments of the Gemm operation.
   * Note: these matrix dimensions are too small to get a performance gain by
   * using SYCL-BLAS, but they are convenient for this sample */
  if (argc < 4) {
    fprintf(stderr, "Usage: %s M N K\n", argv[0]);
    exit(1);
  }
  int batch_size = 36;
  int m = atoi(argv[1]);
  int k = atoi(argv[3]);
  int n = atoi(argv[2]);
  int lda = m;
  int ldb = k;
  int ldc = m;
  float alpha = 1.0;
  float beta = 1.0;

  /* Create the matrices */
  std::vector<float> A = std::vector<float>(lda * k * batch_size);
  std::vector<float> B = std::vector<float>(ldb * n * batch_size);
  std::vector<float> C = std::vector<float>(ldc * n * batch_size);
  auto C_1 = C;
  auto C_2 = C;

  /* Fill the matrices with random values */
  fill_matrix(A, m, k, lda);
  fill_matrix(B, k, n, ldb);
  fill_matrix(C, m, n, ldc);

  /* Print the matrices before the GEMM operation */
/**
  std::cout << "A:\n";
  print_matrix(A, m, k, lda);
  std::cout << "---\nB:\n";
  print_matrix(B, k, n, ldb);
  std::cout << "---\nC (before):\n";
  print_matrix(C, m, n, ldc);
**/

  /* Create the buffers */
  auto a_gpu = blas::make_sycl_iterator_buffer<float>(A.size());
  auto b_gpu = blas::make_sycl_iterator_buffer<float>(B.size());
  auto c_gpu = blas::make_sycl_iterator_buffer<float>(C.size());

  /* Copy the matrices to the device
   * Note: this sample uses explicit copy operations, see the GEMV sample for
   * an alternative way
   */
  // std::cout << "---\nCopying A, B and C to device\n";
  auto event_list = policy_handler.copy_to_device(A.data(), a_gpu, A.size());
  event_list.back().wait_and_throw();
  event_list = policy_handler.copy_to_device(B.data(), b_gpu, B.size());
  event_list.back().wait_and_throw();
  event_list = policy_handler.copy_to_device(C.data(), c_gpu, C.size());
  event_list.back().wait_and_throw();

  GemmArgs<float> args{m,        n,        k,   alpha,      a_gpu,
                          lda,      b_gpu, ldb, beta,       C,
                          c_gpu, C_1, ldc, batch_size, C_2};

  /* Execute the GEMM operation */
  // std::cout << "Executing C = " << alpha << "*A*B + " << beta << "*C\n";
  // blas::_gemm(executor, 'n', 'n', m, n, k, alpha, a_gpu, lda, b_gpu, ldb, beta,
  //             c_gpu, ldc);

#include "generated_combinations.def" 


  /* Copy the result to the host */
  // std::cout << "Copying C to host\n";
  auto event = policy_handler.copy_to_host(c_gpu, C.data(), ldc * n);
  policy_handler.wait(event);

  /* Print the result after the GEMM operation */
/**
  std::cout << "---\nC (after):" << std::endl;
  print_matrix(C, m, n, ldc);
**/
  std::cout << std::endl;

  return 0;
}

#undef BENCH_PARAMS

