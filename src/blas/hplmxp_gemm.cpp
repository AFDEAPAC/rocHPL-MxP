
#include "hplmxp.hpp"
#include <cstdlib>

template <>
void HPLMXP_gemmNT(const int     m,
                   const int     n,
                   const int     k,
                   const double  alpha,
                   const double* a,
                   const int     lda,
                   const double* b,
                   const int     ldb,
                   const double  beta,
                   double*       c,
                   const int     ldc) {
  ROCBLAS_CHECK(rocblas_gemm_ex(blas_hdl,
                                rocblas_operation_none,
                                rocblas_operation_transpose,
                                m,
                                n,
                                k,
                                &alpha,
                                a,
                                rocblas_datatype_f64_r,
                                lda,
                                b,
                                rocblas_datatype_f64_r,
                                ldb,
                                &beta,
                                c,
                                rocblas_datatype_f64_r,
                                ldc,
                                c,
                                rocblas_datatype_f64_r,
                                ldc,
                                rocblas_datatype_f64_r,
                                rocblas_gemm_algo_standard,
                                0,
                                rocblas_gemm_flags_none));
}

template <>
void HPLMXP_gemmNT(const int    m,
                   const int    n,
                   const int    k,
                   const float  alpha,
                   const float* a,
                   const int    lda,
                   const float* b,
                   const int    ldb,
                   const float  beta,
                   float*       c,
                   const int    ldc) {
  ROCBLAS_CHECK(rocblas_gemm_ex(blas_hdl,
                                rocblas_operation_none,
                                rocblas_operation_transpose,
                                m,
                                n,
                                k,
                                &alpha,
                                a,
                                rocblas_datatype_f32_r,
                                lda,
                                b,
                                rocblas_datatype_f32_r,
                                ldb,
                                &beta,
                                c,
                                rocblas_datatype_f32_r,
                                ldc,
                                c,
                                rocblas_datatype_f32_r,
                                ldc,
                                rocblas_datatype_f32_r,
                                rocblas_gemm_algo_standard,
                                0,
                                rocblas_gemm_flags_none));
}

template <>
void HPLMXP_gemmNT(const int     m,
                   const int     n,
                   const int     k,
                   const float   alpha,
                   const __half* a,
                   const int     lda,
                   const __half* b,
                   const int     ldb,
                   const float   beta,
                   float*        c,
                   const int     ldc) {
  if(HPLMXP_gemm_lt('N', 'T', m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) return;
  ROCBLAS_CHECK(rocblas_gemm_ex(blas_hdl,
                                rocblas_operation_none,
                                rocblas_operation_transpose,
                                m,
                                n,
                                k,
                                &alpha,
                                a,
                                rocblas_datatype_f16_r,
                                lda,
                                b,
                                rocblas_datatype_f16_r,
                                ldb,
                                &beta,
                                c,
                                rocblas_datatype_f32_r,
                                ldc,
                                c,
                                rocblas_datatype_f32_r,
                                ldc,
                                rocblas_datatype_f32_r,
                                rocblas_gemm_algo_standard,
                                0,
                                rocblas_gemm_flags_none));
}

template <>
void HPLMXP_gemmNT(const int     m,
                   const int     n,
                   const int     k,
                   const __half  alpha,
                   const __half* a,
                   const int     lda,
                   const __half* b,
                   const int     ldb,
                   const __half  beta,
                   __half*       c,
                   const int     ldc) {
  ROCBLAS_CHECK(rocblas_gemm_ex(blas_hdl,
                                rocblas_operation_none,
                                rocblas_operation_transpose,
                                m,
                                n,
                                k,
                                &alpha,
                                a,
                                rocblas_datatype_f16_r,
                                lda,
                                b,
                                rocblas_datatype_f16_r,
                                ldb,
                                &beta,
                                c,
                                rocblas_datatype_f16_r,
                                ldc,
                                c,
                                rocblas_datatype_f16_r,
                                ldc,
                                rocblas_datatype_f32_r,
                                rocblas_gemm_algo_standard,
                                0,
                                rocblas_gemm_flags_none));
}

bool HPLMXP_panel_tn() {
  static const bool tn = [] {
    const char* e  = getenv("HPLMXP_PANEL_TN");
    const bool  on = e && atoi(e) != 0;
    int         rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) printf("HPLMXP panel layout: %s\n", on ? "TN (HPLMXP_PANEL_TN=1)" : "NT (default)");
    return on;
  }();
  return tn;
}

template <>
void HPLMXP_gemmTN(const int     m,
                   const int     n,
                   const int     k,
                   const float   alpha,
                   const __half* a,
                   const int     lda,
                   const __half* b,
                   const int     ldb,
                   const float   beta,
                   float*        c,
                   const int     ldc) {
  if(HPLMXP_gemm_lt('T', 'N', m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) return;
  ROCBLAS_CHECK(rocblas_gemm_ex(blas_hdl,
                                rocblas_operation_transpose,
                                rocblas_operation_none,
                                m,
                                n,
                                k,
                                &alpha,
                                a,
                                rocblas_datatype_f16_r,
                                lda,
                                b,
                                rocblas_datatype_f16_r,
                                ldb,
                                &beta,
                                c,
                                rocblas_datatype_f32_r,
                                ldc,
                                c,
                                rocblas_datatype_f32_r,
                                ldc,
                                rocblas_datatype_f32_r,
                                rocblas_gemm_algo_standard,
                                0,
                                rocblas_gemm_flags_none));
}

template <>
void HPLMXP_gemmNN(const int     m,
                   const int     n,
                   const int     k,
                   const float   alpha,
                   const __half* a,
                   const int     lda,
                   const __half* b,
                   const int     ldb,
                   const float   beta,
                   float*        c,
                   const int     ldc) {
  if(HPLMXP_gemm_lt('N', 'N', m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) return;
  ROCBLAS_CHECK(rocblas_gemm_ex(blas_hdl,
                                rocblas_operation_none,
                                rocblas_operation_none,
                                m,
                                n,
                                k,
                                &alpha,
                                a,
                                rocblas_datatype_f16_r,
                                lda,
                                b,
                                rocblas_datatype_f16_r,
                                ldb,
                                &beta,
                                c,
                                rocblas_datatype_f32_r,
                                ldc,
                                c,
                                rocblas_datatype_f32_r,
                                ldc,
                                rocblas_datatype_f32_r,
                                rocblas_gemm_algo_standard,
                                0,
                                rocblas_gemm_flags_none));
}

bool HPLMXP_panel_tn2() {
  static const bool tn2 = [] {
    const char* e  = getenv("HPLMXP_PANEL_TN2");
    const bool  on = e && atoi(e) != 0;
    int         rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(on && !HPLMXP_panel_tn()) {
      if(rank == 0) printf("HPLMXP_PANEL_TN2=1 needs HPLMXP_PANEL_TN=1; ignored\n");
      return false;
    }
    if(rank == 0 && on) printf("HPLMXP TRSM GEMMs: L TN, U NN (HPLMXP_PANEL_TN2=1)\n");
    return on;
  }();
  return tn2;
}
