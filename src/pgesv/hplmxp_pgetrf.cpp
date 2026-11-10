/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    HPL - 2.2 - February 24, 2016
 *    Antoine P. Petitet
 *    University of Tennessee, Knoxville
 *    Innovative Computing Laboratory
 *    (C) Copyright 2000-2008 All Rights Reserved
 *
 *    Modified by: Noel Chalmers
 *    (C) 2018-2022 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

#include "hplmxp.hpp"

void HPLMXP_pgetrf(HPLMXP_T_grid&         grid,
                   HPLMXP_T_palg&         algo,
                   HPLMXP_T_pmat<fp32_t>& A) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_pgetrf factors a N-by-N matrix using LU factorization.  The
   * main algorithm  is the "right looking" variant with  look-ahead.
   *
   * Arguments
   * =========
   *
   * GRID    (local input)                 HPLMXP_T_grid *
   *         On entry,  GRID  points  to the data structure containing the
   *         process grid information.
   *
   * ALGO    (global input)                HPLMXP_T_palg *
   *         On entry,  ALGO  points to  the data structure containing the
   *         algorithmic parameters.
   *
   * A       (local input/output)          HPLMXP_T_pmat *
   *         On entry, A points to the data structure containing the local
   *         array information.
   *
   * ---------------------------------------------------------------------
   */

  fp32_t*   Ap      = A.A;
  int const N       = A.n;
  int const b       = A.nb;
  int const nblocks = N / b;
  int const nbrow   = A.nbrow;
  int const nbcol   = A.nbcol;
  int const lda     = A.ld;

  int const myrow = grid.myrow;
  int const mycol = grid.mycol;
  int const nprow = grid.nprow;
  int const npcol = grid.npcol;

  const int ldpiv = b;

  const fp32_t one   = 1.0;
  const fp32_t alpha = -1.0;
  const fp32_t beta  = 1.0;

  // HPLMXP_PANEL_TN=1: keep the fp16 L and U panels K-contiguous (L^T is b x mp, U is b x nq, both with
  // leading dimension b) so every update GEMM is op(A)=T, op(B)=N. The matrix, the fp32 factors in A and the
  // broadcast sizes are unchanged; only the panel copies and the GEMM operand layout differ.
  const bool tn = HPLMXP_panel_tn();
  // HPLMXP_PANEL_TN2=1: the first L copy is also L^T (b x mp) and the first U copy U (b x nq), so the L "TRSM"
  // GEMM is TN against U11^-1 and the U one NN against L11^-1. pivU then holds U11^-1 untransposed; lacpyU
  // writes only its upper triangle, so the strict lower part is reset to zero here.
  const bool tn2 = HPLMXP_panel_tn2();
  if(tn2) HPLMXP_identity(b, A.pivU, ldpiv);
  auto Lp = [&](HPLMXP_T_panel<fp32_t>& P, int rb) -> fp16_t* {
    return tn ? P.L + static_cast<size_t>(rb) * b * b : Mptr(P.L, rb * b, 0, P.ldl);
  };
  auto Up = [&](HPLMXP_T_panel<fp32_t>& P, int cb) -> fp16_t* {
    return tn ? P.U + static_cast<size_t>(cb) * b * b : Mptr(P.U, cb * b, 0, P.ldu);
  };
  auto ldL = [&](HPLMXP_T_panel<fp32_t>& P) { return tn ? b : P.ldl; };
  auto ldU = [&](HPLMXP_T_panel<fp32_t>& P) { return tn ? b : P.ldu; };
  auto upd = [&](int m, int n, const fp16_t* l, int ldl, const fp16_t* u, int ldu, fp32_t* c) {
    if(tn)
      HPLMXP_gemmTN(m, n, b, alpha, l, ldl, u, ldu, beta, c, lda);
    else
      HPLMXP_gemmNT(m, n, b, alpha, l, ldl, u, ldu, beta, c, lda);
  };

  int i = 0, j = 0;
  int ip1 = i + (((1 % grid.nprow) == grid.myrow) ? 1 : 0);
  int jp1 = j + (((1 % grid.npcol) == grid.mycol) ? 1 : 0);

#ifdef HPLMXP_PROGRESS_REPORT
#ifdef HPLMXP_DETAILED_TIMING
  float  DgemmTime, LgemmTime, UgemmTime, TgemmTime;
  double DgemmGflops, LgemmGflops, UgemmGflops, TgemmGflops;

  if(grid.myrow == 0 && grid.mycol == 0) {
    printf("-------------------------------------------------------------------"
           "-------------------------------------------------------------------"
           "------------------------------\n");
    printf("   %%   | Column    | Step Time (s) ||               GEMM GFLOPS   "
           "            || Dbcast (s) | Lbcast (s) | Ubcast (s) |  GPU Sync "
           "(s) | Step GFLOPS | Overall GFLOPS\n");
    printf("       |           |               | Diagonal |    L     |    U    "
           " | Trailing |"
           "            |            |            |               |            "
           " |               \n");
    printf("-------------------------------------------------------------------"
           "-------------------------------------------------------------------"
           "------------------------------\n");
  }
#else
  if(grid.myrow == 0 && grid.mycol == 0) {
    printf("---------------------------------------------------\n");
    printf("   %%   | Column    | Step Time (s) | Overall GFLOPS\n");
    printf("       |           |               |               \n");
    printf("---------------------------------------------------\n");
  }
#endif
#endif

  /* start time */
  double start_time = MPI_Wtime();

  i = 0, j = 0;

  for(int k = 0; k < nblocks; ++k) {
    HPLMXP_ptimer_stepReset(HPLMXP_TIMING_N, HPLMXP_TIMING_BEG);

    const int n = N - k * b;

    double stepStart = MPI_Wtime();

    HPLMXP_T_panel<fp32_t>& prev = A.panels[k % 2];
    HPLMXP_T_panel<fp32_t>& next = A.panels[(k + 1) % 2];

    int const rootrow = k % grid.nprow;
    int const rootcol = k % grid.npcol;

    const bool icurrow = (rootrow == grid.myrow);
    const bool icurcol = (rootcol == grid.mycol);

    ip1 = i + ((icurrow) ? 1 : 0);
    jp1 = j + ((icurcol) ? 1 : 0);

    HPLMXP_pdpanel_init(
        grid, A, n - b, b, (k + 1) * b, (k + 1) * b, ip1 * b, jp1 * b, next);

    if(icurrow && icurcol) {
      /* Update look-ahead */
      if(k > 0) {
        HIP_CHECK(hipEventRecord(DgemmStart, computeStream));
        upd(b, b, prev.L, ldL(prev), prev.U, ldU(prev), Mptr(Ap, i * b, j * b, lda));
        HIP_CHECK(hipEventRecord(DgemmEnd, computeStream));
      }

      /* Factor panel */
      HPLMXP_lacpy(b, b, Mptr(Ap, i * b, j * b, lda), lda, A.piv, ldpiv);
      HPLMXP_getrf(b, b, A.piv, ldpiv);
      HPLMXP_lacpy(b, b, A.piv, ldpiv, Mptr(Ap, i * b, j * b, lda), lda);
      HPLMXP_trtriU(b, A.piv, ldpiv);
      HPLMXP_trtriL(b, A.piv, ldpiv);

      /* Record */
      HIP_CHECK(hipEventRecord(getrf, computeStream));
    }

    if(icurcol) {
      /* gemm L */
      if(k > 0) {
        HIP_CHECK(hipEventRecord(LgemmStart, computeStream));
        upd(A.mp - ip1 * b, b, Lp(prev, ip1 - i), ldL(prev), prev.U, ldU(prev),
            Mptr(Ap, ip1 * b, j * b, lda));
        HIP_CHECK(hipEventRecord(LgemmEnd, computeStream));
      }

      if(tn2)
        HPLMXP_latcpy(b, A.mp - ip1 * b, Mptr(Ap, ip1 * b, j * b, lda), lda, next.L, b);
      else
        HPLMXP_lacpy(A.mp - ip1 * b,
                     b,
                     Mptr(Ap, ip1 * b, j * b, lda),
                     lda,
                     next.L,
                     next.ldl);

      /*recieve piv*/
      if(!icurrow) {
        HPLMXP_TracingPush("D Column Bcast");
        HPLMXP_ptimer(HPLMXP_TIMING_DBCAST);
        HPLMXP_bcast(A.piv, ldpiv * b, rootrow, grid.col_comm, algo.btopo);
        HPLMXP_ptimer(HPLMXP_TIMING_DBCAST);
        HPLMXP_TracingPop("D Column Bcast");
      }

      if(tn2) {
        HPLMXP_lacpyU(b, b, A.piv, ldpiv, A.pivU, ldpiv);
        HPLMXP_gemmTN(A.mp - ip1 * b, b, b, one, next.L, b, A.pivU, ldpiv, fp32_t{0.0},
                      Mptr(Ap, ip1 * b, j * b, lda), lda);
      } else {
        HPLMXP_latcpyU(b, b, A.piv, ldpiv, A.pivU, ldpiv);

        HPLMXP_gemmNT(A.mp - ip1 * b,
                      b,
                      b,
                      one,
                      next.L,
                      next.ldl,
                      A.pivU,
                      ldpiv,
                      fp32_t{0.0},
                      Mptr(Ap, ip1 * b, j * b, lda),
                      lda);
      }

      if(tn)
        HPLMXP_latcpy(b, A.mp - ip1 * b, Mptr(Ap, ip1 * b, j * b, lda), lda, next.L, b);
      else
        HPLMXP_lacpy(A.mp - ip1 * b,
                     b,
                     Mptr(Ap, ip1 * b, j * b, lda),
                     lda,
                     next.L,
                     next.ldl);

      /* Record */
      HIP_CHECK(hipEventRecord(lbcast, computeStream));
    }

    if(icurrow) {
      /* gemm U */
      if(k > 0) {
        HIP_CHECK(hipEventRecord(UgemmStart, computeStream));
        upd(b, A.nq - jp1 * b, prev.L, ldL(prev), Up(prev, jp1 - j), ldU(prev),
            Mptr(Ap, i * b, jp1 * b, lda));
        HIP_CHECK(hipEventRecord(UgemmEnd, computeStream));
      }

      if(tn2)
        HPLMXP_lacpy(b, A.nq - jp1 * b, Mptr(Ap, i * b, jp1 * b, lda), lda, next.U, b);
      else
        HPLMXP_latcpy(A.nq - jp1 * b,
                      b,
                      Mptr(Ap, i * b, jp1 * b, lda),
                      lda,
                      next.U,
                      next.ldu);

      /*recieve piv*/
      if(!icurcol) {
        HPLMXP_TracingPush("D Row Bcast");
        HPLMXP_ptimer(HPLMXP_TIMING_DBCAST);
        HPLMXP_bcast(A.piv, ldpiv * b, rootcol, grid.row_comm, algo.btopo);
        HPLMXP_ptimer(HPLMXP_TIMING_DBCAST);
        HPLMXP_TracingPop("D Row Bcast");
      }

      HPLMXP_lacpyL(b, b, A.piv, ldpiv, A.pivL, ldpiv);

      if(tn2)
        HPLMXP_gemmNN(b, A.nq - jp1 * b, b, one, A.pivL, ldpiv, next.U, b, fp32_t{0.0},
                      Mptr(Ap, i * b, jp1 * b, lda), lda);
      else
        HPLMXP_gemmNT(b,
                      A.nq - jp1 * b,
                      b,
                      one,
                      A.pivL,
                      ldpiv,
                      next.U,
                      next.ldu,
                      fp32_t{0.0},
                      Mptr(Ap, i * b, jp1 * b, lda),
                      lda);

      if(tn)
        HPLMXP_lacpy(b, A.nq - jp1 * b, Mptr(Ap, i * b, jp1 * b, lda), lda, next.U, b);
      else
        HPLMXP_latcpy(A.nq - jp1 * b,
                      b,
                      Mptr(Ap, i * b, jp1 * b, lda),
                      lda,
                      next.U,
                      next.ldu);

      /* Record */
      HIP_CHECK(hipEventRecord(ubcast, computeStream));
    }

    /* Trailing update */
    if(k > 0) {
      HIP_CHECK(hipEventRecord(TgemmStart, computeStream));
      upd(A.mp - ip1 * b, A.nq - jp1 * b, Lp(prev, ip1 - i), ldL(prev), Up(prev, jp1 - j), ldU(prev),
          Mptr(Ap, ip1 * b, jp1 * b, lda));
      HIP_CHECK(hipEventRecord(TgemmEnd, computeStream));
    }

    if(icurrow && icurcol) {
      HPLMXP_ptimer(HPLMXP_TIMING_UPDATE);
      HIP_CHECK(hipEventSynchronize(getrf));
      HPLMXP_ptimer(HPLMXP_TIMING_UPDATE);

      /* broadcast piv */
      HPLMXP_TracingPush("D Row Bcast");
      HPLMXP_ptimer(HPLMXP_TIMING_DBCAST);
      HPLMXP_bcast(A.piv, ldpiv * b, rootcol, grid.row_comm, algo.btopo);
      HPLMXP_ptimer(HPLMXP_TIMING_DBCAST);
      HPLMXP_TracingPop("D Row Bcast");

      /* broadcast piv */
      HPLMXP_TracingPush("D Column Bcast");
      HPLMXP_ptimer(HPLMXP_TIMING_DBCAST);
      HPLMXP_bcast(A.piv, ldpiv * b, rootrow, grid.col_comm, algo.btopo);
      HPLMXP_ptimer(HPLMXP_TIMING_DBCAST);
      HPLMXP_TracingPop("D Column Bcast");
    }

    /* broadcast left panel */
    if(icurcol) {
      HPLMXP_ptimer(HPLMXP_TIMING_UPDATE);
      HIP_CHECK(hipEventSynchronize(lbcast));
      HPLMXP_ptimer(HPLMXP_TIMING_UPDATE);
    }
    HPLMXP_TracingPush("L Bcast");
    HPLMXP_ptimer(HPLMXP_TIMING_LBCAST);
    HPLMXP_bcast(next.L, next.ldl * b, rootcol, grid.row_comm, algo.btopo);
    HPLMXP_ptimer(HPLMXP_TIMING_LBCAST);
    HPLMXP_TracingPop("L Bcast");

    /* broadcast right panel */
    if(icurrow) {
      HPLMXP_ptimer(HPLMXP_TIMING_UPDATE);
      HIP_CHECK(hipEventSynchronize(ubcast));
      HPLMXP_ptimer(HPLMXP_TIMING_UPDATE);
    }
    HPLMXP_TracingPush("U Bcast");
    HPLMXP_ptimer(HPLMXP_TIMING_UBCAST);
    HPLMXP_bcast(next.U, next.ldu * b, rootrow, grid.col_comm, algo.btopo);
    HPLMXP_ptimer(HPLMXP_TIMING_UBCAST);
    HPLMXP_TracingPop("U Bcast");

    /* wait here for the updates to compete */
    HPLMXP_ptimer(HPLMXP_TIMING_UPDATE);
    HIP_CHECK(hipDeviceSynchronize());
    HPLMXP_ptimer(HPLMXP_TIMING_UPDATE);

    double stepEnd = MPI_Wtime();

#ifdef HPLMXP_PROGRESS_REPORT
#ifdef HPLMXP_DETAILED_TIMING
    DgemmTime = 0.0;
    if(k > 0 && icurrow && icurcol) {
      HIP_CHECK(hipEventElapsedTime(&DgemmTime, DgemmStart, DgemmEnd));
      DgemmGflops = (2.0 * b * b * b) / (1.0e6 * (DgemmTime));
    }
    LgemmTime = 0.0;
    if(k > 0 && icurcol && (nbrow - ip1 > 0)) {
      HIP_CHECK(hipEventElapsedTime(&LgemmTime, LgemmStart, LgemmEnd));
      LgemmGflops = (2.0 * b * b * b * (nbrow - ip1)) / (1.0e6 * (LgemmTime));
    }
    UgemmTime = 0.0;
    if(k > 0 && icurrow && (nbcol - jp1 > 0)) {
      HIP_CHECK(hipEventElapsedTime(&UgemmTime, UgemmStart, UgemmEnd));
      UgemmGflops = (2.0 * b * b * b * (nbcol - jp1)) / (1.0e6 * (UgemmTime));
    }
    TgemmTime = 0.0;
    if(k > 0 && (nbrow - ip1 > 0) && (nbcol - jp1 > 0)) {
      HIP_CHECK(hipEventElapsedTime(&TgemmTime, TgemmStart, TgemmEnd));
      TgemmGflops = (2.0 * b * b * (nbrow - ip1) * b * (nbcol - jp1)) /
                    (1.0e6 * (TgemmTime));
    }
#endif
    /* if this is process 0,0 and not the first panel */
    if(grid.myrow == 0 && grid.mycol == 0 && k > 0) {
      double time      = HPLMXP_ptimer_walltime() - start_time;
      double step_time = stepEnd - stepStart;
      /*
      Step FLOP count is (2/3)NB^3 - (1/2)NB^2 - (1/6)NB
                          + 2*n*NB^2 - n*NB + 2*NB*n^2

      Overall FLOP count is (2/3)(N^3-n^3) - (1/2)(N^2-n^2) - (1/6)(N-n)
      */
      double step_gflops =
          ((2.0 / 3.0) * b * b * b - (1.0 / 2.0) * b * b - (1.0 / 6.0) * b +
           2.0 * n * b * b - b * n + 2.0 * b * n * n) /
          (step_time > 0.0 ? step_time : 1.e-6) / 1.e9;
      double gflops = ((2.0 / 3.0) * (N * (double)N * N - n * (double)n * n) -
                       (1.0 / 2.0) * (N * (double)N - n * (double)n) -
                       (1.0 / 6.0) * ((double)N - (double)n)) /
                      (time > 0.0 ? time : 1.e-6) / 1.e9;
      printf("%5.1f%% | %09d | ", k * b * 100.0 / N, k * b);
      printf("   %9.7f  |", stepEnd - stepStart);

#ifdef HPLMXP_DETAILED_TIMING
      if(icurrow && icurcol) {
        printf(" %9.3e|", DgemmGflops);
      } else {
        printf("          |");
      }
      if(icurcol && (nbrow - ip1 > 0)) {
        printf(" %9.3e|", LgemmGflops);
      } else {
        printf("          |");
      }
      if(icurrow && (nbcol - jp1 > 0)) {
        printf(" %9.3e|", UgemmGflops);
      } else {
        printf("          |");
      }
      if((nbrow - ip1 > 0) && (nbcol - jp1 > 0)) {
        printf(" %9.3e|", TgemmGflops);
      } else {
        printf("          |");
      }

      if(icurrow || icurcol) {
        printf("  %9.3e |", HPLMXP_ptimer_getStep(HPLMXP_TIMING_DBCAST));
      } else {
        printf("            |");
      }

      printf("  %9.3e |  %9.3e |    %9.3e  |",
             HPLMXP_ptimer_getStep(HPLMXP_TIMING_LBCAST),
             HPLMXP_ptimer_getStep(HPLMXP_TIMING_UBCAST),
             HPLMXP_ptimer_getStep(HPLMXP_TIMING_UPDATE));

      printf("  %9.3e  |", step_gflops);
#endif

      printf("    %9.3e   \n", gflops);
    }
#endif

    i = ip1;
    j = jp1;
  }
}
