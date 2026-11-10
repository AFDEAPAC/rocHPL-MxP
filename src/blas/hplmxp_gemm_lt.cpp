/* ---------------------------------------------------------------------
 * Direct hipBLASLt path for the fp16 x fp16 -> fp32 GEMMs (HPLMXP_GEMM_LT).
 *
 * With HPLMXP_GEMM_LT unset or 0, nothing here runs and every GEMM goes through
 * rocblas_gemm_ex, which on gfx950 forwards f16 inputs to hipBLASLt itself.
 * With HPLMXP_GEMM_LT=1 the GEMM wrappers call hipblasLtMatmul on computeStream with
 * one cached plan (matmul descriptor, three layouts, algorithm) per shape; a shape the
 * library cannot serve falls back to rocblas_gemm_ex. Same precision, same operands,
 * same FP32 accumulation; only the dispatch layer and, optionally, the algorithm differ.
 *
 *   HPLMXP_LT_WS=<bytes>      workspace handed to hipBLASLt (default 0, as rocBLAS does)
 *   HPLMXP_LT_REQ=<n>         heuristic results requested per shape (default 1)
 *   HPLMXP_LT_TUNE=1          with REQ>1: time the candidates of a geometric sample of the big
 *                             shapes during the warm-up (on the warm-up matrix, which is
 *                             regenerated before the timed solve) and keep the fastest
 *   HPLMXP_LT_INDEX=<i>       force hipBLASLt solution <i> for shapes with m,n >= HPLMXP_LT_MIN
 *                             (default 32768) when the library accepts it there
 *   HPLMXP_LT_VERBOSE=1       every rank prints each plan it builds
 * Rank 0 prints one "HPLMXP_GEMM_LT:" summary line at exit.
 * ---------------------------------------------------------------------
 */

#include "hplmxp.hpp"
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <cstdlib>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct Key {
  char ta, tb;
  int  m, n, k, lda, ldb, ldc;
  bool beta0;
  bool operator<(const Key& o) const {
    return std::tie(ta, tb, m, n, k, lda, ldb, ldc, beta0) <
           std::tie(o.ta, o.tb, o.m, o.n, o.k, o.lda, o.ldb, o.ldc, o.beta0);
  }
};

struct Plan {
  bool                    ok = false;
  hipblasLtMatmulDesc_t   desc{};
  hipblasLtMatrixLayout_t La{}, Lb{}, Lc{};
  hipblasLtMatmulAlgo_t   algo{};
  int                     index = -1;
  bool                    tuned = false;
  long                    calls = 0;
};

long envl(const char* k, long d) {
  const char* v = getenv(k);
  return (v && *v) ? atol(v) : d;
}

struct State {
  bool              init = false, on = false;
  hipblasLtHandle_t lt   = nullptr;
  void*             ws   = nullptr;
  size_t            wsSize = 0;
  int               req = 1, forceIndex = 0, minMN = 32768, verbose = 0;
  bool              tune = false;
  std::map<Key, Plan> plans;
  long calls = 0, fallback = 0, planFail = 0, tuned = 0, tuneSwitched = 0, forced = 0;
  int    lastTunedIndex = -1;
  double lastTunedMN    = 0.0;
} S;

int rank0() {
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}

void init_once() {
  if(S.init) return;
  S.init = true;
  S.on   = envl("HPLMXP_GEMM_LT", 0) != 0;
  if(!S.on) return;
  S.wsSize     = (size_t)envl("HPLMXP_LT_WS", 0);
  S.req        = std::max(1L, envl("HPLMXP_LT_REQ", 1));
  S.tune       = envl("HPLMXP_LT_TUNE", 0) != 0 && S.req > 1;
  S.forceIndex = (int)envl("HPLMXP_LT_INDEX", 0);
  S.minMN      = (int)envl("HPLMXP_LT_MIN", 32768);
  S.verbose    = (int)envl("HPLMXP_LT_VERBOSE", 0);
  if(hipblasLtCreate(&S.lt) != HIPBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "HPLMXP_GEMM_LT: hipblasLtCreate failed; using rocBLAS\n");
    S.on = false;
    return;
  }
  if(S.wsSize) HIP_CHECK(hipMalloc(&S.ws, S.wsSize));
  if(rank0() == 0)
    printf("HPLMXP GEMM path: hipBLASLt direct (HPLMXP_GEMM_LT=1, ws=%zu, req=%d, tune=%d, index=%d)\n",
           S.wsSize, S.req, (int)S.tune, S.forceIndex);
}

hipblasOperation_t op(char t) { return t == 'T' ? HIPBLAS_OP_T : HIPBLAS_OP_N; }

float time_algo(const Key& key, Plan& p, const hipblasLtMatmulAlgo_t& algo, float alpha, const void* a,
                const void* b, float beta, void* c) {
  hipEvent_t e0, e1;
  HIP_CHECK(hipEventCreate(&e0));
  HIP_CHECK(hipEventCreate(&e1));
  float ms = -1.f;
  if(hipblasLtMatmul(S.lt, p.desc, &alpha, a, p.La, b, p.Lb, &beta, c, p.Lc, c, p.Lc, &algo, S.ws, S.wsSize,
                     computeStream) == HIPBLAS_STATUS_SUCCESS) {
    HIP_CHECK(hipEventRecord(e0, computeStream));
    bool ok = true;
    for(int r = 0; r < 2; ++r)
      ok = ok && hipblasLtMatmul(S.lt, p.desc, &alpha, a, p.La, b, p.Lb, &beta, c, p.Lc, c, p.Lc, &algo, S.ws,
                                 S.wsSize, computeStream) == HIPBLAS_STATUS_SUCCESS;
    HIP_CHECK(hipEventRecord(e1, computeStream));
    HIP_CHECK(hipEventSynchronize(e1));
    if(ok) HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
  }
  HIP_CHECK(hipEventDestroy(e0));
  HIP_CHECK(hipEventDestroy(e1));
  return ms;
}

// Build the plan for a shape. With tuning enabled and operand pointers given (warm-up only), the
// candidates are timed on those buffers, which the caller guarantees may be overwritten.
Plan& plan_for(const Key& key, float alpha, float beta, const void* a, const void* b, void* c) {
  auto it = S.plans.find(key);
  if(it != S.plans.end()) return it->second;
  Plan& p = S.plans[key];
  if(hipblasLtMatmulDescCreate(&p.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS) {
    ++S.planFail;
    return p;
  }
  hipblasOperation_t oa = op(key.ta), ob = op(key.tb);
  hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &oa, sizeof(oa));
  hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &ob, sizeof(ob));
  hipblasLtMatrixLayoutCreate(&p.La, HIP_R_16F, key.ta == 'T' ? key.k : key.m, key.ta == 'T' ? key.m : key.k, key.lda);
  hipblasLtMatrixLayoutCreate(&p.Lb, HIP_R_16F, key.tb == 'T' ? key.n : key.k, key.tb == 'T' ? key.k : key.n, key.ldb);
  hipblasLtMatrixLayoutCreate(&p.Lc, HIP_R_32F, key.m, key.n, key.ldc);

  hipblasLtMatmulPreference_t pref;
  hipblasLtMatmulPreferenceCreate(&pref);
  hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &S.wsSize, sizeof(S.wsSize));
  std::vector<hipblasLtMatmulHeuristicResult_t> hr(S.req);
  int nh = 0;
  hipblasLtMatmulAlgoGetHeuristic(S.lt, p.desc, p.La, p.Lb, p.Lc, p.Lc, pref, S.req, hr.data(), &nh);
  hipblasLtMatmulPreferenceDestroy(pref);
  if(nh <= 0) {
    ++S.planFail;
    return p;
  }
  p.algo  = hr[0].algo;
  p.ok    = true;
  const bool big = key.m >= S.minMN && key.n >= S.minMN;

  if(S.forceIndex && big) {
    std::vector<int> ids{S.forceIndex};
    std::vector<hipblasLtMatmulHeuristicResult_t> r;
    size_t need = 0;
    if(hipblaslt_ext::getAlgosFromIndex(S.lt, ids, r) == HIPBLAS_STATUS_SUCCESS && !r.empty() &&
       hipblaslt_ext::matmulIsAlgoSupported(S.lt, p.desc, &alpha, p.La, p.Lb, &beta, p.Lc, p.Lc, r[0].algo, need) ==
           HIPBLAS_STATUS_SUCCESS &&
       need <= S.wsSize) {
      p.algo = r[0].algo;
      ++S.forced;
    }
  } else if(S.tune && big) {
    // Time the candidates on a geometric sample of the shapes (the first big one, then each one whose
    // m*n has dropped below 70% of the last timed one); the others reuse the last winner when it is in
    // their candidate list. That keeps the warm-up to about a dozen timed shapes at 1N.
    const double mn = (double)key.m * key.n;
    if(a && b && c && (S.lastTunedMN == 0.0 || mn < 0.7 * S.lastTunedMN)) {
      S.lastTunedMN = mn;
      float best = time_algo(key, p, hr[0].algo, alpha, a, b, beta, c);
      int   bi   = 0;
      for(int i = 1; i < nh; ++i) {
        float ms = time_algo(key, p, hr[i].algo, alpha, a, b, beta, c);
        if(ms > 0.f && (best < 0.f || ms < 0.99f * best)) best = ms, bi = i;
      }
      p.algo  = hr[bi].algo;
      p.tuned = true;
      ++S.tuned;
      if(bi) ++S.tuneSwitched;
      S.lastTunedIndex = hipblaslt_ext::getIndexFromAlgo(p.algo);
    } else if(S.lastTunedIndex > 0) {
      for(int i = 0; i < nh; ++i)
        if(hipblaslt_ext::getIndexFromAlgo(hr[i].algo) == S.lastTunedIndex) p.algo = hr[i].algo;
    }
  }
  p.index = hipblaslt_ext::getIndexFromAlgo(p.algo);
  if(S.verbose)
    fprintf(stderr, "HPLMXP_GEMM_LT plan rank=%d %c%c m=%d n=%d k=%d lda=%d ldb=%d ldc=%d beta0=%d -> %d %s%s\n",
            rank0(), key.ta, key.tb, key.m, key.n, key.k, key.lda, key.ldb, key.ldc, (int)key.beta0, p.index,
            hipblaslt_ext::getKernelNameFromAlgo(S.lt, p.algo).c_str(), p.tuned ? " (tuned)" : "");
  return p;
}

} // namespace

bool HPLMXP_gemm_lt_on() {
  init_once();
  return S.on;
}

void HPLMXP_gemm_lt_plan(char ta, char tb, int m, int n, int k, float alpha, const __half* a, int lda,
                         const __half* b, int ldb, float beta, float* c, int ldc) {
  if(!HPLMXP_gemm_lt_on() || m <= 0 || n <= 0 || k <= 0) return;
  plan_for(Key{ta, tb, m, n, k, lda, ldb, ldc, beta == 0.f}, alpha, beta, a, b, c);
}

bool HPLMXP_gemm_lt(char ta, char tb, int m, int n, int k, float alpha, const __half* a, int lda,
                    const __half* b, int ldb, float beta, float* c, int ldc) {
  if(!HPLMXP_gemm_lt_on()) return false;
  if(m <= 0 || n <= 0) return true;
  Plan& p = plan_for(Key{ta, tb, m, n, k, lda, ldb, ldc, beta == 0.f}, alpha, beta, nullptr, nullptr, nullptr);
  ++S.calls;
  if(p.ok && hipblasLtMatmul(S.lt, p.desc, &alpha, a, p.La, b, p.Lb, &beta, c, p.Lc, c, p.Lc, &p.algo, S.ws,
                             S.wsSize, computeStream) == HIPBLAS_STATUS_SUCCESS) {
    ++p.calls;
    return true;
  }
  ++S.fallback;
  return false;
}

void HPLMXP_gemm_lt_free() {
  if(!S.on) return;
  std::map<int, long> byIndex;
  for(auto& kv : S.plans)
    if(kv.second.ok) byIndex[kv.second.index] += kv.second.calls;
  if(rank0() == 0) {
    printf("HPLMXP_GEMM_LT: plans=%zu calls=%ld fallback=%ld plan_fail=%ld tuned=%ld tune_switched=%ld forced=%ld\n",
           S.plans.size(), S.calls, S.fallback, S.planFail, S.tuned, S.tuneSwitched, S.forced);
    for(auto& kv : byIndex) {
      std::string name;
      for(auto& p : S.plans)
        if(p.second.ok && p.second.index == kv.first) {
          name = hipblaslt_ext::getKernelNameFromAlgo(S.lt, p.second.algo);
          break;
        }
      printf("HPLMXP_GEMM_LT:   solution %d calls %ld %s\n", kv.first, kv.second, name.c_str());
    }
  }
  for(auto& kv : S.plans) {
    if(kv.second.desc) hipblasLtMatmulDescDestroy(kv.second.desc);
    if(kv.second.La) hipblasLtMatrixLayoutDestroy(kv.second.La);
    if(kv.second.Lb) hipblasLtMatrixLayoutDestroy(kv.second.Lb);
    if(kv.second.Lc) hipblasLtMatrixLayoutDestroy(kv.second.Lc);
  }
  S.plans.clear();
  if(S.ws) HIP_CHECK(hipFree(S.ws));
  hipblasLtDestroy(S.lt);
  S.on = false;
}
