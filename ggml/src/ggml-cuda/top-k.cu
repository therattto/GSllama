#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE


// ---------------------------------------------------------------------------
//  Radix selection, one block per row.
//
//  The fallback used when CCCL has no DeviceTopK (that needs 3.2, here we have
//  3.1.4) sorts the WHOLE row and then throws away 99% of it. For the sparse
//  indexer of Qwen3.8-Flash-Next that means sorting n_blocks elements for each
//  of the n_tps rows just to take the first 513, and the scratch it needs comes
//  from the CUDA pool, so it does not show up in sched_reserve, it grows with
//  the prompt, and at ncmoe 15 it kills the process halfway through prefill with
//  an out of memory inside argsort_f32_i32_cuda_cub.
//
//  Here we look only for the threshold, that is the k-th value, with four 8-bit
//  histogram passes over the monotonic integer representation of the float.
//  Memory: 256 counters in shared, one kilobyte per block, and no global
//  scratch. At the end the k chosen positions are sorted in shared with a
//  bitonic network, because the path this replaces returned indices ordered by
//  decreasing value and not every caller of ggml_top_k promises not to rely on
//  that.
// ---------------------------------------------------------------------------

#define GGML_CUDA_TOPK_RADIX_BLOCK 256
#define GGML_CUDA_TOPK_RADIX_MAX_K 1024

// monotonic float -> uint32 mapping: the larger the float, the larger the
// integer, negative and positive zero included
static __device__ __forceinline__ uint32_t topk_f2u(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

template <bool sort_result>
static __global__ void k_top_k_radix(const float * __restrict__ x,
                                     int   * __restrict__ dst,
                                     const int ncols,
                                     const int k) {
    const int row = blockIdx.x;
    const float * __restrict__ x_row = x + (size_t) row * ncols;
    int * __restrict__ dst_row = dst + (size_t) row * k;

    __shared__ int   s_hist[256];
    __shared__ int   s_above;      // how many elements sit above the chosen bin
    __shared__ int   s_bin;        // the bin holding the k-th element
    __shared__ int   s_n_hi;       // how many already emitted as strictly greater
    __shared__ int   s_n_eq;       // how many emitted among the threshold ties
    __shared__ uint32_t s_prefix;
    __shared__ uint32_t s_mask;
    __shared__ int   s_rem;        // how many are still missing inside the current prefix

    if (threadIdx.x == 0) {
        s_prefix = 0;
        s_mask   = 0;
        s_rem    = k;
        s_n_hi   = 0;
        s_n_eq   = 0;
    }
    __syncthreads();

    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8*pass;

        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            s_hist[i] = 0;
        }
        __syncthreads();

        const uint32_t prefix = s_prefix;
        const uint32_t mask   = s_mask;

        for (int i = threadIdx.x; i < ncols; i += blockDim.x) {
            const uint32_t u = topk_f2u(x_row[i]);
            if ((u & mask) == prefix) {
                atomicAdd(&s_hist[(u >> shift) & 0xFFu], 1);
            }
        }
        __syncthreads();

        // scan from the top: the first bin that overshoots the remaining quota
        if (threadIdx.x == 0) {
            int acc = 0;
            int bin = 0;
            for (int b = 255; b >= 0; --b) {
                if (acc + s_hist[b] >= s_rem) {
                    bin = b;
                    break;
                }
                acc += s_hist[b];
            }
            s_above  = acc;
            s_bin    = bin;
            s_rem   -= acc;
            s_prefix = prefix | ((uint32_t) bin << shift);
            s_mask   = mask   | (0xFFu << shift);
        }
        __syncthreads();
    }

    const uint32_t thr = s_prefix;   // the k-th value, in integer form
    const int      n_eq_take = s_rem;
    const int      n_hi_tot  = k - n_eq_take;

    // emit: first the strictly greater ones, then as many ties at the threshold as needed
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) {
        const uint32_t u = topk_f2u(x_row[i]);
        if (u > thr) {
            const int slot = atomicAdd(&s_n_hi, 1);
            if (slot < n_hi_tot) {
                dst_row[slot] = i;
            }
        } else if (u == thr) {
            const int slot = atomicAdd(&s_n_eq, 1);
            if (slot < n_eq_take) {
                dst_row[n_hi_tot + slot] = i;
            }
        }
    }

    if (!sort_result) {
        return;
    }

    __syncthreads();

    // bitonic sort in shared of the k chosen entries only, by decreasing value
    extern __shared__ int s_dyn[];
    int   * s_idx = s_dyn;
    float * s_val = (float *) (s_dyn + GGML_CUDA_TOPK_RADIX_MAX_K);

    int kpad = 1;
    while (kpad < k) {
        kpad *= 2;
    }

    for (int i = threadIdx.x; i < kpad; i += blockDim.x) {
        if (i < k) {
            const int idx = dst_row[i];
            s_idx[i] = idx;
            s_val[i] = x_row[idx];
        } else {
            s_idx[i] = -1;
            s_val[i] = -INFINITY;
        }
    }
    __syncthreads();

    for (int len = 2; len <= kpad; len *= 2) {
        for (int step = len/2; step > 0; step /= 2) {
            for (int i = threadIdx.x; i < kpad; i += blockDim.x) {
                const int j = i ^ step;
                if (j > i) {
                    const bool up = ((i & len) == 0);          // decreasing in the "up" blocks
                    const bool sw = up ? (s_val[i] < s_val[j]) : (s_val[i] > s_val[j]);
                    if (sw) {
                        const float tv = s_val[i]; s_val[i] = s_val[j]; s_val[j] = tv;
                        const int   ti = s_idx[i]; s_idx[i] = s_idx[j]; s_idx[j] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        dst_row[i] = s_idx[i];
    }
}

static void top_k_radix_cuda(const float * x, int * dst, const int ncols, const int nrows, const int k,
                             const bool sort_result, cudaStream_t stream) {
    const dim3 block(GGML_CUDA_TOPK_RADIX_BLOCK, 1, 1);
    const dim3 grid(nrows, 1, 1);
    if (sort_result) {
        const size_t smem = GGML_CUDA_TOPK_RADIX_MAX_K*(sizeof(int) + sizeof(float));
        k_top_k_radix<true><<<grid, block, smem, stream>>>(x, dst, ncols, k);
    } else {
        k_top_k_radix<false><<<grid, block, 0, stream>>>(x, dst, ncols, k);
    }
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();

    // Radix selection allocates nothing and beats the full sort when k is much
    // smaller than ncols, which is the sparse indexer case. Below 1024 columns
    // the shared-memory bitonic already does everything without scratch, and
    // above 1024 results the final bitonic would not fit in shared, so in both
    // of those we keep the existing path. Turn it off with GGML_CUDA_TOPK_RADIX=0.
    //
    // When it pays, measured with test-backend-ops perf on this 4080. The kernel
    // uses one block per row, so its time is about 3.0 us per 1024 columns and
    // does not depend on nrows as long as the blocks fit in the SMs. The cub sort
    // instead has a fixed cost of about 80 us and then grows with the total
    // element count. The two conditions below follow from that: either there are
    // enough rows to cover cub's fixed cost, or the row is short enough to stay
    // under it. Outside those two, that is few very long rows (the sampler top-k
    // over the vocabulary), the single-block radix loses by up to 5x and we leave
    // it to cub.
    {
        static const bool radix_enabled = [] {
            const char * e = getenv("GGML_CUDA_TOPK_RADIX");
            return e == nullptr || atoi(e) != 0;
        }();

        const bool radix_conviene = nrows >= 8 || ncols <= 24576;

        if (radix_enabled && radix_conviene &&
            ncols > 1024 && k <= GGML_CUDA_TOPK_RADIX_MAX_K && k < ncols && nrows > 0) {
            top_k_radix_cuda(src0_d, dst_d, (int) ncols, (int) nrows, (int) k, /*sort_result =*/ true, stream);
            return;
        }
    }

#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
}
