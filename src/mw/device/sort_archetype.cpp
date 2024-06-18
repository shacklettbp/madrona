#pragma once

#include <madrona/taskgraph.hpp>
#include <madrona/mw_gpu/megakernel_consts.hpp>

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <madrona/mw_gpu/host_print.hpp>

namespace madrona {

namespace sortConsts {
inline constexpr int RADIX_BITS = 8;
inline constexpr int RADIX_DIGITS = 1 << RADIX_BITS;
inline constexpr int ALIGN_BYTES = 256;
inline constexpr int MAX_NUM_PASSES =
    (sizeof(uint32_t) * 8 + RADIX_BITS - 1) / RADIX_BITS;
}

// Copied and modified from AgentRadixSortOnesweep in CUB to support megakernel
/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
using namespace cub;

/**
 * Radix-rank using matching which computes the counts of keys for each digit
 * value early, at the expense of doing more work. This may be useful e.g. for
 * decoupled look-back, where it reduces the time other thread blocks need to
 * wait for digit counts to become available.
 */
template <int BLOCK_DIM_X, int RADIX_BITS, bool IS_DESCENDING,
          BlockScanAlgorithm INNER_SCAN_ALGORITHM = BLOCK_SCAN_WARP_SCANS,
          WarpMatchAlgorithm MATCH_ALGORITHM = WARP_MATCH_ANY, int NUM_PARTS = 1>
struct BlockRadixRankMatchEarlyCountsCustom
{
    // constants
    enum
    {
        BLOCK_THREADS = BLOCK_DIM_X,
        RADIX_DIGITS = 1 << RADIX_BITS,
        BINS_PER_THREAD = (RADIX_DIGITS + BLOCK_THREADS - 1) / BLOCK_THREADS,
        BINS_TRACKED_PER_THREAD = BINS_PER_THREAD,
        FULL_BINS = BINS_PER_THREAD * BLOCK_THREADS == RADIX_DIGITS,
        WARP_THREADS = CUB_PTX_WARP_THREADS,
        PARTIAL_WARP_THREADS = BLOCK_THREADS % WARP_THREADS,
        BLOCK_WARPS = BLOCK_THREADS / WARP_THREADS,
        PARTIAL_WARP_ID = BLOCK_WARPS - 1,
        WARP_MASK = ~0,
        NUM_MATCH_MASKS = MATCH_ALGORITHM == WARP_MATCH_ATOMIC_OR ? BLOCK_WARPS : 0,
        // Guard against declaring zero-sized array:
        MATCH_MASKS_ALLOC_SIZE = NUM_MATCH_MASKS < 1 ? 1 : NUM_MATCH_MASKS,
    };

    // types
    typedef cub::BlockScan<int, BLOCK_THREADS, INNER_SCAN_ALGORITHM> BlockScan;

    

    // temporary storage
    struct TempStorage
    {
        union
        {
            int warp_offsets[BLOCK_WARPS][RADIX_DIGITS];
            int warp_histograms[BLOCK_WARPS][RADIX_DIGITS][NUM_PARTS];
        };

        int match_masks[MATCH_MASKS_ALLOC_SIZE][RADIX_DIGITS];

        typename BlockScan::TempStorage prefix_tmp;
    };

    TempStorage& temp_storage;

    // internal ranking implementation
    template <typename UnsignedBits, int KEYS_PER_THREAD, typename DigitExtractorT,
              typename CountsCallback>
    struct BlockRadixRankMatchInternal
    {
        TempStorage& s;
        DigitExtractorT digit_extractor;
        CountsCallback callback;
        int warp;
        int lane;

        __device__ __forceinline__ int Digit(UnsignedBits key)
        {
            int digit =  digit_extractor.Digit(key);
            return IS_DESCENDING ? RADIX_DIGITS - 1 - digit : digit;
        }

        __device__ __forceinline__ int ThreadBin(int u)
        {
            int bin = threadIdx.x * BINS_PER_THREAD + u;
            return IS_DESCENDING ? RADIX_DIGITS - 1 - bin : bin;
        }

        __device__ __forceinline__
        void ComputeHistogramsWarp(UnsignedBits (&keys)[KEYS_PER_THREAD])
        {
            //int* warp_offsets = &s.warp_offsets[warp][0];
            int (&warp_histograms)[RADIX_DIGITS][NUM_PARTS] = s.warp_histograms[warp];
            // compute warp-private histograms
            #pragma unroll
            for (int bin = lane; bin < RADIX_DIGITS; bin += WARP_THREADS)
            {
                #pragma unroll
                for (int part = 0; part < NUM_PARTS; ++part)
                {
                    warp_histograms[bin][part] = 0;
                }
            }
            if (MATCH_ALGORITHM == WARP_MATCH_ATOMIC_OR)
            {
                int* match_masks = &s.match_masks[warp][0];
                #pragma unroll
                for (int bin = lane; bin < RADIX_DIGITS; bin += WARP_THREADS)
                {
                    match_masks[bin] = 0;
                }                    
            }
            WARP_SYNC(WARP_MASK);

            // compute private per-part histograms
            int part = lane % NUM_PARTS;
            #pragma unroll
            for (int u = 0; u < KEYS_PER_THREAD; ++u)
            {
                atomicAdd(&warp_histograms[Digit(keys[u])][part], 1);
            }
            
            // sum different parts;
            // no extra work is necessary if NUM_PARTS == 1
            if (NUM_PARTS > 1)
            {
                WARP_SYNC(WARP_MASK);
                // TODO: handle RADIX_DIGITS % WARP_THREADS != 0 if it becomes necessary
                const int WARP_BINS_PER_THREAD = RADIX_DIGITS / WARP_THREADS;
                int bins[WARP_BINS_PER_THREAD];
                #pragma unroll
                for (int u = 0; u < WARP_BINS_PER_THREAD; ++u)
                {
                    int bin = lane + u * WARP_THREADS;
                    bins[u] = internal::ThreadReduce(warp_histograms[bin], Sum());
                }
                CTA_SYNC();

                // store the resulting histogram in shared memory
                int* warp_offsets = &s.warp_offsets[warp][0];
                #pragma unroll
                for (int u = 0; u < WARP_BINS_PER_THREAD; ++u)
                {
                    int bin = lane + u * WARP_THREADS;
                    warp_offsets[bin] = bins[u];
                }
            }
        }

        __device__ __forceinline__
        void ComputeOffsetsWarpUpsweep(int (&bins)[BINS_PER_THREAD])
        {
            // sum up warp-private histograms
            #pragma unroll
            for (int u = 0; u < BINS_PER_THREAD; ++u) 
            {
                bins[u] = 0;
                int bin = ThreadBin(u);
                if (FULL_BINS || (bin >= 0 && bin < RADIX_DIGITS))
                {
                    #pragma unroll
                    for (int j_warp = 0; j_warp < BLOCK_WARPS; ++j_warp)
                    {
                        int warp_offset = s.warp_offsets[j_warp][bin];
                        s.warp_offsets[j_warp][bin] = bins[u];
                        bins[u] += warp_offset;
                    }
                }
            }
        }

        __device__ __forceinline__
        void ComputeOffsetsWarpDownsweep(int (&offsets)[BINS_PER_THREAD])
        {
            #pragma unroll
            for (int u = 0; u < BINS_PER_THREAD; ++u)
            {
                int bin = ThreadBin(u);
                if (FULL_BINS || (bin >= 0 && bin < RADIX_DIGITS))
                {
                    int digit_offset = offsets[u];
                    #pragma unroll
                    for (int j_warp = 0; j_warp < BLOCK_WARPS; ++j_warp)
                    {
                        s.warp_offsets[j_warp][bin] += digit_offset;
                    }
                }
            }
        }

        __device__ __forceinline__
        void ComputeRanksItem(
            UnsignedBits (&keys)[KEYS_PER_THREAD], int (&ranks)[KEYS_PER_THREAD],
            Int2Type<WARP_MATCH_ATOMIC_OR>)
        {
            // compute key ranks
            int lane_mask = 1 << lane;
            int* warp_offsets = &s.warp_offsets[warp][0];
            int* match_masks = &s.match_masks[warp][0];
            #pragma unroll
            for (int u = 0; u < KEYS_PER_THREAD; ++u)
            {
                int bin = Digit(keys[u]);
                int* p_match_mask = &match_masks[bin];
                atomicOr(p_match_mask, lane_mask);
                WARP_SYNC(WARP_MASK);
                int bin_mask = *p_match_mask;
                int leader = (WARP_THREADS - 1) - __clz(bin_mask);
                int warp_offset = 0;
                int popc = __popc(bin_mask & LaneMaskLe());
                if (lane == leader)
                {
                    // atomic is a bit faster
                    warp_offset = atomicAdd(&warp_offsets[bin], popc);
                }
                warp_offset = SHFL_IDX_SYNC(warp_offset, leader, WARP_MASK);
                if (lane == leader) *p_match_mask = 0;
                WARP_SYNC(WARP_MASK);
                ranks[u] = warp_offset + popc - 1;
            }
        }

        __device__ __forceinline__
        void ComputeRanksItem(
            UnsignedBits (&keys)[KEYS_PER_THREAD], int (&ranks)[KEYS_PER_THREAD],
            Int2Type<WARP_MATCH_ANY>)
        {
            // compute key ranks
            int* warp_offsets = &s.warp_offsets[warp][0];
            #pragma unroll
            for (int u = 0; u < KEYS_PER_THREAD; ++u)
            {
                int bin = Digit(keys[u]);
                int bin_mask = detail::warp_in_block_matcher_t<RADIX_BITS,
                                                               PARTIAL_WARP_THREADS,
                                                               BLOCK_WARPS - 1>::match_any(bin,
                                                                                           warp);
                int leader = (WARP_THREADS - 1) - __clz(bin_mask);
                int warp_offset = 0;
                int popc = __popc(bin_mask & LaneMaskLe());
                if (lane == leader)
                {
                    // atomic is a bit faster
                    warp_offset = atomicAdd(&warp_offsets[bin], popc);
                }
                warp_offset = SHFL_IDX_SYNC(warp_offset, leader, WARP_MASK);
                ranks[u] = warp_offset + popc - 1;
            }
        }

        __device__ __forceinline__ bool RankKeys(
            UnsignedBits (&keys)[KEYS_PER_THREAD],
            int (&ranks)[KEYS_PER_THREAD],
            int (&exclusive_digit_prefix)[BINS_PER_THREAD])
        {
            ComputeHistogramsWarp(keys);
            
            CTA_SYNC();
            int bins[BINS_PER_THREAD];
            ComputeOffsetsWarpUpsweep(bins);
            bool early_out = callback(bins);
            if (early_out) {
                return true;
            }
            
            BlockScan(s.prefix_tmp).ExclusiveSum(bins, exclusive_digit_prefix);

            ComputeOffsetsWarpDownsweep(exclusive_digit_prefix);
            CTA_SYNC();
            ComputeRanksItem(keys, ranks, Int2Type<MATCH_ALGORITHM>());

            return false;
        }

        __device__ __forceinline__ BlockRadixRankMatchInternal
        (TempStorage& temp_storage, DigitExtractorT digit_extractor, CountsCallback callback)
            : s(temp_storage), digit_extractor(digit_extractor),
              callback(callback), warp(threadIdx.x / WARP_THREADS), lane(LaneId())
            {}
    };

    __device__ __forceinline__ BlockRadixRankMatchEarlyCountsCustom
    (TempStorage& temp_storage) : temp_storage(temp_storage) {}

    /**
     * \brief Rank keys.  For the lower \p RADIX_DIGITS threads, digit counts for each digit are provided for the corresponding thread.
     */
    template <typename UnsignedBits, int KEYS_PER_THREAD, typename DigitExtractorT,
        typename CountsCallback>
    __device__ __forceinline__ bool RankKeys(
        UnsignedBits    (&keys)[KEYS_PER_THREAD],
        int             (&ranks)[KEYS_PER_THREAD],
        DigitExtractorT digit_extractor,
        int             (&exclusive_digit_prefix)[BINS_PER_THREAD],
        CountsCallback  callback)
    {
        BlockRadixRankMatchInternal<UnsignedBits, KEYS_PER_THREAD, DigitExtractorT, CountsCallback>
            internal(temp_storage, digit_extractor, callback);
        return internal.RankKeys(keys, ranks, exclusive_digit_prefix);        
    }

    template <typename UnsignedBits, int KEYS_PER_THREAD, typename DigitExtractorT>
    __device__ __forceinline__ void RankKeys(
        UnsignedBits    (&keys)[KEYS_PER_THREAD],
        int             (&ranks)[KEYS_PER_THREAD],
        DigitExtractorT digit_extractor,
        int             (&exclusive_digit_prefix)[BINS_PER_THREAD])
    {
        typedef BlockRadixRankEmptyCallback<BINS_PER_THREAD> CountsCallback;
        BlockRadixRankMatchInternal<UnsignedBits, KEYS_PER_THREAD, DigitExtractorT, CountsCallback>
            internal(temp_storage, digit_extractor, CountsCallback());
        return internal.RankKeys(keys, ranks, exclusive_digit_prefix);
    }

    template <typename UnsignedBits, int KEYS_PER_THREAD, typename DigitExtractorT>
    __device__ __forceinline__ void RankKeys(
        UnsignedBits    (&keys)[KEYS_PER_THREAD],
        int             (&ranks)[KEYS_PER_THREAD],
        DigitExtractorT digit_extractor)
    {
        int exclusive_digit_prefix[BINS_PER_THREAD];
        return RankKeys(keys, ranks, digit_extractor, exclusive_digit_prefix);
    }
};

struct SortArchetypeNodeBase::RadixSortOnesweepCustom {
    using OffsetT = int32_t;
    using AtomicOffsetT = int32_t;
    using PortionOffsetT = int32_t;
    using KeyT = uint32_t;
    using ValueT = int32_t;
    // constants
    enum
    {
        ITEMS_PER_THREAD = num_elems_per_sort_thread_,
        BLOCK_THREADS = consts::numMegakernelThreads,
        RANK_NUM_PARTS = 1,
        TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
        RADIX_BITS = 8,
        RADIX_DIGITS = 1 << RADIX_BITS,        
        BINS_PER_THREAD = (RADIX_DIGITS + BLOCK_THREADS - 1) / BLOCK_THREADS,
        FULL_BINS = BINS_PER_THREAD * BLOCK_THREADS == RADIX_DIGITS,
        WARP_THREADS = 32,
        BLOCK_WARPS = BLOCK_THREADS / WARP_THREADS,
        WARP_MASK = ~0,
        LOOKBACK_PARTIAL_MASK = 1 << (PortionOffsetT(sizeof(PortionOffsetT)) * 8 - 2),
        LOOKBACK_GLOBAL_MASK = 1 << (PortionOffsetT(sizeof(PortionOffsetT)) * 8 - 1),
        LOOKBACK_KIND_MASK = LOOKBACK_PARTIAL_MASK | LOOKBACK_GLOBAL_MASK,
        LOOKBACK_VALUE_MASK = ~LOOKBACK_KIND_MASK,
    };

    using UnsignedBits = uint32_t;
  
    static const RadixRankAlgorithm RANK_ALGORITHM =
        RADIX_RANK_MATCH_EARLY_COUNTS_ANY;
    static const BlockScanAlgorithm SCAN_ALGORITHM =
        BLOCK_SCAN_RAKING_MEMOIZE;

    typedef RadixSortTwiddle<false, KeyT> Twiddle;

    static_assert(RANK_ALGORITHM == RADIX_RANK_MATCH
                  || RANK_ALGORITHM == RADIX_RANK_MATCH_EARLY_COUNTS_ANY
                  || RANK_ALGORITHM == RADIX_RANK_MATCH_EARLY_COUNTS_ATOMIC_OR,
        "for onesweep agent, the ranking algorithm must warp-strided key arrangement");

    using BlockRadixRankT = BlockRadixRankMatchEarlyCountsCustom<
        BLOCK_THREADS,
        RADIX_BITS,
        false,
        SCAN_ALGORITHM,
        WARP_MATCH_ANY,
        RANK_NUM_PARTS>;

    // temporary storage
    struct TempStorage_
    {
        union
        {
            UnsignedBits keys_out[TILE_ITEMS];
            ValueT values_out[TILE_ITEMS];
            typename BlockRadixRankT::TempStorage rank_temp_storage;
        };
        union
        {
            OffsetT global_offsets[RADIX_DIGITS];
            PortionOffsetT block_idx;
        };
    };

    using TempStorage = Uninitialized<TempStorage_>;

    // thread variables
    TempStorage_& s;

    // kernel parameters
    AtomicOffsetT* d_lookback;
    AtomicOffsetT* d_ctrs;
    OffsetT* d_bins_out;
    const OffsetT*  d_bins_in;
    UnsignedBits* d_keys_out;
    const UnsignedBits* d_keys_in;
    ValueT* d_values_out;
    const ValueT* d_values_in;
    PortionOffsetT num_items;
    ShiftDigitExtractor<KeyT> digit_extractor;

    // other thread variables
    int warp;
    int lane;
    PortionOffsetT block_idx;
    bool full_block;

    // helper methods
    __device__ __forceinline__ int Digit(UnsignedBits key)
    {
        return digit_extractor.Digit(key);
    }

    __device__ __forceinline__ int ThreadBin(int u)
    {
        return threadIdx.x * BINS_PER_THREAD + u;
    }

    __device__ __forceinline__ void LookbackPartial(int (&bins)[BINS_PER_THREAD])
    {
        #pragma unroll
        for (int u = 0; u < BINS_PER_THREAD; ++u) 
        {
            int bin = ThreadBin(u);
            if (FULL_BINS || bin < RADIX_DIGITS)
            {
                // write the local sum into the bin
                AtomicOffsetT& loc = d_lookback[block_idx * RADIX_DIGITS + bin];
                PortionOffsetT value = bins[u] | LOOKBACK_PARTIAL_MASK;
                ThreadStore<STORE_VOLATILE>(&loc, value);
            }
        }
    }

    struct CountsCallback
    {
        RadixSortOnesweepCustom& agent;
        int (&bins)[BINS_PER_THREAD];
        UnsignedBits (&keys)[ITEMS_PER_THREAD];
        static const bool EMPTY = false;
        __device__ __forceinline__ CountsCallback(
                RadixSortOnesweepCustom& agent, int (&bins)[BINS_PER_THREAD],
                UnsignedBits (&keys)[ITEMS_PER_THREAD])
            : agent(agent), bins(bins), keys(keys) {}
        __device__ __forceinline__ bool operator()(int (&other_bins)[BINS_PER_THREAD])
        {
            #pragma unroll
            for (int u = 0; u < BINS_PER_THREAD; ++u)
            {
                bins[u] = other_bins[u];
            }
            agent.LookbackPartial(bins);

            return agent.TryShortCircuit(keys, bins);
        }
    };
  
    __device__ __forceinline__ void LookbackGlobal(int (&bins)[BINS_PER_THREAD])
    {
        #pragma unroll
        for (int u = 0; u < BINS_PER_THREAD; ++u)
        {
            int bin = ThreadBin(u);
            if (FULL_BINS || bin < RADIX_DIGITS)
            {
                PortionOffsetT inc_sum = bins[u];
                int want_mask = ~0;
                // backtrack as long as necessary
                for (PortionOffsetT block_jdx = block_idx - 1; block_jdx >= 0; --block_jdx)
                {
                    // wait for some value to appear
                    PortionOffsetT value_j = 0;
                    AtomicOffsetT& loc_j = d_lookback[block_jdx * RADIX_DIGITS + bin];
                    do {
                        __threadfence_block(); // prevent hoisting loads from loop
                        value_j = ThreadLoad<LOAD_VOLATILE>(&loc_j);
                    } while (value_j == 0);

                    inc_sum += value_j & LOOKBACK_VALUE_MASK;
                    want_mask = WARP_BALLOT((value_j & LOOKBACK_GLOBAL_MASK) == 0, want_mask);
                    if (value_j & LOOKBACK_GLOBAL_MASK) break;
                }
                AtomicOffsetT& loc_i = d_lookback[block_idx * RADIX_DIGITS + bin];
                PortionOffsetT value_i = inc_sum | LOOKBACK_GLOBAL_MASK;
                ThreadStore<STORE_VOLATILE>(&loc_i, value_i);
                s.global_offsets[bin] += inc_sum - bins[u];
            }
        }
    }

    __device__ __forceinline__
    void LoadKeys(OffsetT tile_offset, UnsignedBits (&keys)[ITEMS_PER_THREAD])
    {
        if (full_block)
        {
            LoadDirectWarpStriped(threadIdx.x, d_keys_in + tile_offset, keys);
        }
        else
        {
            LoadDirectWarpStriped(threadIdx.x, d_keys_in + tile_offset, keys,
                                  num_items - tile_offset, Twiddle::DefaultKey());
        }

        #pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u)
        {
            keys[u] = Twiddle::In(keys[u]);
        }
    }

    __device__ __forceinline__
    void LoadValues(OffsetT tile_offset, ValueT (&values)[ITEMS_PER_THREAD])
    {
        if (full_block)
        {
            LoadDirectWarpStriped(threadIdx.x, d_values_in + tile_offset, values);
        }
        else
        {
            int tile_items = num_items - tile_offset;
            LoadDirectWarpStriped(threadIdx.x, d_values_in + tile_offset, values,
                                  tile_items);
        }
    }

    /** Checks whether "short-circuiting" is possible. Short-circuiting happens
     * if all TILE_ITEMS keys fall into the same bin, i.e. have the same digit
     * value (note that it only happens for full tiles). If short-circuiting is
     * performed, the part of the ranking algorithm after the CountsCallback, as
     * well as the rest of the sorting (e.g. scattering keys and values to
     * shared and global memory) are skipped; updates related to decoupled
     * look-back are still performed. Instead, the keys assigned to the current
     * thread block are written cooperatively into a contiguous location in
     * d_keys_out corresponding to their digit. The values (if also sorting
     * values) assigned to the current thread block are similarly copied from
     * d_values_in to d_values_out. */
    __device__ __forceinline__
    bool TryShortCircuit(UnsignedBits (&keys)[ITEMS_PER_THREAD], int (&bins)[BINS_PER_THREAD])
    {
        // check if any bin can be short-circuited
        bool short_circuit = false;
        #pragma unroll
        for (int u = 0; u < BINS_PER_THREAD; ++u)
        {
            if (FULL_BINS || ThreadBin(u) < RADIX_DIGITS)
            {
                short_circuit = short_circuit || bins[u] == TILE_ITEMS;
            }
        }
        short_circuit = CTA_SYNC_OR(short_circuit);
        if (!short_circuit) return false;

        ShortCircuitCopy(keys, bins);

        return true;
    }

    __device__ __forceinline__
    void ShortCircuitCopy(UnsignedBits (&keys)[ITEMS_PER_THREAD], int (&bins)[BINS_PER_THREAD])
    {
        // short-circuit handling; note that global look-back is still required

        // compute offsets
        int common_bin = Digit(keys[0]);
        int offsets[BINS_PER_THREAD];
        #pragma unroll
        for (int u = 0; u < BINS_PER_THREAD; ++u)
        {
            int bin = ThreadBin(u);
            offsets[u] = bin > common_bin ? TILE_ITEMS : 0;
        }

        // global lookback
        LoadBinsToOffsetsGlobal(offsets);
        LookbackGlobal(bins);
        UpdateBinsGlobal(bins, offsets);
        CTA_SYNC();

        // scatter the keys
        OffsetT global_offset = s.global_offsets[common_bin];
        #pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u)
        {
            keys[u] = Twiddle::Out(keys[u]);
        }
        if (full_block)
        {
            StoreDirectWarpStriped(threadIdx.x, d_keys_out + global_offset, keys);
        }
        else
        {
            int tile_items = num_items - block_idx * TILE_ITEMS;
            StoreDirectWarpStriped(threadIdx.x, d_keys_out + global_offset, keys,
                                   tile_items);
        }

        // gather and scatter the values
        ValueT values[ITEMS_PER_THREAD];
        LoadValues(block_idx * TILE_ITEMS, values);
        if (full_block)
        {
            StoreDirectWarpStriped(threadIdx.x, d_values_out + global_offset, values);
        }
        else
        {
            int tile_items = num_items - block_idx * TILE_ITEMS;
            StoreDirectWarpStriped(threadIdx.x, d_values_out + global_offset, values,
                                   tile_items);
        }
    }

    __device__ __forceinline__
    void ScatterKeysShared(UnsignedBits (&keys)[ITEMS_PER_THREAD], int (&ranks)[ITEMS_PER_THREAD])
    {
        // write to shared memory
        #pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u)
        {
            s.keys_out[ranks[u]] = keys[u];
        }
    }

    __device__ __forceinline__
    void ScatterValuesShared(ValueT (&values)[ITEMS_PER_THREAD], int (&ranks)[ITEMS_PER_THREAD])
    {
        // write to shared memory
        #pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u)
        {
            s.values_out[ranks[u]] = values[u];
        }
    }

    __device__ __forceinline__ void LoadBinsToOffsetsGlobal(int (&offsets)[BINS_PER_THREAD])
    {
        // global offset - global part
        #pragma unroll
        for (int u = 0; u < BINS_PER_THREAD; ++u)
        {
            int bin = ThreadBin(u);
            if (FULL_BINS || bin < RADIX_DIGITS)
            {
                s.global_offsets[bin] = d_bins_in[bin] - offsets[u];
            }
        }        
    }

    __device__ __forceinline__ void UpdateBinsGlobal(int (&bins)[BINS_PER_THREAD],
                                                     int (&offsets)[BINS_PER_THREAD])
    {
        bool last_block = (block_idx + 1) * TILE_ITEMS >= num_items;
        if (d_bins_out != NULL && last_block)
        {
            #pragma unroll
            for (int u = 0; u < BINS_PER_THREAD; ++u)
            {
                int bin = ThreadBin(u);
                if (FULL_BINS || bin < RADIX_DIGITS)
                {
                    d_bins_out[bin] = s.global_offsets[bin] + offsets[u] + bins[u];
                }
            }
        }
    }

    template <bool FULL_TILE>
    __device__ __forceinline__ void ScatterKeysGlobalDirect()
    {
        int tile_items = FULL_TILE ? TILE_ITEMS : num_items - block_idx * TILE_ITEMS;
        #pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u)
        {
            int idx = threadIdx.x + u * BLOCK_THREADS;
            UnsignedBits key = s.keys_out[idx];
            OffsetT global_idx = idx + s.global_offsets[Digit(key)];
            if (FULL_TILE || idx < tile_items)
            {
                d_keys_out[global_idx] = Twiddle::Out(key);
            }
            WARP_SYNC(WARP_MASK);
        }
    }

    template <bool FULL_TILE>
    __device__ __forceinline__ void ScatterValuesGlobalDirect(int (&digits)[ITEMS_PER_THREAD])
    {
        int tile_items = FULL_TILE ? TILE_ITEMS : num_items - block_idx * TILE_ITEMS;
        #pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u)
        {
            int idx = threadIdx.x + u * BLOCK_THREADS;
            ValueT value = s.values_out[idx];
            OffsetT global_idx = idx + s.global_offsets[digits[u]];
            if (FULL_TILE || idx < tile_items) d_values_out[global_idx] = value;
            WARP_SYNC(WARP_MASK);
        }
    }

    __device__ __forceinline__ void ScatterKeysGlobalAligned()
    {
        // this only works with full tiles
        const int ITEMS_PER_WARP = TILE_ITEMS / BLOCK_WARPS;
        const int ALIGN = 8;
        const auto CACHE_MODIFIER = STORE_CG;
        
        int warp_start = warp * ITEMS_PER_WARP;
        int warp_end = (warp + 1) * ITEMS_PER_WARP;
        int warp_offset = warp_start;
        while (warp_offset < warp_end - WARP_THREADS)
        {
            int idx = warp_offset + lane;
            UnsignedBits key = s.keys_out[idx];
            UnsignedBits key_out = Twiddle::Out(key);
            OffsetT global_idx = idx + s.global_offsets[Digit(key)];
            int last_lane = WARP_THREADS - 1;
            int num_writes = WARP_THREADS;
            if (lane == last_lane)
            {
                num_writes -= int(global_idx + 1) % ALIGN;
            }
            num_writes = SHFL_IDX_SYNC(num_writes, last_lane, WARP_MASK);
            if (lane < num_writes)
            {
                ThreadStore<CACHE_MODIFIER>(&d_keys_out[global_idx], key_out);
            }
            warp_offset += num_writes;
        }
        {
            int num_writes = warp_end - warp_offset;
            if (lane < num_writes)
            {
                int idx = warp_offset + lane;
                UnsignedBits key = s.keys_out[idx];
                OffsetT global_idx = idx + s.global_offsets[Digit(key)];
                ThreadStore<CACHE_MODIFIER>(&d_keys_out[global_idx], Twiddle::Out(key));
            }
        }
    }

    __device__ __forceinline__ void ScatterKeysGlobal()
    {
        // write block data to global memory
        if (full_block)
        {
            ScatterKeysGlobalDirect<true>();
        }
        else
        {
            ScatterKeysGlobalDirect<false>();
        }
    }

    __device__ __forceinline__ void ScatterValuesGlobal(int (&digits)[ITEMS_PER_THREAD])
    {
        // write block data to global memory
        if (full_block)
        {
            ScatterValuesGlobalDirect<true>(digits);
        }
        else
        {
            ScatterValuesGlobalDirect<false>(digits);
        }
    }

    __device__ __forceinline__ void ComputeKeyDigits(int (&digits)[ITEMS_PER_THREAD])
    {
        #pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u)
        {
            int idx = threadIdx.x + u * BLOCK_THREADS;
            digits[u] = Digit(s.keys_out[idx]);
        }
    }

    __device__ __forceinline__ void GatherScatterValues(
        int (&ranks)[ITEMS_PER_THREAD])
    {
        // compute digits corresponding to the keys
        int digits[ITEMS_PER_THREAD];
        ComputeKeyDigits(digits);
        
        // load values
        ValueT values[ITEMS_PER_THREAD];
        LoadValues(block_idx * TILE_ITEMS, values);
        
        // scatter values
        CTA_SYNC();
        ScatterValuesShared(values, ranks);

        CTA_SYNC();
        ScatterValuesGlobal(digits);
    }
        
    __device__ __forceinline__ void Process()
    {
        // load keys
        // if warp1 < warp2, all elements of warp1 occur before those of warp2
        // in the source array
        UnsignedBits keys[ITEMS_PER_THREAD];
        LoadKeys(block_idx * TILE_ITEMS, keys);

        // rank keys
        int ranks[ITEMS_PER_THREAD];
        int exclusive_digit_prefix[BINS_PER_THREAD];
        int bins[BINS_PER_THREAD];
        bool early_out = BlockRadixRankT(s.rank_temp_storage).RankKeys(
            keys, ranks, digit_extractor, exclusive_digit_prefix,
            CountsCallback(*this, bins, keys));

        if (early_out) {
            return;
        }
        
        // scatter keys in shared memory
        CTA_SYNC();
        ScatterKeysShared(keys, ranks);

        // compute global offsets
        LoadBinsToOffsetsGlobal(exclusive_digit_prefix);
        LookbackGlobal(bins);
        UpdateBinsGlobal(bins, exclusive_digit_prefix);

        // scatter keys in global memory
        CTA_SYNC();
        ScatterKeysGlobal();

        // scatter values if necessaryRadixSortOnesweepCustom
        GatherScatterValues(ranks);
    }

    __device__ __forceinline__ //
    RadixSortOnesweepCustom(TempStorage &temp_storage,
                            AtomicOffsetT *d_lookback,
                            AtomicOffsetT *d_ctrs,
                            OffsetT *d_bins_out,
                            const OffsetT *d_bins_in,
                            KeyT *d_keys_out,
                            const KeyT *d_keys_in,
                            ValueT *d_values_out,
                            const ValueT *d_values_in,
                            PortionOffsetT num_items,
                            int current_bit,
                            int num_bits)
        : s(temp_storage.Alias())
        , d_lookback(d_lookback)
        , d_ctrs(d_ctrs)
        , d_bins_out(d_bins_out)
        , d_bins_in(d_bins_in)
        , d_keys_out(reinterpret_cast<UnsignedBits *>(d_keys_out))
        , d_keys_in(reinterpret_cast<const UnsignedBits *>(d_keys_in))
        , d_values_out(d_values_out)
        , d_values_in(d_values_in)
        , num_items(num_items)
        , digit_extractor(current_bit, num_bits)
        , warp(threadIdx.x / WARP_THREADS)
        , lane(LaneId())
    {
        // initialization
        if (threadIdx.x == 0)
        {
            s.block_idx = atomicAdd(d_ctrs, 1);
        }
        CTA_SYNC();
        block_idx = s.block_idx;
        full_block = (block_idx + 1) * TILE_ITEMS <= num_items;
    }
};

#if 0 && __CUDA_ARCH__ < 800
static uint32_t __reduce_add_sync(uint32_t mask, uint32_t val)
{
    uint32_t lane_id = threadIdx.x % 32;
#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        uint32_t read_lane = lane_id ^ i;

        bool other_active = mask & (1 << read_lane);

        if (!other_active) {
            read_lane = lane_id;
        }

        uint32_t other = __shfl_sync(mask, val, read_lane);

        if (other_active) {
            val += other;
        }
    }

    return val;
}
#endif

SortArchetypeNodeBase::OnesweepNode::OnesweepNode(uint32_t taskgraph_id,
                                                  ParentNodeT parent,
                                                  int32_t pass,
                                                  bool final_pass)
    : taskGraphID(taskgraph_id),
      parentNode(parent),
      passIDX(pass),
      finalPass(final_pass)
{}

SortArchetypeNodeBase::RearrangeNode::RearrangeNode(uint32_t taskgraph_id,
                                                    ParentNodeT parent,
                                                    int32_t col_idx)
    : taskGraphID(taskgraph_id),
      parentNode(parent),
      columnIndex(col_idx)
{}

SortArchetypeNodeBase::ClearCountNode::ClearCountNode(int32_t *offsets,
                                                      int32_t *counts)
    : worldOffsets(offsets),
      worldCounts(counts)
{}

SortArchetypeNodeBase::SortArchetypeNodeBase(uint32_t taskgraph_id,
                                             uint32_t archetype_id,
                                             int32_t col_idx,
                                             uint32_t *keys_col,
                                             int32_t num_passes,
                                             int32_t *sort_offsets,
                                             int32_t *counts)
    :  NodeBase {},
       taskGraphID(taskgraph_id),
       archetypeID(archetype_id),
       sortColumnIndex(col_idx),
       keysCol(keys_col),
       numPasses(num_passes),
       worldOffsets(sort_offsets),
       worldCounts(counts)
{}

void SortArchetypeNodeBase::sortSetup(int32_t)
{
    using namespace sortConsts;

    auto &taskgraph = mwGPU::getTaskGraph(taskGraphID);
    StateManager *state_mgr = mwGPU::getStateManager();
    int32_t num_columns = state_mgr->getArchetypeNumColumns(archetypeID);

    // If this is not a world sort, we want to perform the sort always
    bool world_sort = sortColumnIndex == 1 /* 1 is the WorldID column */;

    if (!state_mgr->archetypeNeedsSort(archetypeID) && world_sort) {
        numDynamicInvocations = 0;

        auto &clear_count_node_data = taskgraph.getNodeData(clearWorldCountData);
        clear_count_node_data.numDynamicInvocations = 0;

        for (int i = 0; i < numPasses; i++) {
            taskgraph.getNodeData(onesweepNodes[i]).numDynamicInvocations = 0;
        }
        return;
    }

    if (world_sort) {
        state_mgr->archetypeClearNeedsSort(archetypeID);
    } else {
        // If this isn't a world sort, this sort will scramble the entities
        // across worlds and therefore we need to set the needs sort
        // (which denotes whether the entity needs to be sorted by world)
        // to true
        state_mgr->archetypeSetNeedsSort(archetypeID);
    }

    int num_rows = state_mgr->numArchetypeRows(archetypeID);

    int32_t num_threads =
        utils::divideRoundUp(num_rows, (int32_t)num_elems_per_sort_thread_);

    uint32_t num_blocks = utils::divideRoundUp((uint32_t)num_threads,
        consts::numMegakernelThreads);

    uint32_t rounded_num_threads = num_blocks * consts::numMegakernelThreads;

    uint64_t indices_final_offset = 0;
    uint64_t total_bytes = indices_final_offset +
        uint64_t(num_rows) * sizeof(int);
    uint64_t column_copy_offset = utils::roundUpPow2(total_bytes, ALIGN_BYTES);
    uint64_t indices_alt_offset = utils::roundUpPow2(total_bytes, ALIGN_BYTES);
    total_bytes = indices_alt_offset +
        uint64_t(num_rows) * sizeof(int);
    uint64_t keys_alt_offset = utils::roundUpPow2(total_bytes, ALIGN_BYTES);
    total_bytes = keys_alt_offset + uint64_t(num_rows) * sizeof(uint32_t);

    uint64_t bins_offset = utils::roundUpPow2(total_bytes, ALIGN_BYTES);
    total_bytes = bins_offset +
        uint64_t(numPasses * RADIX_DIGITS) * sizeof(int32_t);
    uint64_t lookback_offset = utils::roundUpPow2(total_bytes, ALIGN_BYTES);
    total_bytes = lookback_offset + 
        (uint64_t)num_blocks * RADIX_DIGITS * sizeof(int32_t);
    uint64_t counters_offset = utils::roundUpPow2(total_bytes, ALIGN_BYTES);
    total_bytes = counters_offset + uint64_t(numPasses) * sizeof(int32_t);

    uint64_t max_column_bytes =
        (uint64_t)state_mgr->getArchetypeMaxColumnSize(archetypeID) *
        (uint64_t)num_rows;

    uint64_t free_column_bytes = total_bytes - column_copy_offset;

    if (free_column_bytes < max_column_bytes) {
        total_bytes += max_column_bytes - free_column_bytes;
    }

    char *tmp_buffer = (char *)mwGPU::TmpAllocator::get().alloc(total_bytes);

    numRows = num_rows;
    numSortBlocks = num_blocks;
    numSortThreads = rounded_num_threads;
    if (sortColumnIndex == 1) { // World sort
        postBinScanThreads = 1;
    } else {
        postBinScanThreads = numRows;

        // resizeTable won't run so must set the rearrange passes to run
        // over the correct number of rows.
        taskgraph.getNodeData(firstRearrangePassData).numDynamicInvocations =
            numRows;
    }

    // This is only necessary if the global number of entities is 0
    auto &clear_count_node_data = taskgraph.getNodeData(clearWorldCountData);
    if (numRows == 0) {
        clear_count_node_data.numDynamicInvocations =
            mwGPU::GPUImplConsts::get().numWorlds;
    } else {
        clear_count_node_data.numDynamicInvocations = 0;
    }

    indicesFinal = (int *)(tmp_buffer + indices_final_offset);
    columnStaging = tmp_buffer + column_copy_offset;
    bool alt_final = numPasses % 2 == 1;

    if (alt_final) {
        indices = (int *)(tmp_buffer + indices_alt_offset);
        indicesAlt = (int *)(tmp_buffer + indices_final_offset);
    } else {
        indices = (int *)(tmp_buffer + indices_final_offset);
        indicesAlt = (int *)(tmp_buffer + indices_alt_offset);
    }

    keysAlt = (uint32_t *)(tmp_buffer + keys_alt_offset);
    bins = (int32_t *)(tmp_buffer + bins_offset);
    lookback = (int32_t *)(tmp_buffer + lookback_offset);
    counters = (int32_t *)(tmp_buffer + counters_offset);

    uint32_t num_histogram_bins = numPasses * RADIX_DIGITS;

    // Set launch count for next node that zeros the histogram
    numDynamicInvocations = num_histogram_bins;
    // Zero counters
    for (int i = 0; i < numPasses; i++) {
        counters[i] = 0;
    }

    for (int i = 0; i < numPasses; i++) {
        auto &pass_data = taskgraph.getNodeData(onesweepNodes[i]);
        pass_data.numDynamicInvocations = numSortBlocks * RADIX_DIGITS;

        if (i % 2 == 0) {
            pass_data.srcKeys = keysCol;
            pass_data.dstKeys = keysAlt;
            pass_data.srcVals = indices;
            pass_data.dstVals = indicesAlt;
        } else {
            pass_data.srcKeys = keysAlt;
            pass_data.dstKeys = keysCol;
            pass_data.srcVals = indicesAlt;
            pass_data.dstVals = indices;
        }
    } 
}

void SortArchetypeNodeBase::zeroBins(int32_t invocation_idx)
{
    bins[invocation_idx] = 0;

    if (invocation_idx == 0) {
        numDynamicInvocations = numSortBlocks;
    }
}

void SortArchetypeNodeBase::histogram(int32_t block_idx)
{
    using namespace sortConsts;

    struct HistogramSMem {
        uint32_t bins[MAX_NUM_PASSES][RADIX_DIGITS];
    };

    auto smem_tmp = (HistogramSMem *)mwGPU::SharedMemStorage::buffer;

    constexpr int32_t block_items =
        consts::numMegakernelThreads * num_elems_per_sort_thread_;
    for (int pass = 0; pass < numPasses; pass++) {
        smem_tmp->bins[pass][threadIdx.x] = 0;
    }

    __syncthreads();

#pragma unroll
    for (int i = 0; i < num_elems_per_sort_thread_; i++) {
        int32_t row_idx = block_idx * block_items +
            i * consts::numMegakernelThreads + threadIdx.x;

        if (row_idx < numRows) {
            // Initialize indices while we're here
            indices[row_idx] = row_idx;

            int current_bit = 0;
            for (int pass = 0; pass < numPasses; pass++) {
                ShiftDigitExtractor<uint32_t> digit_extractor(
                    current_bit, RADIX_BITS);

                int bin = digit_extractor.Digit(keysCol[row_idx]);
                atomicAdd(&smem_tmp->bins[pass][bin], 1);

                current_bit += RADIX_BITS;
            }
        }
    }

    __syncthreads();

    for (int pass = 0; pass < numPasses; pass++) {
        int32_t bin_count = smem_tmp->bins[pass][threadIdx.x];
        atomicAdd(&bins[pass * RADIX_DIGITS + threadIdx.x], bin_count);
    }

    if (block_idx == 0 && threadIdx.x == 0) {
        numDynamicInvocations = utils::divideRoundUp(
            uint32_t(numPasses * RADIX_DIGITS), consts::numMegakernelThreads);
    }
}

void SortArchetypeNodeBase::binScan(int32_t block_idx)
{
    using namespace sortConsts;

    using BlockScanT = BlockScan<uint32_t, consts::numMegakernelThreads>;
    using SMemTmpT = typename BlockScanT::TempStorage;
    auto smem_tmp = (SMemTmpT *)mwGPU::SharedMemStorage::buffer;

    int32_t invocation_idx = 
        block_idx * consts::numMegakernelThreads + threadIdx.x;

    uint32_t bin_vals[1];
    bin_vals[0] = bins[invocation_idx];

    BlockScanT(*smem_tmp).ExclusiveSum(bin_vals, bin_vals);

    bins[invocation_idx] = bin_vals[0];

    // Setup for resizeTable 
    if (invocation_idx == 0) {
        numDynamicInvocations = postBinScanThreads;
    }
}

void SortArchetypeNodeBase::OnesweepNode::prepareOnesweep(
    int32_t invocation_idx)
{
    auto &parent = mwGPU::getTaskGraph(taskGraphID).getNodeData(parentNode);
    // Zero out the lookback counters
    parent.lookback[invocation_idx]  = 0;

    if (invocation_idx == 0) {
        numDynamicInvocations = parent.numSortBlocks;
    }
}

void SortArchetypeNodeBase::OnesweepNode::onesweep(int32_t block_idx)
{
    using namespace sortConsts;
    using namespace mwGPU;

    static_assert(sizeof(typename RadixSortOnesweepCustom::TempStorage) <=
                  SharedMemStorage::numSMemBytes);
    
    auto smem_tmp =
        (RadixSortOnesweepCustom::TempStorage *)SharedMemStorage::buffer;

    auto &parent = mwGPU::getTaskGraph(taskGraphID).getNodeData(parentNode);

    int32_t pass = passIDX;
    RadixSortOnesweepCustom agent(*smem_tmp,
        parent.lookback,
        parent.counters + pass,
        nullptr,
        parent.bins + pass * RADIX_DIGITS,
        dstKeys,
        srcKeys,
        dstVals,
        srcVals,
        parent.numRows,
        pass * RADIX_BITS,
        RADIX_BITS);

    agent.Process();
}

void SortArchetypeNodeBase::resizeTable(int32_t invocation_idx)
{
    int32_t num_entities = bins[(numPasses - 1) * 256 + 255];
    mwGPU::getStateManager()->resizeArchetype(archetypeID, num_entities);

    auto &taskgraph = mwGPU::getTaskGraph(taskGraphID);
    taskgraph.getNodeData(firstRearrangePassData).numDynamicInvocations =
        num_entities;

    // Set for clearWorldOffsetsAndCounts
    numDynamicInvocations = mwGPU::GPUImplConsts::get().numWorlds;
}

void SortArchetypeNodeBase::ClearCountNode::clearCounts(int32_t invocation_idx)
{
    worldOffsets[invocation_idx] = 0;
    worldCounts[invocation_idx] = 0;

    if (invocation_idx == 0) {
        numDynamicInvocations = 0;
    }
}

void SortArchetypeNodeBase::clearWorldOffsetsAndCounts(int32_t invocation_idx)
{
    int32_t num_entities = bins[(numPasses - 1) * 256 + 255];

    // The counts computation for worldID i works by setting:
    // worldOffsets[i] = entities before world i;
    // worldCounts[i] = entities before and including world i;
    // And then element-wise subtracting worldOffsets from worldCounts.

    // A world with 0 entities will not be written to in either array, 
    // so we ensure counts are correct for those worlds by clearing
    // worldOffsets and worldCounts to the same value.

    // Clear to numEntities in case some number of final worlds
    // have 0 elements (see computeWorldCounts(int32_t invocation_idx)).
    worldOffsets[invocation_idx] = num_entities;
    worldCounts[invocation_idx] = num_entities;
    
    if (invocation_idx == 0) {
        // Set for copyKeys and computeWorldCounts
        numDynamicInvocations = num_entities;
    }
}


void SortArchetypeNodeBase::copyKeys(int32_t invocation_idx)
{
    keysCol[invocation_idx] = keysAlt[invocation_idx];
    
}

void SortArchetypeNodeBase::computeWorldCounts(int32_t invocation_idx)
{
    if (invocation_idx == 0)
    {
        // The offset of the first entity's world must be 0.
        worldOffsets[keysCol[invocation_idx]] = invocation_idx;
        numDynamicInvocations = mwGPU::GPUImplConsts::get().numWorlds;
    }
    else if (keysCol[invocation_idx] != keysCol[invocation_idx - 1])
    {
        // This thread is the index of the first entity in 
        // World "keysCol[invocation_idx]". Its index value is both
        // 1. The offset for its world
        // 2. The total number of entities occuring before it.
        // We write both those cases below.


        // World "keysCol[invocation_idx]" has invocation_idx entities 
        // before it in the sorted list.
        worldOffsets[keysCol[invocation_idx]] = invocation_idx;
        
        // World "keysCol[invocation_idx - 1]" has invocation_idx entities
        // total including entities before it and its entities.
        // For the final world with entities, invocation_idx = numEntities
        // should write to worldCounts, but our final thread index is
        // numEntities - 1. Thus, we initialize to numEntities so the value
        // of worldCounts is correct (see clearWorldOffsetsAndCounts()).
        worldCounts[keysCol[invocation_idx - 1]] = invocation_idx;
    }
}

void SortArchetypeNodeBase::correctWorldCounts(int32_t invocation_idx) {
    // Correct world counts by subtracting "entities before" from "entities
    // before and including" for each world. A world with 0 entities will
    // compute numEntities - numEntities = 0. For worlds with 0 entities,
    // worldOffsets = numEntities. Otherwise, worldOffsets are correct.
    worldCounts[invocation_idx] -= worldOffsets[invocation_idx];
}

void SortArchetypeNodeBase::RearrangeNode::stageColumn(int32_t invocation_idx)
{
    StateManager *state_mgr = mwGPU::getStateManager();
    auto &parent = mwGPU::getTaskGraph(taskGraphID).getNodeData(parentNode);

    uint32_t bytes_per_elem = state_mgr->getArchetypeColumnBytesPerRow(
        parent.archetypeID, columnIndex);

    void *src = state_mgr->getArchetypeColumn(parent.archetypeID, columnIndex);

    int src_idx = parent.indicesFinal[invocation_idx];

    memcpy((char *)parent.columnStaging +
                (uint64_t)bytes_per_elem * (uint64_t)invocation_idx,
           (char *)src + (uint64_t)bytes_per_elem * (uint64_t)src_idx,
           bytes_per_elem);
}

void SortArchetypeNodeBase::RearrangeNode::rearrangeEntities(int32_t invocation_idx)
{
    StateManager *state_mgr = mwGPU::getStateManager();
    auto &parent = mwGPU::getTaskGraph(taskGraphID).getNodeData(parentNode);

    auto entities_staging = (Entity *)parent.columnStaging;
    auto dst = (Entity *)state_mgr->getArchetypeColumn(parent.archetypeID, 0);
    auto worlds = (WorldID *)state_mgr->getArchetypeColumn(parent.archetypeID, 1);

    Entity e = entities_staging[invocation_idx];

    dst[invocation_idx] = e;

    // FIXME: temporary entities still have an entity column, but they need to
    // *NOT* be remapped
    if (e != Entity::none()) {
        state_mgr->remapEntity(e, invocation_idx);
    }

    if (invocation_idx == 0) {
        numDynamicInvocations = 0;
    }
}

void SortArchetypeNodeBase::RearrangeNode::rearrangeColumn(int32_t invocation_idx)
{
    StateManager *state_mgr = mwGPU::getStateManager();
    auto &taskgraph = mwGPU::getTaskGraph(taskGraphID);
    auto &parent = taskgraph.getNodeData(parentNode);

    auto staging = (char *)parent.columnStaging;
    auto dst = (char *)state_mgr->getArchetypeColumn(
        parent.archetypeID, columnIndex);

    uint32_t bytes_per_elem = state_mgr->getArchetypeColumnBytesPerRow(
        parent.archetypeID, columnIndex);

    memcpy(dst + (uint64_t)bytes_per_elem * (uint64_t)invocation_idx,
           staging + (uint64_t)bytes_per_elem * (uint64_t)invocation_idx,
           bytes_per_elem);

    if (invocation_idx == 0) {
        if (nextRearrangeNode.id != -1) {
            taskgraph.getNodeData(nextRearrangeNode).numDynamicInvocations =
                numDynamicInvocations;
        }
        numDynamicInvocations = 0;
    }
}

TaskGraph::NodeID SortArchetypeNodeBase::addToGraph(
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> dependencies,
    uint32_t archetype_id,
    int32_t component_id)
{
    using namespace mwGPU;

    static_assert(consts::numMegakernelThreads ==
                  sortConsts::RADIX_DIGITS);

    StateManager *state_mgr = getStateManager();
    int32_t sort_column_idx = state_mgr->getArchetypeColumnIndex(
        archetype_id, component_id);
    auto keys_col =  (uint32_t *)state_mgr->getArchetypeColumn(
        archetype_id, sort_column_idx);

    int32_t *world_offsets = state_mgr->getArchetypeWorldOffsets(archetype_id);
    int32_t *world_counts = state_mgr->getArchetypeWorldCounts(archetype_id);

    bool world_sort = component_id == TypeTracker::typeID<WorldID>();

    // Optimize for sorts on the WorldID column, where the 
    // max # of worlds is known
    int32_t num_passes;
    if (world_sort) {
        int32_t num_worlds = GPUImplConsts::get().numWorlds;
        // num_worlds + 1 to leave room for columns with WorldID == -1
        int32_t num_bits = 32 - __clz(num_worlds + 1);

        num_passes = utils::divideRoundUp(num_bits, 8);
    } else {
        num_passes = 4;
    }

    uint32_t taskgraph_id = builder.getTaskgraphID();

    auto data_id = builder.constructNodeData<SortArchetypeNodeBase>(
        taskgraph_id, archetype_id, sort_column_idx, keys_col,
        num_passes, world_offsets, world_counts);
    auto &sort_node_data = builder.getDataRef(data_id);

    TaskGraph::NodeID setup = builder.addNodeFn<
        &SortArchetypeNodeBase::sortSetup>(data_id, dependencies,
            Optional<TaskGraph::NodeID>::none(), 1);

    // Create the clear world count pass (which will run optionally 
    // if archetype needs sorting but the number of entities is 0).
    sort_node_data.clearWorldCountData = builder.constructNodeData<
        ClearCountNode>(world_offsets, world_counts);
    auto clear_counts = builder.addNodeFn<
        &ClearCountNode::clearCounts>(sort_node_data.clearWorldCountData,
            {setup}, setup);

    auto zero_bins = builder.addNodeFn<
        &SortArchetypeNodeBase::zeroBins>(data_id, {clear_counts}, setup);

    auto compute_histogram = builder.addNodeFn<
        &SortArchetypeNodeBase::histogram>(data_id, {zero_bins}, setup, 0,
            consts::numMegakernelThreads);

    auto cur_task = builder.addNodeFn<&SortArchetypeNodeBase::binScan>(
        data_id, {compute_histogram}, setup, 0, consts::numMegakernelThreads);

    for (int32_t i = 0; i < num_passes; i++) {
        auto pass_data = builder.constructNodeData<OnesweepNode>(
            taskgraph_id, data_id, i, i == num_passes - 1);
        sort_node_data.onesweepNodes[i] = pass_data;
        cur_task = builder.addNodeFn<
            &OnesweepNode::prepareOnesweep>(pass_data, {cur_task}, setup);

        cur_task = builder.addNodeFn<&OnesweepNode::onesweep>(
            pass_data, {cur_task}, setup, 0,
            consts::numMegakernelThreads);
    }

    // FIXME this could be a fixed-size count
    if (world_sort) {
        cur_task = builder.addNodeFn<&SortArchetypeNodeBase::resizeTable>(
            data_id, {cur_task}, setup);

        cur_task = builder.addNodeFn<&SortArchetypeNodeBase::clearWorldOffsetsAndCounts>(
            data_id, {cur_task}, setup);
    }

    if (num_passes % 2 == 1) {
        cur_task = builder.addNodeFn<&SortArchetypeNodeBase::copyKeys>(
            data_id, {cur_task}, setup);
    }


    if (world_sort) {
        // Compute counts for each world by writing upper ranges to worldCounts
        // and lower ranges to worldOffsets. 
        cur_task = builder.addNodeFn<&SortArchetypeNodeBase::computeWorldCounts>(
            data_id, {cur_task}, setup);

        // Compute final counts by subtracting worldOffsets from worldCounts
        // Worlds with 0 entities have offset numEntities.
        cur_task = builder.addNodeFn<&SortArchetypeNodeBase::correctWorldCounts>(
            data_id, {cur_task}, setup);
    }
    
    int32_t num_columns = state_mgr->getArchetypeNumColumns(archetype_id);

    TaskGraph::TypedDataID<RearrangeNode> prev_rearrange_node { -1 };

    for (int32_t col_idx = 1; col_idx < num_columns; col_idx++) {
        if (col_idx == sort_column_idx) continue;
        auto cur_rearrange_node = builder.constructNodeData<RearrangeNode>(
            taskgraph_id, data_id, col_idx);
        builder.getDataRef(cur_rearrange_node).numDynamicInvocations = 0;

        if (prev_rearrange_node.id == -1) {
            sort_node_data.firstRearrangePassData = cur_rearrange_node;
        } else {
            builder.getDataRef(prev_rearrange_node).nextRearrangeNode =
                cur_rearrange_node;
        }
        prev_rearrange_node = cur_rearrange_node;

        cur_task = builder.addNodeFn<&RearrangeNode::stageColumn>(
            cur_rearrange_node, {cur_task}, setup);

        cur_task = builder.addNodeFn<&RearrangeNode::rearrangeColumn>(
            cur_rearrange_node, {cur_task}, setup);
    }

    auto entities_rearrange_node = builder.constructNodeData<RearrangeNode>(
        taskgraph_id, data_id, 0);

    cur_task = builder.addNodeFn<&RearrangeNode::stageColumn>(
        entities_rearrange_node, {cur_task}, setup);

    cur_task = builder.addNodeFn<&RearrangeNode::rearrangeEntities>(
        entities_rearrange_node, {cur_task}, setup);

    assert(prev_rearrange_node.id != -1);
    builder.getDataRef(prev_rearrange_node).nextRearrangeNode =
        entities_rearrange_node;
    builder.getDataRef(entities_rearrange_node).nextRearrangeNode = { -1 };
    builder.getDataRef(entities_rearrange_node).numDynamicInvocations = 0;

    return cur_task;
}

}
