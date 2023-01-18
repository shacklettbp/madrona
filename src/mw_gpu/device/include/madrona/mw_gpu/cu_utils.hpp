#pragma once

#include <cstdint>

namespace madrona {
namespace mwGPU {

constexpr inline uint32_t numWarpThreads = 32;
constexpr inline uint32_t allActive = 0xFFFF'FFFF;

inline uint32_t warpAllInclusiveScan(uint32_t lane_id, uint32_t val)
{
#pragma unroll
    for (int i = 1; i < numWarpThreads; i *= 2) {
        int tmp = __shfl_up_sync(allActive, val, i);
        if ((int)lane_id >= i) {
            val += tmp;
        }
    }

    return val;
}

inline uint32_t warpAllExclusiveScan(uint32_t lane_id, uint32_t val)
{
    return warpAllInclusiveScan(lane_id, val) - val;
}

inline uint32_t getHighestSetBit(uint32_t mask)
{
    return mask == 0u ? 0u : 31u - __clz(mask);
}

inline uint32_t getLowestSetBit(uint32_t mask)
{
    return mask == 0u ? 0u : __clz(__brev(mask));
}

inline uint32_t getNumHigherSetBits(uint32_t mask, uint32_t idx)
{
    mask >>= idx + 1;

    return __popc(mask);
}

inline uint32_t getNumLowerSetBits(uint32_t mask, uint32_t idx)
{
    mask <<= (32 - idx);

    return __popc(mask);
}

}
}
