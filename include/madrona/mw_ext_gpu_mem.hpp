/*
 * Copyright 2021-2023 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once
#ifdef mw_gpu_EXPORTS
#define MADRONA_MWGPU_EXPORT MADRONA_EXPORT
#else
#define MADRONA_MWGPU_EXPORT MADRONA_IMPORT
#endif

#include <memory>
#include <stdint.h>
#include <madrona/macros.hpp>

namespace madrona {

struct GPUMapping {
    void *basePtr;
    uint64_t numBytes;
};

using GPUExternalVMInstance = uint32_t;

// This is likely going to be unused unless rendering is involved,
// in which this will be a necessity.
class GPUExternalVM {
public:
    // A VM Instance must be registered in order for there to be external
    // VM support.
    MADRONA_MWGPU_EXPORT GPUExternalVMInstance registerInstance();

    // Returns the current state of the allocator state.
    // It is the responsibility of the external system to keep track of
    // differences.
    MADRONA_MWGPU_EXPORT GPUMapping dequeueMapping(GPUExternalVMInstance) const;

    // Gets called by the VM allocator thread in the cuda executor.
    MADRONA_MWGPU_EXPORT void queueMapping(
        GPUExternalVMInstance instance_id,
        const GPUMapping &mapping);

    GPUExternalVM();
    ~GPUExternalVM();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
