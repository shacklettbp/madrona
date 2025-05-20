#include <madrona/mw_gpu/const.hpp>

#include <madrona/taskgraph.hpp>
#include <madrona/mw_gpu/host_print.hpp>
#include <madrona/mw_gpu/tracing.hpp>

extern "C" {
__constant__ madrona::mwGPU::GPUImplConsts madronaMWGPUConsts;
}


extern "C" __global__ void madronaMWGPUComputeConstants(
    uint32_t num_worlds,
    uint32_t num_world_data_bytes,
    uint32_t world_data_alignment,
    uint32_t num_taskgraphs,
    madrona::mwGPU::GPUImplConsts *out_constants,
    size_t *job_system_buffer_size,
    void *mesh_bvhs,
    uint32_t num_mesh_bvhs,
    uint32_t raycast_output_width,
    uint32_t raycast_output_height,
    void *bvh_internal_data,
    uint32_t raycast_rgbd)
{
    using namespace madrona;
    using namespace madrona::mwGPU;

    uint64_t total_bytes = sizeof(TaskGraph) * (uint64_t)num_taskgraphs;

    uint64_t state_mgr_offset = utils::roundUp(total_bytes,
        (uint64_t)alignof(StateManager));

    total_bytes = state_mgr_offset + sizeof(StateManager);

    uint64_t world_data_offset =
        utils::roundUp(total_bytes, (uint64_t)world_data_alignment);

    uint64_t total_world_bytes =
        (uint64_t)num_world_data_bytes * (uint64_t)num_worlds;

    total_bytes = world_data_offset + total_world_bytes;

    uint64_t host_allocator_offset =
        utils::roundUp(total_bytes, (uint64_t)alignof(mwGPU::HostAllocator));

    total_bytes = host_allocator_offset + sizeof(mwGPU::HostAllocator);

    uint64_t host_print_offset =
        utils::roundUp(total_bytes, (uint64_t)alignof(mwGPU::HostPrint));

    total_bytes = host_print_offset + sizeof(mwGPU::HostPrint);

    uint64_t tmp_allocator_offset =
        utils::roundUp(total_bytes, (uint64_t)alignof(TmpAllocator));

    total_bytes = tmp_allocator_offset + sizeof(TmpAllocator);

    uint64_t device_tracing_offset = utils::roundUp(
        total_bytes, (uint64_t)alignof(mwGPU::DeviceTracing));

    total_bytes = device_tracing_offset + sizeof(mwGPU::DeviceTracing);

    *out_constants = GPUImplConsts {
        .jobSystemAddr =                    (void *)0ul,
        .taskGraph =                        (void *)0ul,
        .stateManagerAddr =                 (void *)state_mgr_offset,
        .worldDataAddr =                    (void *)world_data_offset,
        .hostAllocatorAddr =                (void *)host_allocator_offset,
        .hostPrintAddr =                    (void *)host_print_offset,
        .tmpAllocatorAddr =                 (void *)tmp_allocator_offset,
        .deviceTracingAddr =                (void *)device_tracing_offset,
        .meshBVHsAddr =                     (void *)mesh_bvhs,
        .bvhInternalData =                  bvh_internal_data,
        .numWorldDataBytes =                num_world_data_bytes,
        .numWorlds =                        num_worlds,
        .jobGridsOffset =                   (uint32_t)0,
        .jobListOffset =                    (uint32_t)0,
        .maxJobsPerGrid =                   0,
        .sharedJobTrackerOffset =           (uint32_t)0,
        .userJobTrackerOffset =             (uint32_t)0,
        .numMeshBVHs =                      num_mesh_bvhs,
        .raycastOutputWidth =               raycast_output_width,
        .raycastOutputHeight =              raycast_output_height,
        .raycastRGBD =                      raycast_rgbd,
    };

    *job_system_buffer_size = total_bytes;
}
