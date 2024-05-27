#pragma once

#include <madrona/math.hpp>
#include <madrona/taskgraph_builder.hpp>

namespace madrona::render {

// This will be attached to any entity that wants to be a viewer
//
// For ScriptBots, we want multiple camera outputs, one for sensor information
// and another for figuring out which entity is in front of us (finder).
struct RenderCamera {
    Entity cameraEntity;

    // 1.0 / tanf(fovy * 0.5)
    float fovScale;
    float zNear;

    math::Vector3 cameraOffset;
};

// This will be attached to any renderable entity
struct Renderable {
    Entity renderEntity;
};




// TODO: Make sure to move this to private headers which can be
// included by the device code for the BVH.

// For private usage - not to be used by user.
using MortonCode = uint32_t;

// For private usage - not to be used by user.
struct alignas(16) PerspectiveCameraData {
    math::Vector2 position;

    // Eurler angle of viewing direction
    float viewDirPolar;

    uint8_t numForwardRays;
    uint8_t numBackwardRays;

    int32_t worldIDX;

    float zOffset;
};

// For private usage - not to be used by user.
struct alignas(16) InstanceData {
    math::Vector2 position;
    math::Vector2 scale;
    float viewDirPolar;

    Entity owner;

    int32_t objectIDX;
    int32_t worldIDX;

    // We store this separately so that the 1D raytracer doesn't have
    // to be rewritten.
    float zOffset;
};

// This contains the actual render output
struct RenderOutputBufferImpl {
    char buffer[1];
};

struct RenderOutputBuffer : RenderOutputBufferImpl
{
};

// The finder outputs depth and the entity
struct FinderOutput {
    Entity hitEntity;
    float depth;
};

struct FinderOutputBuffer : RenderOutputBufferImpl
{
};

// Reference to an output
struct RenderOutputRef {
    Entity outputEntity;
    Entity finderOutputEntity;
};

// Top level acceleration structure node
struct alignas(16) TLBVHNode {
    math::AABB aabb;
};

// For private usage - not to be used by user.
struct RenderableArchetype : public Archetype<
    InstanceData,

    // For BVH support, we need to sort these not just be world ID,
    // but first by morton code too.
    MortonCode,

    TLBVHNode
> {};

// For private usage - not to be used by user.
struct RenderCameraArchetype : public Archetype<
    PerspectiveCameraData,
    RenderOutputRef
> {};

// This is an unsorted archetype with a runtime-sized component
struct RaycastOutputArchetype : public Archetype<
    RenderOutputBuffer,
    FinderOutputBuffer
> {};



struct RenderECSBridge;

namespace RenderingSystem {
    void registerTypes(ECSRegistry &registry,
                       const RenderECSBridge *bridge);

    TaskGraphNodeID setupTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps);

    void init(Context &ctx,
              const RenderECSBridge *bridge);

    uint32_t * getVoxelPtr(Context &ctx);

    void makeEntityRenderable(Context &ctx,
                              Entity e);

    void attachEntityToView(Context &ctx,
                            Entity e,
                            float vfov_degrees,
                            float z_near,
                            const math::Vector3 &camera_offset);

    // Need to call these before destroying entities
    void cleanupViewingEntity(Context &ctx,
                              Entity e);
    void cleanupRenderableEntity(Context &ctx,
                                 Entity e);

    template <typename OutputT>
    OutputT *getRenderOutput(const RenderCamera &camera);
};

}

#include "ecs.inl"
