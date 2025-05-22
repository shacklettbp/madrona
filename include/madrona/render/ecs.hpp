#pragma once

#include <madrona/math.hpp>
#include <madrona/taskgraph_builder.hpp>

namespace madrona::render {

// This will be attached to any entity that wants to be a viewer
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

struct LightCarrier {
    Entity light;
};




// For private usage - not to be used by user.
using MortonCode = uint32_t;

// For private usage - not to be used by user.
struct alignas(16) PerspectiveCameraData {
    math::Vector3 position;
    math::Quat rotation;
    float xScale;
    float yScale;
    float zNear;
    int32_t worldIDX;
    uint32_t pad;
};

// For private usage - not to be used by user.
struct alignas(16) InstanceData {
    math::Vector3 position;
    math::Quat rotation;
    math::Diag3x3 scale;

    // If this is -1, we just use whatever default material the model
    // has defined for it.
    //
    // If this is -2, we use the color at the end of this struct.
    int32_t matID;

    int32_t objectID;
    int32_t worldIDX;

    uint32_t color;
};

// This is all the data required to configure a light. The actual
// data is read in / written to through SOA.
struct alignas(16) LightDesc {
    // Only affects the spotlight (defaults to 0 0 0).
    math::Vector3 position;

    // Affects both directional/spotlight.
    math::Vector3 direction;

    // Angle for the spotlight (default to pi/4).
    float cutoffAngle;

    // Intensity of the light. (1.f is default)
    float intensity;

    enum Type : uint32_t {
        Directional = 1,
        Spotlight = 0
    };

    // Type of the light.
    Type type;

    // Whether the light casts a shadow.
    uint32_t castShadow;

    // Gives ability to turn light on or off.
    uint32_t active;
};

struct LightDescDirection : math::Vector3 {
    LightDescDirection(math::Vector3 v)
        : Vector3(v)
    {}
};

struct LightDescType {
    LightDesc::Type type;
};

struct LightDescShadow {
    bool castShadow;
};

struct LightDescCutoffAngle {
    float cutoffAngle;
};

struct LightDescIntensity {
    float intensity;
};

struct LightDescActive {
    bool active;
};

struct LightArchetype : public Archetype<
    LightDesc
> {};

struct MaterialOverride {
    // These are values that matID can take on if not some override material ID.
    enum {
        UseDefaultMaterial = -1,
        UseOverrideColor = -2
    };

    int32_t matID;
};

struct ColorOverride {
    uint32_t color;
};

// This contains the actual render output
struct RenderOutputBuffer {
    char buffer[1];
};

struct RGBOutputBuffer : RenderOutputBuffer {};
struct DepthOutputBuffer : RenderOutputBuffer {};

// Reference to an output
struct RenderOutputRef {
    Entity outputEntity;
};

struct RenderOutputIndex {
    uint32_t index;
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
    RenderOutputRef,
    RenderOutputIndex
> {};

// This is an unsorted archetype with a runtime-sized component
struct RaycastOutputArchetype : public Archetype<
    RGBOutputBuffer,
    DepthOutputBuffer
> {};

struct RenderECSBridge;

namespace RenderingSystem {
    void registerTypes(ECSRegistry &registry,
                       const RenderECSBridge *bridge);

    TaskGraphNodeID setupTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps,
        bool update_visual_properties = false);

    void init(Context &ctx,
              const RenderECSBridge *bridge);

    uint32_t * getVoxelPtr(Context &ctx);

    void makeEntityRenderable(Context &ctx,
                              Entity e);
    
    void disableEntityRenderable(Context &ctx,
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


    void makeEntityLightCarrier(Context &ctx,
                                Entity e);
};

}
