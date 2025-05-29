#pragma once

#include <madrona/components.hpp>
#include <madrona/math.hpp>
#include <madrona/rand.hpp>
#include <madrona/physics.hpp>
#include <madrona/render/ecs.hpp>

namespace madMJX {

// Include several madrona types into the simulator namespace for convenience
using madrona::Entity;
using madrona::RandKey;
using madrona::CountT;
using madrona::base::Position;
using madrona::base::Rotation;
using madrona::base::Scale;
using madrona::base::ObjectInstance;
using madrona::base::ObjectID;
using madrona::math::Vector3;
using madrona::math::Quat;
using madrona::render::MaterialOverride;
using madrona::render::ColorOverride;

enum class RenderPrimObjectIDs : uint32_t {
    DebugCam = 0,
    Plane = 1,
    Sphere = 2,
    Box = 3,
    Cylinder = 4,
    Capsule = 5,
    NumPrims,
};

enum class MJXGeomType : uint32_t {
    Plane       = 0,
    Heightfield = 1,
    Sphere      = 2,
    Capsule     = 3,
    Ellipsoid   = 4,
    Cylinder    = 5,
    Box         = 6,
    Mesh        = 7,
};

struct LightEntity : public madrona::Archetype<
    Position,
    madrona::render::LightDescDirection,
    madrona::render::LightDescType,
    madrona::render::LightDescShadow,
    madrona::render::LightDescCutoffAngle,
    madrona::render::LightDescIntensity,
    madrona::render::LightDescActive,
    madrona::render::LightCarrier
> {};

struct RenderEntity : public madrona::Archetype<
    ObjectInstance,
    madrona::render::MaterialOverride,
    madrona::render::ColorOverride,
    // All entities with the Renderable component will be drawn by the
    // viewer and batch renderer
    madrona::render::Renderable
> {};

struct CameraEntity : public madrona::Archetype<
    Position,
    Rotation,
    madrona::render::RenderCamera
> {};

struct DebugCameraEntity : public madrona::Archetype<
    ObjectInstance,
    madrona::render::Renderable,
    madrona::render::RenderCamera
> {};

}
