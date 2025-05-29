#pragma once

#include <madrona/taskgraph_builder.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/rand.hpp>

#include "types.hpp"

namespace madMJX {

class Engine;

// This enum is used by the Sim and Manager classes to track the export slots
// for each component exported to the training code.
enum class ExportID : uint32_t {
    InstancePositions,
    InstanceRotations,
    CameraPositions,
    CameraRotations,
    InstanceScales,
    InstanceMatOverrides,
    InstanceColorOverrides,
    RaycastDepth,
    RaycastRGB,
    LightPositions,
    LightDirections,
    LightTypes,
    LightShadows,
    LightCutoffAngles,
    LightIntensities,
    NumExports,
};

enum class TaskGraphID : uint32_t {
    Init,
    RenderInit,
    Render,
    NumGraphs,
};

// The Sim class encapsulates the per-world state of the simulation.
// Sim is always available by calling ctx.data() given a reference
// to the Engine / Context object that is passed to each ECS system.
//
// Per-World state that is frequently accessed but only used by a few
// ECS systems should be put in a singleton component rather than
// in this class in order to ensure efficient access patterns.
struct Sim : public madrona::WorldBase {
    struct Config {
        int32_t *geomTypes;
        int32_t *geomDataIDs;
        Vector3 *geomSizes;
        uint32_t numGeoms;
        uint32_t numCams;
        uint32_t numLights;
        float *camFovy;
        const madrona::render::RenderECSBridge *renderBridge;
        bool useDebugCamEntity;
        bool useRT;
    };

    // This class would allow per-world custom data to be passed into
    // simulator initialization, but that isn't necessary in this environment
    struct WorldInit {};

    // Sim::registerTypes is called during initialization
    // to register all components & archetypes with the ECS.
    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);

    // Sim::setupTasks is called during initialization to build
    // the system task graph that will be invoked by the 
    // Manager class (src/mgr.hpp) for each step.
    static void setupTasks(madrona::TaskGraphManager &taskgraph_mgr,
                           const Config &cfg);

    // The constructor is called for each world during initialization.
    // Config is global across all worlds, while WorldInit (src/init.hpp)
    // can contain per-world initialization data, created in (src/mgr.cpp)
    Sim(Engine &ctx,
        const Config &cfg,
        const WorldInit &);
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
public:
    using CustomContext::CustomContext;

    // These are convenience helpers for creating renderable
    // entities when rendering isn't necessarily enabled
    template <typename ArchetypeT>
    inline madrona::Entity makeRenderableEntity();
    inline void destroyRenderableEntity(Entity e);
};

}

#include "sim.inl"
