#include "mgr.hpp"
#include "sim.hpp"
#include "geometry.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/physics_loader.hpp>
#include <madrona/tracing.hpp>
#include <madrona/mw_cpu.hpp>
#include <madrona/render/api.hpp>

#include <array>
#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;
using namespace madrona::py;
using namespace madrona::imp;

namespace madMJX {

struct RenderGPUState {
    render::APILibHandle apiLib;
    render::APIManager apiMgr;
    render::GPUHandle gpu;
};

static inline Optional<RenderGPUState> initRenderGPUState(
    const Manager::Config &mgr_cfg,
    const Optional<VisualizerGPUHandles> &viz_gpu_hdls)
{
    if (viz_gpu_hdls.has_value()) {
        return Optional<RenderGPUState>::none();
    }

#ifdef MGR_DISABLE_VULKAN
    return Optional<RenderGPUState>::none();
#endif

    auto render_api_lib = render::APIManager::loadDefaultLib();
    render::APIManager render_api_mgr(render_api_lib.lib());
    render::GPUHandle gpu = render_api_mgr.initGPU(mgr_cfg.gpuID);

    return RenderGPUState {
        .apiLib = std::move(render_api_lib),
        .apiMgr = std::move(render_api_mgr),
        .gpu = std::move(gpu),
    };
}

static inline Optional<render::RenderManager> initRenderManager(
    const Manager::Config &mgr_cfg,
    const MJXModel &mjx_model,
    const Optional<VisualizerGPUHandles> &viz_gpu_hdls,
    const Optional<RenderGPUState> &render_gpu_state)
{
    if (mgr_cfg.useRT && !viz_gpu_hdls.has_value()) {
        return Optional<render::RenderManager>::none();
    }

    render::APIBackend *render_api;
    render::GPUDevice *render_dev;

    if (render_gpu_state.has_value()) {
        render_api = render_gpu_state->apiMgr.backend();
        render_dev = render_gpu_state->gpu.device();
    } else {
        assert(viz_gpu_hdls.has_value());

        render_api = viz_gpu_hdls->renderAPI;
        render_dev = viz_gpu_hdls->renderDev;
    }

    uint32_t max_instances_per_world = mjx_model.numGeoms;
    if (mgr_cfg.addCamDebugGeometry) {
        max_instances_per_world += mjx_model.numCams;
    }

    return render::RenderManager(render_api, render_dev, {
        .enableBatchRenderer = true,
        .renderMode = render::RenderManager::Config::RenderMode::RGBD,
        .agentViewWidth = mgr_cfg.batchRenderViewWidth,
        .agentViewHeight = mgr_cfg.batchRenderViewHeight,
        .numWorlds = mgr_cfg.numWorlds,
        .maxViewsPerWorld = mjx_model.numCams,
        .maxLightsPerWorld = mjx_model.numLights,
        .maxInstancesPerWorld = max_instances_per_world,
        .execMode = ExecMode::CUDA,
        .voxelCfg = {},
    });
}

struct JAXIO {
    Vector3 *geomPositions;
    Quat *geomRotations;
    Vector3 *camPositions;
    Quat *camRotations;
    Diag3x3 *geomSizes;
    int32_t *matIDs;
    uint32_t *geomRGB;
    Vector3 *lightPos;
    Vector3 *lightDir;
    bool *lightIsDir;
    bool *lightCastShadow;
    float *lightCutoff;
    float *lightIntensity;
    uint8_t *rgbOut;
    float *depthOut;

    static inline JAXIO makeInit(void **buffers)
    {
        CountT buf_idx = 0;
        auto geom_positions = (Vector3 *)buffers[buf_idx++];
        auto geom_rotations = (Quat *)buffers[buf_idx++];
        auto cam_positions = (Vector3 *)buffers[buf_idx++];
        auto cam_rotations = (Quat *)buffers[buf_idx++];
        auto mat_ids = (int32_t *)buffers[buf_idx++];
        auto geom_rgb = (uint32_t *)buffers[buf_idx++];
        auto geom_sizes = (Diag3x3 *)buffers[buf_idx++];
        auto light_pos = (Vector3 *)buffers[buf_idx++];
        auto light_dir = (Vector3 *)buffers[buf_idx++];
        auto light_isdir = (bool *)buffers[buf_idx++];
        auto light_castshadow = (bool *)buffers[buf_idx++];
        auto light_cutoff = (float *)buffers[buf_idx++];
        auto light_intensity = (float *)buffers[buf_idx++];
        auto rgb_out = (uint8_t *)buffers[buf_idx++];
        auto depth_out = (float *)buffers[buf_idx++];

        return JAXIO {
            .geomPositions = geom_positions,
            .geomRotations = geom_rotations,
            .camPositions = cam_positions,
            .camRotations = cam_rotations,
            .geomSizes = geom_sizes,
            .matIDs = mat_ids,
            .geomRGB = geom_rgb,
            .lightPos = light_pos,
            .lightDir = light_dir,
            .lightIsDir = light_isdir,
            .lightCastShadow = light_castshadow,
            .lightCutoff = light_cutoff,
            .lightIntensity = light_intensity,
            .rgbOut = rgb_out,
            .depthOut = depth_out,
        };
    }

    static inline JAXIO makeRender(void **buffers)
    {
        CountT buf_idx = 0;
        auto geom_positions = (Vector3 *)buffers[buf_idx++];
        auto geom_rotations = (Quat *)buffers[buf_idx++];
        auto cam_positions = (Vector3 *)buffers[buf_idx++];
        auto cam_rotations = (Quat *)buffers[buf_idx++];
        auto rgb_out = (uint8_t *)buffers[buf_idx++];
        auto depth_out = (float *)buffers[buf_idx++];

        return JAXIO {
            .geomPositions = geom_positions,
            .geomRotations = geom_rotations,
            .camPositions = cam_positions,
            .camRotations = cam_rotations,
            .geomSizes = nullptr,
            .matIDs = nullptr,
            .geomRGB = nullptr,
            .lightPos = nullptr,
            .lightDir = nullptr,
            .lightIsDir = nullptr,
            .lightCastShadow = nullptr,
            .lightCutoff = nullptr,
            .lightIntensity = nullptr,
            .rgbOut = rgb_out,
            .depthOut = depth_out,
        };
    }
};

struct Manager::Impl {
    Config cfg;
    uint32_t numGeoms;
    uint32_t numCams;
    uint32_t numLights;

    Optional<RenderGPUState> renderGPUState;
    Optional<render::RenderManager> renderMgr;

    MWCudaExecutor gpuExec;
    MWCudaLaunchGraph renderGraph;

    Optional<MWCudaLaunchGraph> raytraceGraph;

    static inline Impl * make(
        const Config &cfg,
        const MJXModel &mjx_model,
        const Optional<VisualizerGPUHandles> &viz_gpu_hdls);

    inline Impl(const Manager::Config &mgr_cfg,
                uint32_t num_geoms,
                uint32_t num_cams,
                uint32_t num_lights,
                Optional<RenderGPUState> &&render_gpu_state,
                Optional<render::RenderManager> &&render_mgr,
                MWCudaExecutor &&gpu_exec,
                Optional<MWCudaLaunchGraph> &&raytrace_graph)

        : cfg(mgr_cfg),
          numGeoms(num_geoms),
          numCams(num_cams),
          numLights(num_lights),
          renderGPUState(std::move(render_gpu_state)),
          renderMgr(std::move(render_mgr)),
          gpuExec(std::move(gpu_exec)),
          renderGraph(gpuExec.buildLaunchGraph(TaskGraphID::Render)),
          raytraceGraph(std::move(raytrace_graph))
    {}

    inline ~Impl() {}

    inline void renderImpl()
    {
        if (renderMgr.has_value()) {
            renderMgr->readECS();
            renderMgr->batchRender();
        }

        if (cfg.useRT) {
            gpuExec.run(*raytraceGraph);
        }
    }

    inline void copyInTransforms(Vector3 *geom_positions,
                                 Quat *geom_rotations,
                                 Vector3 *cam_positions,
                                 Quat *cam_rotations,
                                 cudaStream_t strm)
    {
        cudaMemcpyAsync(
            gpuExec.getExported((CountT)ExportID::InstancePositions),
            geom_positions,
            sizeof(Vector3) * numGeoms * cfg.numWorlds,
            cudaMemcpyDeviceToDevice, strm);
        cudaMemcpyAsync(
            gpuExec.getExported((CountT)ExportID::InstanceRotations),
            geom_rotations,
            sizeof(Quat) * numGeoms * cfg.numWorlds,
            cudaMemcpyDeviceToDevice, strm);
        cudaMemcpyAsync(
            gpuExec.getExported((CountT)ExportID::CameraPositions),
            cam_positions,
            sizeof(Vector3) * numCams * cfg.numWorlds,
            cudaMemcpyDeviceToDevice, strm);
        cudaMemcpyAsync(
            gpuExec.getExported((CountT)ExportID::CameraRotations),
            cam_rotations,
            sizeof(Quat) * numCams * cfg.numWorlds,
            cudaMemcpyDeviceToDevice, strm);
    }

    inline void copyInProperties(
        Diag3x3 *geom_sizes,
        int32_t *mat_overrides,
        uint32_t *col_overrides,
        Vector3 *light_pos,
        Vector3 *light_dir,
        bool *light_isdir,
        bool *light_castshadow,
        float *light_cutoff,
        float *light_intensity,
        cudaStream_t strm)
    {
        cudaMemcpyAsync(
            gpuExec.getExported((CountT)ExportID::InstanceMatOverrides),
            mat_overrides,
            sizeof(MaterialOverride) * numGeoms * cfg.numWorlds,
            cudaMemcpyDeviceToDevice, strm);
        cudaMemcpyAsync(
            gpuExec.getExported((CountT)ExportID::InstanceColorOverrides),
            col_overrides,
            sizeof(ColorOverride) * numGeoms * cfg.numWorlds,
            cudaMemcpyDeviceToDevice, strm);
        cudaMemcpyAsync(
            gpuExec.getExported((CountT)ExportID::InstanceScales),
            geom_sizes,
            sizeof(Diag3x3) * numGeoms * cfg.numWorlds,
            cudaMemcpyDeviceToDevice, strm);

        // Copy light properties to GPU
        cudaMemcpyAsync(
            gpuExec.getExported((CountT)ExportID::LightPositions),
            light_pos,
            sizeof(Vector3) * numLights * cfg.numWorlds,
            cudaMemcpyDeviceToDevice, strm);
        cudaMemcpyAsync(
            gpuExec.getExported((CountT)ExportID::LightDirections),
            light_dir,
            sizeof(Vector3) * numLights * cfg.numWorlds,
            cudaMemcpyDeviceToDevice, strm);
        cudaMemcpyAsync(
            gpuExec.getExported((CountT)ExportID::LightTypes),
            light_isdir,
            sizeof(bool) * numLights * cfg.numWorlds,
            cudaMemcpyDeviceToDevice, strm);
        cudaMemcpyAsync(
            gpuExec.getExported((CountT)ExportID::LightShadows),
            light_castshadow,
            sizeof(bool) * numLights * cfg.numWorlds,
            cudaMemcpyDeviceToDevice, strm);
        cudaMemcpyAsync(
            gpuExec.getExported((CountT)ExportID::LightCutoffAngles),
            light_cutoff,
            sizeof(float) * numLights * cfg.numWorlds,
            cudaMemcpyDeviceToDevice, strm);
        cudaMemcpyAsync(
            gpuExec.getExported((CountT)ExportID::LightIntensities),
            light_intensity,
            sizeof(float) * numLights * cfg.numWorlds,
            cudaMemcpyDeviceToDevice, strm);
    }

    inline void init(const Vector3 *geom_positions,
                     const Quat *geom_rotations,
                     const Vector3 *cam_positions,
                     const Quat *cam_rotations,
                     const int32_t *mat_ids,
                     const uint32_t *geom_rgb,
                     const Diag3x3 *geom_sizes,
                     const Vector3 *light_pos,
                     const Vector3 *light_dir,
                     const bool *light_isdir,
                     const bool *light_castshadow,
                     const float *light_cutoff,
                     const float *light_intensity)
    {
        MWCudaLaunchGraph init_graph =
            gpuExec.buildLaunchGraph(TaskGraphID::Init);
        
        MWCudaLaunchGraph render_init_graph =
            gpuExec.buildLaunchGraph(TaskGraphID::RenderInit);

        gpuExec.run(init_graph);

        copyInTransforms(
            const_cast<Vector3 *>(geom_positions),
            const_cast<Quat *>(geom_rotations),
            const_cast<Vector3 *>(cam_positions),
            const_cast<Quat *>(cam_rotations), 0);
        copyInProperties(
            const_cast<Diag3x3 *>(geom_sizes),
            const_cast<int32_t *>(mat_ids),
            const_cast<uint32_t *>(geom_rgb),
            const_cast<Vector3 *>(light_pos),
            const_cast<Vector3 *>(light_dir),
            const_cast<bool *>(light_isdir),
            const_cast<bool *>(light_castshadow),
            const_cast<float *>(light_cutoff),
            const_cast<float *>(light_intensity), 0);

        gpuExec.run(render_init_graph);
        renderImpl();
    }

    inline void render(const Vector3 *geom_pos,
                             const Quat *geom_rot,
                             const Vector3 *cam_pos,
                             const Quat *cam_rot)
    {
        copyInTransforms(
            (Vector3 *)geom_pos,
            (Quat *)geom_rot,
            (Vector3 *)cam_pos,
            (Quat *)cam_rot, 0);

        gpuExec.runAsync(renderGraph, 0);
        // Currently a CPU sync is needed to read back the total number of
        // instances for Vulkan
        // TODO: Can we remove this? where is the total number read after this?
        REQ_CUDA(cudaStreamSynchronize(0));

        renderImpl();
    }

    inline const float * getDepthOut() const
    {
        if (cfg.useRT) {
            return (float *)gpuExec.getExported((uint32_t)ExportID::RaycastDepth);
        } else {
            return renderMgr->batchRendererDepthOut();
        }
    }

    inline const uint8_t * getRGBOut() const
    {
        if (cfg.useRT) {
            return (uint8_t *)gpuExec.getExported((uint32_t)ExportID::RaycastRGB);
        } else {
            return renderMgr->batchRendererRGBOut();
        }
    }

    inline void copyOutRendered(uint8_t *rgb_out, float *depth_out,
                                cudaStream_t strm)
    {  
        cudaMemcpyAsync(depth_out, getDepthOut(),
                        sizeof(float) *
                        (size_t)cfg.batchRenderViewWidth *
                        (size_t)cfg.batchRenderViewHeight *
                        (size_t)cfg.numWorlds *
                        (size_t)numCams,
                        cudaMemcpyDeviceToDevice, strm);

        cudaMemcpyAsync(rgb_out, getRGBOut(),
                        sizeof(uint8_t) *
                        (size_t)cfg.batchRenderViewWidth *
                        (size_t)cfg.batchRenderViewHeight *
                        (size_t)cfg.numWorlds *
                        (size_t)numCams * 4,
                        cudaMemcpyDeviceToDevice, strm);
    }

    inline void gpuStreamInit(cudaStream_t strm, void **buffers)
    {
        MWCudaLaunchGraph init_graph =
            gpuExec.buildLaunchGraph(TaskGraphID::Init);
        
        MWCudaLaunchGraph render_init_graph =
            gpuExec.buildLaunchGraph(TaskGraphID::RenderInit);

        JAXIO jax_io = JAXIO::makeInit(buffers);

        gpuExec.runAsync(init_graph, strm);

        copyInTransforms(
            jax_io.geomPositions,
            jax_io.geomRotations,
            jax_io.camPositions,
            jax_io.camRotations,
            strm);
        
        copyInProperties(
            jax_io.geomSizes,
            jax_io.matIDs,
            jax_io.geomRGB,
            jax_io.lightPos,
            jax_io.lightDir,
            jax_io.lightIsDir,
            jax_io.lightCastShadow,
            jax_io.lightCutoff,
            jax_io.lightIntensity,
            strm);

        gpuExec.runAsync(render_init_graph, strm);

        // Currently a CPU sync is needed to read back the total number of
        // instances for Vulkan
        REQ_CUDA(cudaStreamSynchronize(strm));

        renderImpl();

        copyOutRendered(jax_io.rgbOut, jax_io.depthOut, strm);
    }

    inline void gpuStreamRender(cudaStream_t strm, void **buffers)
    {
        JAXIO jax_io = JAXIO::makeRender(buffers);

        copyInTransforms(jax_io.geomPositions, jax_io.geomRotations,
                         jax_io.camPositions, jax_io.camRotations, strm);

        gpuExec.runAsync(renderGraph, strm);
        // Currently a CPU sync is needed to read back the total number of
        // instances for Vulkan
        REQ_CUDA(cudaStreamSynchronize(strm));

        renderImpl();

        copyOutRendered(jax_io.rgbOut, jax_io.depthOut, strm);
    }

    inline Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dims) const
    {
        void *dev_ptr = gpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, cfg.gpuID);
    }
};

struct RTAssets {
    render::MeshBVHData bvhData;
    render::MaterialData matData;
};

static RTAssets loadRenderObjects(
    const MJXModel &model,
    Optional<render::RenderManager> &render_mgr,
    bool use_rt)
{
    StackAlloc tmp_alloc;

    std::array<std::string, 1> 
        render_asset_paths;
    render_asset_paths[(size_t)RenderPrimObjectIDs::DebugCam] =
        (std::filesystem::path(DATA_DIR) / "debugcam.obj").string();

    std::array<const char *, render_asset_paths.size()> render_asset_cstrs;
    for (size_t i = 0; i < render_asset_paths.size(); i++) {
        render_asset_cstrs[i] = render_asset_paths[i].c_str();
    }

    AssetImporter asset_importer;

    std::array<char, 1024> import_err;
    auto disk_render_assets = asset_importer.importFromDisk(
        render_asset_cstrs, Span<char>(import_err.data(), import_err.size()));

    if (!disk_render_assets.has_value()) {
        FATAL("Failed to load render assets from disk: %s", import_err);
    }

    // Used to store the loaded geomentry data for pointers to remain valid
    ImportedAssets generated_assets {
        .geoData = ImportedAssets::GeometryData {
            .positionArrays { 0 },
            .normalArrays { 0 },
            .tangentAndSignArrays { 0 },
            .uvArrays { 0 },
            .indexArrays { 0 },
            .faceCountArrays { 0 },
            .meshArrays { 0 },
        },
        .objects { 0 },
        .materials { 0 },
        .instances { 0 },
        .textures { 0 },
    };

    HeapArray<SourceMesh> meshes(
        model.meshGeo.numMeshes + (size_t)RenderPrimObjectIDs::NumPrims);
    const CountT num_meshes = (CountT)model.meshGeo.numMeshes;

    meshes[(size_t)RenderPrimObjectIDs::DebugCam] = 
        disk_render_assets->objects[(size_t)RenderPrimObjectIDs::DebugCam].meshes[0];
    meshes[(size_t)RenderPrimObjectIDs::Plane] = CreatePlane(generated_assets);
    meshes[(size_t)RenderPrimObjectIDs::Sphere] = CreateSphere(generated_assets);
    meshes[(size_t)RenderPrimObjectIDs::Box] = CreateBox(generated_assets);
    meshes[(size_t)RenderPrimObjectIDs::Cylinder] = CreateCylinder(generated_assets);
    meshes[(size_t)RenderPrimObjectIDs::Capsule] = CreateCapsule(generated_assets);
    
    for (CountT mesh_idx = 0; mesh_idx < num_meshes; mesh_idx++) {
        uint32_t mesh_vert_offset = model.meshGeo.vertexOffsets[mesh_idx];
        uint32_t next_vert_offset = mesh_idx < num_meshes - 1 ?
            model.meshGeo.vertexOffsets[mesh_idx + 1] : model.meshGeo.numVertices;

        uint32_t mesh_tri_offset = model.meshGeo.triOffsets[mesh_idx];
        uint32_t next_tri_offset = mesh_idx < num_meshes - 1 ?
            model.meshGeo.triOffsets[mesh_idx + 1] : model.meshGeo.numTris;

        uint32_t mesh_num_verts = next_vert_offset - mesh_vert_offset;
        uint32_t mesh_num_tris = next_tri_offset - mesh_tri_offset;
        uint32_t mesh_idx_offset = mesh_tri_offset * 3;

        math::Vector2 *uvs;
        if (model.meshGeo.texCoordOffsets[mesh_idx] != -1) {
            uvs = model.meshGeo.texCoords + model.meshGeo.texCoordOffsets[mesh_idx];
        } else {
            uvs = nullptr;
        }

        meshes[mesh_idx + (size_t)RenderPrimObjectIDs::NumPrims] = {
            .positions = model.meshGeo.vertices + mesh_vert_offset,
            .normals = nullptr,
            .tangentAndSigns = nullptr,
            .uvs = uvs,
            .indices = model.meshGeo.indices + mesh_idx_offset,
            .faceCounts = nullptr,
            .faceMaterials = nullptr,
            .numVertices = mesh_num_verts,
            .numFaces = mesh_num_tris,
            .materialIDX = 0,
        };
    }

    SourceTexture *out_textures = tmp_alloc.allocN<SourceTexture>(model.numTextures);

    for (CountT i = 0; i < model.numTextures; i++) {
        uint32_t tex_offset = model.texOffsets[i];
        Optional<SourceTexture> tex = SourceTexture {
            .data = &model.texData[tex_offset],
            .format = SourceTextureFormat::R8G8B8A8,
            .width = (uint32_t)model.texWidths[i],
            .height = (uint32_t)model.texHeights[i],
            .numBytes = (size_t)(model.texWidths[i] * model.texHeights[i] * 4),
        };
        out_textures[i] = *tex;
    }

    Span<imp::SourceTexture> imported_textures = Span(out_textures, model.numTextures);

    std::vector<imp::SourceMaterial> materials;
    for (CountT i = 0; i < model.numMats; i++) {
        int32_t tex_idx = model.matTexIDs[i * 10];
        SourceMaterial mat = {
            .color = math::Vector4{
                model.matRGBA[i].x, model.matRGBA[i].y,
                model.matRGBA[i].z, model.matRGBA[i].w},
            .textureIdx = tex_idx,
            .roughness = 0.0f,
            .metalness = 0.0f};
        materials.push_back(mat);
    }

    // Create materials for geoms that do not have one assigned
    for (CountT i = 0; i < model.numGeoms; i++) {
        if (model.geomMatIDs[i] == -1) {
            SourceMaterial mat = {
                .color = math::Vector4{
                    model.geomRGBA[i].x, model.geomRGBA[i].y,
                    model.geomRGBA[i].z, model.geomRGBA[i].w},
                .textureIdx = -1,
                .roughness = 0.8f,
                .metalness = 0.2f,
            };
            materials.push_back(mat);
            model.geomMatIDs[i] = materials.size() - 1;

            for (CountT j = i + 1; j < model.numGeoms; j++) {
                // FIX: Should probably implement == op for Vector4
                if (model.geomMatIDs[j] == -1 && 
                    model.geomRGBA[i].x == model.geomRGBA[j].x &&
                    model.geomRGBA[i].y == model.geomRGBA[j].y &&
                    model.geomRGBA[i].z == model.geomRGBA[j].z &&
                    model.geomRGBA[i].w == model.geomRGBA[j].w) 
                {
                    model.geomMatIDs[j] = materials.size() - 1;
                }
            }
        }
    }

    HeapArray<SourceObject> objs(model.numGeoms + 1);

    // Create a new mesh for each geom to avoid geoms that share the same mesh
    // from pointing to the same source mesh
    HeapArray<SourceMesh> dest_meshes(model.numGeoms + 1);
    
    for (CountT i = 0; i < model.numGeoms; i++) {
        int source_mesh_idx = -1;
        switch ((MJXGeomType)model.geomTypes[i]) {
        case MJXGeomType::Plane: {
            source_mesh_idx = (int)RenderPrimObjectIDs::Plane;
        } break;
        case MJXGeomType::Sphere: {
            source_mesh_idx = (int)RenderPrimObjectIDs::Sphere;
        } break;
        case MJXGeomType::Capsule: {
            dest_meshes[i] = CreateCapsule(
                generated_assets,
                model.geomSizes[i].x,
                model.geomSizes[i].y * 2);
            dest_meshes[i].materialIDX = static_cast<uint32_t>(model.geomMatIDs[i]);
        } break;
        case MJXGeomType::Box: {
            source_mesh_idx = (int)RenderPrimObjectIDs::Box;
        } break;
        case MJXGeomType::Cylinder: {
            source_mesh_idx = (int)RenderPrimObjectIDs::Cylinder;
        } break;
        case MJXGeomType::Mesh: {
            source_mesh_idx = (int)RenderPrimObjectIDs::NumPrims + model.geomDataIDs[i];
        } break;
        case MJXGeomType::Heightfield:
        case MJXGeomType::Ellipsoid:
        default:
            FATAL("Unsupported geom type");
            break;
        }

        if (source_mesh_idx != -1) {
            const SourceMesh& source_mesh = meshes[source_mesh_idx];
            dest_meshes[i] = {
                .positions = source_mesh.positions,
                .normals = source_mesh.normals,
                .tangentAndSigns = source_mesh.tangentAndSigns,
                .uvs = source_mesh.uvs,
                .indices = source_mesh.indices,
                .faceCounts = source_mesh.faceCounts,
                .faceMaterials = source_mesh.faceMaterials,
                .numVertices = source_mesh.numVertices,
                .numFaces = source_mesh.numFaces,
                .materialIDX = static_cast<uint32_t>(model.geomMatIDs[i]),
            };
        }

        objs[i] = {
            .meshes = Span<SourceMesh>(&dest_meshes[i], 1),
        };

        model.geomDataIDs[i] = -1;
        for (CountT geom_i = 0; geom_i < model.numEnabledGeomGroups; geom_i++)
        {
            if (model.geomGroups[i] == model.enabledGeomGroups[geom_i])
            {
                model.geomDataIDs[i] = i;
                break;
            }
        }
    }
    
    objs[model.numGeoms] = disk_render_assets->objects[(int)RenderPrimObjectIDs::DebugCam];
    for (auto &mesh : objs[model.numGeoms].meshes) {
        mesh.materialIDX = 0;
    }

    if (render_mgr.has_value()) {
        render_mgr->loadObjects(objs, materials, imported_textures);
    }

    if (use_rt) {
        auto ret = RTAssets {
            render::AssetProcessor::makeBVHData(objs),
            render::AssetProcessor::initMaterialData(materials.data(),
                                     materials.size(),
                                     imported_textures.data(),
                                     imported_textures.size())
        };

        return ret;
    } else {
        return {};
    }
}

Manager::Impl * Manager::Impl::make(
    const Manager::Config &mgr_cfg,
    const MJXModel &mjx_model,
    const Optional<VisualizerGPUHandles> &viz_gpu_hdls)
{
    bool use_rt = mgr_cfg.useRT;

    if (use_rt) {
        printf("Using raytracer\n");
    } else {
        printf("Using rasterizer\n");
    }

    Sim::Config sim_cfg;
    sim_cfg.numGeoms = mjx_model.numGeoms;
    sim_cfg.numCams = mjx_model.numCams;
    sim_cfg.numLights = mjx_model.numLights;
    sim_cfg.useDebugCamEntity = mgr_cfg.addCamDebugGeometry;
    sim_cfg.useRT = use_rt;

    CUcontext cu_ctx = MWCudaExecutor::initCUDA(mgr_cfg.gpuID);

    Optional<RenderGPUState> render_gpu_state =
        initRenderGPUState(mgr_cfg, viz_gpu_hdls);

    Optional<render::RenderManager> render_mgr =
        initRenderManager(mgr_cfg, mjx_model,
                          viz_gpu_hdls, render_gpu_state);

    RTAssets rt_assets = loadRenderObjects(
            mjx_model, render_mgr, use_rt);
    if (render_mgr.has_value()) {
        sim_cfg.renderBridge = render_mgr->bridge();
    } else {
        sim_cfg.renderBridge = nullptr;
    }

    int32_t *geom_types_gpu = (int32_t *)cu::allocGPU(
        sizeof(int32_t) * mjx_model.numGeoms);
    int32_t *geom_data_ids_gpu = (int32_t *)cu::allocGPU(
        sizeof(int32_t) * mjx_model.numGeoms);
    Vector3 *geom_sizes_gpu = (Vector3 *)cu::allocGPU(
        sizeof(Vector3) * mjx_model.numGeoms);
    float *cam_fovy = (float * )cu::allocGPU(
        sizeof(float) * mjx_model.numCams);

    REQ_CUDA(cudaMemcpy(geom_types_gpu, mjx_model.geomTypes,
        sizeof(int32_t) * mjx_model.numGeoms, cudaMemcpyHostToDevice));
    REQ_CUDA(cudaMemcpy(geom_data_ids_gpu, mjx_model.geomDataIDs,
        sizeof(int32_t) * mjx_model.numGeoms, cudaMemcpyHostToDevice));
    REQ_CUDA(cudaMemcpy(geom_sizes_gpu, mjx_model.geomSizes,
        sizeof(Vector3) * mjx_model.numGeoms, cudaMemcpyHostToDevice));
    REQ_CUDA(cudaMemcpy(cam_fovy, mjx_model.camFovy,
        sizeof(float) * mjx_model.numCams, cudaMemcpyHostToDevice));

    sim_cfg.geomTypes = geom_types_gpu;
    sim_cfg.geomDataIDs = geom_data_ids_gpu;
    sim_cfg.geomSizes = geom_sizes_gpu;
    sim_cfg.camFovy = cam_fovy;

    HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);

    Optional<CudaBatchRenderConfig> render_cfg = 
        Optional<CudaBatchRenderConfig>::none();
    if (use_rt) {
        render_cfg = {
            .renderMode = CudaBatchRenderConfig::RenderMode::RGBD,
            .geoBVHData = rt_assets.bvhData,
            .materialData = rt_assets.matData,
            .renderWidth = mgr_cfg.batchRenderViewWidth,
            .renderHeight = mgr_cfg.batchRenderViewHeight,
            .nearPlane = 0.001f,
            .farPlane = 1000.0f,
        };
    }

    MWCudaExecutor gpu_exec({
        .worldInitPtr = world_inits.data(),
        .numWorldInitBytes = sizeof(Sim::WorldInit),
        .userConfigPtr = (void *)&sim_cfg,
        .numUserConfigBytes = sizeof(Sim::Config),
        .numWorldDataBytes = sizeof(Sim),
        .worldDataAlignment = alignof(Sim),
        .numWorlds = mgr_cfg.numWorlds,
        .numTaskGraphs = (uint32_t)TaskGraphID::NumGraphs,
        .numExportedBuffers = (uint32_t)ExportID::NumExports, 
    }, {
        { GPU_HIDESEEK_SRC_LIST },
        { GPU_HIDESEEK_COMPILE_FLAGS },
        CompileConfig::OptMode::LTO,
    }, cu_ctx, render_cfg);

    Optional<MWCudaLaunchGraph> raytrace_graph =
        Optional<MWCudaLaunchGraph>::none();

    if (use_rt) {
        raytrace_graph = gpu_exec.buildRenderGraph();
    }

    cu::deallocGPU(geom_types_gpu);
    cu::deallocGPU(geom_data_ids_gpu);
    cu::deallocGPU(geom_sizes_gpu);
    cu::deallocGPU(cam_fovy);

    return new Impl {
        mgr_cfg,
        mjx_model.numGeoms,
        mjx_model.numCams,
        mjx_model.numLights,
        std::move(render_gpu_state),
        std::move(render_mgr),
        std::move(gpu_exec),
        std::move(raytrace_graph)
    };
}

Manager::Manager(const Config &cfg,
                 const MJXModel &mjx_model,
                 Optional<VisualizerGPUHandles> viz_gpu_hdls)
    : impl_(Impl::make(cfg, mjx_model, viz_gpu_hdls))
{}

Manager::~Manager() {}

void Manager::init(const math::Vector3 *geom_pos, const math::Quat *geom_rot,
                   const math::Vector3 *cam_pos, const math::Quat *cam_rot,
                   const int32_t *mat_ids, const uint32_t *geom_rgb,
                   const math::Diag3x3 *geom_sizes, const math::Vector3 *light_pos,
                   const math::Vector3 *light_dir, const bool *light_isdir,
                   const bool *light_castshadow, const float *light_cutoff,
                   const float *light_intensity)
{
    impl_->init(
        geom_pos, geom_rot, cam_pos, cam_rot, mat_ids, geom_rgb, geom_sizes,
        light_pos, light_dir, light_isdir, light_castshadow, 
        light_cutoff, light_intensity);
}

void Manager::render(const math::Vector3 *geom_pos, const math::Quat *geom_rot,
                     const math::Vector3 *cam_pos, const math::Quat *cam_rot)
{
    impl_->render(geom_pos, geom_rot, cam_pos, cam_rot);
}

#ifdef MADRONA_CUDA_SUPPORT
void Manager::gpuStreamInit(cudaStream_t strm, void **buffers)
{
    impl_->gpuStreamInit(strm, buffers);
}

void Manager::gpuStreamRender(cudaStream_t strm, void **buffers)
{
    impl_->gpuStreamRender(strm, buffers);
}
#endif

Tensor Manager::instancePositionsTensor() const
{
    return impl_->exportTensor(ExportID::InstancePositions,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->numGeoms,
                                   sizeof(Vector3) / sizeof(float),
                               });
}

Tensor Manager::instanceRotationsTensor() const
{
    return impl_->exportTensor(ExportID::InstanceRotations,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->numGeoms,
                                   sizeof(Quat) / sizeof(float),
                               });
}

Tensor Manager::cameraPositionsTensor() const
{
    return impl_->exportTensor(ExportID::CameraPositions,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->numCams,
                                   sizeof(Vector3) / sizeof(float),
                               });
}

Tensor Manager::cameraRotationsTensor() const
{
    return impl_->exportTensor(ExportID::CameraRotations,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->numCams,
                                   sizeof(Quat) / sizeof(float),
                               });
}

Tensor Manager::rgbTensor() const
{
    const uint8_t *rgb_ptr = impl_->getRGBOut();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("1 CUDA error during async copy execution: %s\n", cudaGetErrorString(err));
    }
    auto res = Tensor((void*)rgb_ptr, TensorElementType::UInt8, {
        impl_->cfg.numWorlds,
        impl_->numCams,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        4,
    }, impl_->cfg.gpuID);


    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("1 CUDA error during async copy execution: %s\n", cudaGetErrorString(err));
    }

    return res;
}

Tensor Manager::depthTensor() const
{
    const float *depth_ptr = impl_->getDepthOut();

    return Tensor((void *)depth_ptr, TensorElementType::Float32, {
        impl_->cfg.numWorlds,
        impl_->numCams,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        1,
    }, impl_->cfg.gpuID);
}

uint32_t Manager::numWorlds() const
{
    return impl_->cfg.numWorlds;
}

uint32_t Manager::numCams() const
{
    return impl_->numCams;
}

uint32_t Manager::batchViewWidth() const
{
    return impl_->cfg.batchRenderViewWidth;
}

uint32_t Manager::batchViewHeight() const
{
    return impl_->cfg.batchRenderViewHeight;
}

render::RenderManager & Manager::getRenderManager()
{
    return *(impl_->renderMgr);
}

}
