#ifndef MADRONA_VIEWER_SHADER_COMMON_H_INCLUDED
#define MADRONA_VIEWER_SHADER_COMMON_H_INCLUDED

struct BlurPushConst {
    /* Vertical blur happens first. */
    uint32_t isVertical;
};

struct CullPushConst {
    uint32_t numInstances;
};

struct DeferredLightingPushConst {
    float4 viewDir;
    float4 viewPos;
    float fovy;
    float exposure;
    float fade_dist;
    uint32_t viewIdx;
};

struct DensityLayer {
    float width;
    float expTerm;
    float expScale;
    float linTerm;
    float constTerm;

    float pad[3];
};

struct DensityProfile {
    DensityLayer layers[2];
};

struct SkyData {
    float4 solarIrradiance;
    float4 rayleighScatteringCoef;
    float4 mieScatteringCoef;
    float4 mieExtinctionCoef;
    float4 absorptionExtinctionCoef;
    float4 groundAlbedo;
    float4 wPlanetCenter;
    float4 sunSize;

    DensityProfile rayleighDensity;
    DensityProfile absorptionDensity;
    DensityProfile mieDensity;

    float solarAngularRadius;
    float bottomRadius;
    float topRadius;
    float miePhaseFunctionG;
    float muSunMin;
    float pad[3];
};

struct DrawPushConst {
    uint32_t viewIdx;
};

struct ShadowGenPushConst {
    uint32_t viewIdx;
};

struct VoxelGenPushConst {
    uint32_t worldX;
    uint32_t worldY;
    uint32_t worldZ;
    float blockWidth;
    uint32_t numBlocks;
};

struct Vertex {
    float3 position;
    float3 normal;
    float4 tangentAndSign;
    float2 uv;
};

struct PackedVertex {
    float4 data[2];
};

struct MeshData {
    int32_t vertexOffset;
    int32_t numVertices;
    int32_t indexOffset;
    int32_t numIndices;
    int32_t materialIndex;
    int32_t pad[3];
};

struct MaterialData {
    // For now, just a color
    float4 color;

    int32_t textureIdx;

    float roughness;
    float metalness;

    int32_t pad[1];
};

struct ObjectData {
    int32_t meshOffset;
    int32_t numMeshes;
};

struct PackedInstanceData {
    float4 data[3];
};

struct EngineInstanceData {
    float3 position;
    float4 rotation;
    float3 scale;
    int32_t objectID;
};

struct PackedViewData {
    float4 data[3];
};

struct ShadowViewData {
    float4x4 viewProjectionMatrix;
    float4x4 cameraViewProjectionMatrix;

    float4 cameraRight;
    float4 cameraUp;
    float4 cameraForward;
};

struct DirectionalLight {
    float4 lightDir;
    float4 color;
};

struct PerspectiveCameraData {
    float3 pos;
    float4 rot;
    float xScale;
    float yScale;
    float zNear;
};

struct DrawCmd {
    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t firstIndex;
    int32_t vertexOffset;
    uint32_t firstInstance;
};

struct DrawData {
    int32_t instanceID;
    int32_t materialID;
};

#if 0
struct PackedDrawInstanceData {
    float4 packed[5];
};

struct DrawInstanceData {
    float3x3 toViewRot;
    float3 toViewTranslation;
    float3 objScale;
    int32_t viewIdx;
    float2 projScale;
    float projZNear;
};
#endif

#endif
