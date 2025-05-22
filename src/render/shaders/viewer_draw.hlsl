#include "shader_utils.hlsl"

[[vk::push_constant]]
DrawPushConst push_const;

[[vk::binding(0, 0)]]
StructuredBuffer<PackedViewData> flycamBuffer;

[[vk::binding(1, 0)]]
StructuredBuffer<PackedInstanceData> engineInstanceBuffer;

[[vk::binding(2, 0)]]
StructuredBuffer<DrawData> drawDataBuffer;

[[vk::binding(3, 0)]]
StructuredBuffer<ShadowViewData> shadowViewDataBuffer;

[[vk::binding(4, 0)]]
StructuredBuffer<PackedViewData> viewDataBuffer;

[[vk::binding(5, 0)]]
StructuredBuffer<int> viewOffsetsBuffer;

// Asset descriptor bindings
[[vk::binding(0, 1)]]
StructuredBuffer<PackedVertex> vertexDataBuffer;

[[vk::binding(1, 1)]]
StructuredBuffer<MaterialData> materialBuffer;

// Texture descriptor bindings
[[vk::binding(0, 2)]]
Texture2D<float4> materialTexturesArray[];

[[vk::binding(1, 2)]]
SamplerState linearSampler;

struct V2F {
    [[vk::location(0)]] float3 normal : TEXCOORD0;
    [[vk::location(1)]] float3 position : TEXCOORD1;
    [[vk::location(2)]] float4 color : TEXCOORD2;
    [[vk::location(3)]] float dummy : TEXCOORD3;
    [[vk::location(4)]] float2 uv : TEXCOORD4;
    [[vk::location(5)]] int texIdx : TEXCOORD5;
    [[vk::location(6)]] float roughness : TEXCOORD6;
    [[vk::location(7)]] float metalness : TEXCOORD7;
};

PerspectiveCameraData getCameraData()
{
    PerspectiveCameraData camera_data;

    if (push_const.viewIdx == 0) {
        camera_data = unpackViewData(flycamBuffer[0]);
    } else {
        PerspectiveCameraData fly_cam = unpackViewData(flycamBuffer[0]);

        int view_idx = (push_const.viewIdx - 1) + viewOffsetsBuffer[push_const.worldIdx];
        camera_data = unpackViewData(viewDataBuffer[view_idx]);

        // We want to inherit the aspect ratio from the flycam camera
        camera_data.xScale = fly_cam.xScale;
        camera_data.yScale = fly_cam.yScale;
    }

    return camera_data;
}

[shader("vertex")]
float4 vert(in uint vid : SV_VertexID,
            in uint draw_id : SV_InstanceID,
            out V2F v2f) : SV_Position
{
    DrawData draw_data = drawDataBuffer[draw_id];

    Vertex vert = unpackVertex(vertexDataBuffer[vid]);

    uint instance_id = draw_data.instanceID;
    EngineInstanceData instance_data = unpackEngineInstanceData(
        engineInstanceBuffer[instance_id]);

    PerspectiveCameraData view_data = getCameraData();

    float3 to_view_translation;
    float4 to_view_rotation;
    computeCompositeTransform(instance_data.position, instance_data.rotation,
        view_data.pos, view_data.rot,
        to_view_translation, to_view_rotation);

    float3 view_pos =
        rotateVec(to_view_rotation, instance_data.scale * vert.position) +
            to_view_translation;

#if 0
    float4 clip_pos = float4(
        view_data.xScale * view_pos.x,
        view_data.yScale * view_pos.z,
        view_data.zNear,
        view_pos.y);
#endif

    float4 clip_pos;

    if (push_const.isOrtho == 1) {
        float x_max = push_const.xMax * view_data.xScale;
        float x_min = push_const.xMin * view_data.xScale;

        float y_max = push_const.yMax;
        float y_min = push_const.yMin;

        float z_max = push_const.zMax * (1.0f / -view_data.yScale);
        float z_min = push_const.zMin * (1.0f / -view_data.yScale);

        float4x4 m1 = float4x4(
                float4(2.0f / (x_max - x_min),             0.0f,                       0.0f,                        -(x_max + x_min) / (x_max - x_min)),
                float4(0.0f,                               0.0f,                      -2.0f / (z_max - z_min),      -(z_max+z_min) / (z_max - z_min)),
                float4(0.0f,                               1.0f / (y_max - y_min),     0.0f,                        -(y_min) / (y_max - y_min)),
                float4(0.0f,                               0.0f,                       0.0f,                        1.0f));

        clip_pos = mul(m1, float4(view_pos.x, view_pos.y, view_pos.z, 1.0f));
        clip_pos.z = 1.0 - clip_pos.z;
    }
    else {
#if 0
        clip_pos = float4( view_data.xScale * view_pos.x,
                       view_data.yScale * view_pos.z,
                       view_data.zNear,
                       1.0);
#endif

        clip_pos = float4(
            view_data.xScale * view_pos.x,
            view_data.yScale * view_pos.z,
            view_data.zNear,
            view_pos.y);
    }

    v2f.normal = normalize(
        rotateVec(instance_data.rotation, (vert.normal / instance_data.scale)));
    v2f.uv = vert.uv;

    v2f.position = rotateVec(instance_data.rotation,
                             instance_data.scale * vert.position) + instance_data.position;
    v2f.dummy = shadowViewDataBuffer[0].viewProjectionMatrix[0][0];



    v2f.texIdx = -1;
    // Defaults for now
    v2f.roughness = 0.8;
    v2f.metalness = 0.2;

    if (draw_data.materialID == -2) {
        v2f.color = hexToRgb(draw_data.color);
    } else {
        int32_t material_id = draw_data.materialID;
        
        float4 color = materialBuffer[material_id].color;

        // Material
        v2f.color = color;
        v2f.texIdx = materialBuffer[material_id].textureIdx;
        v2f.roughness = materialBuffer[material_id].roughness;
        v2f.metalness = materialBuffer[material_id].metalness;
    }

    return clip_pos;
}

struct PixelOutput {
    float4 color : SV_Target0;
    float4 normal : SV_Target1;
    float4 position : SV_Target2;
};

[shader("pixel")]
PixelOutput frag(in V2F v2f)
{
    PixelOutput output;
    output.color = v2f.color;
    output.color.a = v2f.roughness;
    output.normal = float4(normalize(v2f.normal), 1.f);
    output.position = float4(v2f.position, v2f.dummy * 0.0000001f);
    output.position.a += v2f.metalness;

    // output.color.rgb = v2f.normal.xyz;

    if (v2f.texIdx != -1) {
        output.color *= materialTexturesArray[v2f.texIdx].SampleLevel(
            linearSampler, float2(v2f.uv.x, 1.f - v2f.uv.y), 0);
    }

    return output;
}
