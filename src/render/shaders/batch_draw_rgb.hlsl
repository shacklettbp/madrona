#include "shader_utils.hlsl"

[[vk::push_constant]]
BatchDrawPushConst pushConst;

// Instances and views
[[vk::binding(0, 0)]]
StructuredBuffer<PackedViewData> viewDataBuffer;

[[vk::binding(1, 0)]]
StructuredBuffer<PackedInstanceData> engineInstanceBuffer;

[[vk::binding(2, 0)]]
StructuredBuffer<uint32_t> instanceOffsets;

// TODO: Make this part of lighting shader
[[vk::binding(3, 0)]]
StructuredBuffer<PackedLightData> lightDataBuffer;

// Draw information
[[vk::binding(0, 1)]]
RWStructuredBuffer<uint32_t> drawCount;

[[vk::binding(1, 1)]]
RWStructuredBuffer<DrawCmd> drawCommandBuffer;

[[vk::binding(2, 1)]]
RWStructuredBuffer<DrawDataBR> drawDataBuffer;

// Asset descriptor bindings
[[vk::binding(0, 2)]]
StructuredBuffer<PackedVertex> vertexDataBuffer;

[[vk::binding(1, 2)]]
StructuredBuffer<MeshData> meshDataBuffer;

[[vk::binding(2, 2)]]
StructuredBuffer<MaterialData> materialBuffer;

[[vk::binding(0, 3)]]
Texture2D<float4> materialTexturesArray[];

[[vk::binding(1, 3)]]
SamplerState linearSampler;

struct V2F {
    [[vk::location(0)]] float4 position : SV_Position;
    [[vk::location(1)]] float3 worldPos : TEXCOORD0;
    [[vk::location(2)]] float2 uv : TEXCOORD1;
    [[vk::location(3)]] int materialIdx : TEXCOORD2;
    [[vk::location(4)]] uint color : TEXCOORD3;
    [[vk::location(5)]] float3 worldNormal : TEXCOORD4;
    [[vk::location(6)]] uint worldIdx : TEXCOORD5;
};

[shader("vertex")]
void vert(in uint vid : SV_VertexID,
            in uint draw_id : SV_InstanceID,
            out V2F v2f)
{
    DrawDataBR draw_data = drawDataBuffer[draw_id + pushConst.drawDataOffset];

    Vertex vert = unpackVertex(vertexDataBuffer[vid]);
    uint instance_id = draw_data.instanceID;

    PerspectiveCameraData view_data =
        unpackViewData(viewDataBuffer[draw_data.viewID]);

    EngineInstanceData instance_data = unpackEngineInstanceData(
        engineInstanceBuffer[instance_id]);

    float3 to_view_translation;
    float4 to_view_rotation;
    computeCompositeTransform(instance_data.position, instance_data.rotation,
        view_data.pos, view_data.rot,
        to_view_translation, to_view_rotation);

    float3 view_pos =
        rotateVec(to_view_rotation, instance_data.scale * vert.position) +
            to_view_translation;

    float4 clip_pos = float4(
        view_data.xScale * view_pos.x,
        view_data.yScale * view_pos.z,
        view_data.zNear,
        view_pos.y);

#if 1
    uint something = min(0, instanceOffsets[0]) +
                     min(0, drawCount[0]) +
                     min(0, drawCommandBuffer[0].vertexOffset) +
                     min(0, int(ceil(meshDataBuffer[0].vertexOffset)));

    // v2f.meshID = draw_data.meshID;
#endif

    clip_pos.x += min(0.0, abs(float(draw_data.meshID))) +
                  min(0.0, abs(float(draw_data.instanceID))) +
                  something;

    v2f.worldPos = rotateVec(instance_data.rotation, instance_data.scale * vert.position) + instance_data.position;
    v2f.position = clip_pos;
    v2f.uv = vert.uv;
    v2f.worldNormal = rotateVec(instance_data.rotation, vert.normal);
    v2f.worldIdx = instance_data.worldID;
#if 0
    if (instance_data.matID == -1) {
        v2f.materialIdx = meshDataBuffer[draw_data.meshID].materialIndex;
    } else {
        v2f.materialIdx = instance_data.matID;
    }
#endif

    if (instance_data.matID == -2) {
        v2f.materialIdx = -2;
        v2f.color = instance_data.color;
    } else if (instance_data.matID == -1) {
        v2f.materialIdx = meshDataBuffer[draw_data.meshID].materialIndex;
        v2f.color = 0;
    } else {
        v2f.materialIdx = instance_data.matID;
        v2f.color = 0;
    }
}

struct PixelOutput {
    float4 rgbOut : SV_Target0;
};

[shader("pixel")]
PixelOutput frag(in V2F v2f,
                 in uint prim_id : SV_PrimitiveID)
{
    PixelOutput output;

    if (v2f.materialIdx == -2) {
        output.rgbOut = hexToRgb(v2f.color);

        return output;
    } else {
        MaterialData mat_data = materialBuffer[v2f.materialIdx];
        float4 color = mat_data.color;
        
        if (mat_data.textureIdx != -1) {
            color *= materialTexturesArray[mat_data.textureIdx].SampleLevel(
                    linearSampler, v2f.uv, 0);
        }

        float3 normal = normalize(v2f.worldNormal);
        float3 totalLighting = 0;
        uint numLights = pushConst.numLights;

        [unroll(1)]
        for (uint i = 0; i < numLights; i++) {
            LightDesc light = unpackLightData(lightDataBuffer[v2f.worldIdx * numLights + i]);
            if(!light.active) {
                continue;
            }
            
            float3 ray_dir;            
            if (light.isDirectional) { // Directional light
                ray_dir = normalize(light.direction.xyz);
            } else { // Spot light
                ray_dir = normalize(v2f.worldPos.xyz - light.position.xyz);
                if(light.cutoffAngle >= 0) {
                    float angle = acos(dot(normalize(ray_dir), normalize(light.direction.xyz)));
                    if (abs(angle) > light.cutoffAngle) {
                        continue;
                    }
                }
            }

            float n_dot_l = max(0.0, dot(normal, -ray_dir));
            totalLighting += n_dot_l * light.intensity;
        }

        float3 lighting = totalLighting * color.rgb;

        // Add ambient term
        float ambient = 0.2;
        lighting += color.rgb * ambient;
        
        color.rgb = lighting;
        output.rgbOut = color;

        return output;
    }
}
