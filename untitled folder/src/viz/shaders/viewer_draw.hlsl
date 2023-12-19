#include "shader_common.h"
#include "../../render/vk/shaders/utils.hlsl"

[[vk::push_constant]]
DrawPushConst push_const;

[[vk::binding(0, 0)]]
StructuredBuffer<PackedViewData> viewDataBuffer;

[[vk::binding(1, 0)]]
StructuredBuffer<PackedInstanceData> engineInstanceBuffer;

[[vk::binding(2, 0)]]
StructuredBuffer<DrawData> drawDataBuffer;

[[vk::binding(3, 0)]]
StructuredBuffer<ShadowViewData> shadowViewDataBuffer;

[[vk::binding(4, 0)]]
cbuffer TimeBuffer : register(b4) {
    float time;
};




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

float4 composeQuats(float4 a, float4 b)
{
    return float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

float3 rotateVec(float4 q, float3 v)
{
    float3 pure = q.xyz;
    float scalar = q.w;
    
    float3 pure_x_v = cross(pure, v);
    float3 pure_x_pure_x_v = cross(pure, pure_x_v);
    
    return v + 2.f * ((pure_x_v * scalar) + pure_x_pure_x_v);
}

PerspectiveCameraData unpackViewData(PackedViewData packed)
{
    const float4 d0 = packed.data[0];
    const float4 d1 = packed.data[1];
    const float4 d2 = packed.data[2];

    PerspectiveCameraData cam;
    cam.pos = d0.xyz;
    cam.rot = float4(d1.xyz, d0.w);
    cam.xScale = d1.w;
    cam.yScale = d2.x;
    cam.zNear = d2.y;

    return cam;
}

Vertex unpackVertex(PackedVertex packed)
{
    const float4 d0 = packed.data[0];
    const float4 d1 = packed.data[1];

    uint3 packed_normal_tangent = uint3(
        asuint(d0.w), asuint(d1.x), asuint(d1.y));

    float3 normal;
    float4 tangent_and_sign;
    decodeNormalTangent(packed_normal_tangent, normal, tangent_and_sign);

    Vertex vert;
    vert.position = float3(d0.x, d0.y, d0.z);
    vert.normal = normal;
    vert.tangentAndSign = tangent_and_sign;
    vert.uv = float2(d1.z, d1.w);

    return vert;
}

EngineInstanceData unpackEngineInstanceData(PackedInstanceData packed)
{
    const float4 d0 = packed.data[0];
    const float4 d1 = packed.data[1];
    const float4 d2 = packed.data[2];

    EngineInstanceData o;
    o.position = d0.xyz;
    o.rotation = float4(d1.xyz, d0.w);
    o.scale = float3(d1.w, d2.xy);
    o.objectID = asint(d2.z);

    return o;
}

#if 1
float3x3 toMat(float4 r)
{
    float x2 = r.x * r.x;
    float y2 = r.y * r.y;
    float z2 = r.z * r.z;
    float xz = r.x * r.z;
    float xy = r.x * r.y;
    float yz = r.y * r.z;
    float wx = r.w * r.x;
    float wy = r.w * r.y;
    float wz = r.w * r.z;

    return float3x3(
        float3(
            1.f - 2.f * (y2 + z2),
            2.f * (xy - wz),
            2.f * (xz + wy)),
        float3(
            2.f * (xy + wz),
            1.f - 2.f * (x2 + z2),
            2.f * (yz - wx)),
        float3(
            2.f * (xz - wy),
            2.f * (yz + wx),
            1.f - 2.f * (x2 + y2)));
}
#endif

void computeCompositeTransform(float3 obj_t,
                               float4 obj_r,
                               float3 cam_t,
                               float4 cam_r_inv,
                               out float3 to_view_translation,
                               out float4 to_view_rotation)
{
    to_view_translation = rotateVec(cam_r_inv, obj_t - cam_t);
    to_view_rotation = normalize(composeQuats(cam_r_inv, obj_r));
}

float hash(float3 p) {
    p*=.02;
    p = frac(p * 0.3183099 + float3(1.0, 17.0, 113.0));
    p *= dot(p, p + 19.19);
    return frac(p.x + p.y + p.z);
}/*
float hash(float3 p) {
    // Simple gradient noise function
    float3 i = floor(p);
    float3 f = frac(p);

    // Smooth interpolation curve
    f = f * f * (3.0 - 2.0 * f);

    // Hash function for corners of the cube
    float n = dot(i, float3(1.0, 57.0, 113.0));

    // Random gradients
    float3 g000 = frac(sin(float3(n, n + 1.0, n + 2.0)) * 43758.5453);
    float3 g001 = frac(sin(float3(n + 3.0, n + 4.0, n + 5.0)) * 43758.5453);
    float3 g010 = frac(sin(float3(n + 6.0, n + 7.0, n + 8.0)) * 43758.5453);
    float3 g011 = frac(sin(float3(n + 9.0, n + 10.0, n + 11.0)) * 43758.5453);
    float3 g100 = frac(sin(float3(n + 12.0, n + 13.0, n + 14.0)) * 43758.5453);
    float3 g101 = frac(sin(float3(n + 15.0, n + 16.0, n + 17.0)) * 43758.5453);
    float3 g110 = frac(sin(float3(n + 18.0, n + 19.0, n + 20.0)) * 43758.5453);
    float3 g111 = frac(sin(float3(n + 21.0, n + 22.0, n + 23.0)) * 43758.5453);

    // Interpolate gradients
    float n000 = dot(g000, f);
    float n100 = dot(g100, f - float3(1.0, 0.0, 0.0));
    float n010 = dot(g010, f - float3(0.0, 1.0, 0.0));
    float n110 = dot(g110, f - float3(1.0, 1.0, 0.0));
    float n001 = dot(g001, f - float3(0.0, 0.0, 1.0));
    float n101 = dot(g101, f - float3(1.0, 0.0, 1.0));
    float n011 = dot(g011, f - float3(0.0, 1.0, 1.0));
    float n111 = dot(g111, f - float3(1.0, 1.0, 1.0));

    // Linear interpolation manually
    float nx00 = n000 + f.x * (n100 - n000);
    float nx01 = n001 + f.x * (n101 - n001);
    float nx10 = n010 + f.x * (n110 - n010);
    float nx11 = n011 + f.x * (n111 - n011);

    float nxy0 = nx00 + f.y * (nx10 - nx00);
    float nxy1 = nx01 + f.y * (nx11 - nx01);

    float nxyz = nxy0 + f.z * (nxy1 - nxy0);

    return nxyz;
}*/



[shader("vertex")]
float4 vert(in uint vid : SV_VertexID,
            in uint draw_id : SV_InstanceID,
            out V2F v2f) : SV_Position
{
    DrawData draw_data = drawDataBuffer[draw_id];

    Vertex vert = unpackVertex(vertexDataBuffer[vid]);

    if (draw_data.materialID == 7) {
        
        float topZ = 1.0; // The Z value of the cube's top edge in its local space
        float edgeThreshold = 0.01; // A small threshold to identify the top face vertices
        /*float waveAmplitude = 1.0; // The height of the wave
        float waveFrequency = 0.5; // The frequency of the wave

        // Check if the vertex is part of the top face
        if (abs(vert.position.z - topZ) < edgeThreshold) {
            // Determine the phase shift based on the x-coordinate
            float phaseShift = (vert.position.x > 0) ? time : -time;

            // Apply the wave transformation
            vert.position.z += waveAmplitude * sin(vert.position.x + phaseShift * waveFrequency * 3.14159 * 2.0);
        }*/
        float waveAmplitude = 1.0; // The height of the wave
        float waveFrequency = 10.0; // The frequency of the wave
        float waveSpeed = 1.0; // Speed at which the wave propagates

        // Check if the vertex is part of the top edge
        if (abs(vert.position.z - topZ) < edgeThreshold) {
            // Calculate the phase based on the vertex's position along the edge and time
            // This will make the wave move along the edge over time
            float phase = vert.position.x * waveFrequency + time * waveSpeed;

            // Apply the wave transformation
            // The sine function will make the vertex move up and down, creating a wave effect
            vert.position.z += waveAmplitude * sin(phase * 3.14159 * 2.0);
        }
        float3 positionFactor = vert.position * 0.5 + 0.5; // Normalize position to 0-1 range

        // Use the original hash function
        float noiseValue = hash(vert.position * 1.0);

        // Map the noise value to a full color spectrum from red to yellow
        float4 colorA = float4(1.0, 0.0, 0.0, 1.0); // Red
        float4 colorB = float4(1.0, 0.1, 0.0, 1.0); // Yellow
        // Introduce flowy motion by offsetting the time for each vertex
        float offsetTime = time + positionFactor.x * 0.5; // Adjust the factor to control the flow speed
        float flowyMotion = 1.0 * sin(offsetTime * .5 * 3.14159 * 2.0);

        // Apply the flowy motion to the color
        noiseValue = saturate(noiseValue);
        
        float4 finalColor = lerp(colorA, colorB, smoothstep(0.0, 1.0, noiseValue)) + float4(flowyMotion, flowyMotion * 0.1, flowyMotion*.005, 0.0);
        v2f.color = finalColor;
        
    } else {
        v2f.color = materialBuffer[draw_data.materialID].color;
    }
    
    uint instance_id = draw_data.instanceID;

    PerspectiveCameraData view_data =
        unpackViewData(viewDataBuffer[push_const.viewIdx]);

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

    // v2f.viewPos = view_pos;
#if 0
    v2f.normal = normalize(
        rotateVec(to_view_rotation, (vert.normal / instance_data.scale)));
#endif
    v2f.normal = normalize(
        rotateVec(instance_data.rotation, (vert.normal / instance_data.scale)));
    v2f.uv = vert.uv;
    v2f.color = v2f.color;
    v2f.position = rotateVec(instance_data.rotation,
                             instance_data.scale * vert.position) + instance_data.position;
    v2f.dummy = shadowViewDataBuffer[0].viewProjectionMatrix[0][0];
    v2f.texIdx = materialBuffer[draw_data.materialID].textureIdx;
    v2f.roughness = materialBuffer[draw_data.materialID].roughness;
    v2f.metalness = materialBuffer[draw_data.materialID].metalness;
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

#if 0
DrawInstanceData unpackDrawInstanceData(PackedDrawInstanceData data)
{
    const float4 d0 = data.packed[0];
    const float4 d1 = data.packed[1];
    const float4 d2 = data.packed[2];
    const float4 d3 = data.packed[3];
    const float4 d4 = data.packed[4];

    DrawInstanceData out;

    float3 rot_col0 = d0.xyz;
    float3 rot_col1 = float3(d0.w, d1.xy);
    float3 rot_col2 = float3(d1.zw, d2.x);

    out.toViewRot = float3x3(
        float3(rot_col0.x, rot_col1.x, rot_col2.x),
        float3(rot_col0.y, rot_col1.y, rot_col2.y),
        float3(rot_col0.z, rot_col1.z, rot_col2.z),
    );
    out.toViewTranslation = d2.yzw;
    out.objScale = d3.xyz;
    out.viewIdx = asint(d3.w);
    out.projScale = d4.xy;
    out.projZNear = d4.z;

    return out;
}
#endif
