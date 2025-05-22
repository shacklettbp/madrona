#include "shader_utils.hlsl"

[[vk::push_constant]]
TexturedQuadPushConst pushConst;

[[vk::binding(0, 0)]]
Texture2D<float4> toDisplay;

[[vk::binding(1, 0)]]
SamplerState samplerState;

#if 0
[numThreads(32, 32, 1)]
[shader("compute")]
void main(uint3 idx : SV_DispatchThreadID)
{
    if (idx.x >= pushConst.extentPixels.x || idx.y >= pushConst.extentPixels.y) {
        return;
    }

    uint2 dst_pixel = pushConst.startPixels + idx.xy;
    uint2 src_pixel = idx.xy;
    
    // Source pixel UV coordinate for sampling
    float2 src_pixel_uv = float2(idx.xy) / float2(pushConst.extentPixels);
    float4 src_pixel_value = toDisplay.SampleLevel(samplerState, src_pixel_uv, 0);

    outputImage[dst_pixel] = src_pixel_value;
}
#endif

struct V2F {
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

[shader("vertex")]
float4 vert(in uint vid : SV_VertexID,
            out V2F v2f) : SV_Position
{
    float2 vertices[4] = {
        pushConst.startPixels,
        pushConst.startPixels + float2(pushConst.extentPixels.x, 0),
        pushConst.startPixels + float2(0, pushConst.extentPixels.y),
        pushConst.startPixels + pushConst.extentPixels,
    };

    v2f.uv = (vertices[vid] - pushConst.startPixels) / pushConst.extentPixels;

    float4 pos = float4(2.0f * (vertices[vid] / pushConst.targetExtent) - float2(1.0, 1.0), 0.0, 1.0);

    return pos;
}

struct PixelOutput {
    float4 color : SV_Target0;
};

[shader("pixel")]
PixelOutput frag(in V2F v2f)
{
    PixelOutput output;
    output.color = toDisplay.SampleLevel(samplerState, v2f.uv, 0);
    return output;
}
