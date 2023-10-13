#include "shader_common.h"
#include "../../render/vk/shaders/utils.hlsl"

[[vk::push_constant]]
PushConstant pushConst;

[[vk::binding(0, 0)]]
RWTexture2D<uint2> vizBuffer;

[[vk::binding(1, 0)]]
RWTexture2D<float4> colorBuffer;

uint hash(uint x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

float4 intToColor(uint in_col)
{
    if(in_col==0xffffffff) return float4(1,1,1,1);

    float a=((in_col & 0xff000000) >>24);
    float r=((in_col & 0xff0000) >> 16);
    float g=((in_col & 0xff00) >> 8);
    float b=((in_col & 0xff));
    return float4(r,g,b,255.0)/255.0;
}

[numThreads(32, 32, 1)]
[shader("compute")]
void visualize(uint3 idx : SV_DispatchThreadID)
{
    uint2 target_dim;
    vizBuffer.GetDimensions(target_dim.x, target_dim.y);

    if (idx.x >= target_dim.x || idx.y >= target_dim.y) {
        return;
    }

    uint2 target_pixel = uint2(idx.x, idx.y);

    uint2 in_value = vizBuffer[target_pixel];
    float4 generated_color = intToColor(hash(in_value.x + in_value.y));

    colorBuffer[target_pixel] = generated_color;
}
