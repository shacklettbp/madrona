#include "shader_utils.hlsl"

[[vk::binding(0, 0)]]
RWTexture2D<uint2> vizBuffer;

[[vk::binding(1, 0)]]
RWTexture2D<float4> colorBuffer;

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
