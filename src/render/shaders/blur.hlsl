#include "shader_utils.hlsl"

[[vk::push_constant]]
BlurPushConst pushConst;

[[vk::binding(0, 0)]]
RWTexture2D<float2> attachment;

[[vk::binding(1, 0)]]
RWTexture2D<float2> intermediate;

#define WEIGHT_COUNT 5

[numThreads(32, 32, 1)]
[shader("compute")]
void blur(uint3 idx : SV_DispatchThreadID)
{
    uint2 targetDim;
    attachment.GetDimensions(targetDim.x, targetDim.y);

    if (idx.x < targetDim.x && idx.y < targetDim.y) {
        uint2 targetPixel = idx.xy;

        const float weights[WEIGHT_COUNT] = {
#if 0
            20.0 / 64.0,
            15.0 / 64.0,
            6.0 / 64.0,
            1.0 / 64.0
#endif
            0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216 
            // 1.0
        };

        if (pushConst.isVertical) {
            // Read from  intermediate, write to attachment
            float2 output = intermediate[targetPixel].rg * weights[0];
            
            for (int i = 1; i < WEIGHT_COUNT; ++i) {
                output += intermediate[targetPixel + uint2(0, i)].rg * weights[i];
                output += intermediate[targetPixel - uint2(0, i)].rg * weights[i];
            }

            attachment[targetPixel] = output;
        }
        else {
            // Read from attachment, write to intermediate
            float2 output = attachment[targetPixel].rg * weights[0];
            
            for (int i = 1; i < WEIGHT_COUNT; ++i) {
                output += attachment[targetPixel + uint2(i, 0)].rg * weights[i];
                output += attachment[targetPixel - uint2(i, 0)].rg * weights[i];
            }

            intermediate[targetPixel] = output;
        }
    }
}
