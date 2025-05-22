#include "shader_utils.hlsl"

[[vk::push_constant]]
GridDrawPushConst pushConst;

[[vk::binding(0, 0)]]
RWTexture2D<float4> gridOutput;

[[vk::binding(1, 0)]]
StructuredBuffer<uint32_t> batchRenderRGBOut;

[[vk::binding(2, 0)]]
StructuredBuffer<float> batchRenderDepthOut;

float vizDepth(float v)
{
    return log(v + 0.9f) / log(10000.f);
}

float srgbToLinear(float srgb)
{
    if (srgb <= 0.04045f) {
        return srgb / 12.92f;
    }

    return pow((srgb + 0.055f) / 1.055f, 2.4f);
}

float4 rgb8ToFloat(uint r, uint g, uint b)
{
    return float4(
        srgbToLinear((float)r / 255.f),
        srgbToLinear((float)g / 255.f),
        srgbToLinear((float)b / 255.f),
        1.f);
}

float4 fetchBatchRenderPixel(uint3 pixel_idx)
{
    pixel_idx.x = clamp(pixel_idx.x, 0, pushConst.viewWidth - 1);
    pixel_idx.y = clamp(pixel_idx.y, 0, pushConst.viewHeight - 1);

    uint linear_pixel_idx =
        pixel_idx.z * pushConst.viewHeight * pushConst.viewWidth +
        pixel_idx.y * pushConst.viewWidth + pixel_idx.x;

    if (pushConst.showDepth == 1) {
        float depth = vizDepth(batchRenderDepthOut[linear_pixel_idx]);
        return float4(float3(depth, depth, depth), 1);
    } else {
        uint packed = batchRenderRGBOut[linear_pixel_idx];

        uint r = packed & 0xFF;
        uint g = (packed >> 8) & 0xFF;
        uint b = (packed >> 16) & 0xFF;

        return rgb8ToFloat(r, g, b);
    }
}

float4 sampleBatchRenderOutput(float2 uv, uint view_idx)
{
    float2 coords =
        uv * float2(pushConst.viewWidth, pushConst.viewHeight);

    float2 base = floor(coords);
    float2 diff = coords - base;
    uint2 c00 = (uint2)base;
    uint2 c01 = c00 + uint2(1, 0);
    uint2 c10 = c00 + uint2(0, 1);
    uint2 c11 = c00 + uint2(1, 1);

    float4 p00 = fetchBatchRenderPixel(uint3(c00, view_idx));
    float4 p01 = fetchBatchRenderPixel(uint3(c01, view_idx));
    float4 p10 = fetchBatchRenderPixel(uint3(c10, view_idx));
    float4 p11 = fetchBatchRenderPixel(uint3(c11, view_idx));

    float4 a = p00 + diff.x * (p01 - p00);
    float4 b = p10 + diff.x * (p11 - p10);

    return a + diff.y * (b - a);
}

[numThreads(32, 32, 1)]
[shader("compute")]
void gridDraw(uint3 idx : SV_DispatchThreadID)
{
    uint2 target_dim;
    gridOutput.GetDimensions(target_dim.x, target_dim.y);

    if (idx.x >= target_dim.x || idx.y >= target_dim.y) {
        return;
    }

    // Get the view index that this pixel is going to sample from
    float global_pixel_x = pushConst.offsetX + float(idx.x);
    float global_pixel_y = pushConst.offsetY + float(idx.y);

    float ratio_x = global_pixel_x / float(pushConst.gridViewSize);
    float ratio_y = global_pixel_y / float(pushConst.gridViewSize);

    int view_idx_x = floor(ratio_x);
    int view_idx_y = floor(ratio_y);

    float uv_x = ratio_x - float(view_idx_x);
    float uv_y = ratio_y - float(view_idx_y);

    if (view_idx_x < 0 || view_idx_y < 0 || view_idx_x >= pushConst.gridWidth) {
        gridOutput[idx.xy] = float4(0, 0, 0, 0);
        return;
    }

    // Get linear view index
    int view_idx = view_idx_x + view_idx_y * pushConst.gridWidth;

    if (view_idx >= pushConst.numViews) {
        gridOutput[idx.xy] = float4(0, 0, 0, 0);
        return;
    }

    gridOutput[idx.xy] = sampleBatchRenderOutput(float2(uv_x, uv_y), view_idx);
}
