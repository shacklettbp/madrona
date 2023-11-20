#include "shader_common.h"

[[vk::push_constant]]
GridDrawPushConst pushConst;

[[vk::binding(0, 0)]]
RWTexture2D<float4> gridOutput;

[[vk::binding(1, 0)]]
Texture2DArray<float4> batchOutputs[];

[[vk::binding(2, 0)]]
SamplerState samplerState;

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

    uint layered_image_idx = view_idx / pushConst.maxViewsPerImage;

    float4 color = batchOutputs[layered_image_idx].SampleLevel(
        samplerState, float3(uv_x, uv_y, float(view_idx % pushConst.maxViewsPerImage)), 0);

    gridOutput[idx.xy] = color;
}
