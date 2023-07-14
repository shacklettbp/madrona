#include "shader_common.h"
#include "../../render/vk/shaders/utils.hlsl"

[[vk::push_constant]]
ShadowGenPushConst pushConst;

[[vk::binding(0, 0)]]
RWStructuredBuffer<ShadowViewData> shadowViewDataBuffer;

[[vk::binding(1, 0)]]
StructuredBuffer<PackedViewData> viewDataBuffer;

[[vk::binding(2, 0)]]
StructuredBuffer<DirectionalLight> lights;

float4 invQuat(float4 rot)
{
    return float4(-rot.x, -rot.y, -rot.z, rot.w);
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

#if 0
float4x4 lookAt(float3 dir, float3 up)
{
    float3 f = normalize(dir);
    float3 s = normalize(cross(f, up));
    float3 u = normalize(cross(f, s));

    return float4x4(
        float4(s.x, s.y, s.z, 0.0),
        float4(u.x, u.y, u.z, 0.0),
        float4(f.x, f.y, f.z, 0.0),
        float4(0,   0,   0,   1)
    );
}
#endif

float4 toQuat(float3 from, float3 to)
{
    float3 a = cross(from, to);

    float4 q;

    q.xyz = a;
    q.w = sqrt(dot(from, from) * dot(to, to)) + dot(from, to);

    return normalize(q);
}

float4x4 toMat(float4 r)
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

    return float4x4(
        float4(
            1.f - 2.f * (y2 + z2),
            2.f * (xy - wz),
            2.f * (xz + wy), 0.0),
        float4(
            2.f * (xy + wz),
            1.f - 2.f * (x2 + z2),
            2.f * (yz - wx), 0.0),
        float4(
            2.f * (xz - wy),
            2.f * (yz + wx),
            1.f - 2.f * (x2 + y2), 0.0),
        float4(0, 0, 0, 1));
}

[numThreads(32, 1, 1)]
[shader("compute")]
void shadowGen(uint3 idx : SV_DispatchThreadID)
{
    /* Assume that the sun is from lights[0] */
    if (idx.x >= pushConst.numViews+1)
        return;

    PackedViewData viewData = viewDataBuffer[idx.x];

    PerspectiveCameraData unpackedView = unpackViewData(viewData);

    float3 cam_pos = unpackedView.pos;
    float4 cam_rot = invQuat(unpackedView.rot);

    float3 cam_fwd = rotateVec(cam_rot, float3(0.0f, 1.0f, 0.0f));
    float3 cam_up = rotateVec(cam_rot, float3(0.0f, 0.0f, 1.0f));
    float3 cam_right = rotateVec(cam_rot, float3(1.0f, 0.0f, 0.0f));

    // Construct orthonormal basis
    float3 light_fwd = normalize(lights[0].lightDir.xyz);
    float3 light_right = (light_fwd.z < 0.9999f) ?
        normalize(cross(float3(0.f, 0.f, 1.f), light_fwd)) :
        float3(1.f, 0.f, 0.f);
    float3 light_up = cross(light_right, light_fwd);

    // Note that we use the basis vectors as the *rows* of the to_light
    // transform matrix, because we want the inverse of the light to world
    // matrix (which is just the transpose for rotation matrices).
    float3x3 to_light = float3x3(
        light_right.x, light_right.y, light_right.z,
        light_fwd.x,   light_fwd.y, light_fwd.z,
        light_up.x, light_up.y, light_up.z
    );

    float far_width, near_width, far_height, near_height;

    float tan_half_fov = -1.0f / unpackedView.yScale;
    float aspect = -unpackedView.yScale / unpackedView.xScale;
    float near = 1.0f;
    float far = 50.0f;

    far_height = 2.0f * far * tan_half_fov;
    near_height = 2.0f * near * tan_half_fov;
    far_width = far_height * aspect;
    near_width = near_height * aspect;

    float3 center_near = cam_pos + cam_fwd * near;
    float3 center_far = cam_pos + cam_fwd * far;

    float far_width_half = far_width / 2.0f;
    float near_width_half = near_width / 2.0f;
    float far_height_half = far_height / 2.0f;
    float near_height_half = near_height / 2.0f;

    // f = far, n = near, l = left, r = right, t = top, b = bottom
    enum OrthoCorner {
        flt, flb,
        frt, frb,
        nlt, nlb,
        nrt, nrb
    };    

    float3 ls_corners[8];

#if 0
    ls_corners[flt] = mul(view, float4(cam_pos + ws_direction * far - right_view_ax * far_width_half + up_view_ax * far_height_half, 1.0f));
    ls_corners[flb] = mul(view, float4(ws_position + ws_direction * far - right_view_ax * far_width_half - up_view_ax * far_height_half, 1.0f));
    ls_corners[frt] = mul(view, float4(ws_position + ws_direction * far + right_view_ax * far_width_half + up_view_ax * far_height_half, 1.0f));
    ls_corners[frb] = mul(view, float4(ws_position + ws_direction * far + right_view_ax * far_width_half - up_view_ax * far_height_half, 1.0f));
    ls_corners[nlt] = mul(view, float4(ws_position + ws_direction * near - right_view_ax * near_width_half + up_view_ax * near_height_half, 1.0f));
    ls_corners[nlb] = mul(view, float4(ws_position + ws_direction * near - right_view_ax * near_width_half - up_view_ax * near_height_half, 1.0f));
    ls_corners[nrt] = mul(view, float4(ws_position + ws_direction * near + right_view_ax * near_width_half + up_view_ax * near_height_half, 1.0f));
    ls_corners[nrb] = mul(view, float4(ws_position + ws_direction * near + right_view_ax * near_width_half - up_view_ax * near_height_half, 1.0f));
#endif

    ls_corners[flt] = mul(to_light, center_far - cam_right * far_width_half + cam_up * far_height_half);
    ls_corners[flb] = mul(to_light, center_far - cam_right * far_width_half - cam_up * far_height_half);
    ls_corners[frt] = mul(to_light, center_far + cam_right * far_width_half + cam_up * far_height_half);
    ls_corners[frb] = mul(to_light, center_far + cam_right * far_width_half - cam_up * far_height_half);
    ls_corners[nlt] = mul(to_light, center_near - cam_right * near_width_half + cam_up * near_height_half);
    ls_corners[nlb] = mul(to_light, center_near - cam_right * near_width_half - cam_up * near_height_half);
    ls_corners[nrt] = mul(to_light, center_near + cam_right * near_width_half + cam_up * near_height_half);
    ls_corners[nrb] = mul(to_light, center_near + cam_right * near_width_half - cam_up * near_height_half);

    float x_min, x_max, y_min, y_max, z_min, z_max;

    x_min = x_max = ls_corners[0].x;
    y_min = y_max = ls_corners[0].y;
    z_min = z_max = ls_corners[0].z;

    for (uint32_t i = 1; i < 8; ++i) {
        if (x_min > ls_corners[i].x) x_min = ls_corners[i].x;
        if (x_max < ls_corners[i].x) x_max = ls_corners[i].x;

        if (y_min > ls_corners[i].y) y_min = ls_corners[i].y;
        if (y_max < ls_corners[i].y) y_max = ls_corners[i].y;

        if (z_min > ls_corners[i].z) z_min = ls_corners[i].z;
        if (z_max < ls_corners[i].z) z_max = ls_corners[i].z;
    }

    {
        float tmp = y_max;
        y_max = y_min;
        y_min = tmp;
    }
    

    float4x4 projection =(float4x4(
        float4(2.0f / (x_max - x_min),             0.0f,                       0.0f,                        -(x_max + x_min) / (x_max - x_min)),
        float4(0.0f,                               0.0f,                      -2.0f / (z_max - z_min),      (z_max+z_min) / (z_max - z_min)),
        float4(0.0f,                               1.0f / (y_max - y_min),     0.0f,                        -(y_min) / (y_max - y_min)),
        float4(0.0f,                               0.0f,                       0.0f,                        1.0f)));

    shadowViewDataBuffer[idx.x].viewProjectionMatrix = mul(
        projection, float4x4(
            float4(to_light[0].xyz, 0.f),
            float4(to_light[1].xyz, 0.f),
            float4(to_light[2].xyz, 0.f),
            float4(0.f, 0.f, 0.f, 1.f)
        )
    );

    {
        shadowViewDataBuffer[idx.x].cameraRight = float4(cam_right, 1.f);
        shadowViewDataBuffer[idx.x].cameraUp = float4(cam_up, 1.f);
        shadowViewDataBuffer[idx.x].cameraForward = float4(cam_fwd, 1.f);
    }
}
