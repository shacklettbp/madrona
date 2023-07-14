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

    float3 cam_position = unpackedView.pos;
    float4 rotation = invQuat(unpackedView.rot);

    float3 cam_view = rotateVec(rotation, float3(0.0f, 1.0f, 0.0f));

    float3 ws_position = cam_position;
    float3 ws_direction = normalize(cam_view);
    float3 ws_up = float3(0.000000001f, 0.000000001f, 1.0f);
    float3 ws_light_dir = normalize(lights[0].lightDir.xyz);
    // ws_light_dir.xy *= -1;

    // ws_position -= ws_direction;

    // float4x4 view = lookAt(ws_light_dir, ws_up);

    // float4 view_quat = toQuat(float3(0, 1, 0), ws_light_dir);
    float4 view_quat = toQuat(ws_light_dir, float3(0, 1, 0));

    float far_width, near_width, far_height, near_height;


    float tan_half_fov = -1.0f / unpackedView.yScale;
    float aspect = -unpackedView.yScale / unpackedView.xScale;
    float near = 1.0f;
    float far = 50.0f;



    far_height = 2.0f * far * tan_half_fov;
    near_height = 2.0f * near * tan_half_fov;
    far_width = far_height * aspect;
    near_width = near_height * aspect;




    float3 center_near = ws_position + ws_direction * near;
    float3 center_far = ws_position + ws_direction * far;
    float3 right_view_ax = normalize(cross(ws_direction, ws_up));
    float3 up_view_ax = -normalize(cross(ws_direction, right_view_ax));



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
    ls_corners[flt] = mul(view, float4(ws_position + ws_direction * far - right_view_ax * far_width_half + up_view_ax * far_height_half, 1.0f));
    ls_corners[flb] = mul(view, float4(ws_position + ws_direction * far - right_view_ax * far_width_half - up_view_ax * far_height_half, 1.0f));
    ls_corners[frt] = mul(view, float4(ws_position + ws_direction * far + right_view_ax * far_width_half + up_view_ax * far_height_half, 1.0f));
    ls_corners[frb] = mul(view, float4(ws_position + ws_direction * far + right_view_ax * far_width_half - up_view_ax * far_height_half, 1.0f));
    ls_corners[nlt] = mul(view, float4(ws_position + ws_direction * near - right_view_ax * near_width_half + up_view_ax * near_height_half, 1.0f));
    ls_corners[nlb] = mul(view, float4(ws_position + ws_direction * near - right_view_ax * near_width_half - up_view_ax * near_height_half, 1.0f));
    ls_corners[nrt] = mul(view, float4(ws_position + ws_direction * near + right_view_ax * near_width_half + up_view_ax * near_height_half, 1.0f));
    ls_corners[nrb] = mul(view, float4(ws_position + ws_direction * near + right_view_ax * near_width_half - up_view_ax * near_height_half, 1.0f));
#endif
    ls_corners[flt] = rotateVec(view_quat, ws_position + ws_direction * far - right_view_ax * far_width_half + up_view_ax * far_height_half);
    ls_corners[flb] = rotateVec(view_quat, ws_position + ws_direction * far - right_view_ax * far_width_half - up_view_ax * far_height_half);
    ls_corners[frt] = rotateVec(view_quat, ws_position + ws_direction * far + right_view_ax * far_width_half + up_view_ax * far_height_half);
    ls_corners[frb] = rotateVec(view_quat, ws_position + ws_direction * far + right_view_ax * far_width_half - up_view_ax * far_height_half);
    ls_corners[nlt] = rotateVec(view_quat, ws_position + ws_direction * near - right_view_ax * near_width_half + up_view_ax * near_height_half);
    ls_corners[nlb] = rotateVec(view_quat, ws_position + ws_direction * near - right_view_ax * near_width_half - up_view_ax * near_height_half);
    ls_corners[nrt] = rotateVec(view_quat, ws_position + ws_direction * near + right_view_ax * near_width_half + up_view_ax * near_height_half);
    ls_corners[nrb] = rotateVec(view_quat, ws_position + ws_direction * near + right_view_ax * near_width_half - up_view_ax * near_height_half);

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

    shadowViewDataBuffer[idx.x].viewProjectionMatrix = mul(projection, toMat(view_quat));

    {
        float3 f = normalize(ws_direction);
        float3 s = normalize(cross(f, ws_up));
        float3 u = normalize(cross(f, s));

        shadowViewDataBuffer[idx.x].cameraRight = float4(s.x, s.y, s.z, 1.0f);
        shadowViewDataBuffer[idx.x].cameraUp = -float4(u.x, u.y, u.z, 1.0f);
        shadowViewDataBuffer[idx.x].cameraForward = float4(f.x, f.y, f.z, 1.0f);
    }
}
