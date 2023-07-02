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
    return float4(rot.w, -rot.x, -rot.y, -rot.z);
}

float3 rotateVec(float4 rot, float3 v)
{
    float3 pure = float3(rot.x, rot.y, rot.z);
    float scalar = rot.w;

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

// From GLM
float4x4 lookAt(float3 eye, float3 center, float3 up)
{
    float3 f = normalize(center - eye);
    float3 s = normalize(cross(f, up));
    float3 u = normalize(cross(s, f));

    float4x4 m;
    m[0][0] = s.x;
    m[1][0] = s.y;
    m[2][0] = s.z;
    m[0][1] = u.x;
    m[1][1] = u.y;
    m[2][1] = u.z;
    m[0][2] =-f.x;
    m[1][2] =-f.y;
    m[2][2] =-f.z;
    m[3][0] =-dot(s, eye);
    m[3][1] =-dot(u, eye);
    m[3][2] = dot(f, eye);
    m[3][3] = 1.0f;
    return m;
}

float3x3 getCameraToWorldMatrix(float3 eye,
    float3 center,
    float3 up)
{
    float3 f = normalize(center - eye);
    float3 s = normalize(cross(f, up));
    float3 u = normalize(cross(s, f));

    return float3x3(
        s, u, f
    );
}

// From Brendan Galea
float4x4 perspective(float fovy, float aspect, float near, float far)
{
    float4x4 projection;

    const float tanHalfFovy = tan(radians(fovy) / 2.f);
    projection[0][0] = 1.f / (aspect * tanHalfFovy);
    projection[1][1] = 1.f / (tanHalfFovy);
    projection[2][2] = far / (far - near);
    projection[2][3] = 1.f;
    projection[3][2] = -(far * near) / (far - near);   

    return projection;
}

[numThreads(32, 1, 1)]
[shader("compute")]
void shadowGen(uint3 idx : SV_DispatchThreadID)
{
    // Assume that the sun is from lights[0]
    if (idx.x >= pushConst.numViews+1)
        return;

    PackedViewData viewData = viewDataBuffer[idx.x];

    PerspectiveCameraData unpackedView = unpackViewData(viewData);

    float3 cam_position = unpackedView.pos;
    float4 rotation = unpackedView.rot;

    float3 cam_view = rotateVec(rotation, float3(0.0f, 1.0f, 0.0f));

    float3 ws_position = cam_position;
    float3 ws_direction = normalize(cam_view);
    float3 ws_up = float3(0.000000001f, 0.000000001f, 1.0f);
    float3 ws_light_dir = normalize(lights[0].lightDir.xyz);

#if 1
    ws_position = ws_position.xzy;
    ws_direction = ws_direction.xzy;
    ws_up = ws_up.xzy;
    ws_light_dir = ws_light_dir.xzy;
#endif

    float4x4 view = lookAt(float3(0.0f, 0.0f, 0.0f), ws_light_dir, ws_up);

    float far_width, near_width, far_height, near_height;


    float tan_half_fov = -1.0f / unpackedView.yScale;
    float aspect = -unpackedView.yScale / unpackedView.xScale;
    float near = 1.0f;
    float far = 50.0f;


#if 0
    shadowViewDataBuffer[viewIdx].viewProjectionMatrix[0][0] = tan_half_fov;
    shadowViewDataBuffer[viewIdx].viewProjectionMatrix[1][0] = aspect;
    shadowViewDataBuffer[viewIdx].viewProjectionMatrix[2][0] = near;
    shadowViewDataBuffer[viewIdx].viewProjectionMatrix[3][0] = far;
#endif


#if 0
    far_height = 2.0f * far * tan(radians(cam.fov) / 2.0f);
    near_height = 2.0f * near * tan(radians(cam.fov) / 2.0f);
    far_width = far_height * aspect;
    near_width = near_height * aspect;
#endif
    far_height = 2.0f * far * tan_half_fov;
    near_height = 2.0f * near * tan_half_fov;
    far_width = far_height * aspect;
    near_width = near_height * aspect;




    float3 center_near = ws_position + ws_direction * near;
    float3 center_far = ws_position + ws_direction * far;
    float3 right_view_ax = normalize(cross(ws_direction, ws_up));
    float3 up_view_ax = -normalize(cross(ws_direction, right_view_ax));


#if 0
    shadowViewDataBuffer[viewIdx].viewProjectionMatrix[0] = float4(center_near, 1.0f);
    shadowViewDataBuffer[viewIdx].viewProjectionMatrix[1] = float4(center_far, 1.0f);
    shadowViewDataBuffer[viewIdx].viewProjectionMatrix[3] = float4(up_view_ax, 1.0f);
    shadowViewDataBuffer[viewIdx].viewProjectionMatrix[2] = float4(right_view_ax, 1.0f);
#endif


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

    float4 ls_corners[8];

    // Light space
    ls_corners[flt] = mul(view, float4(ws_position + ws_direction * far - right_view_ax * far_width_half + up_view_ax * far_height_half, 1.0f));
    ls_corners[flb] = mul(view, float4(ws_position + ws_direction * far - right_view_ax * far_width_half - up_view_ax * far_height_half, 1.0f));
    ls_corners[frt] = mul(view, float4(ws_position + ws_direction * far + right_view_ax * far_width_half + up_view_ax * far_height_half, 1.0f));
    ls_corners[frb] = mul(view, float4(ws_position + ws_direction * far + right_view_ax * far_width_half - up_view_ax * far_height_half, 1.0f));
    ls_corners[nlt] = mul(view, float4(ws_position + ws_direction * near - right_view_ax * near_width_half + up_view_ax * near_height_half, 1.0f));
    ls_corners[nlb] = mul(view, float4(ws_position + ws_direction * near - right_view_ax * near_width_half - up_view_ax * near_height_half, 1.0f));
    ls_corners[nrt] = mul(view, float4(ws_position + ws_direction * near + right_view_ax * near_width_half + up_view_ax * near_height_half, 1.0f));
    ls_corners[nrb] = mul(view, float4(ws_position + ws_direction * near + right_view_ax * near_width_half - up_view_ax * near_height_half, 1.0f));

#if 0
    shadowViewDataBuffer[0].viewProjectionMatrix[0] = ls_corners[flt];
    shadowViewDataBuffer[0].viewProjectionMatrix[1] = ls_corners[flb];
    shadowViewDataBuffer[0].viewProjectionMatrix[2] = ls_corners[frt];
    shadowViewDataBuffer[0].viewProjectionMatrix[3] = ls_corners[frb];
#endif

#if 0
    shadowViewDataBuffer[0].viewProjectionMatrix[0] = 
        mul(view, float4(float3(1.0f, 1.0f, 1.0f), 1.0f));
#endif

    // shadowViewDataBuffer[idx.x].viewProjectionMatrix[0] = mul(view, float4(1.0f, 1.0f, 1.0f, 1.0f));
    // shadowViewDataBuffer[idx.x].viewProjectionMatrix[0][0] = float(idx.x);
    shadowViewDataBuffer[idx.x].viewProjectionMatrix = view;
    shadowViewDataBuffer[idx.x].cameraViewProjectionMatrix[0][0] = view[0].w + view[1].w + view[2].w + view[3].w;

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
    
    z_min = z_min - (z_max - z_min);

    // Y is up
    float4x4 projection = float4x4(
            float4( 2.0f / (x_max - x_min), 0.0f,                   0.0f,                   0.0f),
            float4(0.0f,                    2.0f / (y_max - y_min), 0.0f,                   0.0f),
            float4(0.0f,                    0.0f,                   1.0f / (z_max - z_min), 0.0f ),
            float4(-(x_max + x_min) / (x_max - x_min), -(y_max + y_min) / (y_max - y_min), -(z_min) / (z_max - z_min), 1.0f));

    // shadowViewDataBuffer[viewIdx].viewProjectionMatrix = mul(projection, view);

#if 0
    shadowViewDataBuffer[viewIdx].cameraViewProjectionMatrix = perspective(cam.fov, aspect, 1.0f, 1000.0f).
        compose(lookAt(ws_position, ws_position + ws_direction, ws_up));
#endif

    {
        float3 f = normalize(ws_direction);
        float3 s = normalize(cross(f, ws_up));
        float3 u = normalize(cross(s, f));

        shadowViewDataBuffer[idx.x].cameraRight = float4(s.x, s.y, s.z, 1.0f);
        shadowViewDataBuffer[idx.x].cameraUp = float4(u.x, u.y, u.z, 1.0f);
        shadowViewDataBuffer[idx.x].cameraForward = float4(f.x, f.y, f.z, 1.0f);
    }
}
