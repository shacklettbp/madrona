#ifndef RLPBR_VK_GEOMETRY_GLSL_INCLUDED
#define RLPBR_VK_GEOMETRY_GLSL_INCLUDED

#include "inputs.glsl"
#include "unpack.glsl"
#include "camera.glsl"

struct HitInfo {
    vec3 position;
    vec3 geoNormal;
    float triArea;
    TangentFrame tangentFrame;
    Material material;
};

struct RayCone {
    vec3 curOrigin;
    float totalDistance;
    float pixelSpread;
};

u32vec3 fetchTriangleIndices(IdxRef idx_ref, uint32_t index_offset)
{
    // FIXME: maybe change all this to triangle offset
    return u32vec3(
        idx_ref[nonuniformEXT(index_offset)].idx,
        idx_ref[nonuniformEXT(index_offset + 1)].idx,
        idx_ref[nonuniformEXT(index_offset + 2)].idx);
}

Triangle fetchTriangle(VertRef vert_ref, IdxRef idx_ref, uint32_t index_offset)
{
    u32vec3 indices = fetchTriangleIndices(idx_ref, index_offset);

    return Triangle(
        unpackVertex(vert_ref, indices.x),
        unpackVertex(vert_ref, indices.y),
        unpackVertex(vert_ref, indices.z));
}

#define INTERPOLATE_ATTR(a, b, c, barys) \
    (a + barys.x * (b - a) + \
     barys.y * (c - a))

vec3 interpolatePosition(vec3 a, vec3 b, vec3 c, vec2 barys)
{
    return INTERPOLATE_ATTR(a, b, c, barys);
}

vec3 interpolateNormal(vec3 a, vec3 b, vec3 c, vec2 barys)
{
    return INTERPOLATE_ATTR(a, b, c, barys);
}

vec4 interpolateCombinedTangent(vec4 a, vec4 b, vec4 c, vec2 barys)
{
    return INTERPOLATE_ATTR(a, b, c, barys);
}

vec2 interpolateUV(vec2 a, vec2 b, vec2 c, vec2 barys)
{
    return INTERPOLATE_ATTR(a, b, c, barys);
}

#undef INTERPOLATE_ATTR

void computeTriangleProperties(in vec3 a, in vec3 b, in vec3 c,
                               out vec3 geo_normal,
                               out float area)
{
    vec3 v1 = b - a;
    vec3 v2 = c - a;

    vec3 cp = cross(v1, v2);
    float len = length(cp);

    geo_normal = cp / len;
    area = 0.5f * len;
}

TangentFrame computeTangentFrame(Triangle hit_tri,
                                 vec2 barys,
                                 MaterialParams mat_params,
                                 uint32_t base_tex_idx,
                                 vec2 uv,
                                 TextureDerivatives tex_derivs)
{
    vec3 n = interpolateNormal(hit_tri.a.normal,
                               hit_tri.b.normal,
                               hit_tri.c.normal,
                               barys);

    vec4 combined = interpolateCombinedTangent(hit_tri.a.tangentAndSign,
                                               hit_tri.b.tangentAndSign,
                                               hit_tri.c.tangentAndSign,
                                               barys);

    vec3 t = combined.xyz;

    // Need to extend to 1 or -1 (or 0) since interpolation can produce
    // something in between
    float bitangent_sign = sign(combined.w);

    vec3 b = cross(n, t) * bitangent_sign;

    vec3 perturb = vec3(0, 0, 1);
    if (bool(mat_params.flags & MaterialFlagsHasNormalMap)) {
        vec2 xy = fetchSceneTexture(base_tex_idx + TextureConstantsNormalOffset,
                                    uv, tex_derivs).xy;

        vec2 centered = xy * 2.0 - 1.0;
        float length2 = clamp(dot(centered, centered), 0.0, 1.0);

        perturb = vec3(centered.x, centered.y, sqrt(1.0 - length2));
    } 

    // Perturb normal
    n = normalize(t * perturb.x + b * perturb.y + n * perturb.z);

    // At this point, two things can have gone wrong (due to bad UVs or normal
    // 1. bitangent_sign interpolated to 0 => 0 length bitangent
    // 2. Perturbed normal is parallel to original tangent
    // Just reinvent the tangent space in both these cases
    // FIXME: perhaps we should always recreate the tangent space,
    // that would avooid the branch and need to transform
    // the normal and bitangent

    float n_d_t = dot(n, t);

    if (n_d_t > 0.9999f || bitangent_sign == 0.f) {
        bitangent_sign = 1.f;
        t = normalize(getOrthogonalVec(n));
    } else {
        // Make tangent perpendicular to perturbed normal
        t = normalize(t - n * n_d_t);
    }
    b = cross(n, t) * bitangent_sign;

    return TangentFrame(t, b, n);
}

TangentFrame tangentFrameToWorld(mat4x3 o2w, mat4x3 w2o, TangentFrame frame,
                                 vec3 geo_normal)
{
    frame.tangent = normalize(transformVector(o2w, frame.tangent));
    frame.bitangent = normalize(transformVector(o2w, frame.bitangent));
    frame.normal = normalize(transformNormal(w2o, frame.normal));

    // There's some stupidity with frame.normal being backfacing relative
    // to the outgoing vector when it shouldn't be due to normal
    // interpolation / mapping. Therefore instead of flipping the tangent
    // frame based on ray direction, flip it based on the geo normal,
    // which has been aligned with the outgoing vector already
    if (dot(frame.normal, geo_normal) < 0.f) {
        frame.tangent *= -1.f;
        frame.bitangent *= -1.f;
        frame.normal *= -1.f;
    }
                                                               
    return frame;                                              
}

void getHitParams(in rayQueryEXT ray_query, out vec2 barys,
                  out uint32_t tri_idx, out uint32_t material_offset,
                  out uint32_t geo_idx, out uint32_t mesh_offset,
                  out mat4x3 o2w, out mat4x3 w2o)
{
    barys = rayQueryGetIntersectionBarycentricsEXT(ray_query, true);

    tri_idx =
        uint32_t(rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, true));

    material_offset = uint32_t(
        rayQueryGetIntersectionInstanceCustomIndexEXT(ray_query, true));

    geo_idx = 
        uint32_t(rayQueryGetIntersectionGeometryIndexEXT(ray_query, true));

    mesh_offset = uint32_t(
        rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(
            ray_query, true));

    o2w = rayQueryGetIntersectionObjectToWorldEXT(ray_query, true);
    w2o = rayQueryGetIntersectionWorldToObjectEXT(ray_query, true);
}

uint32_t getHitInstance(in rayQueryEXT ray_query)
{
    return uint32_t(rayQueryGetIntersectionInstanceIdEXT(ray_query, true));
}

#if 0
void barycentricWorldDerivatives(
    vec3 A1, vec3 A2,
    out vec3 du_dx, out vec3 dv_dx)
{
    vec3 Nt = cross(A1, A2) / dot(Nt, Nt);
    du_dx = cross(A2, Nt);
    dv_dx = cross(Nt, A1);
}

mat3 WorldScreenDerivatives(vec4 x)
{
    float wMx = dot(WorldToTargetMatrix[3], x);
    maat3 dx_dxt = mat3(TargetToWorldMatrix);
    dx_dxt[0] -= x.x * TargetToWorldMatrix[3].xyz;
    dx_dxt[1] -= x.y * TargetToWorldMatrix[3].xyz;
    dx_dxt[2] -= x.z * TargetToWorldMatrix[3].xyz;

    return dx_dxt;
}

// Ray Tracing Gems II Chapter 7
vec4 UVDerivsFromRayCone(vec3 ray_dir,
                       vec3 world_normal,
                       float tri_area,
                       float cone_width,
                       vec2 a_uv,
                       vec2 b_uv,
                       vec2 c_uv)
{
    vec2 vUV10 = b_uv - a_uv;
	vec2 vUV20 = c_uv - a_uv;
	float fQuadUVArea = abs(vUV10.x * vUV20.y - vUV20.x * vUV10.y);

	// Since the ray cone's width is in world-space, we need to compute the quad
	// area in world-space as well to enable proper ratio calculation
	float fQuadArea = 2.f * tri_area;

	float fDistTerm = abs(cone_width);
	float fNormalTerm = abs(dot(ray_dir, world_normal));
	float fProjectedConeWidth = cone_width / fNormalTerm;
	float fVisibleAreaRatio =
        (fProjectedConeWidth * fProjectedConeWidth) / fQuadArea;

	float fVisibleUVArea = fQuadUVArea * fVisibleAreaRatio;
	float fULength = sqrt(fVisibleUVArea);
	return vec4(fULength, 0, 0, fULength);
}

RayCone initRayCone(Camera cam)
{
    return RayCone(cam.origin, 0, atan(2.f * cam.upScale / float(RES_Y)));
}

float updateRayCone(in vec3 position, inout RayCone cone)
{
    float new_dist = length(position - cone.curOrigin);

    cone.totalDistance += new_dist;
    cone.curOrigin = position;

    return cone.pixelSpread * cone.totalDistance;
}

void updateRayDifferential(inout RayDifferential ray_diff,
                           in vec3 d, in float t, in vec3 n)
{
    vec3 dodx = ray_diff.dOdX + t * ray_diff.dDdX;
    vec3 dody = ray_diff.dOdY + t * ray_diff.dDdY;

    float inv_cos_dir = 1.f / dot(d, n);
    float dtdx = -dot(dodx, n) * inv_cos_dir;
    float dtdy = -dot(dody, n) * inv_cos_dir;

    ray_diff.dOdX = dodx + d * dtdx;
    ray_diff.dOdY = dody + d * dtdy;
}

#endif

TextureDerivatives rayDiffDerivsAndUpdate(
    inout RayDifferential ray_diff,
    vec3 d, float t,
    vec3 geo_normal,
    vec3 p0, vec3 p1, vec3 p2,
    vec2 uv0, vec2 uv1, vec2 uv2)
{
    vec3 e1 = p1 - p0;
    vec3 e2 = p2 - p0;

    vec2 g1 = uv1 - uv0;
    vec2 g2 = uv2 - uv0;

    float inv_k = 1.f / dot(cross(e1, e2), d);

    vec3 cu = cross(e2, d);
    vec3 cv = cross(d, e1);

    vec3 q = ray_diff.dOdX + t * ray_diff.dDdX;
    vec3 r = ray_diff.dOdY + t * ray_diff.dDdY;

    vec2 dBarydX = inv_k * vec2(
        dot(cu, q),
        dot(cv, q));

    vec2 dBarydY = inv_k * vec2(
        dot(cu, r),
        dot(cv, r));

    ray_diff.dOdX = dBarydX.x * e1 + dBarydX.y * e2;
    ray_diff.dOdY = dBarydY.x * e1 + dBarydY.y * e2;

    TextureDerivatives result;
    result.dUVdX = dBarydX.x * g1 + dBarydX.y * g2;
    result.dUVdY = dBarydY.x * g1 + dBarydY.y * g2;

    return result;
}

HitInfo processHit(in rayQueryEXT ray_query, in Environment env,
                   in vec3 prev_origin,
                   in vec3 outgoing_dir,
                   inout RayDifferential ray_diff)
{
    vec2 barys;
    uint32_t tri_idx, material_offset, geo_idx, mesh_offset;
    mat4x3 o2w, w2o;
    getHitParams(ray_query, barys, tri_idx,
                 material_offset, geo_idx, mesh_offset, o2w, w2o);

    GPUSceneInfo scene_info = sceneInfos[env.sceneID];

    MeshInfo mesh_info =
        unpackMeshInfo(scene_info.meshAddr, mesh_offset + geo_idx);

    uint32_t index_offset = mesh_info.indexOffset + tri_idx * 3;
    Triangle hit_tri =
        fetchTriangle(scene_info.vertAddr, scene_info.idxAddr, index_offset);
    vec3 world_a = transformPosition(o2w, hit_tri.a.position);
    vec3 world_b = transformPosition(o2w, hit_tri.b.position);
    vec3 world_c = transformPosition(o2w, hit_tri.c.position);
    vec3 world_geo_normal;
    float world_tri_area;
    computeTriangleProperties(world_a, world_b, world_c, world_geo_normal,
                              world_tri_area);
    vec3 world_position =
        interpolatePosition(world_a, world_b, world_c, barys);

    if (dot(world_geo_normal, -outgoing_dir) < 0.f) {
        world_geo_normal *= -1.f;
    }

    float hit_t = distance(world_position, prev_origin);

#if 0
    float cone_width = updateRayCone(world_position, ray_cone);
    vec4 uv_derivs = UVDerivsFromRayCone(outgoing_dir,
                                         world_geo_normal,
                                         world_tri_area,
                                         cone_width,
                                         hit_tri.a.uv, hit_tri.b.uv,
                                         hit_tri.c.uv);
#endif

    TextureDerivatives tex_derivs = rayDiffDerivsAndUpdate(
        ray_diff, outgoing_dir,
        hit_t, world_geo_normal,
        world_a, world_b, world_c,
        hit_tri.a.uv, hit_tri.b.uv, hit_tri.c.uv);

    vec2 uv = interpolateUV(hit_tri.a.uv, hit_tri.b.uv, hit_tri.c.uv,
                            barys);
    // Unpack materials
    uint32_t material_id = unpackMaterialID(
        env.baseMaterialOffset + material_offset + geo_idx);

    MaterialParams material_params =
        unpackMaterialParams(scene_info.matAddr, material_id);

    uint32_t mat_texture_offset = env.baseTextureOffset +
        material_id * TextureConstantsTexturesPerMaterial;

    TangentFrame obj_tangent_frame =
        computeTangentFrame(hit_tri, barys, material_params,
                            mat_texture_offset, uv, tex_derivs);

    TangentFrame world_tangent_frame =
        tangentFrameToWorld(o2w, w2o, obj_tangent_frame, world_geo_normal);

    Material material = processMaterial(material_params,
        mat_texture_offset, uv, tex_derivs);

    return HitInfo(world_position, world_geo_normal,
                   world_tri_area, world_tangent_frame, material);
}

vec3 worldToLocalIncoming(vec3 v, TangentFrame frame) 
{
    return vec3(dot(v, frame.tangent), dot(v, frame.bitangent),
        dot(v, frame.normal));
}

vec3 worldToLocalOutgoing(vec3 v, TangentFrame frame)
{
    // Hack from frostbite / filament
    // Consider Falcor strategy if can find reference
    return vec3(dot(v, frame.tangent), dot(v, frame.bitangent),
                abs(dot(v, frame.normal)) + 1e-5f);
}

vec3 localToWorld(vec3 v, TangentFrame frame)
{
    return v.x * frame.tangent + v.y * frame.bitangent +
        v.z * frame.normal;
}

#endif
