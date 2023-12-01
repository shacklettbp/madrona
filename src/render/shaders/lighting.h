#ifndef MADRONA_VIEWER_LIGHTING_H_INCLUDED
#define MADRONA_VIEWER_LIGHTING_H_INCLUDED

/* Eric Bruneton's multiple scattering atmosphere model. */

#define TRANSMITTANCE_TEXTURE_WIDTH 256
#define TRANSMITTANCE_TEXTURE_HEIGHT 64
#define SCATTERING_TEXTURE_R_SIZE 32
#define SCATTERING_TEXTURE_MU_SIZE 128
#define SCATTERING_TEXTURE_MU_S_SIZE 32
#define SCATTERING_TEXTURE_NU_SIZE 8
#define SCATTERING_TEXTURE_WIDTH (SCATTERING_TEXTURE_NU_SIZE * SCATTERING_TEXTURE_MU_S_SIZE)
#define SCATTERING_TEXTURE_HEIGHT SCATTERING_TEXTURE_MU_SIZE
#define SCATTERING_TEXTURE_DEPTH SCATTERING_TEXTURE_R_SIZE
#define IRRADIANCE_TEXTURE_WIDTH 64
#define IRRADIANCE_TEXTURE_HEIGHT 16

/* Make sure this file gets included after these declarations
   [[vk::binding(###)]]
   RWTexture2D<float4> transmittanceLUT;

   [[vk::binding(###)]]
   RWTexture2D<float4> irradianceLUT;

   [[vk::binding(###)]]
   RWTexture3D<float4> mieLUT;

   [[vk::binding(###)]]
   RWTexture3D<float4> scatteringLUT;

   [[vk::binding(###)]]
   StructuredBuffer<SkyData> skyBuffer;

   [[vk::binding(###)]]
   SamplerState linearSampler;
*/

#define M_PI   (3.14159265358979323846264338327950288)

/* Some utility functions */
float clamp0To1(float a) 
{
    return clamp(a, -1.0, 1.0);
}

float clampPositive(float a) 
{
    return max(a, 0.0);
}

float clampRadius(in SkyData sky, float radius) 
{
    return clamp(radius, sky.bottomRadius, sky.topRadius);
}

float safeSqrt(float a) 
{
    return sqrt(max(a, 0.0));
}

float getTextureCoordFromUnit(float x, int textureSize) 
{
    return 0.5 / float(textureSize) + x * (1.0 - 1.0 / float(textureSize));
}

float getUnitFromTextureCoord(float u, int textureSize) 
{
    return (u - 0.5 / float(textureSize)) / (1.0 - 1.0 / float(textureSize));
}

float distToSkyBoundary(
        in SkyData sky,
        float centreToPointDist,
        float mu) 
{
    float r = centreToPointDist;
    float delta = r * r * (mu * mu - 1.0) + sky.topRadius * sky.topRadius;
    return clampPositive(-r * mu + safeSqrt(delta));
}

float distToGroundBoundary(
        in SkyData sky,
        float centreToPointDist,
        float mu) 
{
    float r = centreToPointDist;
    float delta = r * r * (mu * mu - 1.0) + sky.bottomRadius * sky.bottomRadius;
    return clampPositive(-r * mu - safeSqrt(delta));
}

bool doesRayIntersectGround(
        in SkyData sky,
        float centreToPointDist,
        float mu) 
{
    float r = centreToPointDist;
    float delta = r * r * (mu * mu - 1.0) + sky.bottomRadius * sky.bottomRadius;

    return mu < 0.0 && delta >= 0.0;
}

float2 getTransmittanceTextureUVFromRMu(
        in SkyData sky,
        float centreToPointDist,
        float mu) 
{
    float r = centreToPointDist;

    /*
       Distance from ground level to top sky boundary along a horizontal ray 
       tangent to the ground
       Simple Pythagoras
       */
    float h = sqrt(
            sky.topRadius * sky.topRadius - sky.bottomRadius * sky.bottomRadius);

    // Distance to the horizon
    float rho = safeSqrt(
            r * r - sky.bottomRadius * sky.bottomRadius);

    float d = distToSkyBoundary(sky, r, mu);
    float dMin = sky.topRadius - r;
    float dMax = rho + h;

    // The mapping for mu is done in terms of dMin and dMax (0 -> 1)
    float xMu = (d - dMin) / (dMax - dMin);
    float xR = rho / h;

    return float2(
            getTextureCoordFromUnit(xMu, TRANSMITTANCE_TEXTURE_WIDTH),
            getTextureCoordFromUnit(xR, TRANSMITTANCE_TEXTURE_HEIGHT));
}

float3 getTransmittanceToSkyBoundary(
        in SkyData sky,
        in Texture2D<float4> transmittanceTexture,
        float centreToPointDist, float mu) 
{
    float2 uv = getTransmittanceTextureUVFromRMu(sky, centreToPointDist, mu);
    return float3(transmittanceTexture.SampleLevel(linearSampler, float2(uv.x, 1.0 - uv.y), 0).xyz);
}

float3 getTransmittance(
        in SkyData sky,
        in Texture2D<float4> transmittanceTexture,
        float r, float mu,
        float d,
        bool doesRMuIntersectGround) 
{
    float rD = clampRadius(
            sky,
            sqrt(d * d + 2.0 * r * mu * d + r * r));

    float muD = clamp0To1((r * mu + d) / rD);

    if (doesRMuIntersectGround) {
        return min(
                getTransmittanceToSkyBoundary(
                    sky, transmittanceTexture, rD, -muD) /
                getTransmittanceToSkyBoundary(
                    sky, transmittanceTexture, r, -mu),
                float3(1.0, 1.0, 1.0));
    }
    else {
        return min(
                getTransmittanceToSkyBoundary(
                    sky, transmittanceTexture, r, mu) /
                getTransmittanceToSkyBoundary(
                    sky, transmittanceTexture, rD, muD),
                float3(1.0, 1.0, 1.0));
    }
}

float3 getTransmittanceToSun(
        in SkyData sky,
        in Texture2D<float4> transmittanceTexture,
        float r, float muSun) 
{
    float sinThetaH = sky.bottomRadius / r;
    float cosThetaH = -sqrt(max(1.0 - sinThetaH * sinThetaH, 0.0));

    float visibleFactor = smoothstep(
            -sinThetaH * sky.solarAngularRadius,
            sinThetaH * sky.solarAngularRadius,
            muSun - cosThetaH);

    return getTransmittanceToSkyBoundary(
            sky, transmittanceTexture, r, muSun) * visibleFactor;
}

float rayleighPhase(float nu) 
{
    float k = 3.0 / (16.0 * M_PI);
    return k * (1.0 + nu * nu);
}

float miePhase(float g, float nu) 
{
    float k = 3.0 / (8.0 * M_PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + nu * nu) / pow(1.0 + g * g - 2.0 * g * nu, 1.5);
}

float4 getScatteringTextureUVWZFromRMuMuSunNu(
        in SkyData sky,
        float r, float mu, float muSun, float nu,
        bool doesRMuIntersectGround) 
{
    float h = sqrt(
            sky.topRadius * sky.topRadius - sky.bottomRadius * sky.bottomRadius);

    float rho = safeSqrt(r * r - sky.bottomRadius * sky.bottomRadius);

    float rMapping = getTextureCoordFromUnit(rho / h, SCATTERING_TEXTURE_R_SIZE);

    float rMu = r * mu;
    float delta = rMu * rMu - r * r + sky.bottomRadius * sky.bottomRadius;

    float muMapping;

    if (doesRMuIntersectGround) {
        float d = -rMu - safeSqrt(delta);
        float dMin = r - sky.bottomRadius;
        float dMax = rho;
        muMapping = 0.5 - 0.5 * getTextureCoordFromUnit(
                dMax == dMin ? 0.0 : (d - dMin) / (dMax - dMin),
                SCATTERING_TEXTURE_MU_SIZE / 2);
    }
    else {
        float d = -rMu + safeSqrt(delta + h * h);
        float dMin = sky.topRadius - r;
        float dMax = rho + h;
        muMapping = 0.5 + 0.5 * getTextureCoordFromUnit(
                (d - dMin) / (dMax - dMin), SCATTERING_TEXTURE_MU_SIZE / 2);
    }

    float d = distToSkyBoundary(sky, sky.bottomRadius, muSun);
    float dMin = sky.topRadius - sky.bottomRadius;
    float dMax = h;
    float a = (d - dMin) / (dMax - dMin);
    float dMuSunMin = distToSkyBoundary(sky, sky.bottomRadius, sky.muSunMin);
    float aMuSunMin = (dMuSunMin - dMin) / (dMax - dMin);

    float muSunMapping = getTextureCoordFromUnit(
            max(1.0 - a / aMuSunMin, 0.0) / (1.0 + a), SCATTERING_TEXTURE_MU_S_SIZE);

    float nuMapping = (nu + 1.0) / 2.0;

    return float4(nuMapping, muSunMapping, muMapping, rMapping);
}

float3 getIrradiance(
        in SkyData sky,
        in Texture2D<float4> irradianceTexture,
        float r, float muSun);

float4 getScatteringTextureUvwzFromRMuMuSNu(
        in SkyData sky,
        float r, float mu, float muS, float nu,
        bool rayRMuIntersectsGround) 
{
    float h = sqrt(
            sky.topRadius * sky.topRadius - sky.bottomRadius * sky.bottomRadius);

    float rho = safeSqrt(r * r - sky.bottomRadius * sky.bottomRadius);
    float uR = getTextureCoordFromUnit(rho / h, SCATTERING_TEXTURE_R_SIZE);

    float rMu = r * mu;

    float discriminant = rMu * rMu - r * r + sky.bottomRadius * sky.bottomRadius;

    float uMu;

    if (rayRMuIntersectsGround) {
        float d = -rMu - safeSqrt(discriminant);
        float dMin = r - sky.bottomRadius;
        float dMax = rho;
        uMu = 0.5 - 0.5 * getTextureCoordFromUnit(
                dMax == dMin ? 0.0 :
                (d - dMin) / (dMax - dMin), SCATTERING_TEXTURE_MU_SIZE / 2);
    }
    else {
        float d = -rMu + safeSqrt(discriminant + h * h);
        float dMin = sky.topRadius - r;
        float dMax = rho + h;
        uMu = 0.5 + 0.5 * getTextureCoordFromUnit(
                (d - dMin) / (dMax - dMin), SCATTERING_TEXTURE_MU_SIZE / 2);
    }

    float d = distToSkyBoundary(
            sky, sky.bottomRadius, muS);
    float dMin = sky.topRadius - sky.bottomRadius;
    float dMax = h;
    float a = (d - dMin) / (dMax - dMin);
    float A =
        -2.0 * sky.muSunMin * sky.bottomRadius / (dMax - dMin);
    float uMuS = getTextureCoordFromUnit(
            max(1.0 - a / A, 0.0) / (1.0 + a), SCATTERING_TEXTURE_MU_S_SIZE);

    float uNu = (nu + 1.0) / 2.0;
    return float4(uNu, uMuS, uMu, uR);
}

float3 getScattering(
        SkyData sky,
        Texture3D<float4> scatteringTexture,
        float r, float mu, float muS, float nu,
        bool rayRMuIntersectsGround) 
{
    float4 uvwz = getScatteringTextureUvwzFromRMuMuSNu(
            sky, r, mu, muS, nu, rayRMuIntersectsGround);

    float texCoordX = uvwz.x * float(SCATTERING_TEXTURE_NU_SIZE - 1);

    float texX = floor(texCoordX);
    float lerp = texCoordX - texX;

    float3 uvw0 = float3(
            (texX + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);

    float3 uvw1 = float3(
            (texX + 1.0 + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);

    uvw0.y = 1.0 - uvw0.y;
    uvw1.y = 1.0 - uvw1.y;

    return float3(
            scatteringTexture.SampleLevel(linearSampler, uvw0, 0).xyz * (1.0 - lerp) +
            scatteringTexture.SampleLevel(linearSampler, uvw1, 0).xyz * lerp);
}

float3 getScattering(
        SkyData sky,
        Texture3D<float4> singleRayleighScatteringTexture,
        Texture3D<float4> singleMieScatteringTexture,
        Texture3D<float4> multipleScatteringTexture,
        float r, float mu, float muS, float nu,
        bool rayRMuIntersectsGround,
        int scatteringOrder) 
{
    if (scatteringOrder == 1) {
        float3 rayleigh = getScattering(
                sky, singleRayleighScatteringTexture, r, mu, muS, nu,
                rayRMuIntersectsGround);

        float3 mie = getScattering(
                sky, singleMieScatteringTexture, r, mu, muS, nu,
                rayRMuIntersectsGround);

        return rayleigh * rayleighPhase(nu) +
            mie * miePhase(sky.miePhaseFunctionG, nu);
    }
    else {
        return getScattering(
                sky, multipleScatteringTexture, r, mu, muS, nu,
                rayRMuIntersectsGround);
    }
}

float2 getIrradianceTextureUVFromRMuSun(
        in SkyData sky, float r, float muSun) 
{
    float rMapping = (r - sky.bottomRadius) /
        (sky.topRadius - sky.bottomRadius);
    float muSunMapping = muSun * 0.5 + 0.5;

    return float2(
            getTextureCoordFromUnit(muSunMapping, IRRADIANCE_TEXTURE_WIDTH),
            getTextureCoordFromUnit(rMapping, IRRADIANCE_TEXTURE_HEIGHT));
}

#define IRRADIANCE_TEXTURE_SIZE float2(IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT)

float3 getIrradiance(
        in SkyData sky,
        in Texture2D<float4> irradianceTexture,
        float r, float muSun) 
{
    float2 uv = getIrradianceTextureUVFromRMuSun(sky, r, muSun);
    return float3(irradianceTexture.SampleLevel(linearSampler, float2(uv.x, 1.0 - uv.y), 0).xyz);
}

float3 getSolarRadiance(in SkyData sky) 
{
    return sky.solarIrradiance.xyz /
        (M_PI * sky.solarAngularRadius * sky.solarAngularRadius);
}

float3 getExtrapolatedSingleMieScattering(
        in SkyData sky, in float4 scattering) 
{
    if (scattering.r == 0.0) {
        return float3(0.0, 0.0, 0.0);
    }

    return scattering.rgb * scattering.a / scattering.r *
        (sky.rayleighScatteringCoef.r / sky.mieScatteringCoef.r) *
        (sky.mieScatteringCoef.xyz / sky.rayleighScatteringCoef.xyz);
}

float3 getCombinedScattering(
        in SkyData sky,
        in Texture3D<float4> scatteringTexture,
        in Texture3D<float4> singleMieScatteringTexture,
        float r, float mu, float muSun, float nu,
        bool doesRMuIntersectGround,
        out float3 singleMieScattering) 
{
    float4 uvwz = getScatteringTextureUVWZFromRMuMuSunNu(
            sky, r, mu, muSun, nu, doesRMuIntersectGround);

    float texCoordX = uvwz.x * float(SCATTERING_TEXTURE_NU_SIZE - 1);
    float texX = floor(texCoordX);
    float lerp = texCoordX - texX;
    float3 uvw0 = float3((texX + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE),
            uvwz.z, uvwz.w);
    float3 uvw1 = float3((texX + 1.0 + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE),
            uvwz.z, uvwz.w);

    uvw0.y = 1.0 - uvw0.y;
    uvw1.y = 1.0 - uvw1.y;

    float4 combinedScattering =
        scatteringTexture.SampleLevel(linearSampler, uvw0, 0) * (1.0 - lerp) +
        scatteringTexture.SampleLevel(linearSampler, uvw1, 0) * lerp;
    float3 scattering = float3(combinedScattering.xyz);

    singleMieScattering = getExtrapolatedSingleMieScattering(
            sky, combinedScattering);
    return scattering;
}

float3 getSkyRadiance(
        in SkyData sky,
        in Texture2D<float4> transmittanceTexture,
        in Texture3D<float4> scatteringTexture,
        in Texture3D<float4> singleMieScateringTexture,
        float3 camera, float3 viewRay, float shadowLength,
        float3 sunDirection, out float3 transmittance) 
{
    float r = length(camera);
    float rMu = dot(camera, viewRay);

    float distToTopSkyBoundary = -rMu -
        sqrt(rMu * rMu - r * r + sky.topRadius * sky.topRadius);

    if (distToTopSkyBoundary > 0.0) {
        camera = camera + viewRay * distToTopSkyBoundary;
        r = sky.topRadius;
        rMu += distToTopSkyBoundary;
    }
    else if (r > sky.topRadius) {
        transmittance = float3(1.0, 1.0, 1.0);
        // SPACE!
        return float3(0.0, 0.0, 0.0);
    }

    // retrieve cos of the zenith angle
    float mu = rMu / r;
    float muSun = dot(camera, sunDirection) / r;
    float nu = dot(viewRay, sunDirection);
    bool doesRMuIntersectGround = doesRayIntersectGround(sky, r, mu);

    transmittance = doesRMuIntersectGround ? float3(0.0, 0.0, 0.0) :
        getTransmittanceToSkyBoundary(sky, transmittanceTexture, r, mu);

    float3 singleMieScattering;
    float3 scattering;

    if (shadowLength == 0.0) {
        scattering = getCombinedScattering(
                sky, scatteringTexture, singleMieScateringTexture,
                r, mu, muSun, nu, doesRMuIntersectGround, singleMieScattering);
    }
    else {
        float d = shadowLength;
        float rP = clampRadius(sky, sqrt(d * d + 2.0 * r * mu * d + r * r));
        float muP = (r * mu + d) / rP;
        float muSunP = (r * muSun + d * nu) / rP;

        scattering = getCombinedScattering(
                sky, scatteringTexture, singleMieScateringTexture,
                rP, muP, muSunP, nu, doesRMuIntersectGround, singleMieScattering);

        float3 shadowTransmittance = getTransmittance(
                sky, transmittanceTexture, r, mu, shadowLength, doesRMuIntersectGround);

        scattering = scattering * shadowTransmittance;
        singleMieScattering = singleMieScattering * shadowTransmittance;
    }

    return scattering * rayleighPhase(nu) + singleMieScattering *
        miePhase(sky.miePhaseFunctionG, nu);
}

float3 getSkyRadianceToPoint(
        in SkyData sky,
        in Texture2D<float4> transmittanceTexture,
        in Texture3D<float4> scatteringTexture,
        in Texture3D<float4> singleMieScatteringTexture,
        float3 camera, float3 p, float shadowLength,
        float3 sunDirection, out float3 transmittance) 
{
    float3 viewRay = normalize(p - camera);
    float r = length(camera);
    float rmu = dot(camera, viewRay);
    float distToTopSkyBoundary = -rmu -
        sqrt(rmu * rmu - r * r + sky.topRadius * sky.topRadius);

    if (distToTopSkyBoundary > 0.0) {
        camera = camera + viewRay * distToTopSkyBoundary;
        r = sky.topRadius;
        rmu += distToTopSkyBoundary;
    }

    float mu = rmu / r;
    float muSun = dot(camera, sunDirection) / r;
    float nu = dot(viewRay, sunDirection);
    float d = length(p - camera);
    bool doesRMuIntersectGround = doesRayIntersectGround(sky, r, mu);

    transmittance = getTransmittance(
            sky, transmittanceTexture, r, mu, d, doesRMuIntersectGround);

    float3 singleMieScattering;
    float3 scattering = getCombinedScattering(
            sky, scatteringTexture, singleMieScatteringTexture,
            r, mu, muSun, nu, doesRMuIntersectGround, singleMieScattering);

    d = max(d - shadowLength, 0.0);
    float rP = clampRadius(sky, sqrt(d * d + 2.0 * r * mu * d + r * r));
    float muP = (r * mu + d) / rP;
    float muSunP = (r * muSun + d * nu) / rP;

    float3 singleMieScatteringP;
    float3 scatteringP = getCombinedScattering(
            sky, scatteringTexture, singleMieScatteringTexture,
            rP, muP, muSunP, nu, doesRMuIntersectGround, singleMieScatteringP);

    float3 shadowTransmittance = transmittance;
    if (shadowLength > 0.0) {
        shadowTransmittance = getTransmittance(
                sky, transmittanceTexture,
                r, mu, d, doesRMuIntersectGround);
    }

    scattering = scattering - shadowTransmittance * scatteringP;
    singleMieScattering = singleMieScattering -
        shadowTransmittance * singleMieScatteringP;

    singleMieScattering = getExtrapolatedSingleMieScattering(
            sky, float4(scattering, singleMieScattering.r));

    singleMieScattering = singleMieScattering * smoothstep(0.0, 0.01f, muSun);

    return scattering * rayleighPhase(nu) + singleMieScattering *
        miePhase(sky.miePhaseFunctionG, nu);
}

float3 getSunAndSkyIrradiance(
        in SkyData sky,
        in Texture2D<float4> transmittanceTexture,
        in Texture2D<float4> irradianceTexture,
        float3 p, float3 normal, float3 sunDirection,
        out float3 skyIrradiance) 
{
    float r = length(p);
    float muSun = dot(p, sunDirection) / r;

    skyIrradiance = getIrradiance(sky, irradianceTexture, r, muSun) *
        (1.0 + dot(normal, p) / r) * 0.5;

    float incidentIntensity = max(dot(normal, sunDirection), 0.0);

    return sky.solarIrradiance.xyz *
        getTransmittanceToSun(sky, transmittanceTexture, r, muSun) *
        max(dot(normal, sunDirection), 0.0);
}

#endif
