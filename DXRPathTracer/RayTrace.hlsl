//=================================================================================================
//
//  DXR Path Tracer
//  by MJP
//  http://mynameismjp.wordpress.com/
//
//  All code and content licensed under the MIT license
//
//=================================================================================================

//=================================================================================================
// Includes
//=================================================================================================
#include <DescriptorTables.hlsl>
#include <Constants.hlsl>
#include <Quaternion.hlsl>
#include <BRDF.hlsl>
#include <RayTracing.hlsl>
#include <Sampling.hlsl>

#include "SharedTypes.h"
#include "AppSettings.hlsl"

struct RayTraceConstants
{
    row_major float4x4 InvViewProjection;

    float3 SunDirectionWS;
    float CosSunAngularRadius;
    float3 SunIrradiance;
    float SinSunAngularRadius;
    float3 SunRenderColor;
    uint Padding;
    float3 CameraPosWS;
    uint CurrSampleIdx;
    uint TotalNumPixels;

    uint VtxBufferIdx;
    uint IdxBufferIdx;
    uint GeometryInfoBufferIdx;
    uint MaterialBufferIdx;
    uint SkyTextureIdx;
};

struct HitConstants
{
    uint GeometryIdx;
    uint MaterialIdx;
};

RaytracingAccelerationStructure Scene : register(t0, space200);
RWTexture2D<float4> RenderTarget : register(u0);
StructuredBuffer<GeometryInfo> GeometryInfoBuffers[] : register(t0, space100);
StructuredBuffer<MeshVertex> VertexBuffers[] : register(t0, space101);
StructuredBuffer<Material> MaterialBuffers[] : register(t0, space102);

ConstantBuffer<RayTraceConstants> RayTraceCB : register(b0);
ConstantBuffer<HitConstants> HitCB : register(b0, space200);

SamplerState MeshSampler : register(s0);
SamplerState LinearSampler : register(s1);

typedef BuiltInTriangleIntersectionAttributes HitAttributes;
struct PrimaryPayload
{
    HitAttributes Attributes;
    uint          GeometryIdx;
    uint          PrimitiveIdx;
};

struct ShadowPayload
{
    float Visibility;
};

enum RayTypes {
    RayTypeRadiance = 0,
    RayTypeShadow = 1,

    NumRayTypes
};

static float2 SamplePoint(in uint pixelIdx, inout uint setIdx)
{
    const uint permutation = setIdx * RayTraceCB.TotalNumPixels + pixelIdx;
    setIdx += 1;
    return SampleCMJ2D(RayTraceCB.CurrSampleIdx, AppSettings.SqrtNumSamples, AppSettings.SqrtNumSamples, permutation);
}

[shader("closesthit")]
void ClosestHitShader(inout PrimaryPayload payload, in HitAttributes attr)
{
    payload.Attributes   = attr;
    payload.GeometryIdx  = HitCB.GeometryIdx;
    payload.PrimitiveIdx = PrimitiveIndex();
}

void ProcessHit(in PrimaryPayload payload, out MeshVertex hitSurface, out Material material)
{
    const HitAttributes attr = payload.Attributes;

    float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

    StructuredBuffer<GeometryInfo> geoInfoBuffer = GeometryInfoBuffers[RayTraceCB.GeometryInfoBufferIdx];
    const GeometryInfo geoInfo = geoInfoBuffer[payload.GeometryIdx];

    StructuredBuffer<MeshVertex> vtxBuffer = VertexBuffers[RayTraceCB.VtxBufferIdx];
    Buffer<uint> idxBuffer = BufferUintTable[RayTraceCB.IdxBufferIdx];

    StructuredBuffer<Material> materialBuffer = MaterialBuffers[RayTraceCB.MaterialBufferIdx];
    material = materialBuffer[geoInfo.MaterialIdx];

    const uint primIdx = payload.PrimitiveIdx;
    const uint idx0 = idxBuffer[primIdx * 3 + geoInfo.IdxOffset + 0];
    const uint idx1 = idxBuffer[primIdx * 3 + geoInfo.IdxOffset + 1];
    const uint idx2 = idxBuffer[primIdx * 3 + geoInfo.IdxOffset + 2];

    const MeshVertex vtx0 = vtxBuffer[idx0 + geoInfo.VtxOffset];
    const MeshVertex vtx1 = vtxBuffer[idx1 + geoInfo.VtxOffset];
    const MeshVertex vtx2 = vtxBuffer[idx2 + geoInfo.VtxOffset];

    hitSurface = BarycentricLerp(vtx0, vtx1, vtx2, barycentrics);
}

[shader("miss")]
void MissShader(inout PrimaryPayload payload)
{
    payload.GeometryIdx  = 0xFFFFFFFF;
    payload.PrimitiveIdx = 0xFFFFFFFF;
}

[shader("closesthit")]
void ShadowHitShader(inout ShadowPayload payload, in HitAttributes attr)
{
    payload.Visibility = 0.0f;
}

[shader("miss")]
void ShadowMissShader(inout ShadowPayload payload)
{
    payload.Visibility = 1.0f;
}

bool IsAlphaGeometry(in HitAttributes attr)
{
    float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

    StructuredBuffer<GeometryInfo> geoInfoBuffer = GeometryInfoBuffers[RayTraceCB.GeometryInfoBufferIdx];
    const GeometryInfo geoInfo = geoInfoBuffer[HitCB.GeometryIdx];

    StructuredBuffer<Material> materialBuffer = MaterialBuffers[RayTraceCB.MaterialBufferIdx];

    const Material material = materialBuffer[geoInfo.MaterialIdx];

    Texture2D OpacityMap = Tex2DTable[NonUniformResourceIndex(material.Opacity)];

    StructuredBuffer<MeshVertex> vtxBuffer = VertexBuffers[RayTraceCB.VtxBufferIdx];
    Buffer<uint> idxBuffer = BufferUintTable[RayTraceCB.IdxBufferIdx];

    const uint primIdx = PrimitiveIndex();
    const uint idx0 = idxBuffer[primIdx * 3 + geoInfo.IdxOffset + 0];
    const uint idx1 = idxBuffer[primIdx * 3 + geoInfo.IdxOffset + 1];
    const uint idx2 = idxBuffer[primIdx * 3 + geoInfo.IdxOffset + 2];

    const MeshVertex vtx0 = vtxBuffer[idx0 + geoInfo.VtxOffset];
    const MeshVertex vtx1 = vtxBuffer[idx1 + geoInfo.VtxOffset];
    const MeshVertex vtx2 = vtxBuffer[idx2 + geoInfo.VtxOffset];

    const MeshVertex hitSurface = BarycentricLerp(vtx0, vtx1, vtx2, barycentrics);

    return (OpacityMap.SampleLevel(LinearSampler, hitSurface.UV, 0.0f).x < 0.35f);
}

[shader("anyhit")]
void AnyHitColor(inout PrimaryPayload payload, in HitAttributes attr)
{
    if (IsAlphaGeometry(attr))
    {
        IgnoreHit();
    }
}

[shader("anyhit")]
void AnyHitShadow(inout ShadowPayload payload, in HitAttributes attr)
{
    if (IsAlphaGeometry(attr))
    {
        IgnoreHit();
    }
}

void ImportanceSampleBRDF(
    in float3x3 tangentToWorld, in float3 positionWS, in uint pixelIdx, in uint sampleSetIdx, in bool enableSpecular, in bool enableDiffuse, in float roughness, in float3 diffuseAlbedo, in float3 specularAlbedo,
    inout RayDesc ray,
    out float3 throughput)
{
    float3 incomingRayDirWS = ray.Direction;

    float3 rayDirTS = 0.0f;

    // Choose our next path by importance sampling our BRDFs
    float2 brdfSample = SamplePoint(pixelIdx, sampleSetIdx);

    float selector = brdfSample.x;
    if (enableSpecular == false)
    {
        selector = 0.0f;
    }
    else if (enableDiffuse == false)
    {
        selector = 1.0f;
    }

    if (selector < 0.5f)
    {
        // We're sampling the diffuse BRDF, so sample a cosine-weighted hemisphere
        if (enableSpecular)
            brdfSample.x *= 2.0f;
        rayDirTS = SampleDirectionCosineHemisphere(brdfSample.x, brdfSample.y);

        // The PDF of sampling a cosine hemisphere is NdotL / Pi, which cancels out those terms
        // from the diffuse BRDF and the irradiance integral
        throughput = diffuseAlbedo;
    }
    else
    {
        // We're sampling the GGX specular BRDF by sampling the distribution of visible normals. See this post
        // for more info: https://schuttejoe.github.io/post/ggximportancesamplingpart2/.
        // Also see: https://hal.inria.fr/hal-00996995v1/document and https://hal.archives-ouvertes.fr/hal-01509746/document
        if (enableDiffuse)
        {
            brdfSample.x = (brdfSample.x - 0.5f) * 2.0f;
        }

        float3 incomingRayDirTS = normalize(mul(incomingRayDirWS, transpose(tangentToWorld)));
        float3 microfacetNormalTS = SampleGGXVisibleNormal(-incomingRayDirTS, roughness, roughness, brdfSample.x, brdfSample.y);
        float3 sampleDirTS = reflect(incomingRayDirTS, microfacetNormalTS);

        float3 normalTS = float3(0.0f, 0.0f, 1.0f);

        float3 F = Fresnel(specularAlbedo, microfacetNormalTS, sampleDirTS);
        float G1 = SmithGGXMasking(normalTS, sampleDirTS, -incomingRayDirTS, roughness * roughness);
        float G2 = SmithGGXMaskingShadowing(normalTS, sampleDirTS, -incomingRayDirTS, roughness * roughness);

        throughput = (F * (G2 / G1));
        rayDirTS = sampleDirTS;
    }

    const float3 rayDirWS = normalize(mul(rayDirTS, tangentToWorld));

    if (enableDiffuse && enableSpecular)
    {
        throughput *= 2.0f;
    }

    // Shoot another ray to get the next path
    ray.Origin    = positionWS;
    ray.Direction = rayDirWS;
    ray.TMin      = 0.00001f;
    ray.TMax      = FP32Max;
}

float QueryVisibility(float3 origin, float3 direction)
{
    RayDesc shadowRay;
    shadowRay.Origin    = origin;
    shadowRay.Direction = direction;
    shadowRay.TMin      = 0.00001f;
    shadowRay.TMax      = FP32Max;

    ShadowPayload shadowPayload;
    shadowPayload.Visibility = 1.0f;

    const uint hitGroupOffset        = RayTypeShadow;
    const uint hitGroupGeoMultiplier = NumRayTypes;
    const uint missShaderIdx         = RayTypeShadow;
    TraceRay(Scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFFFFFFFF, hitGroupOffset, hitGroupGeoMultiplier, missShaderIdx, shadowRay, shadowPayload);

    return shadowPayload.Visibility;
}

[shader("raygeneration")]
void RaygenShader()
{
    const uint2 pixelCoord = DispatchRaysIndex().xy;
    const uint pixelIdx = pixelCoord.y * DispatchRaysDimensions().x + pixelCoord.x;

    uint sampleSetIdx = 0;

    // Form a primary ray by un-projecting the pixel coordinate using the inverse view * projection matrix
    float2 primaryRaySample = SamplePoint(pixelIdx, sampleSetIdx);

    float2 rayPixelPos = pixelCoord + primaryRaySample;
    float2 ncdXY       = (rayPixelPos / (DispatchRaysDimensions().xy * 0.5f)) - 1.0f;
    ncdXY.y *= -1.0f;
    float4 rayStart = mul(float4(ncdXY, 0.0f, 1.0f), RayTraceCB.InvViewProjection);
    float4 rayEnd   = mul(float4(ncdXY, 1.0f, 1.0f), RayTraceCB.InvViewProjection);

    rayStart.xyz    /= rayStart.w;
    rayEnd.xyz      /= rayEnd.w;
    float3 rayDir    = normalize(rayEnd.xyz - rayStart.xyz);
    float rayLength  = length(rayEnd.xyz - rayStart.xyz);

    // Trace a primary ray
    RayDesc ray;
    ray.Origin    = rayStart.xyz;
    ray.Direction = rayDir;
    ray.TMin      = 0.0f;
    ray.TMax      = rayLength;

    float3 accumRadiance = 0.0f;
    uint pathLength = 1;

    PrimaryPayload payload;

    while (pathLength <= 3)
    {
        // Trace next sample on ray
        const uint hitGroupOffset        = RayTypeRadiance;
        const uint hitGroupGeoMultiplier = NumRayTypes;
        const uint missShaderIdx         = RayTypeRadiance;

        TraceRay(Scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFFFFFFFF, hitGroupOffset, hitGroupGeoMultiplier, missShaderIdx, ray, payload);

        if (payload.GeometryIdx != 0xFFFFFFFF)
        {
            MeshVertex hitSurface;
            Material   material;
            ProcessHit(payload, hitSurface, material);

            float3 radiance = 0.0;

            if ((!AppSettings.EnableDiffuse && !AppSettings.EnableSpecular) ||
                (!AppSettings.EnableDirect && !AppSettings.EnableIndirect))
            {
                radiance = 0.0;
                break;
            }
            else if (pathLength > 1 && !AppSettings.EnableIndirect)
            {
                radiance = 0.0;
                break;
            }
            else
            {
                float3x3 tangentToWorld = float3x3(hitSurface.Tangent, hitSurface.Bitangent, hitSurface.Normal);

                const float3 positionWS          = hitSurface.Position;
                const float3 incomingRayOriginWS = ray.Origin;
                const float3 incomingRayDirWS    = ray.Direction;

                float3 normalWS = hitSurface.Normal;
                if (AppSettings.EnableNormalMaps)
                {
                    // Sample the normal map, and convert the normal to world space
                    Texture2D normalMap = Tex2DTable[NonUniformResourceIndex(material.Normal)];

                    float3 normalTS;
                    normalTS.xy = normalMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).xy * 2.0f - 1.0f;
                    normalTS.z  = sqrt(1.0f - saturate(normalTS.x * normalTS.x + normalTS.y * normalTS.y));
                    normalWS    = normalize(mul(normalTS, tangentToWorld));

                    tangentToWorld._31_32_33 = normalWS;
                }

                float3 baseColor = 1.0f;
                if (AppSettings.EnableAlbedoMaps)
                {
                    Texture2D albedoMap = Tex2DTable[NonUniformResourceIndex(material.Albedo)];
                    baseColor = albedoMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).xyz;
                }

                Texture2D metallicMap = Tex2DTable[NonUniformResourceIndex(material.Metallic)];
                const float metallic  = metallicMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).x;

                const bool enableDiffuse  = (AppSettings.EnableDiffuse && metallic < 1.0f);
                const bool enableSpecular = (AppSettings.EnableSpecular && (AppSettings.EnableIndirectSpecular ? true : (pathLength == 1)));
                if (enableDiffuse || enableSpecular)
                {
                    Texture2D roughnessMap = Tex2DTable[NonUniformResourceIndex(material.Roughness)];
                    const float sqrtRoughness = roughnessMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).x* AppSettings.RoughnessScale;

                    const float3 diffuseAlbedo = lerp(baseColor, 0.0f, metallic) * (enableDiffuse ? 1.0f : 0.0f);
                    const float3 specularAlbedo = lerp(0.03f, baseColor, metallic) * (enableSpecular ? 1.0f : 0.0f);
                    const float roughness = sqrtRoughness * sqrtRoughness;

                    Texture2D emissiveMap = Tex2DTable[NonUniformResourceIndex(material.Emissive)];
                    float3 radiance = emissiveMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).xyz;

                    if (AppSettings.EnableSun)
                    {
                        float3 sunDirection = RayTraceCB.SunDirectionWS;

                        if (AppSettings.SunAreaLightApproximation)
                        {
                            float3 D = RayTraceCB.SunDirectionWS;
                            float3 R = reflect(incomingRayDirWS, normalWS);
                            float  r = RayTraceCB.SinSunAngularRadius;
                            float  d = RayTraceCB.CosSunAngularRadius;
                            float3 DDotR = dot(D, R);
                            float3 S = R - DDotR * D;
                            sunDirection = DDotR < d ? normalize(d * D + normalize(S) * r) : R;
                        }

                        // Shoot a shadow ray to see if the sun is occluded
                        const float visibility = QueryVisibility(positionWS, RayTraceCB.SunDirectionWS);

                        radiance += CalcLighting(normalWS, sunDirection, RayTraceCB.SunIrradiance, diffuseAlbedo, specularAlbedo,
                            roughness, positionWS, incomingRayOriginWS) * visibility;
                    }

                    accumRadiance = radiance;

                    float3 throughput = 0.0f;

                    ImportanceSampleBRDF(
                        tangentToWorld, positionWS, pixelIdx, sampleSetIdx, enableSpecular, enableDiffuse, roughness, diffuseAlbedo, specularAlbedo, ray, throughput);

                    if (pathLength == 1 && !AppSettings.EnableDirect)
                    {
                        radiance = 0.0f;
                    }

                    if (AppSettings.EnableIndirect)
                    {
                        pathLength++;
                        accumRadiance += radiance * throughput;
                    }
                    else
                    {
                        const float visibility = QueryVisibility(ray.Origin, ray.Direction);

                        TextureCube skyTexture = TexCubeTable[RayTraceCB.SkyTextureIdx];
                        float3 skyRadiance = skyTexture.SampleLevel(LinearSampler, ray.Direction, 0.0f).xyz;

                        accumRadiance += visibility * skyRadiance * throughput;

                        break;
                    }
                }
            }
        }
        else
        {
            TextureCube skyTexture = TexCubeTable[RayTraceCB.SkyTextureIdx];
            float3 radiance = skyTexture.SampleLevel(LinearSampler, ray.Direction, 0.0f).xyz;

            if (pathLength == 1)
            {
                float cosSunAngle = dot(rayDir, RayTraceCB.SunDirectionWS);
                if (cosSunAngle >= RayTraceCB.CosSunAngularRadius)
                {
                    radiance = RayTraceCB.SunRenderColor;
                }
            }

            accumRadiance += radiance;
            break;
        }
    }

    accumRadiance = clamp(accumRadiance, 0.0f, FP16Max);

    // Update the progressive result with the new radiance sample
    const float lerpFactor = RayTraceCB.CurrSampleIdx / (RayTraceCB.CurrSampleIdx + 1.0f);
    float3 newSample = accumRadiance;
    float3 currValue = RenderTarget[pixelCoord].xyz;
    float3 newValue = lerp(newSample, currValue, lerpFactor);

    RenderTarget[pixelCoord] = float4(newValue, 1.0f);
}