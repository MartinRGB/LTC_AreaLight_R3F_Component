import { Box, OrbitControls, Plane, SpotLight, shaderMaterial, useDepthBuffer, useGLTF } from "@react-three/drei"
import { Canvas, useThree, useFrame, createPortal, useLoader } from "@react-three/fiber"
import { useControls } from "leva";
import {  useEffect, useMemo, useRef, useState } from "react";
import * as THREE from 'three'
import { Perf } from "r3f-perf";
import { SpotLightProxy } from "@/SpotLightProxy/SpotLightProxy";

import { RectAreaLightHelper } from "three/examples/jsm/helpers/RectAreaLightHelper";
import { RectAreaLightUniformsLib } from "three/examples/jsm/lights/RectAreaLightUniformsLib.js";
import * as React from "react";
import mergeRefs from 'react-merge-refs';

const RECT_AREALIGHT_PREFIX = `
struct PhysicalMaterial {
	vec3 diffuseColor;
	float roughness;
	vec3 specularColor;
	float specularF90;

	#ifdef USE_CLEARCOAT
		float clearcoat;
		float clearcoatRoughness;
		vec3 clearcoatF0;
		float clearcoatF90;
	#endif

	#ifdef USE_IRIDESCENCE
		float iridescence;
		float iridescenceIOR;
		float iridescenceThickness;
		vec3 iridescenceFresnel;
		vec3 iridescenceF0;
	#endif

	#ifdef USE_SHEEN
		vec3 sheenColor;
		float sheenRoughness;
	#endif

	#ifdef IOR
		float ior;
	#endif

	#ifdef USE_TRANSMISSION
		float transmission;
		float transmissionAlpha;
		float thickness;
		float attenuationDistance;
		vec3 attenuationColor;
	#endif

	#ifdef USE_ANISOTROPY
		float anisotropy;
		float alphaT;
		vec3 anisotropyT;
		vec3 anisotropyB;
	#endif

};

// temporary
vec3 clearcoatSpecular = vec3( 0.0 );
vec3 sheenSpecular = vec3( 0.0 );

vec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {
    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );
    float x2 = x * x;
    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );

    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );
}

// Moving Frostbite to Physically Based Rendering 3.0 - page 12, listing 2
// https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
float V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {

	float a2 = pow2( alpha );

	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );

	return 0.5 / max( gv + gl, EPSILON );

}

// Microfacet Models for Refraction through Rough Surfaces - equation (33)
// http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
// alpha is "roughness squared" in Disney’s reparameterization
float D_GGX( const in float alpha, const in float dotNH ) {

	float a2 = pow2( alpha );

	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0; // avoid alpha = 0 with dotNH = 1

	return RECIPROCAL_PI * a2 / pow2( denom );

}

// https://google.github.io/filament/Filament.md.html#materialsystem/anisotropicmodel/anisotropicspecularbrdf
#ifdef USE_ANISOTROPY

	float V_GGX_SmithCorrelated_Anisotropic( const in float alphaT, const in float alphaB, const in float dotTV, const in float dotBV, const in float dotTL, const in float dotBL, const in float dotNV, const in float dotNL ) {

		float gv = dotNL * length( vec3( alphaT * dotTV, alphaB * dotBV, dotNV ) );
		float gl = dotNV * length( vec3( alphaT * dotTL, alphaB * dotBL, dotNL ) );
		float v = 0.5 / ( gv + gl );

		return saturate(v);

	}

	float D_GGX_Anisotropic( const in float alphaT, const in float alphaB, const in float dotNH, const in float dotTH, const in float dotBH ) {

		float a2 = alphaT * alphaB;
		highp vec3 v = vec3( alphaB * dotTH, alphaT * dotBH, a2 * dotNH );
		highp float v2 = dot( v, v );
		float w2 = a2 / v2;

		return RECIPROCAL_PI * a2 * pow2 ( w2 );

	}

#endif

#ifdef USE_CLEARCOAT

	// GGX Distribution, Schlick Fresnel, GGX_SmithCorrelated Visibility
	vec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {

		vec3 f0 = material.clearcoatF0;
		float f90 = material.clearcoatF90;
		float roughness = material.clearcoatRoughness;

		float alpha = pow2( roughness ); // UE4's roughness

		vec3 halfDir = normalize( lightDir + viewDir );

		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float dotNH = saturate( dot( normal, halfDir ) );
		float dotVH = saturate( dot( viewDir, halfDir ) );

		vec3 F = F_Schlick( f0, f90, dotVH );

		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );

		float D = D_GGX( alpha, dotNH );

		return F * ( V * D );

	}

#endif

vec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {

	vec3 f0 = material.specularColor;
	float f90 = material.specularF90;
	float roughness = material.roughness;

	float alpha = pow2( roughness ); // UE4's roughness

	vec3 halfDir = normalize( lightDir + viewDir );

	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );

	vec3 F = F_Schlick( f0, f90, dotVH );

	#ifdef USE_IRIDESCENCE

		F = mix( F, material.iridescenceFresnel, material.iridescence );

	#endif

	#ifdef USE_ANISOTROPY

		float dotTL = dot( material.anisotropyT, lightDir );
		float dotTV = dot( material.anisotropyT, viewDir );
		float dotTH = dot( material.anisotropyT, halfDir );
		float dotBL = dot( material.anisotropyB, lightDir );
		float dotBV = dot( material.anisotropyB, viewDir );
		float dotBH = dot( material.anisotropyB, halfDir );

		float V = V_GGX_SmithCorrelated_Anisotropic( material.alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );

		float D = D_GGX_Anisotropic( material.alphaT, alpha, dotNH, dotTH, dotBH );

	#else

		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );

		float D = D_GGX( alpha, dotNH );

	#endif

	return F * ( V * D );

}

// Rect Area Light

// Real-Time Polygonal-Light Shading with Linearly Transformed Cosines
// by Eric Heitz, Jonathan Dupuy, Stephen Hill and David Neubelt
// code: https://github.com/selfshadow/ltc_code/

vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {

	const float LUT_SIZE = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS = 0.5 / LUT_SIZE;

	float dotNV = saturate( dot( N, V ) );

	// texture parameterized by sqrt( GGX alpha ) and sqrt( 1 - cos( theta ) )
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );

	uv = uv * LUT_SCALE + LUT_BIAS;

	return uv;

}

float LTC_ClippedSphereFormFactor( const in vec3 f ) {

	// Real-Time Area Lighting: a Journey from Research to Production (p.102)
	// An approximation of the form factor of a horizon-clipped rectangle.

	float l = length( f );

	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );

}

vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {

	float x = dot( v1, v2 );

	float y = abs( x );

	// rational polynomial approximation to theta / sin( theta ) / 2PI
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;

	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;

	return cross( v1, v2 ) * theta_sintheta;

}

vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {

	// bail if point is on back side of plane of light
	// assumes ccw winding order of light vertices
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );

	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );

	// construct orthonormal basis around N
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 ); // negated from paper; possibly due to a different handedness of world coordinate system

	// compute transform
	mat3 mat = mInv * transposeMat3( mat3( T1, T2, N ) );

	// transform rect
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );

	// project rect onto sphere
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );

	// calculate vector form factor
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );

	// adjust for horizon clipping
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );

    /*
	// alternate method of adjusting for horizon clipping (see referece)
	// refactoring required
	float len = length( vectorFormFactor );
	float z = vectorFormFactor.z / len;

	const float LUT_SIZE = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS = 0.5 / LUT_SIZE;

	// tabulated horizon-clipped sphere, apparently...
	vec2 uv = vec2( z * 0.5 + 0.5, len );
	uv = uv * LUT_SCALE + LUT_BIAS;

	float scale = texture2D( ltc_2, uv ).w;

	float result = len * scale;
    */

	return vec3( result );

}

// End Rect Area Light
`

const LTC_AREALIGHT_CORE=`

// *************** START LTC AREA LIGHT ***************

void ClipQuadToHorizon(inout vec3 L[5], out int n)
{
    // detect clipping config
    int config = 0;
    if (L[0].z > 0.0) config += 1;
    if (L[1].z > 0.0) config += 2;
    if (L[2].z > 0.0) config += 4;
    if (L[3].z > 0.0) config += 8;

    // clip
    n = 0;

    if (config == 0)
    {
        // clip all
    }
    else if (config == 1) // V1 clip V2 V3 V4
    {
        n = 3;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[3].z * L[0] + L[0].z * L[3];
    }
    else if (config == 2) // V2 clip V1 V3 V4
    {
        n = 3;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    }
    else if (config == 3) // V1 V2 clip V3 V4
    {
        n = 4;
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
        L[3] = -L[3].z * L[0] + L[0].z * L[3];
    }
    else if (config == 4) // V3 clip V1 V2 V4
    {
        n = 3;
        L[0] = -L[3].z * L[2] + L[2].z * L[3];
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
    }
    else if (config == 5) // V1 V3 clip V2 V4) impossible
    {
        n = 0;
    }
    else if (config == 6) // V2 V3 clip V1 V4
    {
        n = 4;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    }
    else if (config == 7) // V1 V2 V3 clip V4
    {
        n = 5;
        L[4] = -L[3].z * L[0] + L[0].z * L[3];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    }
    else if (config == 8) // V4 clip V1 V2 V3
    {
        n = 3;
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
        L[1] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] =  L[3];
    }
    else if (config == 9) // V1 V4 clip V2 V3
    {
        n = 4;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[2].z * L[3] + L[3].z * L[2];
    }
    else if (config == 10) // V2 V4 clip V1 V3) impossible
    {
        n = 0;
    }
    else if (config == 11) // V1 V2 V4 clip V3
    {
        n = 5;
        L[4] = L[3];
        L[3] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    }
    else if (config == 12) // V3 V4 clip V1 V2
    {
        n = 4;
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
    }
    else if (config == 13) // V1 V3 V4 clip V2
    {
        n = 5;
        L[4] = L[3];
        L[3] = L[2];
        L[2] = -L[1].z * L[2] + L[2].z * L[1];
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
    }
    else if (config == 14) // V2 V3 V4 clip V1
    {
        n = 5;
        L[4] = -L[0].z * L[3] + L[3].z * L[0];
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
    }
    else if (config == 15) // V1 V2 V3 V4
    {
        n = 4;
    }
    
    if (n == 3)
        L[3] = L[0];
    if (n == 4)
        L[4] = L[0];
}

float IntegrateEdge(vec3 v1, vec3 v2)
{
    float cosTheta = dot(v1, v2);
    float theta = acos(cosTheta);    
    float res = cross(v1, v2).z * ((theta > 0.001) ? theta/sin(theta) : 1.0);

	return res;
}

vec3 IntegrateEdgeVec(vec3 v1, vec3 v2)
{
    float x = dot(v1, v2);
    float y = abs(x);

    float a = 0.8543985 + (0.4965155 + 0.0145206*y)*y;
    float b = 3.4175940 + (4.1616724 + y)*y;
    float v = a / b;

    float theta_sintheta = (x > 0.0) ? v : 0.5*inversesqrt(max(1.0 - x*x, 1e-7)) - v;

    return cross(v1, v2)*theta_sintheta;
}

// ******* Filtered Border Region 
// https://advances.realtimerendering.com/s2016/s2016_ltc_rnd.pdf p-104  -> filtered border region
// https://www.shadertoy.com/view/dd2SDd

float maskBox(vec2 _st, vec2 _size, float _smoothEdges){
    _size = vec2(0.5)-_size*0.5;
    vec2 aa = vec2(_smoothEdges*0.5);
    vec2 uv = smoothstep(_size,_size+aa,_st);
    uv *= smoothstep(_size,_size+aa,vec2(1.0)-_st);
    return uv.x*uv.y;
}

vec3 draw(vec2 uv,in sampler2D tex) {
    // return texture(tex,vec2(1.- uv.x,uv.y)).rgb;   
    return textureLod(tex,vec2(1.- uv.x,uv.y),8.).rgb;   
}

float grid(float var, float size) {
    return floor(var*size)/size;
}

float blurRand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

// *** the 'repeats' affect the performance
vec3 blurredImage( in float roughness,in vec2 uv , in sampler2D tex)
{
    
    float bluramount = 0.2 * roughness;
    //float dists = 5.;
    vec3 blurred_image = vec3(0.);
    #define repeats 30.
    for (float i = 0.; i < repeats; i++) { 
        //Older:
        //vec2 q = vec2(cos(degrees((grid(i,dists)/repeats)*360.)),sin(degrees((grid(i,dists)/repeats)*360.))) * (1./(1.+mod(i,dists)));
        vec2 q = vec2(cos(degrees((i/repeats)*360.)),sin(degrees((i/repeats)*360.))) *  (blurRand(vec2(i,uv.x+uv.y))+bluramount); 
        vec2 uv2 = uv+(q*bluramount);
        blurred_image += draw(uv2,tex)/2.;
        //One more to hide the noise.
        q = vec2(cos(degrees((i/repeats)*360.)),sin(degrees((i/repeats)*360.))) *  (blurRand(vec2(i+2.,uv.x+uv.y+24.))+bluramount); 
        uv2 = uv+(q*bluramount);
        blurred_image += draw(uv2,tex)/2.;
    }
    blurred_image /= repeats;
        
    return blurred_image;
}


vec3 filterBorderRegion(in float roughness,in vec2 uv,in sampler2D tex){
    // this is useless now
	float scale = 1.;
    float error = 0.4; //0.45

    // Convert uv range to -1 to 1
    vec2 UVC = uv * 2.0 - 1.0;
    UVC *= (1. * 0.5 + 0.5) * (1. + (1. - scale));
    // Convert back to 0 to 1 range
    UVC = UVC * 0.5 + 0.5;

    vec4 ClearCol;
    vec4 BlurCol;
    
    BlurCol.rgb = blurredImage(2.,uv,tex);
	if(UVC.x < 1. && UVC.x > 0. && UVC.y > 0. && UVC.y < 1.){
        ClearCol.rgb = blurredImage(min(2.,roughness),UVC,tex);
    }
	//ClearCol.rgb = blurredImage(roughness,UVC,tex);
	float boxMask = maskBox(UVC,vec2(scale+0.),error);
    BlurCol.rgb = mix(BlurCol.rgb, ClearCol.rgb, boxMask);
    return BlurCol.rgb;
    
    // # Method 2
	//return blurredImage(min(2.,roughness),uv,tex).rgb;
}

vec3 FetchDiffuseFilteredTexture(float roughness,vec3 L[5],vec3 vLooupVector,sampler2D tex)
{
	vec3 V1 = L[1] - L[0];
	vec3 V2 = L[3] - L[0];
	// Plane's normal
	vec3 PlaneOrtho = cross(V1, V2);
	float PlaneAreaSquared = dot(PlaneOrtho, PlaneOrtho);
	float planeDistxPlaneArea = dot(PlaneOrtho, L[0]);
	// orthonormal projection of (0,0,0) in area light space
	vec3 P = planeDistxPlaneArea * PlaneOrtho / PlaneAreaSquared - L[0];

	// find tex coords of P
	float dot_V1_V2 = dot(V1, V2);
	float inv_dot_V1_V1 = 1.0 / dot(V1, V1);
	vec3 V2_ = V2 - V1 * dot_V1_V2 * inv_dot_V1_V1;
	vec2 UV;
	UV.y = dot(V2_, P) / dot(V2_, V2_);
	UV.x = dot(V1, P) * inv_dot_V1_V1 - dot_V1_V2 * inv_dot_V1_V1 * UV.y;

	// float scale = 1.;
    // float error = 0.45;
    // // Convert uv range to -1 to 1
    // vec2 UVC = UV * 2.0 - 1.0;
    // UVC *= (1. * 0.5 + 0.5) * (1. + (1. - scale));
    // // Convert back to 0 to 1 range
    // UVC = UVC * 0.5 + 0.5;

    // vec4 ClearCol;
    // vec4 BlurCol;
    
    // BlurCol.rgb = blurredImage(2.,UV,tex);
	// if(UVC.x < 1. && UVC.x > 0. && UVC.y > 0. && UVC.y < 1.){
    //     ClearCol.rgb = blurredImage(min(2.,roughness),UVC,tex);
    // }
	// //ClearCol.rgb = blurredImage(roughness,UVC,tex);
	// float boxMask = maskBox(UVC,vec2(scale+0.),error);
    // BlurCol.rgb = mix(BlurCol.rgb, ClearCol.rgb, boxMask);

    // to delete border light even the canvas is dark
    // UV -= .5;
    // UV /= 1.1;
    // UV += .5;

	return filterBorderRegion(roughness,UV,tex);
}

vec3 LTC_Evaluate_With_Texture_0(const in float roughness, const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ],in sampler2D tex,in sampler2D blur_tex) {

	// bail if point is on back side of plane of light
	// assumes ccw winding order of light vertices
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );

	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );

	// construct orthonormal basis around N
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 ); // negated from paper; possibly due to a different handedness of world coordinate system

	// compute transform
	mat3 mat = mInv * transposeMat3( mat3( T1, T2, N ) );

	// transform rect
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );

	// project rect onto sphere
    
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );

	// calculate vector form factor
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );

	// adjust for horizon clipping
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );

    result = max(0.,result);

    // Calculate color

    vec3 L[5];
	L[0] = mat * ( rectCoords[0] - P );
	L[1] = mat * ( rectCoords[1] - P );
	L[2] = mat * ( rectCoords[2] - P );
	L[3] = mat * ( rectCoords[3] - P );

    vec3 e1 = normalize(L[0] - L[1]);
    vec3 e2 = normalize(L[2] - L[1]);
    vec3 N2 = cross(e1, e2); // Normal to light
    vec3 V2 = N2 * dot(L[1], N2); // Vector to some point in light rect
    vec2 Tlight_shape = vec2(length(L[0] - L[1]), length(L[2] - L[1]));
    V2 = V2 - L[1];
    float b = e1.y*e2.x - e1.x*e2.y + .1; // + .1 to remove artifacts
	vec2 pLight = vec2((V2.y*e2.x - V2.x*e2.y)/b, (V2.x*e1.y - V2.y*e1.x)/b);
   	pLight /= Tlight_shape;
    // pLight -= .5;
    // pLight /= 2.5;
    // pLight += .5;
    
    vec3 ref_col;
    
    ref_col = texture(tex, pLight).xyz;

    //ref_col = filterBorderRegion(roughness + 2.,vec2(saturate(pLight.x),saturate(pLight.y)),tex);

	//ref_col = FetchDiffuseFilteredTexture(roughness,L,vec3(result),tex);

    //ref_col = textureLod(tex,pLight,4.).rgb;

    vec3 Lo_i = vec3(result) * ref_col;

    return Lo_i;


}

vec3 LTC_Evaluate_With_Texture( 
    const in float roughness,
    const in vec3 N, 
    const in vec3 V, 
    const in vec3 P, 
    const in mat3 Minv, 
    const in vec3 points[ 4 ], 
    in sampler2D tex) {
	
	// construct orthonormal basis around N
	vec3 T1, T2;
    T1 = normalize(V - N*dot(V, N));
    T2 = cross(N, T1);
    // rotate area light in (T1, T2, N) basis
	mat3 mInv = Minv * transpose(mat3(T1, T2, N));
	
	vec3 L[5];
    L[0] = mInv * (points[0] - P);
    L[1] = mInv * (points[1] - P);
    L[2] = mInv * (points[2] - P);
	L[3] = mInv * (points[3] - P);

	int n=0;
    ClipQuadToHorizon(L, n);
	
	if (n == 0)
		return vec3(0, 0, 0);

	vec3 PL[5];
	PL[0] = normalize(L[0]);
	PL[1] = normalize(L[1]);
	PL[2] = normalize(L[2]);
	PL[3] = normalize(L[3]);
	PL[4] = normalize(L[4]);


    // *** Method I

    // calculate vector form factor
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += IntegrateEdgeVec( PL[ 0 ], PL[ 1 ] );
	vectorFormFactor += IntegrateEdgeVec( PL[ 1 ], PL[ 2 ] );
	vectorFormFactor += IntegrateEdgeVec( PL[ 2 ], PL[ 3 ] );
	vectorFormFactor += IntegrateEdgeVec( PL[ 3 ], PL[ 0 ] );

	// adjust for horizon clipping
	float sum = LTC_ClippedSphereFormFactor( vectorFormFactor );

    // *** Method II
	// *** Integrate for every edge.
	// float sum = 0.0;
	
	// sum += IntegrateEdge(PL[0], PL[1]);
    // sum += IntegrateEdge(PL[1], PL[2]);
    // sum += IntegrateEdge(PL[2], PL[3]);
    // if (n >= 4)
    //     sum += IntegrateEdge(PL[3], PL[4]);
    // if (n == 5)
	// 	sum += IntegrateEdge(PL[4], PL[0]);
		
	// sum =  max(0.0, sum);
    
    // *** Calculate color
    vec3 e1 = normalize(L[0] - L[1]);
    vec3 e2 = normalize(L[2] - L[1]);
    vec3 N2 = cross(e1, e2); // Normal to light
    vec3 V2 = N2 * dot(L[1], N2); // Vector to some point in light rect
    vec2 Tlight_shape = vec2(length(L[0] - L[1]), length(L[2] - L[1]));
    V2 = V2 - L[1];
    float b = e1.y*e2.x - e1.x*e2.y + 0.1; // + .1 to remove artifacts
	vec2 pLight = vec2((V2.y*e2.x - V2.x*e2.y)/b, (V2.x*e1.y - V2.y*e1.x)/b);
   	pLight /= Tlight_shape;
    // pLight -= .5;
    // pLight /= 2.5;
    // pLight += .5;
    
	vec3 ref_col;
	//ref_col = filterBorderRegion(roughness + 2.,vec2(saturate(pLight.x),saturate(pLight.y)),tex);

	//ref_col = FetchDiffuseFilteredTexture(roughness,L,vec3(sum),tex);

    ref_col = textureLod(tex,pLight,4.).rgb;

	vec3 Lo_i = vec3(sum)*ref_col;
	
    return Lo_i;

}

// *************** END LTC AREA LIGHT ***************
`

const RECT_AREALIGHT_SUFFIX_0 = `
#if defined( USE_SHEEN )

// https://github.com/google/filament/blob/master/shaders/src/brdf.fs
float D_Charlie( float roughness, float dotNH ) {

	float alpha = pow2( roughness );

	// Estevez and Kulla 2017, "Production Friendly Microfacet Sheen BRDF"
	float invAlpha = 1.0 / alpha;
	float cos2h = dotNH * dotNH;
	float sin2h = max( 1.0 - cos2h, 0.0078125 ); // 2^(-14/2), so sin2h^2 > 0 in fp16

	return ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );

}

// https://github.com/google/filament/blob/master/shaders/src/brdf.fs
float V_Neubelt( float dotNV, float dotNL ) {

	// Neubelt and Pettineo 2013, "Crafting a Next-gen Material Pipeline for The Order: 1886"
	return saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );

}

vec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {

	vec3 halfDir = normalize( lightDir + viewDir );

	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );

	float D = D_Charlie( sheenRoughness, dotNH );
	float V = V_Neubelt( dotNV, dotNL );

	return sheenColor * ( D * V );

}

#endif

// This is a curve-fit approxmation to the "Charlie sheen" BRDF integrated over the hemisphere from 
// Estevez and Kulla 2017, "Production Friendly Microfacet Sheen BRDF". The analysis can be found
// in the Sheen section of https://drive.google.com/file/d/1T0D1VSyR4AllqIJTQAraEIzjlb5h4FKH/view?usp=sharing
float IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {

	float dotNV = saturate( dot( normal, viewDir ) );

	float r2 = roughness * roughness;

	float a = roughness < 0.25 ? -339.2 * r2 + 161.4 * roughness - 25.9 : -8.48 * r2 + 14.3 * roughness - 9.95;

	float b = roughness < 0.25 ? 44.0 * r2 - 23.7 * roughness + 3.26 : 1.97 * r2 - 3.27 * roughness + 0.72;

	float DG = exp( a * dotNV + b ) + ( roughness < 0.25 ? 0.0 : 0.1 * ( roughness - 0.25 ) );

	return saturate( DG * RECIPROCAL_PI );

}

// Analytical approximation of the DFG LUT, one half of the
// split-sum approximation used in indirect specular lighting.
// via 'environmentBRDF' from "Physically Based Shading on Mobile"
// https://www.unrealengine.com/blog/physically-based-shading-on-mobile
vec2 DFGApprox( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {

	float dotNV = saturate( dot( normal, viewDir ) );

	const vec4 c0 = vec4( - 1, - 0.0275, - 0.572, 0.022 );

	const vec4 c1 = vec4( 1, 0.0425, 1.04, - 0.04 );

	vec4 r = roughness * c0 + c1;

	float a004 = min( r.x * r.x, exp2( - 9.28 * dotNV ) ) * r.x + r.y;

	vec2 fab = vec2( - 1.04, 1.04 ) * a004 + r.zw;

	return fab;

}

vec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {

	vec2 fab = DFGApprox( normal, viewDir, roughness );

	return specularColor * fab.x + specularF90 * fab.y;

}

// Fdez-Agüera's "Multiple-Scattering Microfacet Model for Real-Time Image Based Lighting"
// Approximates multiscattering in order to preserve energy.
// http://www.jcgt.org/published/0008/01/03/
#ifdef USE_IRIDESCENCE
void computeMultiscatteringIridescence( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif

	vec2 fab = DFGApprox( normal, viewDir, roughness );

	#ifdef USE_IRIDESCENCE

		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );

	#else

		vec3 Fr = specularColor;

	#endif

	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;

	float Ess = fab.x + fab.y;
	float Ems = 1.0 - Ess;

	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619; // 1/21
	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );

	singleScatter += FssEss;
	multiScatter += Fms * Ems;

}
`

const RECT_AREALIGHT_HACK = `

// *************** HACK THE RECT AREA LIGHT ***************

uniform bool isLTCWithTexture;
uniform sampler2D ltc_tex;
uniform sampler2D ltc_blur_tex;

#if NUM_RECT_AREA_LIGHTS > 0

	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {

		vec3 normal = geometry.normal;
		vec3 viewDir = geometry.viewDir;
		vec3 position = geometry.position;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.roughness;

		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight; // counterclockwise; light shines in local neg z direction
		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;



        if(isLTCWithTexture){



            // *** METHOD I ***


            // float LUTSIZE = 128.0;
            // float MATRIX_PARAM_OFFSET = 0.01;


            // float theta = acos(dot(normal, viewDir));
    
            // vec2 uv_0 = vec2(roughness, theta/(0.5*PI)) * float(LUTSIZE-1.);
            // uv_0 += vec2(0.5);
            
            // vec4 params = texture(ltc_1, (uv_0+vec2(MATRIX_PARAM_OFFSET, 0.0))/vec2(LUTSIZE,LUTSIZE));
            
            // mat3 mInv = mat3(
            //     vec3(  1,        0,      params.y),
            //     vec3(  0,     params.z,   0),
            //     vec3(params.w,   0,      params.x)
            // );

            // vec4 t2 = texture(ltc_2, uv_0);

            // vec3 fresnel = ( material.specularColor * t2.x + ( vec3( 1.0 ) - material.specularColor ) * t2.y );

            // **** METHOD II ****
            // const float LUT_SIZE = 8.0;
            // const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
            // const float LUT_BIAS = 0.5 / LUT_SIZE;
            // float MATRIX_PARAM_OFFSET = 1.0;
        
            // float dotNV = saturate( dot( normal, viewDir ) );
        
            // // texture parameterized by sqrt( GGX alpha ) and sqrt( 1 - cos( theta ) )
            // vec2 uv_0 = vec2( roughness, sqrt( 1.0 - dotNV ) );
        
            // uv_0 = uv_0 * LUT_SCALE + LUT_BIAS;
            // uv_0 +=  vec2(MATRIX_PARAM_OFFSET,0)/vec2(LUT_SIZE,LUT_SIZE);

            // vec4 t1 = texture2D( ltc_1, uv_0 );
            // vec4 t2 = texture2D( ltc_2, uv_0 );

            // mat3 mInv = mat3(
            //     vec3( t1.x, 0, t1.y ),
            //     vec3(    0, 1,    0 ),
            //     vec3( t1.z, 0, t1.w )
            // );
    
            // vec3 fresnel = ( material.specularColor * t2.x + ( vec3( 1.0 ) - material.specularColor ) * t2.y );
            // *** result ***
            
    
            // LTC Fresnel Approximation by Stephen Hill
            // http://blog.selfshadow.com/publications/s2016-advances/s2016_ltc_fresnel.pdf


            // reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate_With_Texture(roughness, normal, viewDir, position, mInv, rectCoords , ltc_tex);
            // reflectedLight.directDiffuse += lightColor * material.diffuseColor * LTC_Evaluate_With_Texture(roughness,normal, viewDir, position, mat3( 1.0 ), rectCoords, ltc_tex );

           
            // *** METHOD III ***

            // vec2 uv = LTC_Uv( normal, viewDir, roughness );

            // vec4 t1 = texture2D( ltc_1, uv );
            // vec4 t2 = texture2D( ltc_2, uv );
    
            const float LUT_SIZE = 64.0;
            const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
            const float LUT_BIAS = 0.5 / LUT_SIZE;
            float MATRIX_PARAM_OFFSET = 16.0;

            float dotNV = saturate( dot( normal, viewDir ) );
        
            // texture parameterized by sqrt( GGX alpha ) and sqrt( 1 - cos( theta ) )
            vec2 uv_0 = vec2( roughness , sqrt( 1.0 - dotNV ) );
        
            uv_0 = uv_0 * LUT_SCALE + LUT_BIAS;
            //uv_0 +=  vec2(MATRIX_PARAM_OFFSET,0)/vec2(LUT_SIZE,LUT_SIZE);
            

            vec4 t1 = texture2D( ltc_1, uv_0 );
            vec4 t2 = texture2D( ltc_2, uv_0 );

            // LTC Fresnel Approximation by Stephen Hill
            // http://blog.selfshadow.com/publications/s2016-advances/s2016_ltc_fresnel.pdf
            vec3 fresnel = ( material.specularColor * t2.x + ( vec3( 1.0 ) - material.specularColor ) * t2.y );

            mat3 mInv = mat3(
                vec3( t1.x, 0, t1.y ),
                vec3(    0, 1,    0 ),
                vec3( t1.z, 0, t1.w )
            );
            
            reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate_With_Texture_0( roughness, normal, viewDir, position, mInv, rectCoords , ltc_tex,ltc_blur_tex);
            reflectedLight.directDiffuse += lightColor * material.diffuseColor * LTC_Evaluate_With_Texture_0( roughness, normal, viewDir, position, mat3( 1.0 ), rectCoords, ltc_tex,ltc_blur_tex);


            // *** METHOD IV 
            // const float LUT_SIZE  = 64.0;
            // const float LUT_SCALE = (LUT_SIZE - 1.0)/LUT_SIZE;
            // const float LUT_BIAS  = 0.5/LUT_SIZE;

            // float theta = acos(dot(normal, viewDir));
            // vec2 uv = vec2(roughness, theta/(0.5*PI));
            // uv = uv*LUT_SCALE + LUT_BIAS;
            
            // vec4 t1 = texture2D( ltc_1, uv );
            // vec4 t2 = texture2D( ltc_2, uv );
            // mat3 Minv = mat3(
            //     vec3(  1,   0, t1.y),
            //     vec3(  0, t1.z,   0),
            //     vec3(t1.w,   0, t1.x)
            // );

            // vec3 fresnel = ( material.specularColor * t2.x + ( vec3( 1.0 ) - material.specularColor ) * t2.y );
            // reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate_With_Texture( roughness, normal, viewDir, position, Minv, rectCoords , ltc_tex);
            // reflectedLight.directDiffuse += lightColor * material.diffuseColor * LTC_Evaluate_With_Texture( roughness, normal, viewDir, position, mat3( 1.0 ), rectCoords, ltc_tex);
           
        }
        else{

            vec2 uv = LTC_Uv( normal, viewDir, roughness );

            vec4 t1 = texture2D( ltc_1, uv );
            vec4 t2 = texture2D( ltc_2, uv );
    
            // LTC Fresnel Approximation by Stephen Hill
            // http://blog.selfshadow.com/publications/s2016-advances/s2016_ltc_fresnel.pdf
            vec3 fresnel = ( material.specularColor * t2.x + ( vec3( 1.0 ) - material.specularColor ) * t2.y );

            mat3 mInv = mat3(
                vec3( t1.x, 0, t1.y ),
                vec3(    0, 1,    0 ),
                vec3( t1.z, 0, t1.w )
            );

		    reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );
		    reflectedLight.directDiffuse += lightColor * material.diffuseColor * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );
        
        }

	}

#endif

// *************** HACK THE RECT AREA LIGHT ***************
`
const RECT_AREALIGHT_SUFFIX_1 = `

void RE_Direct_Physical( const in IncidentLight directLight, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {

	float dotNL = saturate( dot( geometry.normal, directLight.direction ) );

	vec3 irradiance = dotNL * directLight.color;

	#ifdef USE_CLEARCOAT

		float dotNLcc = saturate( dot( geometry.clearcoatNormal, directLight.direction ) );

		vec3 ccIrradiance = dotNLcc * directLight.color;

		clearcoatSpecular += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometry.viewDir, geometry.clearcoatNormal, material );

	#endif

	#ifdef USE_SHEEN

		sheenSpecular += irradiance * BRDF_Sheen( directLight.direction, geometry.viewDir, geometry.normal, material.sheenColor, material.sheenRoughness );

	#endif

	reflectedLight.directSpecular += irradiance * BRDF_GGX( directLight.direction, geometry.viewDir, geometry.normal, material );

	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}

void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {

	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );

}

void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {

	#ifdef USE_CLEARCOAT

		clearcoatSpecular += clearcoatRadiance * EnvironmentBRDF( geometry.clearcoatNormal, geometry.viewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );

	#endif

	#ifdef USE_SHEEN

		sheenSpecular += irradiance * material.sheenColor * IBLSheenBRDF( geometry.normal, geometry.viewDir, material.sheenRoughness );

	#endif

	// Both indirect specular and indirect diffuse light accumulate here

	vec3 singleScattering = vec3( 0.0 );
	vec3 multiScattering = vec3( 0.0 );
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;

	#ifdef USE_IRIDESCENCE

		computeMultiscatteringIridescence( geometry.normal, geometry.viewDir, material.specularColor, material.specularF90, material.iridescence, material.iridescenceFresnel, material.roughness, singleScattering, multiScattering );

	#else

		computeMultiscattering( geometry.normal, geometry.viewDir, material.specularColor, material.specularF90, material.roughness, singleScattering, multiScattering );

	#endif

	vec3 totalScattering = singleScattering + multiScattering;
	vec3 diffuse = material.diffuseColor * ( 1.0 - max( max( totalScattering.r, totalScattering.g ), totalScattering.b ) );

	reflectedLight.indirectSpecular += radiance * singleScattering;
	reflectedLight.indirectSpecular += multiScattering * cosineWeightedIrradiance;

	reflectedLight.indirectDiffuse += diffuse * cosineWeightedIrradiance;

}

#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical

// ref: https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {

	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );

}

`

const prefix_vertex = `
    varying vec2 vUv;
    varying vec3 v_pos;

`

const common_vertex_main = `
    void main()	{
        vUv = uv;
        v_pos = position;
        gl_Position = vec4(position, 1.);
    }
`

const prefix_frag = `
    #ifdef GL_ES
    precision mediump float;
    #endif

    varying vec3 v_pos;
    varying vec2 vUv;
`

const DOWNSAMPLE_BLUR=`
uniform sampler2D buff_tex;
uniform float blurOffset;

#define sampleScale (1. + blurOffset*0.1)
#define pixelOffset 1.

void main() {
    vec2 uv = vUv*sampleScale;
    vec2 halfpixel = pixelOffset / ((gl_FragCoord.xy/vUv.xy) / sampleScale);

    vec4 sum;
    sum = texture(buff_tex, uv) * 4.0;
    sum += texture(buff_tex, uv - halfpixel.xy * blurOffset);
    sum += texture(buff_tex, uv + halfpixel.xy * blurOffset);
    sum += texture(buff_tex, uv + vec2(halfpixel.x, -halfpixel.y) * blurOffset);
    sum += texture(buff_tex, uv - vec2(halfpixel.x, -halfpixel.y) * blurOffset);

    gl_FragColor = sum / 8.0;
}
`

const UPSAMPLE_BLUR = `
uniform sampler2D buff_tex;
uniform float blurOffset;

#define sampleScale (1. + blurOffset*0.1)
#define pixelOffset 1.

void main() {

    vec2 uv = vUv/sampleScale;
    vec2 halfpixel = pixelOffset / ((gl_FragCoord.xy/vUv.xy) * sampleScale);

    vec4 sum;
    
    sum =  texture(buff_tex, uv +vec2(-halfpixel.x * 2.0, 0.0) * blurOffset);
    sum += texture(buff_tex, uv + vec2(-halfpixel.x, halfpixel.y) * blurOffset) * 2.0;
    sum += texture(buff_tex, uv + vec2(0.0, halfpixel.y * 2.0) * blurOffset);
    sum += texture(buff_tex, uv + vec2(halfpixel.x, halfpixel.y) * blurOffset) * 2.0;
    sum += texture(buff_tex, uv + vec2(halfpixel.x * 2.0, 0.0) * blurOffset);
    sum += texture(buff_tex, uv + vec2(halfpixel.x, -halfpixel.y) * blurOffset) * 2.0;
    sum += texture(buff_tex, uv + vec2(0.0, -halfpixel.y * 2.0) * blurOffset);
    sum += texture(buff_tex, uv + vec2(-halfpixel.x, -halfpixel.y) * blurOffset) * 2.0;

    gl_FragColor = sum / 12.0;
}
`


const LTCAreaLightWithHelper = React.forwardRef(({ children,position,rotation, color,intensity,width,height,isEnableHelper }:{
    children?: React.ReactNode;
    position?: [number, number, number];
    rotation?: [number, number, number];


    color?: string;
    intensity?: number;
    width?: number;
    height?: number;
    texture?: THREE.Texture | string;
    isEnableHelper?:boolean;
},
ref: React.ForwardedRef<any>
) => {
    // Besides the useThree hook, all of this is taken straight from one of the examples on threejs.org: https://threejs.org/examples/#webgl_lights_rectarealight.
  
    const { scene,gl,size,camera} = useThree();

    const rectAreaLightRef = useRef<any>();
    const childrenRef = useRef<any>(null!);

    const image_Tex = useLoader(THREE.TextureLoader,'./light_test.png')
    const [threeTexture,setThreeTexture] = useState<THREE.Texture | null>(null);

    // # Material Ref
    const kawaseBlurMaterialRefA = useRef<THREE.ShaderMaterial | null>(null)
    const kawaseBlurMaterialRefB = useRef<THREE.ShaderMaterial | null>(null)
    const kawaseBlurMaterialRefC = useRef<THREE.ShaderMaterial | null>(null)
    const kawaseBlurMaterialRefD = useRef<THREE.ShaderMaterial | null>(null)
    const finalMaterialRef = useRef<THREE.ShaderMaterial | null>(null)
    
    // # Scene
    const [
            DKDownSceneA,
            DKDownSceneB,
            DKUpSceneA,
            DKUpSceneB,

    ] = useMemo(()=>{
        return [
            new THREE.Scene(),
            new THREE.Scene(),
            new THREE.Scene(),
            new THREE.Scene()
        ]
    },[])

    // # FBO
    let FBOSettings = { format: THREE.RGBAFormat,minFilter: THREE.LinearFilter,magFilter: THREE.LinearFilter,type: THREE.FloatType,}

    let [
        blurFBOA,
        blurFBOB,
        blurFBOC,
        blurFBOD
    ] = useMemo(()=>{
        return [
            new THREE.WebGLRenderTarget(size.width/.2,size.height/2,FBOSettings),
            new THREE.WebGLRenderTarget(size.width/.2,size.height/2,FBOSettings),
            new THREE.WebGLRenderTarget(size.width/.2,size.height/2,FBOSettings),
            new THREE.WebGLRenderTarget(size.width/.2,size.height/2,FBOSettings)
        ]
    },[])



    const HackRectAreaLight = (tex:THREE.Texture) =>{
        if(childrenRef.current){
            childrenRef.current.traverse((obj:any)=>{
                if(obj.isMesh){
                    obj.material.onBeforeCompile = (shader:any) => {
                            shader.uniforms.isLTCWithTexture = { value: true };
                            shader.uniforms.ltc_tex = { value: tex };
                            //
                            shader.fragmentShader = shader.fragmentShader.replace(`#include <lights_physical_pars_fragment>`,
                            RECT_AREALIGHT_PREFIX
                            + LTC_AREALIGHT_CORE
                            + RECT_AREALIGHT_SUFFIX_0
                            + RECT_AREALIGHT_HACK
                            + RECT_AREALIGHT_SUFFIX_1
                            )
                    }
                }
            })
        }
    }

    // *** Load Video Texture ***
    useEffect(()=>{
        const videoE = document.createElement('video');
        videoE.src = 'https://localhost:8222/external/Video/capture_1.mp4';
        videoE.crossOrigin = 'Anonymous'
        videoE.loop = true
        videoE.muted = true
        videoE.load();
        videoE.play();     
        const vidTex = new THREE.VideoTexture( videoE );
        vidTex.minFilter = THREE.NearestFilter;
        vidTex.magFilter = THREE.LinearFilter;
        vidTex.wrapS = vidTex.wrapT = THREE.ClampToEdgeWrapping;
        console.log(vidTex)
        setThreeTexture(vidTex)
    },[])

    // *** Init LTC Texture ***
    useEffect(()=>{
        if(rectAreaLightRef.current){
            RectAreaLightUniformsLib.init();
            if(isEnableHelper)
                scene.add(new RectAreaLightHelper(rectAreaLightRef.current));
        }

    },[rectAreaLightRef])


    // *** Init Shader ***
    useEffect(()=>{
        HackRectAreaLight(image_Tex)

    },[childrenRef])

    useFrame(() => {

        if(threeTexture){

            if(kawaseBlurMaterialRefA.current){
                kawaseBlurMaterialRefA.current.uniforms.buff_tex.value = threeTexture
                // Transformation Pass Buffer
                gl.setRenderTarget(blurFBOA);
                gl.render(DKDownSceneA,camera)
                gl.setRenderTarget(null)
                            
            }

            if(kawaseBlurMaterialRefB.current){
                kawaseBlurMaterialRefB.current.uniforms.buff_tex.value = blurFBOA.texture
                // Transformation Pass Buffer
                gl.setRenderTarget(blurFBOB);
                gl.render(DKDownSceneB,camera)
                gl.setRenderTarget(null)
            }

            if(kawaseBlurMaterialRefC.current){
                kawaseBlurMaterialRefC.current.uniforms.buff_tex.value = blurFBOB.texture
                // Transformation Pass Buffer
                gl.setRenderTarget(blurFBOC);
                gl.render(DKUpSceneA,camera)
                gl.setRenderTarget(null)
            }

            if(kawaseBlurMaterialRefD.current){
                kawaseBlurMaterialRefD.current.uniforms.buff_tex.value = blurFBOC.texture
                // Transformation Pass Buffer
                gl.setRenderTarget(blurFBOD);
                gl.render(DKUpSceneB,camera)
                gl.setRenderTarget(null)
            }

            // if(finalMaterialRef.current){
            //     finalMaterialRef.current.uniforms.buff_tex.value = blurFBOD.texture
            // }


            HackRectAreaLight(threeTexture)

        }
    },)


    const {blurOffset,roughness} = useControls('blur',{
        blurOffset:{
            value:1.0,
            min:0.,
            max:10,
            step:0.1,
            onChange:(v:any)=>{
                if(kawaseBlurMaterialRefA.current){
                    kawaseBlurMaterialRefA.current.uniforms.blurOffset.value = v
                }
                if(kawaseBlurMaterialRefB.current){
                    kawaseBlurMaterialRefB.current.uniforms.blurOffset.value = v
                }
                if(kawaseBlurMaterialRefC.current){
                    kawaseBlurMaterialRefC.current.uniforms.blurOffset.value = v
                }
                if(kawaseBlurMaterialRefD.current){
                    kawaseBlurMaterialRefD.current.uniforms.blurOffset.value = v
                }
                
            }
        },
        roughness:{
            value:0.0,
            min:-10.,
            max:10.,
            step:0.01,
        }

    })
  
    //return null;
    return(
        <>
            {threeTexture != null?
            <>
            <rectAreaLight
                ref={rectAreaLightRef}
                rotation={rotation?rotation:[0,0,0]}
                position={position?position:[0,0,0]}
                width={width?width:4}
                height={height?height:4}
                color={color?color:'white'}
                intensity={intensity?intensity:15}
            />

            {
                createPortal(
                <>
                    <Plane args={[2,2]}>
                        <shaderMaterial
                            ref={kawaseBlurMaterialRefA}
                            vertexShader={
                                prefix_vertex+common_vertex_main
                            }
                            fragmentShader={
                                prefix_frag
                                + DOWNSAMPLE_BLUR
                            }
                            uniforms={{
                                buff_tex:{value:null},
                                blurOffset:{value:blurOffset}
                            }}
                        ></shaderMaterial>
                    </Plane>
                </>
                ,DKDownSceneA)
            }
            {
                createPortal(
                <>
                    <Plane args={[2,2]}>
                        <shaderMaterial
                            ref={kawaseBlurMaterialRefB}
                            vertexShader={
                                prefix_vertex+common_vertex_main
                            }
                            fragmentShader={
                                prefix_frag
                                + DOWNSAMPLE_BLUR
                            }
                            uniforms={{
                                buff_tex:{value:null},
                                blurOffset:{value:blurOffset}
                            }}
                        ></shaderMaterial>
                    </Plane>
                </>
                ,DKDownSceneB)
            }
            {
                createPortal(
                <>
                    <Plane args={[2,2]}>
                        <shaderMaterial
                            ref={kawaseBlurMaterialRefC}
                            vertexShader={
                                prefix_vertex+common_vertex_main
                            }
                            fragmentShader={
                                prefix_frag
                                + UPSAMPLE_BLUR
                            }
                            uniforms={{
                                buff_tex:{value:null},
                                blurOffset:{value:blurOffset}
                            }}
                        ></shaderMaterial>
                    </Plane>
                </>
                ,DKUpSceneA)
            }
            {
                createPortal(
                <>
                    <Plane args={[2,2]}>
                        <shaderMaterial
                            ref={kawaseBlurMaterialRefD}
                            vertexShader={
                                prefix_vertex+common_vertex_main
                            }
                            fragmentShader={
                                prefix_frag
                                + UPSAMPLE_BLUR
                            }
                            uniforms={{
                                buff_tex:{value:null},
                                blurOffset:{value:blurOffset}
                            }}
                        ></shaderMaterial>
                    </Plane>
                </>
                ,DKUpSceneB)
            }

            {/* <Plane 
                args={[width?width:1,height?height:1]}
                position={position?position:[0,0,0]}
                >
                        <shaderMaterial
                            ref={finalMaterialRef}
                            uniforms = {{
                                buff_tex:{value:null},
                            }}
                            vertexShader={
                                prefix_vertex+common_vertex_main
                            }
                            fragmentShader={
                                prefix_frag
                                + `
                                uniform sampler2D buff_tex;
                                void main(){
                                    gl_FragColor = texture(buff_tex,vUv);
                                }
                                `
                            }
                            
   
                        ></shaderMaterial>
            </Plane> */}

            <Plane 
                args={[width?width:4,height?height:4]}
                position={position?position:[0,0,0]}
                >
                <meshBasicMaterial color={color?color:'white'} map={threeTexture} />
            </Plane>
            
            <group ref={childrenRef}>
                {children}
            </group>
            </>:
            <></>}
        </>
    )
  
});

LTCAreaLightWithHelper.displayName = 'LTCAreaLightWithHelper'

const LTCTexturedLightDemo = () =>{

    const {size,gl,camera} = useThree()
    const depthBuffer = useDepthBuffer({ frames: 1 })
    const { nodes, materials } = useGLTF('./model.gltf')
    const ref = useRef();

    const {dragon_position} = useControls('attributes',{
        dragon_position:{
            value:[0,0,0],
            step:0.1,
        }
    })

    return(
        <>
        <LTCAreaLightWithHelper 
            ref={ref} 
            position={[0, 3, -5]} 
            rotation={[0,-Math.PI,0]} 
            // rotation={[0,-0,0]} 
            color="white" 
            isEnableHelper={false}    
        >
            <mesh position={dragon_position} castShadow receiveShadow geometry={nodes.dragon.geometry} material={materials['Default OBJ.001']} dispose={null} />
            <mesh receiveShadow position={[0, -1, 0]} rotation-x={-Math.PI / 2}>
                <planeGeometry args={[50, 50]} />
                <meshPhongMaterial />
            </mesh>
            <Plane args={[100, 100]} rotation={[-Math.PI / 2, 0, 0]}>
                <meshStandardMaterial color="#808080" roughness={0.1} metalness={0} />
            </Plane>
        </LTCAreaLightWithHelper>
        </>
    )
}


export const Effect = (props:any) =>{

    return(
      <>
          <Canvas 
            camera={{ position: [0, 1, 8], fov: 50, near: 0.1, far: 1000 }}
            className={props.className} 
            style={{...props.style}}>
            <Perf style={{position:'absolute',top:'10px',left:'10px',width:'360px',borderRadius:'10px'}}/>
            <ambientLight intensity={0.015}></ambientLight>
            <color attach="background" args={['#202020']} />
            <fog attach="fog" args={['#202020', 5, 20]} />
            <LTCTexturedLightDemo/>
            <OrbitControls></OrbitControls>
          </Canvas>
      </>
  
    )
}


