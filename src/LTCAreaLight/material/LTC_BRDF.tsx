export const ltc_shadertoy=`
// Code modified EvilRyu by JuliaPoo
// https://www.shadertoy.com/view/4tBBDK

#ifdef GL_ES
//precision lowp float;
#endif

#define lut_tex iChannel0
#define floor_tex iChannel1
#define light_tex iChannel2

const float intensity = 1.5;
const float light_width = .7;
const float light_height = 0.5;

const vec3 light_col = vec3(1.)*intensity;
const vec3 light_pos = vec3(0., 0.3, 0.);
const vec3 light_normal = vec3(0., 0., 1.);


const float PI = 3.1415926;
const float LUTSIZE  = 8.0;
const float MATRIX_PARAM_OFFSET = 8.0;

const mat2 R_obj2 = mat2(
        			0.955336489125606,-0.295520206661339,
        			0.295520206661339, 0.955336489125606
                  	);



float rect(vec3 p, vec3 b)
{
  	vec3 d = abs(p) - b;
  	return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

void init_rect_points(out vec3 points[4])
{
    // get the orthogonal basis of polygon light
    vec3 right=normalize(cross(light_normal, vec3(0.0, 1.0, 0.0)));
    vec3 up=normalize(cross(right, light_normal));
    
    vec3 ex = light_width * right;
    vec3 ey = light_height * up;

    points[0] = light_pos - ex - ey;
    points[1] = light_pos + ex - ey;
    points[2] = light_pos + ex + ey;
    points[3] = light_pos - ex + ey;
}


#define LIGHT 0.
#define FLOOR 1.
#define OBJ1  2.
#define OBJ2  3.

float object_id = 0.;


float map(vec3 p)
{
    vec3 p0 = p;
    p0.xz *= R_obj2;
    
    float d0=rect(p-light_pos, vec3(light_width, light_height, 0.));
    
    float d1;
    if (abs(p.y + .5) > .015) d1 = abs(p.y+.5);
    else d1=abs(p.y+0.5+texture(floor_tex, p.xz).x*.01)*.9;
    
   	float d = d0;
    object_id = LIGHT;
    
    if(d > d1)
    {
        d = d1;
        object_id=FLOOR;
    }
    
    
    return d;
}

vec3 get_normal(vec3 p) {
	const vec2 e = vec2(0.002, 0);
	return normalize(vec3(map(p + e.xyy)-map(p - e.xyy), 
                          map(p + e.yxy)-map(p - e.yxy),	
                          map(p + e.yyx)-map(p - e.yyx)));
}

float intersect( in vec3 ro, in vec3 rd )
{
    float t = 0.01;
    for( int i=0; i<32; i++ )
    {
        float c = map(ro + rd*t);
        if( c < 0.005 ) break;
        t += c;
        if( t>50.0 ) return -1.0;
    }
    return t;
}


// Linearly Transformed Cosines 

float IntegrateEdge(vec3 v1, vec3 v2)
{
    float cosTheta = dot(v1, v2);
    float theta = acos(cosTheta);    
    float res = cross(v1, v2).z * ((theta > 0.001) ? theta/sin(theta) : 1.0);

    return res;
}

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


vec3 LTC_Evaluate(vec3 N, vec3 V, vec3 P, mat3 Minv, vec3 points[4])
{
    // construct orthonormal basis around N
    vec3 T1, T2;
    T1 = normalize(V - N*dot(V, N));
    T2 = cross(N, T1);

    // rotate area light in (T1, T2, N) basis
    Minv = Minv * transpose(mat3(T1, T2, N));

    // polygon (allocate 5 vertices for clipping)
    vec3 L[5];
    L[0] = Minv * (points[0] - P);
    L[1] = Minv * (points[1] - P);
    L[2] = Minv * (points[2] - P);
    L[3] = Minv * (points[3] - P);

    int n=0;
    // The integration is assumed on the upper hemisphere
    // so we need to clip the frustum, the clipping will add 
    // at most 1 edge, that's why L is declared 5 elements.
    ClipQuadToHorizon(L, n);
    
    if (n == 0)
        return vec3(0, 0, 0);

    // project onto sphere
    vec3 PL[5];
    PL[0] = normalize(L[0]);
    PL[1] = normalize(L[1]);
    PL[2] = normalize(L[2]);
    PL[3] = normalize(L[3]);
    PL[4] = normalize(L[4]);

    // integrate for every edge.
    float sum = 0.0;

    sum += IntegrateEdge(PL[0], PL[1]);
    sum += IntegrateEdge(PL[1], PL[2]);
    sum += IntegrateEdge(PL[2], PL[3]);
    if (n >= 4)
        sum += IntegrateEdge(PL[3], PL[4]);
    if (n == 5)
        sum += IntegrateEdge(PL[4], PL[0]);

    sum =  max(0.0, sum);
    
    // Calculate colour
    vec3 e1 = normalize(L[0] - L[1]);
    vec3 e2 = normalize(L[2] - L[1]);
    vec3 N2 = cross(e1, e2); // Normal to light
    vec3 V2 = N2 * dot(L[1], N2); // Vector to some point in light rect
    vec2 Tlight_shape = vec2(length(L[0] - L[1]), length(L[2] - L[1]));
    V2 = V2 - L[1];
    float b = e1.y*e2.x - e1.x*e2.y + .1; // + .1 to remove artifacts
	vec2 pLight = vec2((V2.y*e2.x - V2.x*e2.y)/b, (V2.x*e1.y - V2.y*e1.x)/b);
   	pLight /= Tlight_shape;
    //pLight -= .5;
    //pLight /= 2.5;
    //pLight += .5;
    
    vec3 ref_col = texture(light_tex, pLight).xyz;

    vec3 Lo_i = vec3(sum) * ref_col;

    return Lo_i;
}

    
/////////////////////////////////////////////


void  LTC_shading_Diff(float roughness, 
                 vec3 N, 
                 vec3 V, 
                 mat3 Minv,
                 vec3 pos, 
                 vec3[4] points, 
                 vec3 m_diff,
                 inout vec3 col
                 )
{
    

    vec3 diff = LTC_Evaluate(N, V, pos, Minv, points)*m_diff; 

    col  = light_col*m_diff*diff;
    col /= 2.0*PI;
}

void LTC_shading_Spec(float roughness, 
                 vec3 N, 
                 vec3 V, 
                 mat3 Minv,
                 vec3 pos, 
                 vec3[4] points, 
                 vec3 m_spec,
                 inout vec3 col
                 )
{
    
    float theta = acos(dot(N, V));
    
    vec2 uv = vec2(roughness, theta/(0.5*PI)) * float(LUTSIZE-1.);
    uv += vec2(0.5 );
    
    vec3 spec = LTC_Evaluate(N, V, pos, Minv, points)*m_spec;

    spec *= texture(lut_tex, uv/iChannelResolution[0].xy).x;


    col  = light_col*m_spec*spec;
    col /= 2.0*PI;
}



void LTC_shading(float roughness, 
                 vec3 N, 
                 vec3 V, 
                 mat3 Minv,
                 vec3 pos, 
                 vec3[4] points, 
                 vec3 m_spec,
                 vec3 m_diff,
                 vec2 fragCoord,
                 inout vec3 col
                 )
{


    vec3 spec = LTC_Evaluate(N, V, pos, Minv, points)*m_spec;

    //spec *= texture(lut_tex, uv/iChannelResolution[0].xy).x;
    
    vec3 diff = LTC_Evaluate(N, V, pos, mat3(1), points)*m_diff; 

    col  = light_col*(m_spec*spec+m_diff*diff);
    col /= 2.0*PI;
    
}


mat3 caculatedMInv(float roughness,vec3 N,vec3 V){

    float theta = acos(dot(N, V));
    
    vec2 uv = vec2(roughness, theta/(0.5*PI)) * float(LUTSIZE-1.);
    uv += vec2(0.5 );
    
    vec4 params = texture(lut_tex, (uv+vec2(MATRIX_PARAM_OFFSET, 0.0))/iChannelResolution[0].xy);
    
    mat3 Minv = mat3(
        vec3(  1,        0,      params.y),
        vec3(  0,     params.z,   0),
        vec3(params.w,   0,      params.x)
    );
    
    return Minv;
}

`

export const ltc_lighting=`

// ############################## LTC TextureAreaLight ##############################

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

// *** Linearly Transformed Cosines *** A.K.A IntergrateEdge

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

// ############################## LTC TextureAreaLight ##############################

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


vec3 LTC_Evaluate_Without_Texture(vec3 N, vec3 V, vec3 P, mat3 Minv, vec3 points[4]) {
    
    // construct orthonormal basis around N
    vec3 T1, T2;
    T1 = normalize(V - N*dot(V, N));
    T2 = cross(N, T1);

    // rotate area light in (T1, T2, N) basis
    mat3 mInv = Minv * (transpose(mat3(T1, T2, N)));

    // polygon (allocate 5 vertices for clipping)
    vec3 L[5];
    L[0] = mInv * (points[0] - P);
    L[1] = mInv * (points[1] - P);
    L[2] = mInv * (points[2] - P);
    L[3] = mInv * (points[3] - P);

    int n;
    ClipQuadToHorizon(L, n);
    
    if (n == 0)
        return vec3(0, 0, 0);

    // project onto sphere
    L[0] = normalize(L[0]);
    L[1] = normalize(L[1]);
    L[2] = normalize(L[2]);
    L[3] = normalize(L[3]);
    L[4] = normalize(L[4]);

    // ### METHOD I

    // calculate vector form factor
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += IntegrateEdgeVec( L[ 0 ], L[ 1 ] );
	vectorFormFactor += IntegrateEdgeVec( L[ 1 ], L[ 2 ] );
	vectorFormFactor += IntegrateEdgeVec( L[ 2 ], L[ 3 ] );
	vectorFormFactor += IntegrateEdgeVec( L[ 3 ], L[ 0 ] );

	// adjust for horizon clipping
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );

    return vec3( result );

    // // ### METHOD II 
    // // integrate
    // float sum = 0.0;

    // sum += IntegrateEdge(L[0], L[1]);
    // sum += IntegrateEdge(L[1], L[2]);
    // sum += IntegrateEdge(L[2], L[3]);
    // if (n >= 4)
    //     sum += IntegrateEdge(L[3], L[4]);
    // if (n == 5)
    //     sum += IntegrateEdge(L[4], L[0]);

    // //sum = twoSided ? abs(sum) : max(0.0, sum);
	// sum = max(0.0, sum);

    // vec3 Lo_i = vec3(sum, sum, sum);
	
	// return Lo_i;
}


float maskBox(vec2 _st, vec2 _size, float _smoothEdges){
    _size = vec2(0.5)-_size*0.5;
    vec2 aa = vec2(_smoothEdges*0.5);
    vec2 uv = smoothstep(_size,_size+aa,_st);
    uv *= smoothstep(_size,_size+aa,vec2(1.0)-_st);
    return uv.x*uv.y;
}

vec3 draw(vec2 uv,in sampler2D tex) {
    return texture(tex,vec2(1.- uv.x,uv.y)).rgb;   
}

float grid(float var, float size) {
    return floor(var*size)/size;
}

float blurRand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

vec3 blurredImage( in float roughness,in vec2 uv , in sampler2D tex)
{
    
    float bluramount = 0.2 * roughness;
    //float dists = 5.;
    vec3 blurred_image = vec3(0.);
    #define repeats 60.
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

// https://advances.realtimerendering.com/s2016/s2016_ltc_rnd.pdf p-104  -> filtered border region
// https://www.shadertoy.com/view/dd2SDd
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


vec3 LTC_Evaluate_With_Texture( in bool isDiffuse,in float roughness,const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 Minv, const in vec3 points[ 4 ], in sampler2D tex) {
	
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


    // ### Method I

    // calculate vector form factor
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += IntegrateEdgeVec( PL[ 0 ], PL[ 1 ] );
	vectorFormFactor += IntegrateEdgeVec( PL[ 1 ], PL[ 2 ] );
	vectorFormFactor += IntegrateEdgeVec( PL[ 2 ], PL[ 3 ] );
	vectorFormFactor += IntegrateEdgeVec( PL[ 3 ], PL[ 0 ] );

	// adjust for horizon clipping
	float sum = LTC_ClippedSphereFormFactor( vectorFormFactor );

    // // ### Method II
	// // integrate for every edge.
	// float sum = 0.0;
	
	// sum += IntegrateEdge(PL[0], PL[1]);
    // sum += IntegrateEdge(PL[1], PL[2]);
    // sum += IntegrateEdge(PL[2], PL[3]);
    // if (n >= 4)
    //     sum += IntegrateEdge(PL[3], PL[4]);
    // if (n == 5)
	// 	sum += IntegrateEdge(PL[4], PL[0]);
		
	// sum =  max(0.0, sum);
    
    // Calculate colour
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
	//ref_col = filterBorderRegion(roughness,vec2(saturate(pLight.x),saturate(pLight.y)),tex);

	ref_col = FetchDiffuseFilteredTexture(roughness,L,vec3(sum),tex);

	vec3 Lo_i = vec3(sum)*ref_col;
	
    return Lo_i;

}

// ############################## LTC TextureAreaLight ##############################
`

export const ltc_implementation=`
uniform sampler2D ltc_1; // RGBA Float
uniform sampler2D ltc_2; // RGBA Float

struct TextureAreaLight {
    vec3 areaLightColor;
    vec3 areaLightPosition;
    vec3 halfWidth;
    vec3 halfHeight;
    vec3 areaLightNormal;
    vec3 areaLightRight;
    vec3 areaLightUp;
    vec2 areaLightSize;
    float areaLightIntensity;
    float areaLightRoughness;
    bool areaLightTextureIsNull;
    bool areaLightRouhnessControllable;
    vec3 areaLightAttenuation;
};

// Pre-computed values of LinearTransformedCosine approximation of BRDF
// BRDF approximation Texture is 64x64
uniform sampler2D areaLightTextures;
uniform TextureAreaLight textureAreaLights;

void TextureAreaLightCaculation( const in TextureAreaLight textureAreaLight, 
    const in GeometricContext geometry, 
    const in PhysicalMaterial material, 
    inout ReflectedLight reflectedLight, 
    sampler2D areaLightTexture ){

        // *** geom properties ***
        vec3 normal = geometry.normal;
		vec3 viewDir = geometry.viewDir;
		vec3 position = geometry.position;

        // *** area light properties ***
		vec3 halfWidth = textureAreaLight.halfWidth;
		vec3 halfHeight = textureAreaLight.halfHeight;
		vec3 lightColor = textureAreaLight.areaLightColor;
		vec3 lightPosition = textureAreaLight.areaLightPosition;
		vec3 lightNormal = textureAreaLight.areaLightNormal;
		vec3 lightRight = textureAreaLight.areaLightRight;
		vec3 lightUp = textureAreaLight.areaLightUp;
		vec2 lightSize = textureAreaLight.areaLightSize;
		vec3 lightAttenuation = textureAreaLight.areaLightAttenuation;
		float lightIntensity = textureAreaLight.areaLightIntensity;
		float lightRoughness = textureAreaLight.areaLightRoughness;
		bool lightTextureIsNull = textureAreaLight.areaLightTextureIsNull;
		bool lightRouhnessControllable =  textureAreaLight.areaLightRouhnessControllable;

        // *** light coords ***
        vec3 rectCoords[ 4 ];

		rectCoords[ 0 ] = lightPosition + halfWidth - halfHeight; // counterclockwise; light shines in local neg z direction
		rectCoords[ 1 ] = lightPosition - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPosition - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPosition + halfWidth + halfHeight;

        // *** material roughness ***
        float roughnessMIN = 0.35;
		float M_Roughness = material.roughness; //material.roughness //lightRoughness
		//M_Roughness *= M_Roughness;
		M_Roughness += roughnessMIN;

		//float L_Roughness = lightRoughness; //material.roughness //
		//L_Roughness *= L_Roughness;
		//L_Roughness += roughnessMIN;

		// comment this for material lightness;
		if(lightRouhnessControllable){
			M_Roughness = lightRoughness;
			M_Roughness += roughnessMIN;
		}

        // *** LTC ***

        // LTC Fresnel Approximation by Stephen Hill
		// http://blog.selfshadow.com/publications/s2016-advances/s2016_ltc_fresnel.pdf
		if(lightTextureIsNull){

			vec2 uv = LTC_Uv( normal, viewDir, M_Roughness ); //mRoughness

			vec4 t1 = texture2D( ltc_1, uv );
			vec4 t2 = texture2D( ltc_2, uv );

			mat3 mInv = mat3(
				vec3( t1.x, 0, t1.y ),
				vec3(    0, 1,    0 ),
				vec3( t1.z, 0, t1.w )
			);

			vec3 fresnel = ( material.specularColor * t2.x + ( vec3( 1.0 ) - material.specularColor ) * t2.y );
			reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate_Without_Texture( normal, viewDir, position, mInv, rectCoords );
			reflectedLight.directDiffuse += lightColor * material.diffuseColor * LTC_Evaluate_Without_Texture( normal, viewDir, position, mat3( 1.0 ), rectCoords );
		}
		else{

			vec2 uv = LTC_Uv( normal, viewDir, M_Roughness ); //mRoughness

			vec4 t1 = texture2D( ltc_1, uv );
			vec4 t2 = texture2D( ltc_2, uv );

			mat3 mInv = mat3(
				vec3( t1.x, 0, t1.y ),
				vec3(    0, 1,    0 ),
				vec3( t1.z, 0, t1.w )
			);
			
			vec3 fresnel = ( material.specularColor * t2.x + ( vec3( 1.0 ) - material.specularColor ) * t2.y );
			float diffuseRoughness = 2.;
			reflectedLight.directSpecular +=  lightColor * fresnel * LTC_Evaluate_With_Texture( false,M_Roughness,normal, viewDir, position, mInv, rectCoords,areaLightTexture);
			reflectedLight.directDiffuse += lightColor *  material.diffuseColor * LTC_Evaluate_With_Texture( false,2.,normal, viewDir, position, mat3(1.0), rectCoords,areaLightTexture);

		}
        
    
    }

`