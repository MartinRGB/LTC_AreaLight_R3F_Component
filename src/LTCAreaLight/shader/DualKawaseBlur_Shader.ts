// ***********************
// this is the main shader code for the LTC Area Light
// the LTC code mainly from selfshadow's 'ltc_code' repo
// the rest code from 'lights_physical_pars_fragment' from three.js
// ***********************

export const DOWNSAMPLE_BLUR=`
uniform sampler2D buff_tex;
uniform float blurOffset;
uniform vec2 resolution;

#define sampleScale (1. + blurOffset*0.1)
#define pixelOffset 1.

void main() {
    vec2 uv = gl_FragCoord.xy/resolution.xy;
    uv *= sampleScale;
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

export const UPSAMPLE_BLUR = `
uniform sampler2D buff_tex;
uniform float blurOffset;
uniform vec2 resolution;

#define sampleScale (1. + blurOffset*0.1)
#define pixelOffset 1.

void main() {
    vec2 uv = gl_FragCoord.xy/resolution.xy;
    uv /= sampleScale;
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