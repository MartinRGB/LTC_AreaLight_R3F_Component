# LTC_AreaLight_R3F_Component

[live preview](https://martinrgb.github.io/LTC_AreaLight_R3F_Component/)


There are three ways to implement LTC Arealight in Three.JS

1. modify the Three.JS engine and compile it -> [TextureAreaLight_ThreeJS](https://github.com/MartinRGB/TextureAreaLight_ThreeJS)
2. Hack the RectAreaLight related shader for all 3D Objects in onBeforeCompile -> this repo
3. Write your own shaderMaterial and consider all cases  -> [WebGL implemetation](https://martinrgb.github.io/ltc_code_videoTexture/)

This Repo uses the second method.


## How to use

1. import `LTCAreaLight` & `LTCAreaLightProxy`

```tsx
import { LTCAreaLight,LTCAreaLightProxy } from "@/LTCAreaLight/LTCAreaLight";
```

2.Put `LTCAreaLight` & `3D Objects` into `LTCAreaLightProxy`

```tsx
{/* LTCAreaLightProxy contains LTCAreaLight Objects & 3D Objects */}
<LTCAreaLightProxy>
    {/* LTCAreaLight Objects */}
    <LTCAreaLight
        ref={ltc3Ref}  // ref
        isEnableHelper={true} // is Enable Visual Helper
        position={position} 
        rotation={rotation} 
        color={color} 
        width={width}
        height={height}
        intensity={intensity} // lightIntensity
        texture={null} // Image Texture | Video Texture | Null(Only color works)
        blurSize={64} // 64' means 64 x 64 FBO used for KawaseBlur
        doubleSide={true} // is enable 'DoubleSide' or not
        clipless={false} // is enable 'Clipless Approximation' or not
    ></LTCAreaLight>
    {/* 3D Objects Below */}
</LTCAreaLightProxy>
```

3.Props that can be modified at runtime

- isEnableHelper
- position
- rotation
- color
- width
- height
- intensity

## Core ideas

### What `LTCAreaLight` did

 1. According to the properties of the 'LTCAreaLight' object，add the following properties to the Ref object:
      - isDoubleSide
      - isClipless
      - rectAreaLightTexture (To improve performance, the texture is blurred in advance using DualKawaseBlur,and the FBO size can be set)
 2. In 'UseFrame',Continuous Blur Material with DualKawaseBlur
 3. Create a Plane Mesh to display the image or video texture as LightHelper.

### What `LTCAreaLightProxy` did

 1. Init the LTC Texture  -> RectAreaLightUniformsLib.init() -> which will generate ltc1/ltc2 Texture for LTC_Evaluate
 2. Traverse all objects in the scene, and find the 'LTCAreaLight' object, and add the following properties to the Ref object(Array):
     - isDoubleSide
     - isClipless
     - rectAreaLightTexture 
 3. Traverse all objects in the scene, and find the 'Mesh' object,
     - add these properties(in step 2) as uniforms into them:
     - modify shaders in onBeforeCompile

### What has been changed in shaderChunk
 The three main shaders in Three.js have been modified"
 
    - `lights_pars_begin`
    - `lights_fragment_begin`
    - `lights_physical_pars_fragment`

 1. In `lights_pars_begin`, add the following uniforms:
    ```glsl
      uniform bool enableRectAreaLightTextures[ NUM_RECT_AREA_LIGHTS ];
      uniform bool isCliplesses[ NUM_RECT_AREA_LIGHTS ];
      uniform bool isDoubleSides[ NUM_RECT_AREA_LIGHTS ];
    ```

 2. In `lights_fragment_begin`, add these uniforms into the 'rectAreaLight' function:
    
    ```glsl
      RE_Direct_RectArea( rectAreaLight, geometry, material, rectAreaLightTextures[ i ],enableRectAreaLightTextures[i],isDoubleSides[i],isCliplesses[i],reflectedLight );
    ```

 3. In `lights_physical_pars_fragment`,this part mainly based on SelfShadow's [ltc_code](https://github.com/selfshadow/ltc_code):
    
     - Modified the calculation of roughness in ‘RE_Direct_RectArea’ so that it is influenced by the map color
     - Modified ‘LTC_Evaluate's algorithm to include texture sampling 
     - 'FilteredBorderRegion' is used to blur & clamp the edges of the texture when it is scaled according to roughness in 'LTC_Evaluate'

## TODO

PCSS Shadow
