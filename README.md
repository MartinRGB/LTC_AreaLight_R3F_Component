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

## TODO

PCSS Shadow
