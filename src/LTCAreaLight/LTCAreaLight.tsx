import { Plane, useDepthBuffer } from "@react-three/drei"
import { useThree, useFrame, createPortal } from "@react-three/fiber"
import {  useEffect, useMemo, useRef, useState } from "react";
import * as THREE from 'three'
import { RectAreaLightUniformsLib } from "three/examples/jsm/lights/RectAreaLightUniformsLib.js";
import * as React from "react";
import { RECT_AREALIGHT_PREFIX,LTC_AREALIGHT_CORE,RECT_AREALIGHT_SUFFIX_0,RECT_AREALIGHT_HACK,RECT_AREALIGHT_SUFFIX_1,HACKED_LIGHTS_PARS_BEGIN,HACKED_LIGHTS_FRAGMENT_BEGIN} from './shader/LTC_Shader';
import { DOWNSAMPLE_BLUR, UPSAMPLE_BLUR } from "./shader/DualKawaseBlur_Shader";
import { common_vertex_main, prefix_frag, prefix_vertex } from "./shader/Utils";
import mergeRefs from 'react-merge-refs';


// *** The Process of LTCAreaLight ***

// *** What LTCAreaLight did ***
// 1. According to the properties of the 'LTCAreaLight' object，add the following properties to the Ref object:
//      - isDoubleSide
//      - isClipless
//      - rectAreaLightTexture (To improve performance, the texture is blurred in advance using DualKawaseBlur,and the FBO size can be set)
// 2. In 'UseFrame',Continuous Blur Material with DualKawaseBlur
// 3. Create a Plane Mesh to display the image or video texture as LightHelper.

// *** What LTCAreaLightProxy did ***
// 1. Init the LTC Texture  -> RectAreaLightUniformsLib.init() -> which will generate ltc1/ltc2 Texture for LTC_Evaluate
// 2. Traverse all objects in the scene, and find the 'LTCAreaLight' object, and add the following properties to the Ref object(Array):
//     - isDoubleSide
//     - isClipless
//     - rectAreaLightTexture 
// 3. Traverse all objects in the scene, and find the 'Mesh' object,
//     - add these properties(in step 2) as uniforms into them:
//     - modify shaders in onBeforeCompile

// *** What the shaders have changed ***
// The three main shaders in Three.js have been modified
//     -lights_pars_begin
//     -lights_fragment_begin
//     -lights_physical_pars_fragment

// 1. In 'lights_pars_begin', add the following uniforms:
//      uniform bool enableRectAreaLightTextures[ NUM_RECT_AREA_LIGHTS ];
//      uniform bool isCliplesses[ NUM_RECT_AREA_LIGHTS ];
//      uniform bool isDoubleSides[ NUM_RECT_AREA_LIGHTS ];

// 2. In 'lights_fragment_begin', add these uniforms into the 'rectAreaLight' function:
//      RE_Direct_RectArea( rectAreaLight, geometry, material, rectAreaLightTextures[ i ],enableRectAreaLightTextures[i],isDoubleSides[i],isCliplesses[i],reflectedLight );

// 3. In 'lights_physical_pars_fragment',this part mainly based on SelfShadow's 'ltc_code'(https://github.com/selfshadow/ltc_code):
//     - Modified the calculation of roughness in ‘RE_Direct_RectArea’ so that it is influenced by the map color
//     - Modified ‘LTC_Evaluate's algorithm to include texture sampling 
//     - 'FilteredBorderRegion' is used to blur & clamp the edges of the texture when it is scaled according to roughness in 'LTC_Evaluate'

export const LTCAreaLightProxy = React.forwardRef(({ 
    children,
}:{
    children?: React.ReactNode;

},
ref: React.ForwardedRef<any>
) => {
    
    const depthBuffer = useDepthBuffer({ frames: 1 })
    
    const childrenRef = useRef<THREE.Object3D>();

    // *** Init The LTC Texture ***

    const initLTCTexture = () =>{
        RectAreaLightUniformsLib.init();
    }

    const texArrRef = useRef<(THREE.Texture | null)[]>([]);
    const texEnableArrRef = useRef<boolean[]>([]);
    const texIsDoubleSideArrRef = useRef<boolean[]>([]);
    const texIsCliplessArrRef = useRef<boolean[]>([]);
    const [texIsPrepared,SetTexIsPrepared] = useState<boolean>(false);


    useEffect(()=>{
        initLTCTexture();
    },[])

    useFrame(() => {

        
            
            if(childrenRef.current){
                texArrRef.current = [];
                texEnableArrRef.current=[];
                texIsDoubleSideArrRef.current=[];
                texIsCliplessArrRef.current=[];
                SetTexIsPrepared(false);                
                childrenRef.current.traverse((obj:any)=>{
                    if(obj.isRectAreaLight){
                        if(obj.rectAreaLightTexture){
                            texEnableArrRef.current.push(true);
                            texArrRef.current.push(obj.rectAreaLightTexture);
                        }
                        else{
                            texEnableArrRef.current.push(false);
                            texArrRef.current.push(null);
                        }
                        texIsDoubleSideArrRef.current.push(obj.isDoubleSide);
                        texIsCliplessArrRef.current.push(obj.isClipless);
                    }

                })

                SetTexIsPrepared(true);

            }

            if(childrenRef.current && texIsPrepared ){

                childrenRef.current.traverse((obj:any)=>{
                    if(obj.isMesh){
                        obj.material.onBeforeCompile = (shader:any) => {
                            shader.uniforms.enableRectAreaLightTextures = { value: texEnableArrRef.current };
                            shader.uniforms.rectAreaLightTextures = { value:texArrRef.current};
                            shader.uniforms.isDoubleSides = { value:texIsDoubleSideArrRef.current};
                            shader.uniforms.isCliplesses = { value:texIsCliplessArrRef.current};
                            

                            shader.fragmentShader = shader.fragmentShader.replace(`#include <lights_pars_begin>`,
                            HACKED_LIGHTS_PARS_BEGIN
                            )

                            shader.fragmentShader = shader.fragmentShader.replace(`#include <lights_fragment_begin>`,
                            HACKED_LIGHTS_FRAGMENT_BEGIN
                            )
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
    },)


    return (
        <>
            {/* All objects in scene */}
            <group ref={mergeRefs([childrenRef,ref])}>
                {children}
            </group>
        </>
    )
})

LTCAreaLightProxy.displayName='LTCAreaLightProxy'

export const LTCAreaLight = React.forwardRef(({ 
    position,
    rotation,
    texture,
    isEnableHelper,
    width,
    height,
    color,
    intensity,
    blurSize,
    doubleSide,
    clipless,
}:{
    position?: [number, number, number];
    rotation?: [number, number, number];
    texture?: THREE.Texture | null;
    isEnableHelper?:boolean;
    color?: string;
    intensity?: number;
    width?: number;
    height?: number;
    blurSize?:number;
    doubleSide?:boolean;
    clipless?:boolean;
},
ref: React.ForwardedRef<any>
) => {

    const rectAreaLightRef = useRef<any>();
    const rectAreLightHelperRef = useRef<any>();
    
    const TextureType = texture?(texture.constructor.name === 'VideoTexture')?'VideoTexture':'Texture':'Null';

    // # Material Ref
    const kawaseBlurMaterialRefA = useRef<THREE.ShaderMaterial | null>(null)
    const kawaseBlurMaterialRefB = useRef<THREE.ShaderMaterial | null>(null)
    const kawaseBlurMaterialRefC = useRef<THREE.ShaderMaterial | null>(null)
    const kawaseBlurMaterialRefD = useRef<THREE.ShaderMaterial | null>(null)
    
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
            new THREE.WebGLRenderTarget(blurSize?blurSize:64,blurSize?blurSize:64,FBOSettings),
            new THREE.WebGLRenderTarget(blurSize?blurSize:64,blurSize?blurSize:64,FBOSettings),
            new THREE.WebGLRenderTarget(blurSize?blurSize:64,blurSize?blurSize:64,FBOSettings),
            new THREE.WebGLRenderTarget(blurSize?blurSize:64,blurSize?blurSize:64,FBOSettings)
        ]
    },[])

    const { gl,camera } = useThree();

    // *** Dual Kaswase Blur Pass ***
    const DualKawaseBlurPass = (tex:THREE.Texture) =>{
        if(kawaseBlurMaterialRefA.current){
            kawaseBlurMaterialRefA.current.uniforms.buff_tex.value = tex
            gl.setRenderTarget(blurFBOA);
            gl.render(DKDownSceneA,camera)
            gl.setRenderTarget(null)
                        
        }

        if(kawaseBlurMaterialRefB.current){
            kawaseBlurMaterialRefB.current.uniforms.buff_tex.value = blurFBOA.texture
            gl.setRenderTarget(blurFBOB);
            gl.render(DKDownSceneB,camera)
            gl.setRenderTarget(null)
        }

        if(kawaseBlurMaterialRefC.current){
            kawaseBlurMaterialRefC.current.uniforms.buff_tex.value = blurFBOB.texture
            gl.setRenderTarget(blurFBOC);
            gl.render(DKUpSceneA,camera)
            gl.setRenderTarget(null)
        }

        if(kawaseBlurMaterialRefD.current){
            kawaseBlurMaterialRefD.current.uniforms.buff_tex.value = blurFBOC.texture
            gl.setRenderTarget(blurFBOD);
            gl.render(DKUpSceneB,camera)
            gl.setRenderTarget(null)
        }

        if( rectAreaLightRef.current){
            rectAreaLightRef.current.rectAreaLightTexture = blurFBOD.texture;
        }
    }


    useEffect(()=>{
        if(rectAreaLightRef.current){
            rectAreaLightRef.current.isDoubleSide = doubleSide?doubleSide:false;
            rectAreaLightRef.current.isClipless = clipless?clipless:true;
        }
    },[rectAreaLightRef])

    //TODO: Img in UseEffect,Vid in UseFrame
    useFrame(() => {
        
            if((TextureType === 'VideoTexture' || TextureType === 'Texture') && texture){
                DualKawaseBlurPass(texture)
            }
    
    },)

    return (
        <>
         {/* The DualKawaseBlur Render Pass */}
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
                                blurOffset:{value:0.},
                                resolution:{value:[blurSize?blurSize:64,blurSize?blurSize:64]}
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
                                blurOffset:{value:0.},
                                resolution:{value:[blurSize?blurSize:64,blurSize?blurSize:64]}
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
                                blurOffset:{value:0.},
                                resolution:{value:[blurSize?blurSize:64,blurSize?blurSize:64]}
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
                                blurOffset:{value:0.},
                                resolution:{value:[blurSize?blurSize:64,blurSize?blurSize:64]}
                            }}
                        ></shaderMaterial>
                    </Plane>
                </>
                ,DKUpSceneB)
            }
            {/* The Hacking of Rect AreaLight -> LTC Area Light */}
            <group ref={ref}
                rotation={rotation?[rotation[0],rotation[1],rotation[2]]:[0,0,0]}
                position={position?position:[0,0,0]}
            >
            <rectAreaLight
                ref={rectAreaLightRef}
                width={width?width:4}
                height={height?height:4}
                color={color?color:'white'}
                intensity={intensity?intensity:15}
            />

            {/* LTC Area Light Helper -> Screen */}
            {isEnableHelper && <Plane 
                args={[width?width:4,height?height:4]} 
                
            >
       
                <meshBasicMaterial 
                    ref={rectAreLightHelperRef} 
                    side={doubleSide?THREE.DoubleSide:THREE.FrontSide}
                    color={color?color:'white'} 
                    map={texture} />
            </Plane>}
            </group>
        </>
    )
})

LTCAreaLight.displayName = 'LTCAreaLight'