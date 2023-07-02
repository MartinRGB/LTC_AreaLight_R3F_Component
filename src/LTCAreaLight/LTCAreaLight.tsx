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
    const texIsDoubleSide = useRef<boolean[]>([]);
    const [texIsPrepared,SetTexIsPrepared] = useState<boolean>(false);


    useEffect(()=>{
        initLTCTexture();
    },[])

    useFrame(() => {

        
            
            if(childrenRef.current){
                texArrRef.current = [];
                texEnableArrRef.current=[];
                texIsDoubleSide.current=[];
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
                        texIsDoubleSide.current.push(obj.isDoubleSide);
                    }

                })

                SetTexIsPrepared(true);

            }

            if(childrenRef.current && texIsPrepared ){

                childrenRef.current.traverse((obj:any)=>{
                    if(obj.isMesh){
                        obj.material.onBeforeCompile = (shader:any) => {
                            console.log(texArrRef.current)
                            shader.uniforms.enableRectAreaLightTextures = { value: texEnableArrRef.current };
                            shader.uniforms.rectAreaLightTextures = { value:texArrRef.current};
                            shader.uniforms.isDoubleSides = { value:texIsDoubleSide.current};

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
        }
    },[rectAreaLightRef])


    useEffect(()=>{
            //TODO: Img in UseEffect,Vid in UseFrame
            if(TextureType === 'Texture' && texture){
                DualKawaseBlurPass(texture)
            }

    },[texture])


    useFrame(() => {
        
            if((TextureType === 'VideoTexture') && texture){
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
            <rectAreaLight
                ref={mergeRefs([ref,rectAreaLightRef])}
                rotation={rotation?[rotation[0],rotation[1],rotation[2]]:[0,0,0]}
                position={position?position:[0,0,0]}
                width={width?width:4}
                height={height?height:4}
                color={color?color:'white'}
                intensity={intensity?intensity:15}
            />

            {/* LTC Area Light Helper -> Screen */}
            {isEnableHelper && <Plane 
                args={[width?width:4,height?height:4]} 
                rotation={rotation?[rotation[0],rotation[1],rotation[2]]:[0,0,0]}
                position={position?position:[0,0,0]}
                
            >
       
                <meshBasicMaterial 
                    ref={rectAreLightHelperRef} 
                    side={doubleSide?THREE.DoubleSide:THREE.FrontSide}
                    color={color?color:'white'} 
                    map={texture} />
            </Plane>}

        </>
    )
})

LTCAreaLight.displayName = 'LTCAreaLight'