import { Box, OrbitControls, Plane, SpotLight, shaderMaterial, useDepthBuffer, useGLTF } from "@react-three/drei"
import { Canvas, useThree, useFrame, createPortal, useLoader } from "@react-three/fiber"
import { useControls } from "leva";
import {  useEffect, useMemo, useRef, useState } from "react";
import * as THREE from 'three'
import { Perf } from "r3f-perf";
import { RectAreaLightUniformsLib } from "three/examples/jsm/lights/RectAreaLightUniformsLib.js";
import * as React from "react";
import { RECT_AREALIGHT_PREFIX,LTC_AREALIGHT_CORE,RECT_AREALIGHT_SUFFIX_0,RECT_AREALIGHT_HACK,RECT_AREALIGHT_SUFFIX_1} from '../LTCAreaLight/shader/LTC_Shader';
import { DOWNSAMPLE_BLUR, UPSAMPLE_BLUR } from "../LTCAreaLight/shader/DualKawaseBlur_Shader";
import { common_vertex_main, prefix_frag, prefix_vertex } from "../LTCAreaLight/shader/Utils";

// TODO：Implement the Shader Code in this Component
const LTCAreaLightContainer = ({ children }:{
    children?: React.ReactNode;

}) => {

    // *** Init LTC Texture ***

    const initLTCTexture = () =>{
        RectAreaLightUniformsLib.init();
    }
    
    useEffect(()=>{
        initLTCTexture();

    },[])


    return(
        <>
            {children}
        </>
    )
  
};

LTCAreaLightContainer.displayName = 'LTCAreaLightContainer'

// TODO：Implement the LTC Light Only In This Component

const LTCAreaLight = React.forwardRef(({ position,rotation, color,intensity,width,height,isEnableHelper }:{
    position?: [number, number, number];
    rotation?: [number, number, number];
    width?: number;
    height?: number;
    texture?: THREE.Texture | string;
    isEnableHelper?:boolean;
    color?: string;
    intensity?: number;

},
ref: React.ForwardedRef<any>
) => {
    const {gl,camera} = useThree();

    const rectAreaLightRef = useRef<any>();
    const rectAreLightHelperRef = useRef<any>();
    
    const blurBufferSize = 128.;

    const videoUrl = './test.mp4';
    const imageUrl = './test.png';

    const isVideoTexture = true;

    const image_Tex = useLoader(THREE.TextureLoader,imageUrl);
    const [copyVideo,setCopyVideo] = useState<boolean>(false);
    const videoRef = useRef<any>(null);

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
            new THREE.WebGLRenderTarget(blurBufferSize,blurBufferSize,FBOSettings),
            new THREE.WebGLRenderTarget(blurBufferSize,blurBufferSize,FBOSettings),
            new THREE.WebGLRenderTarget(blurBufferSize,blurBufferSize,FBOSettings),
            new THREE.WebGLRenderTarget(blurBufferSize,blurBufferSize,FBOSettings)
        ]
    },[])


    // *** Load Video Texture 
    // *** from 'Animating textures in WebGL'
    // *** https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/Tutorial/Animating_textures_in_WebGL
    const setupVideo = (src:string) =>{
        videoRef.current = document.createElement('video');
        videoRef.current.src = 'src';
        videoRef.current.crossOrigin = 'Anonymous'
        videoRef.current.loop = true
        videoRef.current.muted = true
        videoRef.current.playsInline = true;
        var playing =false;
        var timeupdate = false;
        videoRef.current.addEventListener(
            "playing",
            () => {
              playing = true;
              checkUpdate();
            },
            true
        );
        
        videoRef.current.addEventListener(
            "timeupdate",
            () => {
                timeupdate = true;
                checkUpdate();
            },
            true
        );        
    
        videoRef.current.src = src;
        videoRef.current.play();


        function checkUpdate() {
            if (playing && timeupdate) {
                // * tik tok tik tok *
                setCopyVideo(true)
              }
            
        }
       
    }

    
    // *** Dual Kaswase Blur Pass ***
    const DualKawaseBlurPass = (tex:THREE.Texture):THREE.Texture =>{
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

        return blurFBOD.texture;
    }

  
    //return null;
    return(
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
                                resolution:{value:[blurBufferSize,blurBufferSize]}
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
                                resolution:{value:[blurBufferSize,blurBufferSize]}
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
                                resolution:{value:[blurBufferSize,blurBufferSize]}
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
                                resolution:{value:[blurBufferSize,blurBufferSize]}
                            }}
                        ></shaderMaterial>
                    </Plane>
                </>
                ,DKUpSceneB)
            }
            {/* The Hacked Rect AreaLight -> LTC Area Light */}
            <rectAreaLight
                ref={rectAreaLightRef}
                rotation={rotation?rotation:[0,0,0]}
                position={position?position:[0,0,0]}
                width={width?width:4}
                height={height?height:4}
                color={color?color:'white'}
                intensity={intensity?intensity:15}
            />

            {/* LTC Area Light Helper -> Screen */}
            {isEnableHelper && <Plane args={[width?width:4,height?height:4]} position={position?position:[0,0,0]}>
                <meshBasicMaterial ref={rectAreLightHelperRef} color={color?color:'white'} />
            </Plane>}
            
        </>
    )
  
});

LTCAreaLight.displayName = 'LTCAreaLight'

