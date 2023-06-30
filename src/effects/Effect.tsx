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


const LTCAreaLightContainer = ({ children }:{
    children?: React.ReactNode;

},
ref: React.ForwardedRef<any>
) => {

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



const LTCAreaLightWithHelper = React.forwardRef(({ children,position,rotation, color,intensity,width,height,isEnableHelper }:{
    children?: React.ReactNode;
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
    const childrenRef = useRef<any>(null!);
    
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

    const HackRectAreaLight = (tex:THREE.Texture,blur_tex:THREE.Texture) =>{
   
        // *** Hacking Children's Material
        if(childrenRef.current){
            childrenRef.current.traverse((obj:any)=>{
                if(obj.isMesh){
                    obj.material.onBeforeCompile = (shader:any) => {
                            shader.uniforms.isLTCWithTexture = { value: true };
                            shader.uniforms.ltc_tex = { value: blur_tex };
                            shader.uniforms.external_roughness = {value:0.}
                            shader.uniforms.light_intensity = {value:1.0}
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

        // *** The Pseudo Helper of RectAreaLight(In fact,it is a Plane mesh) 
        if(rectAreLightHelperRef.current){
            rectAreLightHelperRef.current.onBeforeCompile = (shader:any) => {
                shader.uniforms.vid_tex = {value:tex};
                shader.vertexShader = shader.vertexShader.replace(`#include <common>`,
                `#include <common>
                varying vec2 vUv;`
                )
                shader.vertexShader = shader.vertexShader.replace(`#include <fog_vertex>`,
                `
                #include <fog_vertex>
                vUv = uv;
                `
                )
                shader.fragmentShader = shader.fragmentShader.replace(`uniform float opacity;`,
                `uniform float opacity;
                 uniform sampler2D vid_tex;
                 varying vec2 vUv;
                `
                )
                shader.fragmentShader = shader.fragmentShader.replace(`#include <dithering_fragment>`,
                    `#include <dithering_fragment>
                    gl_FragColor = texture2D(vid_tex,vUv);`
                )

            }
        }
    }

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

    // *** Init The LTC Texture ***

    const initLTCTexture = () =>{
        RectAreaLightUniformsLib.init();
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

    // *** Init LTC Texture ***

    useEffect(()=>{
        if(rectAreaLightRef.current){
            initLTCTexture();
            if(isVideoTexture)
                setupVideo(videoUrl)
            else{
                HackRectAreaLight(image_Tex,DualKawaseBlurPass(image_Tex))
            }
        }

    },[rectAreaLightRef])


    useFrame(() => {

            if(isVideoTexture){
                // *** Update Video Texture ***
                var vidTex = new THREE.VideoTexture( videoRef.current );
                vidTex.minFilter = THREE.NearestFilter;
                vidTex.magFilter = THREE.LinearFilter;
                vidTex.wrapS = vidTex.wrapT = THREE.ClampToEdgeWrapping;
                HackRectAreaLight(vidTex,DualKawaseBlurPass(vidTex))
            }

    },)


  
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
            
            {/* All objects in scene */}
            <group ref={childrenRef}>
                {children}
            </group>
        </>
    )
  
});

LTCAreaLightWithHelper.displayName = 'LTCAreaLightWithHelper'

const LTCTexturedLightDemo = () =>{

    const {size,gl,camera} = useThree()
    const depthBuffer = useDepthBuffer({ frames: 1 })
    const { nodes, materials } = useGLTF('./model.gltf')
    const ltcRef = useRef<any>();
    const dragonRef = useRef<any>();

    const {floor_roughness,dragon_roughness} = useControls('Object Material',{
  
        floor_roughness:{
            value:0.1,
            min:0.0,
            max:10.0,
        },
        dragon_roughness:{
            value:0.5,
            min:0.0,
            max:10.0,
            onChange:(v:any)=>{
            
                if(dragonRef.current){
                    dragonRef.current.traverse((obj:any)=>{
                        if(obj.isMesh){
                            obj.material.roughness = v;
                        }
                    })
                }
            }
        }
    }) as {
        floor_roughness:number,
        dragon_roughness:number
    }


    useFrame(({gl}) => {
        const time =  performance.now() * 0.001;
        if(dragonRef.current){
            //dragonRef.current.position.x = 1. * Math.sin(time) ;
            //dragonRef.current.position.y = 1. + 1. * Math.cos(time);
        }
    })

    const floorMap = useLoader(THREE.TextureLoader,'./floor2.jpg');
    floorMap.repeat.set(20,20);
    floorMap.wrapS = floorMap.wrapT = THREE.RepeatWrapping;
    

    return(
        <>
        <LTCAreaLightWithHelper 
            ref={ltcRef} 
            position={[0, 3, -5]} 
            rotation={[0,0,0]} 
            color="white" 
            isEnableHelper={true}    
        >
            <mesh ref={dragonRef} position={[0,0,0]} castShadow receiveShadow geometry={nodes.dragon.geometry} material={materials['Default OBJ.001']} dispose={null} />
            <mesh receiveShadow position={[0, -1, 0]} rotation-x={-Math.PI / 2}>
                <planeGeometry args={[50, 50]} />
                <meshPhongMaterial />
            </mesh>
            <Plane args={[100, 100]} rotation={[-Math.PI / 2, 0, 0]}>
                <meshStandardMaterial 
                    color="#ffffff" 
                    roughness={floor_roughness} 
                    map={floorMap}
                />
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
            <LTCTexturedLightDemo/>
            <OrbitControls></OrbitControls>
          </Canvas>
      </>
  
    )
}


