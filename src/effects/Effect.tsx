import { Box, OrbitControls, Plane, shaderMaterial, useDepthBuffer, useGLTF } from "@react-three/drei"
import { Canvas, useThree, useFrame, createPortal, useLoader } from "@react-three/fiber"
import { useControls } from "leva";
import {  useEffect, useMemo, useRef, useState,Suspense } from "react";
import * as THREE from 'three'
import { Perf } from "r3f-perf";
import { RectAreaLightUniformsLib } from "three/examples/jsm/lights/RectAreaLightUniformsLib.js";
import * as React from "react";
import { RECT_AREALIGHT_PREFIX,LTC_AREALIGHT_CORE,RECT_AREALIGHT_SUFFIX_0,RECT_AREALIGHT_HACK,RECT_AREALIGHT_SUFFIX_1,HACKED_LIGHTS_PARS_BEGIN,HACKED_LIGHTS_FRAGMENT_BEGIN} from '../LTCAreaLight/shader/LTC_Shader';
import { DOWNSAMPLE_BLUR, UPSAMPLE_BLUR } from "../LTCAreaLight/shader/DualKawaseBlur_Shader";
import { common_vertex_main, prefix_frag, prefix_vertex } from "../LTCAreaLight/shader/Utils";
import mergeRefs from 'react-merge-refs';

const LTCAreaLightProxy = React.forwardRef(({ 
    children,

}:{
    children?: React.ReactNode;

},
ref: React.ForwardedRef<any>
) => {


    const childrenRef = useRef();

    // *** Init The LTC Texture ***

    const initLTCTexture = () =>{
        RectAreaLightUniformsLib.init();
    }

    const texArrRef = useRef<any>([]);
    const texEnableArrRef = useRef<any>([]);
    const texIsDoubleSide = useRef<any>([]);
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

const LTCAreaLight = React.forwardRef(({ 
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
    index,
    dst,

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
    index?:number;
    dst?:THREE.WebGLRenderTarget;

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

        // TODO:Write the buffer as an output
        // if(ref.current){
        //     ref.current.rectAreaLightTexture = blurFBOD.texture;
        // }
        if( rectAreaLightRef.current){
            rectAreaLightRef.current.rectAreaLightTexture = blurFBOD.texture;
        }
        //return blurFBOD.texture;
    }


    useEffect(()=>{
        if(rectAreaLightRef.current){
            rectAreaLightRef.current.isDoubleSide = doubleSide?doubleSide:false;
        }
    },[rectAreaLightRef])


    useEffect(()=>{
            if(TextureType === 'Texture' && texture){
                //DualKawaseBlurPass(texture)

                // if(isEnableHelper)  
                //     updateHelper(texture);
            }

    },[texture])


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

const LTCTexturedLightDemo = () =>{

    const {size,gl,camera,scene} = useThree()
    const depthBuffer = useDepthBuffer({ frames: 1 })
    const { nodes, materials } = useGLTF('./model.gltf')
    const dragonRef = useRef<any>();


    const videoUrl = './test3.mp4';
    const imageUrl = './test.png';

    const [copyVideo,setCopyVideo] = useState<boolean>(false);
    const [vid_tex,setVidTexture] = useState<THREE.Texture | null>(null);
    const img_tex = useLoader(THREE.TextureLoader,imageUrl);

    // *** Load Video Texture 
    // *** from 'Animating textures in WebGL'
    // *** https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/Tutorial/Animating_textures_in_WebGL
    const setupVideo = (src:string) =>{
        const videoE = document.createElement('video');
        videoE.src = 'src';
        videoE.crossOrigin = 'Anonymous'
        videoE.loop = true
        videoE.muted = true
        videoE.playsInline = true;
        var playing =false;
        var timeupdate = false;
        videoE.addEventListener(
            "playing",
            () => {
              playing = true;
              checkUpdate();
            },
            true
        );
        
        videoE.addEventListener(
            "timeupdate",
            () => {
                timeupdate = true;
                checkUpdate();
            },
            true
        );        
    
        videoE.src = src;
        videoE.play();


        function checkUpdate() {
            if (playing && timeupdate) {
                // * tik tok tik tok *
                setCopyVideo(true)
              }
            
        }

        const vidTex = new THREE.VideoTexture( videoE );
        vidTex.minFilter = THREE.NearestFilter;
        vidTex.magFilter = THREE.LinearFilter;
        vidTex.wrapS = vidTex.wrapT = THREE.ClampToEdgeWrapping;

        setVidTexture(vidTex)

    }

    useEffect(()=>{
        setupVideo(videoUrl)
    },[])

    // *** Object Material Properties
    const {floor_roughness,dragon_roughness} = useControls('Object Material',{
  
        floor_roughness:{
            value:0.5,
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

    // *** Video AreaLight Properties
    var {position0,rotation0,color0,intensity0,width0,height0} = useControls('Video LTC AreaLight',{
        position0:{
            value:[0,3,-5],
            label:'Position',
        },
        rotation0:{
            value:[0,0,0],
            step:0.1,
            label:'Rotation',
        },
        color0:{
            value:'#ffffff',
            label:'Color',
        },

        intensity0:{
            value:15,
            min:0.01,
            max:100.0,
            step:0.01,
            label:'Intensity',
        },
        width0:{
            value:6.4,
            min:0.01,
            max:100.0,
            step:0.01,
            label:'Width',
        },
        height0:{
            value:4,
            min:0.01,
            max:100.0,
            step:0.01,
            label:'Height',
        },
    }) as {
        position0:[number,number,number],
        rotation0:[number,number,number],
        color0:string
        intensity0:number,
        width0:number,
        height0:number,
    }

    // *** Image AreaLight Properties
    var {position1,rotation1,color1,intensity1,width1,height1} = useControls('Image LTC AreaLight',{
        position1:{
            value:[8,3,0],
            label:'Position',
        },
        rotation1:{
            value:[0,-Math.PI/2,0],
            step:0.1,
            label:'Rotation',
        },
        color1:{
            value:'#ffffff',
            label:'Color',
        },

        intensity1:{
            value:15,
            min:0.01,
            max:100.0,
            step:0.01,
            label:'Intensity',
        },
        width1:{
            value:4,
            min:0.01,
            max:100.0,
            step:0.01,
            label:'Width',
        },
        height1:{
            value:4,
            min:0.01,
            max:100.0,
            step:0.01,
            label:'Height',
        },
    }) as {
        position1:[number,number,number],
        rotation1:[number,number,number],
        color1:string
        intensity1:number,
        width1:number,
        height1:number,
    }

    // *** Color AreaLight Properties
    var {position2,rotation2,color2,intensity2,width2,height2} = useControls('Color LTC AreaLight',{

        position2:{
            value:[-8,3,0],
            label:'Position',
        },
        rotation2:{
            value:[0,Math.PI/2,0],
            step:0.1,
            label:'Rotation',
        },
        color2:{
            value:'#ffffff',
            label:'Color',
        },

        intensity2:{
            value:5,
            min:0.01,
            max:100.0,
            step:0.01,
            label:'Intensity',
        },
        width2:{
            value:4,
            min:0.01,
            max:100.0,
            step:0.01,
            label:'Width',
        },
        height2:{
            value:4,
            min:0.01,
            max:100.0,
            step:0.01,
            label:'Height',
        },
    }) as {
        position2:[number,number,number],
        rotation2:[number,number,number],
        color2:string
        intensity2:number,
        width2:number,
        height2:number,
    }

    // *** floor texture
    const floorMap = useLoader(THREE.TextureLoader,'./floor2.jpg');
    floorMap.repeat.set(20,20);
    floorMap.wrapS = floorMap.wrapT = THREE.RepeatWrapping;

    useFrame((state, delta) => {
        const time = state.clock.getElapsedTime();
        if(dragonRef.current){
            dragonRef.current.rotation.y += 0.01;
            dragonRef.current.position.y = 1. + Math.sin(time);
            dragonRef.current.position.x = Math.cos(time);
        }
    },)
    

    return(
        <>
        
        <Suspense fallback={null}>
        <LTCAreaLightProxy>
            <LTCAreaLight
                isEnableHelper={true}
                position={position0} 
                rotation={rotation0} 
                color={color0} 
                width={width0}
                height={height0}
                intensity={intensity0}
                texture={vid_tex}
                blurSize={64}
            ></LTCAreaLight>

            <LTCAreaLight
                isEnableHelper={true}
                position={position1} 
                rotation={rotation1} 
                color={color1} 
                width={width1}
                height={height1}
                intensity={intensity1}
                texture={img_tex}
                blurSize={64}
                doubleSide={true}
            ></LTCAreaLight>

            <LTCAreaLight
                isEnableHelper={true}
                position={position2} 
                rotation={rotation2} 
                color={color2} 
                width={width2}
                height={height2}
                intensity={intensity2}
                texture={null}
                blurSize={64}
                doubleSide={true}
            ></LTCAreaLight>
            <mesh ref={dragonRef} position={[0,0.5,0]} castShadow receiveShadow geometry={nodes.dragon.geometry} material={materials['Default OBJ.001']} dispose={null} />
   
            <Plane args={[100, 100]} rotation={[-Math.PI / 2, 0, 0]}>
                <meshStandardMaterial 
                    color="#ffffff" 
                    roughness={floor_roughness} 
                    map={floorMap}
                />
            </Plane>
        </LTCAreaLightProxy>
        </Suspense>
        </>
    )
    
}


export const Effect = (props:any) =>{

    
    // *** Utils 
    const {auto_rotate} = useControls('Utils',{
        auto_rotate:{
            value:true,
        },
    }) as {
        auto_rotate:boolean,
    }

    return(
      <>
          <Canvas 
            camera={{ position: [0, 15, 35], fov: 50, near: 0.1, far: 1000 }}
            className={props.className} 
            style={{...props.style}}>
            <Perf style={{position:'absolute',top:'10px',left:'10px',width:'360px',borderRadius:'10px'}}/>
            <ambientLight intensity={0.015}></ambientLight>
            <color attach="background" args={['#202020']} />
            <LTCTexturedLightDemo/>
            <OrbitControls autoRotate={auto_rotate}></OrbitControls>
          </Canvas>
      </>
  
    )
}


