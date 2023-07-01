import { Box, OrbitControls, Plane, SpotLight, shaderMaterial, useDepthBuffer, useGLTF } from "@react-three/drei"
import { Canvas, useThree, useFrame, createPortal, useLoader, invalidate } from "@react-three/fiber"
import { useControls } from "leva";
import {  useEffect, useMemo, useRef, useState } from "react";
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
    const [texIsPrepared,SetTexIsPrepared] = useState<boolean>(false);

    useEffect(()=>{
        initLTCTexture();
    },[])

    useFrame(() => {

        
            
            if(childrenRef.current){
                texArrRef.current = [];
                SetTexIsPrepared(false);                
                childrenRef.current.traverse((obj:any)=>{
                    if(obj.isRectAreaLight && obj.rectAreaLightTexture){
                        texArrRef.current.push(obj.rectAreaLightTexture);
                    }

                })

                SetTexIsPrepared(true);

            }

            if(childrenRef.current && texIsPrepared ){

                childrenRef.current.traverse((obj:any)=>{
                    if(obj.isMesh){
                        obj.material.onBeforeCompile = (shader:any) => {

                            if(texArrRef.current.length > 0){
                                shader.uniforms.isLTCWithTexture = { value: true };
                                shader.uniforms.rectAreaLightTextures = { value:texArrRef.current};
                            }
                            else{
                                shader.uniforms.isLTCWithTexture = { value: false };
                                shader.uniforms.rectAreaLightTextures = { value:[null]};
                            }

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
                {/* <meshBasicMaterial ref={rectAreLightHelperRef} color={color?color:'white'} /> */}
       
                <meshBasicMaterial ref={rectAreLightHelperRef} color={color?color:'white'} map={texture} />
            </Plane>}

        </>
    )
})

LTCAreaLight.displayName = 'LTCAreaLight'


const LTCAreaLightContainer = React.forwardRef(({ 
    children,
    position,
    rotation,
    texture,
    isEnableHelper,
    width,
    height,
    color,
    intensity,
    blurSize,
    index

}:{
    children?: React.ReactNode;
    position?: [number, number, number];
    rotation?: [number, number, number];
    texture?: THREE.Texture | null;
    isEnableHelper?:boolean;
    color?: string;
    intensity?: number;
    width?: number;
    height?: number;
    blurSize?:number;
    index:number;

},
ref: React.ForwardedRef<any>
) => {
    const {gl,camera,scene} = useThree();

    const rectAreaLightRef = useRef<any>();
    const rectAreLightHelperRef = useRef<any>();
    const childrenRef = useRef<any>(null!);
    
    const TextureType = texture?(texture.constructor.name === 'VideoTexture')?'VideoTexture':'Texture':'Null';

    // # Material Ref
    const kawaseBlurMaterialRefA = useRef<THREE.ShaderMaterial | null>(null)
    const kawaseBlurMaterialRefB = useRef<THREE.ShaderMaterial | null>(null)
    const kawaseBlurMaterialRefC = useRef<THREE.ShaderMaterial | null>(null)
    const kawaseBlurMaterialRefD = useRef<THREE.ShaderMaterial | null>(null)


    useEffect(()=>{
        console.log(scene)
    },[])
    
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

    const HackRectAreaLight = (tex:THREE.Texture,blur_tex:THREE.Texture) =>{
   
        // *** Hacking Children's Material
        if(childrenRef.current){
            childrenRef.current.traverse((obj:any)=>{
                
                if(obj.isMesh){
                    obj.material.onBeforeCompile = (shader:any) => {
                            shader.uniforms.isLTCWithTexture = { value: true };
                            shader.uniforms.rectAreaLightTextures = { value:[blur_tex] };

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

        rectAreaLightRef.current.rectAreaLightTexture = blurFBOD.texture

        return blurFBOD.texture;
    }


    console.log('TextureType',TextureType)
    // *** Init LTC Texture ***

    useEffect(()=>{
        if(rectAreaLightRef.current){
            initLTCTexture();
            if(TextureType === 'Texture' && texture){
                HackRectAreaLight(texture,DualKawaseBlurPass(texture))
            }
        }

    },[rectAreaLightRef,texture])


    useFrame(() => {
            if(TextureType === 'VideoTexture' && texture){
                //HackRectAreaLight(texture,DualKawaseBlurPass(texture))
                HackRectAreaLight(texture,DualKawaseBlurPass(texture))
            }
    },)

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
            {/* The Hacked Rect AreaLight -> LTC Area Light */}
            <rectAreaLight
                ref={rectAreaLightRef}
                rotation={rotation?[rotation[0],rotation[1] + (TextureType != 'Null'?0.:Math.PI),rotation[2]]:[0,(TextureType != 'Null'?0.:Math.PI),0]}
                position={position?position:[0,0,0]}
                width={width?width:4}
                height={height?height:4}
                color={color?color:'white'}
                intensity={intensity?intensity:15}
            />

            {/* LTC Area Light Helper -> Screen */}
            {isEnableHelper && <Plane 
                args={[width?width:4,height?height:4]} 
                rotation={rotation?[rotation[0],rotation[1] + (TextureType != 'Null'?0.:Math.PI),rotation[2]]:[0,(TextureType != 'Null'?0.:Math.PI),0]}
                position={position?position:[0,0,0]}>
                <meshBasicMaterial ref={rectAreLightHelperRef} color={color?color:'white'} />
            </Plane>}
            
            {/* All objects in scene */}
            <group ref={childrenRef}>
                {children}
            </group>
        </>
    )
  
});

LTCAreaLightContainer.displayName = 'LTCAreaLightContainer'

const LTCTexturedLightDemo = () =>{

    const {size,gl,camera,scene} = useThree()
    const depthBuffer = useDepthBuffer({ frames: 1 })
    // const { nodes, materials } = useGLTF('./model.gltf')
    const ltcRef = useRef<any>();
    const dragonRef = useRef<any>();


    const videoUrl = './test3.mp4';
    const imageUrl = './test.png';

    const [copyVideo,setCopyVideo] = useState<boolean>(false);
    const [texture,setTexture] = useState<THREE.Texture | null>(null);
    const img_tex = useLoader(THREE.TextureLoader,'./test.png');


    // *** Load Image Texture
    // *** https://threejs.org/docs/#api/en/loaders/TextureLoader    
    const setupImage = (src:string) =>{

        // instantiate a loader
        const loader = new THREE.TextureLoader();

        // load a resource
        loader.load(
            // resource URL
            src,

            // onLoad callback
            function ( texture ) {
                // in this example we create the material when the texture is loaded
                setTexture(texture)
            },

            // onProgress callback currently not supported
            undefined,

            // onError callback
            function ( err ) {
                console.error( 'An error happened.' );
            }
        );



    }

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

        setTexture(vidTex)

    }

    useEffect(()=>{
        //setupImage(imageUrl)
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

    // *** AreaLight Properties
    var {position,rotation,color,intensity,width,height} = useControls('LTC AreaLight',{
        position:{
            value:[0,3,-5],
        },
        rotation:{
            value:[0,0,0],
            step:0.1,
        },
        color:{
            value:'#ffffff',
        },

        intensity:{
            value:15,
            min:0.01,
            max:100.0,
            step:0.01,
        },
        width:{
            value:6.4,
            min:0.01,
            max:100.0,
            step:0.01,
        },
        height:{
            value:4,
            min:0.01,
            max:100.0,
            step:0.01,
        },

    }) as {
        position:[number,number,number],
        rotation:[number,number,number],
        color:string
        intensity:number,
        width:number,
        height:number,
    
    }

    useFrame(()=>{
        //console.log(scene)
    },)

    // *** floor texture
    const floorMap = useLoader(THREE.TextureLoader,'./floor2.jpg');
    floorMap.repeat.set(20,20);
    floorMap.wrapS = floorMap.wrapT = THREE.RepeatWrapping;
    

    return(
        <>
        {/* <LTCAreaLightContainer 
            ref={ltcRef} 
            isEnableHelper={true}
            position={position} 
            rotation={rotation} 
            color={color} 
            width={width}
            height={height}
            intensity={intensity}
            texture={texture}
            blurSize={64}
            index={0}
        > */}
        <LTCAreaLightProxy>
            <LTCAreaLight
                isEnableHelper={true}
                position={position} 
                rotation={rotation} 
                color={color} 
                width={width}
                height={height}
                intensity={intensity}
                texture={texture}
                blurSize={64}
            ></LTCAreaLight>

            <LTCAreaLight
                isEnableHelper={true}
                position={[10,3,0]} 
                rotation={[0,-Math.PI/2,0]} 
                color={color} 
                width={width}
                height={height}
                intensity={intensity}
                texture={img_tex}
                blurSize={64}
            ></LTCAreaLight>

            <LTCAreaLight
                isEnableHelper={true}
                position={[-10,3,0]} 
                rotation={[0,Math.PI/2,0]} 
                color={color} 
                width={width}
                height={height}
                intensity={intensity}
                texture={null}
                blurSize={64}
            ></LTCAreaLight>
            {/* <mesh ref={dragonRef} position={[0,0.5,0]} castShadow receiveShadow geometry={nodes.dragon.geometry} material={materials['Default OBJ.001']} dispose={null} /> */}
   
            <Plane args={[100, 100]} rotation={[-Math.PI / 2, 0, 0]}>
                <meshStandardMaterial 
                    color="#ffffff" 
                    roughness={floor_roughness} 
                    map={floorMap}
                />
            </Plane>
        </LTCAreaLightProxy>
        {/* </LTCAreaLightContainer> */}
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


