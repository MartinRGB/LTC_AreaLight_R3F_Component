import { Box, OrbitControls, Plane, SpotLight, shaderMaterial, useDepthBuffer, useGLTF } from "@react-three/drei"
import { Canvas, useThree, useFrame, createPortal, useLoader, invalidate } from "@react-three/fiber"
import { useControls } from "leva";
import {  useEffect, useMemo, useRef, useState } from "react";
import * as THREE from 'three'
import { Perf } from "r3f-perf";
import { RectAreaLightUniformsLib } from "three/examples/jsm/lights/RectAreaLightUniformsLib.js";
import * as React from "react";
import { RECT_AREALIGHT_PREFIX,LTC_AREALIGHT_CORE,RECT_AREALIGHT_SUFFIX_0,RECT_AREALIGHT_HACK,RECT_AREALIGHT_SUFFIX_1} from '../LTCAreaLight/shader/LTC_Shader';
import { DOWNSAMPLE_BLUR, UPSAMPLE_BLUR } from "../LTCAreaLight/shader/DualKawaseBlur_Shader";
import { common_vertex_main, prefix_frag, prefix_vertex } from "../LTCAreaLight/shader/Utils";

const standard_frag = `
#define STANDARD
#ifdef PHYSICAL
	#define IOR
	#define USE_SPECULAR
#endif
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;
#ifdef IOR
	uniform float ior;
#endif
#ifdef USE_SPECULAR
	uniform float specularIntensity;
	uniform vec3 specularColor;
	#ifdef USE_SPECULAR_COLORMAP
		uniform sampler2D specularColorMap;
	#endif
	#ifdef USE_SPECULAR_INTENSITYMAP
		uniform sampler2D specularIntensityMap;
	#endif
#endif
#ifdef USE_CLEARCOAT
	uniform float clearcoat;
	uniform float clearcoatRoughness;
#endif
#ifdef USE_IRIDESCENCE
	uniform float iridescence;
	uniform float iridescenceIOR;
	uniform float iridescenceThicknessMinimum;
	uniform float iridescenceThicknessMaximum;
#endif
#ifdef USE_SHEEN
	uniform vec3 sheenColor;
	uniform float sheenRoughness;
	#ifdef USE_SHEEN_COLORMAP
		uniform sampler2D sheenColorMap;
	#endif
	#ifdef USE_SHEEN_ROUGHNESSMAP
		uniform sampler2D sheenRoughnessMap;
	#endif
#endif
#ifdef USE_ANISOTROPY
	uniform vec2 anisotropyVector;
	#ifdef USE_ANISOTROPYMAP
		uniform sampler2D anisotropyMap;
	#endif
#endif
varying vec3 vViewPosition;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <iridescence_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_physical_pars_fragment>
#include <transmission_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <clearcoat_pars_fragment>
#include <iridescence_pars_fragment>
#include <roughnessmap_pars_fragment>
#include <metalnessmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	#include <clipping_planes_fragment>
	vec4 diffuseColor = vec4( diffuse, opacity );
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <roughnessmap_fragment>
	#include <metalnessmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <clearcoat_normal_fragment_begin>
	#include <clearcoat_normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_physical_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;
	#include <transmission_fragment>
	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;
	#ifdef USE_SHEEN
		float sheenEnergyComp = 1.0 - 0.157 * max3( material.sheenColor );
		outgoingLight = outgoingLight * sheenEnergyComp + sheenSpecular;
	#endif
	#ifdef USE_CLEARCOAT
		float dotNVcc = saturate( dot( geometry.clearcoatNormal, geometry.viewDir ) );
		vec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );
		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + clearcoatSpecular * material.clearcoat;
	#endif
	#include <output_fragment>
	#include <tonemapping_fragment>
	#include <encodings_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}
`
const standard_vert=`
#define STANDARD
varying vec3 vViewPosition;
#ifdef USE_TRANSMISSION
	varying vec3 vWorldPosition;
#endif
#include <common>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
#ifdef USE_TRANSMISSION
	vWorldPosition = worldPosition.xyz;
#endif
}
`

const LTCAreaLightWithHelper = React.forwardRef(({ 
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

},
ref: React.ForwardedRef<any>
) => {
    const {gl,camera} = useThree();

    const rectAreaLightRef = useRef<any>();
    const rectAreLightHelperRef = useRef<any>();
    const childrenRef = useRef<any>(null!);
    
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

    const HackRectAreaLight = (tex:THREE.Texture,blur_tex:THREE.Texture) =>{
   
        // *** Hacking Children's Material
        if(childrenRef.current){
            childrenRef.current.traverse((obj:any)=>{
                
                if(obj.isMesh){
                    obj.material.onBeforeCompile = (shader:any) => {
                            shader.uniforms.isLTCWithTexture = { value: true };
                            shader.uniforms.ltc_tex = { value: blur_tex };
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


    const videoUrl = './test3.mp4';
    const imageUrl = './test.png';

    const [copyVideo,setCopyVideo] = useState<boolean>(false);
    const [texture,setTexture] = useState<THREE.Texture | null>(null);

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
        setupImage(imageUrl)
        //setupVideo(videoUrl)
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

    // *** floor texture
    const floorMap = useLoader(THREE.TextureLoader,'./floor2.jpg');
    floorMap.repeat.set(20,20);
    floorMap.wrapS = floorMap.wrapT = THREE.RepeatWrapping;
    

    return(
        <>
        <LTCAreaLightWithHelper 
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
        >
            <mesh ref={dragonRef} position={[0,0.5,0]} castShadow receiveShadow geometry={nodes.dragon.geometry} material={materials['Default OBJ.001']} dispose={null} />
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


