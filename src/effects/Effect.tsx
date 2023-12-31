import { Box, OrbitControls, Plane, TransformControls, useGLTF } from "@react-three/drei"
import { Canvas, useThree, useFrame, useLoader } from "@react-three/fiber"
import { useControls } from "leva";
import {  useEffect, useRef, useState,Suspense } from "react";
import * as THREE from 'three'
import { Perf } from "r3f-perf";
import * as React from "react";
import { LTCAreaLight,LTCAreaLightProxy } from "@/LTCAreaLight/LTCAreaLight";
import { Html, useProgress } from '@react-three/drei'

const Loader = () => {
  const { progress } = useProgress()
  return <Html center>{progress} % loaded</Html>
}


const LTCTexturedLightDemo = () =>{

    // Init model
    const { nodes, materials }:any = useGLTF('./model.gltf')
    const { scene }:any = useGLTF('./vr_gallery_test.glb')
    const dragonRef = useRef<any>();
    const roomRef = useRef<any>();
    const ltc1Ref = useRef<any>();
    const ltc2Ref = useRef<any>();
    const ltc3Ref = useRef<any>();
    const ltc4Ref = useRef<any>();

    // Init textures
    const videoUrl1 = './test1.mp4';
    const videoUrl2 = './test2.mp4';
    const imageUrl = './test.png';

    const [copyVideo,setCopyVideo] = useState<boolean>(false);
    const [vid_tex1,setVidTexture1] = useState<THREE.Texture | null>(null);
    const [vid_tex2,setVidTexture2] = useState<THREE.Texture | null>(null);
    const img_tex = useLoader(THREE.TextureLoader,imageUrl);

    const floorMap = useLoader(THREE.TextureLoader,'./floor.jpg');
    floorMap.repeat.set(16,16);
    floorMap.wrapS = floorMap.wrapT = THREE.RepeatWrapping;

    // *** Load Video Texture 
    // *** from 'Animating textures in WebGL'
    // *** https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/Tutorial/Animating_textures_in_WebGL
    const setupVideo = (src:string,callback:(tex:THREE.Texture)=>void) =>{
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
        callback(vidTex)
        //setVidTexture1(vidTex);
    }

    useEffect(()=>{
        //setupVideo(videoUrl1)
        setupVideo(videoUrl1,(tex:THREE.Texture)=>{setVidTexture1(tex)})
        setupVideo(videoUrl2,(tex:THREE.Texture)=>{setVidTexture2(tex)})
    },[])

    useFrame((state, delta) => {
        const time = state.clock.getElapsedTime();
        if(dragonRef.current){
            dragonRef.current.rotation.y += 0.01;
        }
        if(ltc1Ref.current){
            //ltc1Ref.current.position.x = 5.* Math.sin(time);
        }

        if(ltc2Ref.current){
            //ltc2Ref.current.position.y =  2.* Math.sin(time);
        }

        if(ltc3Ref.current){
            //ltc3Ref.current.position.y =  2.* Math.cos(time);
        }

        if(ltc4Ref.current){
            //ltc4Ref.current.rotation.y += 0.02;
        }
    },)

    
    const [isControlEnabled,setControlEnabled] = useState(false)
    



    function Mirror({ ...props }) {
        const mirror = useGLTF('./mirror_b.glb')
        
        return <primitive object={mirror.scene} {...props} />
    }

    // *** Utils
    const {auto_rotate} = useControls('Utils',{
  
        auto_rotate:{
            value:true,
        },
    }) as {
        auto_rotate:boolean,
    }


    // *** Object Material Properties
    const {model_roughness,dragon_roughness} = useControls('Object Material',{
  
        model_roughness:{
            value:1.,
            min:0.0,
            max:10.0,

            onChange:(v:any)=>{
                if(roomRef.current){
                    roomRef.current.traverse((obj:any)=>{

                        if(obj.isMesh){
                            obj.material.roughness = v;
                        }
                    })
                }
            }
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
        model_roughness:number,
        dragon_roughness:number,
    }

    // *** Video1 AreaLight Properties
    const {position0,rotation0,color0,intensity0,width0,height0,addtionalRoughness0,isDoubleSide0,isClipless0} = useControls('Video1 LTC AreaLight',{
        position0:{
            value:[0,2.5,-9],
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
        addtionalRoughness0:{
            value:0.0,
            min:0.0,
            max:1.0,
            step:0.01,
            label:'Addtional Roughness',
        },
        isDoubleSide0:{
            value:true,
            label:'Double Side',
        },
        isClipless0:{
            value:false,
            label:'Clipless Approximation',
        }
    }) as {
        position0:[number,number,number],
        rotation0:[number,number,number],
        color0:string
        intensity0:number,
        width0:number,
        height0:number,
        addtionalRoughness0:number,
        isDoubleSide0:boolean,
        isClipless0:boolean,
    }

    // *** Image AreaLight Properties
    const {position1,rotation1,color1,intensity1,width1,height1,addtionalRoughness1,isDoubleSide1,isClipless1} = useControls('Image LTC AreaLight',{
        position1:{
            value:[8,2.5,0],
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
        addtionalRoughness1:{
            value:0.0,
            min:0.0,
            max:1.0,
            step:0.01,
            label:'Addtional Roughness',
        },
        isDoubleSide1:{
            value:true,
            label:'Double Side',
        },
        isClipless1:{
            value:false,
            label:'Clipless Approximation',
        }
    }) as {
        position1:[number,number,number],
        rotation1:[number,number,number],
        color1:string
        intensity1:number,
        width1:number,
        height1:number,
        addtionalRoughness1:number,
        isDoubleSide1:boolean,
        isClipless1:boolean,
    }

    // *** Color AreaLight Properties
    const {position2,rotation2,color2,intensity2,width2,height2,addtionalRoughness2,isDoubleSide2,isClipless2} = useControls('Color LTC AreaLight',{

        position2:{
            value:[-8,2.5,0],
            label:'Position',
        },
        rotation2:{
            value:[0,Math.PI/2,0],
            step:0.1,
            label:'Rotation',
        },
        color2:{
            value:'#00ff00',
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
        addtionalRoughness2:{
            value:0.0,
            min:0.0,
            max:1.0,
            step:0.01,
            label:'Addtional Roughness',
        },
        isDoubleSide2:{
            value:true,
            label:'Double Side',
        },
        isClipless2:{
            value:false,
            label:'Clipless Approximation',
        }
    }) as {
        position2:[number,number,number],
        rotation2:[number,number,number],
        color2:string
        intensity2:number,
        width2:number,
        height2:number,
        addtionalRoughness2:number,
        isDoubleSide2:boolean,
        isClipless2:boolean,
    }

    // *** Video2 AreaLight Properties
    // const {position3,rotation3,color3,intensity3,width3,height3} = useControls('Video2 LTC AreaLight',{

    //     position3:{
    //         value:[0,2.5,9],
    //         label:'Position',
    //     },
    //     rotation3:{
    //         value:[0,0,0],
    //         step:0.1,
    //         label:'Rotation',
    //     },
    //     color3:{
    //         value:'#ffffff',
    //         label:'Color',
    //     },

    //     intensity3:{
    //         value:15,
    //         min:0.01,
    //         max:100.0,
    //         step:0.01,
    //         label:'Intensity',
    //     },
    //     width3:{
    //         value:6.4,
    //         min:0.01,
    //         max:100.0,
    //         step:0.01,
    //         label:'Width',
    //     },
    //     height3:{
    //         value:4,
    //         min:0.01,
    //         max:100.0,
    //         step:0.01,
    //         label:'Height',
    //     },
    // }) as {
    //     position3:[number,number,number],
    //     rotation3:[number,number,number],
    //     color3:string
    //     intensity3:number,
    //     width3:number,
    //     height3:number,
    // }

    const {position4,rotation4,scale4} = useControls('Room Props',{
        position4:{
            value:[0,0,-1],
            label:'Position',
            step:0.001,
        },
        rotation4:{
            value:[0,0,0],
            step:0.1,
            label:'Rotation',
        },
        scale4:{
            value:[1.2,2.0,1.2],
            label:'Scale',
            step:0.001,
        },

    }) as {
        position4:[number,number,number],
        rotation4:[number,number,number],
        scale4:[number,number,number]
    }



    // TODO: Remove 3D Objects from Proxy
    return(
        <>
        
        <Suspense fallback={null}>
        {/* LTCAreaLightProxy contains LTCAreaLight Objects & 3D Objects */}
        <LTCAreaLightProxy>
            {/* LTCAreaLight Objects */}
            <LTCAreaLight
                ref={ltc1Ref}
                isEnableHelper={true}
                position={position0} 
                rotation={rotation0} 
                color={color0} 
                width={width0}
                height={height0}
                intensity={intensity0}
                addtionalRoughness={addtionalRoughness0}
                texture={vid_tex1}
                blurSize={64}
                doubleSide={isDoubleSide0}
                clipless={isClipless0}
            ></LTCAreaLight>

            <LTCAreaLight
                ref={ltc2Ref}
                isEnableHelper={true}
                position={position1} 
                rotation={rotation1} 
                color={color1} 
                width={width1}
                height={height1}
                intensity={intensity1}
                addtionalRoughness={addtionalRoughness1}
                texture={img_tex}
                blurSize={64}
                doubleSide={isDoubleSide1}
                clipless={isClipless1}
            ></LTCAreaLight>

            <LTCAreaLight
                ref={ltc3Ref}
                isEnableHelper={true}
                position={position2} 
                rotation={rotation2} 
                color={color2} 
                width={width2}
                height={height2}
                intensity={intensity2}
                addtionalRoughness={addtionalRoughness2}
                texture={null}
                blurSize={64}
                doubleSide={isDoubleSide2}
                clipless={isClipless2}
            ></LTCAreaLight>

            {/* <LTCAreaLight
                ref={ltc4Ref}
                isEnableHelper={true}
                position={position3} 
                rotation={rotation3} 
                color={color3} 
                width={width3}
                height={height3}
                intensity={intensity3}
                texture={vid_tex2}
                blurSize={64}
                doubleSide={true}
                clipless={true}
            ></LTCAreaLight> */}

            {/* 3D Objects */}
            
            {isControlEnabled && <TransformControls mode="translate" enabled={isControlEnabled} object={dragonRef.current}/>}

            <group dispose={null}>
            <primitive ref={roomRef} object={scene} position={position4} scale={scale4} rotation={rotation4} />
            </group>
            <Mirror position={[0,-8.,9]} scale={[8.0,8.0,8.0]} rotation={[0,-Math.PI,0]}/>
            <mesh ref={dragonRef} 
                    position={[0,0,0]} 
                    castShadow 
                    receiveShadow 
                    onClick={(e)=>{setControlEnabled(!isControlEnabled)}}
                    geometry={nodes.dragon.geometry} 
                    material={materials['Default OBJ.001']}
                     dispose={null} />
        </LTCAreaLightProxy>
        
        <OrbitControls autoRotate={auto_rotate} enabled={!isControlEnabled}></OrbitControls>
        </Suspense>
        </>
    )
    
}


export const Effect = (props:any) =>{


    return(
      <>
          <Canvas 
            camera={{ position: [0, 5, 30], fov: 50, near: 0.1, far: 1000 }}
            className={props.className} 
            style={{...props.style}}>
            <Suspense fallback={<Loader />}>
                <Perf style={{position:'absolute',top:'10px',left:'10px',width:'360px',borderRadius:'10px'}}/>
                <ambientLight intensity={0.015}></ambientLight>
                <color attach="background" args={['#000000']} />
                <LTCTexturedLightDemo/>
            </Suspense>
          </Canvas>
      </>
  
    )
}


