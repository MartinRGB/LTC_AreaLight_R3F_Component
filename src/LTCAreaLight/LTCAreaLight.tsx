// SpotLight Inspired by http://john-chapman-graphics.blogspot.com/2013/01/good-enough-volumetrics-for-spotlights.html

import * as React from 'react'
import {
  Mesh,
  DepthTexture,
  Vector3,
  CylinderGeometry,
  Matrix4,
  RectAreaLight as RectAreaLightImpl,
  DoubleSide,
  Texture,
  WebGLRenderTarget,
  ShaderMaterial,
  RGBAFormat,
  LinearEncoding,
  RepeatWrapping,
  Object3D,
} from 'three'
import { useFrame, useThree } from '@react-three/fiber'
import { FullScreenQuad } from 'three-stdlib'
import mergeRefs from 'react-merge-refs'
import { LTCAreaLightMaterial } from './material/LTCAreaLightMaterial'

// eslint-disable-next-line
// @ts-ignore
// import SpotlightShadowShader from './glsl/DefaultSpotlightProxyShadows.glsl'

type LTCAreaLightProps = JSX.IntrinsicElements['ltcAreaLight'] & {
  depthBuffer?: DepthTexture
  attenuation?: number
  anglePower?: number
  radiusTop?: number
  radiusBottom?: number
  opacity?: number
  color?: string | number
  volumetric?: boolean
  debug?: boolean
}

const isLTCAreaLight = (child: Object3D | null): child is RectAreaLightImpl => {
  return (child as RectAreaLightImpl)?.isRectAreaLight
}

function VolumetricMesh({
  opacity = 1,
  radiusTop,
  radiusBottom,
  depthBuffer,
  color = 'white',
  distance = 5,
  angle = 0.15,
  attenuation = 5,
  anglePower = 5,
}: Omit<LTCAreaLightProps, 'volumetric'>) {
  const mesh = React.useRef<Mesh>(null!)
  const size = useThree((state) => state.size)
  const camera = useThree((state) => state.camera)
  const dpr = useThree((state) => state.viewport.dpr)
  const [material] = React.useState(() => new LTCAreaLightMaterial())
  const [vec] = React.useState(() => new Vector3())

  console.log(material)

  radiusTop = radiusTop === undefined ? 0.1 : radiusTop
  radiusBottom = radiusBottom === undefined ? angle * 7 : radiusBottom

  useFrame(() => {
    material.uniforms.spotPosition.value.copy(mesh.current.getWorldPosition(vec))
    mesh.current.lookAt((mesh.current.parent as any).target.getWorldPosition(vec))
  })

  const geom = React.useMemo(() => {
    const geometry = new CylinderGeometry(radiusTop, radiusBottom, distance, 128, 64, true)
    geometry.applyMatrix4(new Matrix4().makeTranslation(0, -distance / 2, 0))
    geometry.applyMatrix4(new Matrix4().makeRotationX(-Math.PI / 2))
    return geometry
  }, [distance, radiusTop, radiusBottom])

  return (
    <>
      <mesh ref={mesh} geometry={geom} raycast={() => null}>
        <primitive
          object={material}
          attach="material"
          uniforms-opacity-value={opacity}
          uniforms-lightColor-value={color}
          uniforms-attenuation-value={attenuation}
          uniforms-anglePower-value={anglePower}
          uniforms-depth-value={depthBuffer}
          uniforms-cameraNear-value={camera.near}
          uniforms-cameraFar-value={camera.far}
          uniforms-resolution-value={depthBuffer ? [size.width * dpr, size.height * dpr] : [0, 0]}
        />
      </mesh>
    </>
  )
}

function useCommon(
  ltcAreaLight: React.MutableRefObject<RectAreaLightImpl>,
  mesh: React.MutableRefObject<Mesh>,
  width: number,
  height: number,
  distance: number
) {
  const [[pos, dir]] = React.useState(() => [new Vector3(), new Vector3()])

  React.useLayoutEffect(() => {
    if (isLTCAreaLight(ltcAreaLight.current)) {
      ltcAreaLight.current.shadow.mapSize.set(width, height)
      ltcAreaLight.current.shadow.needsUpdate = true
    } else {
      throw new Error('LTCAreaLightShadow must be a child of a LTCAreaLight')
    }
  }, [ltcAreaLight, width, height])

  useFrame(() => {
    if (!ltcAreaLight.current) return

    const A = ltcAreaLight.current.position
    const B = ltcAreaLight.current.target.position

    dir.copy(B).sub(A)
    var len = dir.length()
    dir.normalize().multiplyScalar(len * distance)
    pos.copy(A).add(dir)

    mesh.current.position.copy(pos)
    mesh.current.lookAt(ltcAreaLight.current.target.position)
  })
}

interface ShadowMeshProps {
  distance?: number
  alphaTest?: number
  scale?: number
  map?: Texture
  shader?: string
  width?: number
  height?: number
}

function LTCAreaLightShadowWithShader({
  distance = 0.4,
  alphaTest = 0.5,
  map,
  // shader = SpotlightShadowShader,
  width = 512,
  height = 512,
  scale = 1,
  children,
  ...rest
}: React.PropsWithChildren<ShadowMeshProps>) {
  const mesh = React.useRef<Mesh>(null!)
  const ltcAreaLight = (rest as any).ltcAreaLightRef
  const debug = (rest as any).debug

  const ltcAreaLightProxyShadowsFrag=`varying vec2 vUv;

  uniform sampler2D uShadowMap;
  uniform float uTime;
  
  void main() {
      vec3 color = texture2D(uShadowMap, vUv).xyz;
      gl_FragColor = vec4(color, 1.);
  }`

  useCommon(ltcAreaLight, mesh, width, height, distance)

  const renderTarget = React.useMemo(
    () =>
      new WebGLRenderTarget(width, height, {
        format: RGBAFormat,
        encoding: LinearEncoding,
        stencilBuffer: false,
        // depthTexture: null!
      }),
    [width, height]
  )

  const uniforms = React.useRef({
    uShadowMap: {
      value: map,
    },
    uTime: {
      value: 0,
    },
  })

  React.useEffect(() => void (uniforms.current.uShadowMap.value = map), [map])

  const fsQuad = React.useMemo(
    () =>
      new FullScreenQuad(
        new ShaderMaterial({
          uniforms: uniforms.current,
          vertexShader: /* glsl */ `
          varying vec2 vUv;

          void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
          `,
          fragmentShader: ltcAreaLightProxyShadowsFrag, //shader
        })
      ),
    [ltcAreaLightProxyShadowsFrag]
  )

  React.useEffect(
    () => () => {
      fsQuad.material.dispose()
      fsQuad.dispose()
    },
    [fsQuad]
  )

  React.useEffect(() => () => renderTarget.dispose(), [renderTarget])

  useFrame(({ gl }, dt) => {
    uniforms.current.uTime.value += dt

    gl.setRenderTarget(renderTarget)
    fsQuad.render(gl)
    gl.setRenderTarget(null)
  })

  return (
    <>
      <mesh ref={mesh} scale={scale} castShadow>
        <planeGeometry />
        <meshBasicMaterial
          transparent
          side={DoubleSide}
          alphaTest={alphaTest}
          alphaMap={renderTarget.texture}
          alphaMap-wrapS={RepeatWrapping}
          alphaMap-wrapT={RepeatWrapping}
          opacity={debug ? 1 : 0}
        >
          {children}
        </meshBasicMaterial>
      </mesh>
    </>
  )
}

function LTCAreaLightShadowWithoutShader({
  distance = 0.4,
  alphaTest = 0.5,
  map,
  width = 512,
  height = 512,
  scale,
  children,
  ...rest
}: React.PropsWithChildren<ShadowMeshProps>) {
  const mesh = React.useRef<Mesh>(null!)
  const ltcAreaLight = (rest as any).ltcAreaLightRef
  const debug = (rest as any).debug

  useCommon(ltcAreaLight, mesh, width, height, distance)

  return (
    <>
      <mesh ref={mesh} scale={scale} castShadow>
        <planeGeometry />
        <meshBasicMaterial
          transparent
          side={DoubleSide}
          alphaTest={alphaTest}
          alphaMap={map}
          alphaMap-wrapS={RepeatWrapping}
          alphaMap-wrapT={RepeatWrapping}
          opacity={debug ? 1 : 0}
        >
          {children}
        </meshBasicMaterial>
      </mesh>
    </>
  )
}

export function LTCAreaLightShadow(props: React.PropsWithChildren<ShadowMeshProps>) {
  if (props.shader) return <LTCAreaLightShadowWithShader {...props} />
  return <LTCAreaLightShadowWithoutShader {...props} />
}

const LTCAreaLight = React.forwardRef(
  (
    {
      // Volumetric
      opacity = 1,
      radiusTop,
      radiusBottom,
      depthBuffer,
      color = 'white',
      distance = 5,
      angle = 0.15,
      attenuation = 5,
      anglePower = 5,
      volumetric = true,
      debug = false,
      children,
      ...props
    }: React.PropsWithChildren<LTCAreaLightProps>,
    ref: React.ForwardedRef<RectAreaLightImpl>
  ) => {
    const ltcAreaLight = React.useRef<any>(null!)

    return (
      <group>
        {debug && ltcAreaLight.current && <spotLightHelper args={[ltcAreaLight.current]} />}

        <spotLight
          ref={mergeRefs([ref, ltcAreaLight])}
          angle={angle}
          color={color}
          distance={distance}
          castShadow
          {...props}
        >
          {volumetric && (
            <VolumetricMesh
              debug={debug}
              opacity={opacity}
              radiusTop={radiusTop}
              radiusBottom={radiusBottom}
              depthBuffer={depthBuffer}
              color={color}
              distance={distance}
              angle={angle}
              attenuation={attenuation}
              anglePower={anglePower}
            />
          )}
        </spotLight>
        {children &&
          React.cloneElement(children as any, {
            ltcAreaLightRef: ltcAreaLight,
            debug: debug,
          })}
      </group>
    )
  }
)

LTCAreaLight.displayName='LTCAreaLight'

export { LTCAreaLight }