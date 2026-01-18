import React, { useRef, useMemo, useEffect, useLayoutEffect } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { useSpring, animated } from '@react-spring/three'
import * as THREE from 'three'
import { LODLevel, PIN_CONFIG, CIRCUIT_TRACE_CONFIG } from '../../types'
import {
  sharedMaterials,
  easeInOutCubic,
  lastCameraState,
  setLastCameraState,
  CameraAnimationTarget,
  ChipPinData,
} from './shared'

// ============================================
// 带动画的透明材质组件 (react-spring)
// ============================================

const AnimatedMeshStandardMaterial = animated.meshStandardMaterial

interface FadingMaterialProps {
  targetOpacity: number
  color: string
  metalness?: number
  roughness?: number
  emissive?: string
  emissiveIntensity?: number
  toneMapped?: boolean
  side?: THREE.Side
}

export const FadingMaterial: React.FC<FadingMaterialProps> = ({
  targetOpacity,
  color,
  metalness = 0.5,
  roughness = 0.5,
  emissive,
  emissiveIntensity = 0,
  toneMapped = true,
  side = THREE.FrontSide,
}) => {
  const { opacity } = useSpring({
    from: { opacity: 0 },  // 从透明开始，确保淡入效果
    to: { opacity: targetOpacity },
    config: { tension: 120, friction: 20 }
  })

  return (
    <AnimatedMeshStandardMaterial
      transparent
      opacity={opacity}
      color={color}
      metalness={metalness}
      roughness={roughness}
      emissive={emissive}
      emissiveIntensity={emissiveIntensity}
      toneMapped={toneMapped}
      side={side}
    />
  )
}

// 带动画的 meshBasicMaterial
const AnimatedMeshBasicMaterial = animated.meshBasicMaterial

interface FadingBasicMaterialProps {
  targetOpacity: number
  color: string
}

export const FadingBasicMaterial: React.FC<FadingBasicMaterialProps> = ({
  targetOpacity,
  color,
}) => {
  const { opacity } = useSpring({
    from: { opacity: 0 },  // 从透明开始，确保淡入效果
    to: { opacity: targetOpacity },
    config: { tension: 120, friction: 20 }
  })

  return (
    <AnimatedMeshBasicMaterial
      transparent
      opacity={opacity}
      color={color}
    />
  )
}

// ============================================
// InstancedMesh 组件 - 批量渲染引脚
// ============================================

export const InstancedPins: React.FC<{
  chips: ChipPinData[]
  lodLevel: LODLevel
}> = ({ chips, lodLevel }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null)

  // 低细节模式不渲染引脚
  if (lodLevel === 'low' || chips.length === 0) return null

  // 中等细节模式减少引脚数
  const actualPinsPerSide = lodLevel === 'medium' ? 3 : PIN_CONFIG.pinsPerSide
  const actualPinsPerChip = actualPinsPerSide * 4
  const actualTotalPins = chips.length * actualPinsPerChip

  // 创建变换矩阵
  const matrices = useMemo(() => {
    const tempMatrix = new THREE.Matrix4()
    const result: THREE.Matrix4[] = []

    chips.forEach(({ position, dimensions }) => {

      // 左右两侧引脚
      for (let i = 0; i < actualPinsPerSide; i++) {
        const zOffset = (i - (actualPinsPerSide - 1) / 2) * (dimensions[2] / (actualPinsPerSide + 1))

        // 左侧
        tempMatrix.makeTranslation(
          position.x - dimensions[0] / 2 - PIN_CONFIG.pinWidth / 2,
          position.y,
          position.z + zOffset
        )
        result.push(tempMatrix.clone())

        // 右侧
        tempMatrix.makeTranslation(
          position.x + dimensions[0] / 2 + PIN_CONFIG.pinWidth / 2,
          position.y,
          position.z + zOffset
        )
        result.push(tempMatrix.clone())
      }

      // 前后两侧引脚
      for (let i = 0; i < actualPinsPerSide; i++) {
        const xOffset = (i - (actualPinsPerSide - 1) / 2) * (dimensions[0] / (actualPinsPerSide + 1))

        // 前侧
        tempMatrix.makeTranslation(
          position.x + xOffset,
          position.y,
          position.z - dimensions[2] / 2 - PIN_CONFIG.pinDepth / 2
        )
        result.push(tempMatrix.clone())

        // 后侧
        tempMatrix.makeTranslation(
          position.x + xOffset,
          position.y,
          position.z + dimensions[2] / 2 + PIN_CONFIG.pinDepth / 2
        )
        result.push(tempMatrix.clone())
      }
    })

    return result
  }, [chips, actualPinsPerSide])

  // 更新 InstancedMesh
  useEffect(() => {
    if (!meshRef.current) return
    matrices.forEach((matrix, i) => {
      meshRef.current!.setMatrixAt(i, matrix)
    })
    meshRef.current.instanceMatrix.needsUpdate = true
  }, [matrices])

  // 使用第一个芯片的尺寸作为引脚尺寸参考
  const pinDimensions = chips[0] ? [
    PIN_CONFIG.pinWidth,
    chips[0].dimensions[1] * PIN_CONFIG.pinHeightRatio,
    PIN_CONFIG.pinDepth
  ] as [number, number, number] : [0.006, 0.006, 0.004] as [number, number, number]

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, actualTotalPins]} castShadow material={sharedMaterials.pin}>
      <boxGeometry args={pinDimensions} />
    </instancedMesh>
  )
}

// ============================================
// InstancedMesh 组件 - 批量渲染电路纹理
// ============================================

export const InstancedCircuitTraces: React.FC<{
  chips: ChipPinData[]
  lodLevel: LODLevel
}> = ({ chips, lodLevel }) => {
  const hMeshRef = useRef<THREE.InstancedMesh>(null)
  const vMeshRef = useRef<THREE.InstancedMesh>(null)

  // 非高细节模式不渲染电路纹理
  if (lodLevel !== 'high' || chips.length === 0) return null

  const hCount = CIRCUIT_TRACE_CONFIG.horizontalCount
  const vCount = CIRCUIT_TRACE_CONFIG.verticalCount
  const totalHTraces = chips.length * hCount
  const totalVTraces = chips.length * vCount

  // 创建水平纹理变换矩阵
  const hMatrices = useMemo(() => {
    const tempMatrix = new THREE.Matrix4()
    const result: THREE.Matrix4[] = []

    chips.forEach(({ position, dimensions }) => {
      for (let i = 0; i < hCount; i++) {
        const zOffset = (i - (hCount - 1) / 2) * (dimensions[2] * 0.25)
        tempMatrix.makeTranslation(
          position.x,
          position.y + dimensions[1] / 2 + 0.001,
          position.z + zOffset
        )
        result.push(tempMatrix.clone())
      }
    })

    return result
  }, [chips])

  // 创建垂直纹理变换矩阵
  const vMatrices = useMemo(() => {
    const tempMatrix = new THREE.Matrix4()
    const result: THREE.Matrix4[] = []

    chips.forEach(({ position, dimensions }) => {
      for (let i = 0; i < vCount; i++) {
        const xOffset = (i - (vCount - 1) / 2) * (dimensions[0] * 0.25)
        tempMatrix.makeTranslation(
          position.x + xOffset,
          position.y + dimensions[1] / 2 + 0.001,
          position.z
        )
        result.push(tempMatrix.clone())
      }
    })

    return result
  }, [chips])

  // 更新 InstancedMesh
  useEffect(() => {
    if (hMeshRef.current) {
      hMatrices.forEach((matrix, i) => {
        hMeshRef.current!.setMatrixAt(i, matrix)
      })
      hMeshRef.current.instanceMatrix.needsUpdate = true
    }
    if (vMeshRef.current) {
      vMatrices.forEach((matrix, i) => {
        vMeshRef.current!.setMatrixAt(i, matrix)
      })
      vMeshRef.current.instanceMatrix.needsUpdate = true
    }
  }, [hMatrices, vMatrices])

  // 使用第一个芯片的尺寸
  const firstChip = chips[0]
  if (!firstChip) return null

  const hTraceWidth = firstChip.dimensions[0] * 0.7
  const vTraceDepth = firstChip.dimensions[2] * 0.7

  return (
    <>
      {/* 水平纹理 */}
      <instancedMesh ref={hMeshRef} args={[undefined, undefined, totalHTraces]} material={sharedMaterials.circuitTrace}>
        <boxGeometry args={[hTraceWidth, CIRCUIT_TRACE_CONFIG.traceHeight, CIRCUIT_TRACE_CONFIG.traceWidth]} />
      </instancedMesh>
      {/* 垂直纹理 */}
      <instancedMesh ref={vMeshRef} args={[undefined, undefined, totalVTraces]} material={sharedMaterials.circuitTrace}>
        <boxGeometry args={[CIRCUIT_TRACE_CONFIG.traceWidth, CIRCUIT_TRACE_CONFIG.traceHeight, vTraceDepth]} />
      </instancedMesh>
    </>
  )
}

// ============================================
// 相机动画控制器
// ============================================

export const CameraController: React.FC<{
  target: CameraAnimationTarget
  baseDuration?: number  // 基础动画时长
  onAnimationComplete?: () => void
  enabled?: boolean
  resetTrigger?: number  // 变化时强制重置，即使目标位置相同
  visible?: boolean  // 是否可见，隐藏时直接跳转不执行动画
}> = ({ target, baseDuration = 1.0, onAnimationComplete, enabled = true, resetTrigger = 0, visible = true }) => {
  const { camera } = useThree()
  const controlsRef = useRef<any>(null)

  // 动画状态
  const isAnimating = useRef(false)
  const startPosition = useRef(new THREE.Vector3())
  const startTarget = useRef(new THREE.Vector3())
  const progress = useRef(0)
  const actualDuration = useRef(baseDuration)  // 根据距离动态计算的实际时长
  const lastTarget = useRef<CameraAnimationTarget | null>(null)
  const lastResetTrigger = useRef(resetTrigger)
  const pendingCallback = useRef<(() => void) | null>(null)
  const isFirstRender = useRef(true)  // 首次渲染标记
  const needsInitialTarget = useRef(false)  // 首次渲染时 OrbitControls 未挂载，需要延迟设置 target

  // 记录上一次的 visible 状态
  const lastVisible = useRef(visible)

  // 持续追踪 OrbitControls 的 target（因为卸载时 ref 可能已失效）
  // 初始化为目标 lookAt，确保即使 useFrame 没来得及更新也有正确的值
  const lastKnownTarget = useRef(target.lookAt.clone())
  // 标记是否已经设置过有效的 target（避免 useFrame 在 useEffect 之前用默认值覆盖）
  const hasValidTarget = useRef(false)

  useFrame(() => {
    // 每帧更新已知的 target 位置，但只有在已设置过有效 target 后才更新
    if (controlsRef.current && hasValidTarget.current) {
      lastKnownTarget.current.copy(controlsRef.current.target)
    }
  })

  // 当 target.lookAt 变化时，同步更新 lastKnownTarget（作为后备）
  // 但只在 hasValidTarget 为 false 时更新（一旦 controls 设置过，就不再用 target.lookAt 覆盖）
  useEffect(() => {
    if (!hasValidTarget.current) {
      lastKnownTarget.current.copy(target.lookAt)
    }
  }, [target.lookAt.x, target.lookAt.y, target.lookAt.z])

  // 组件卸载时保存相机状态到模块级变量
  useEffect(() => {
    return () => {
      // 保存优先级（WebGL Context Lost 时 controlsRef.target 会被重置为 0,0,0）：
      // 1. lastTarget.current.lookAt（我们设置的目标值，最可靠）
      // 2. lastKnownTarget（useFrame 更新的值）
      // 3. controlsRef.current.target（可能被重置，作为最终后备）
      const lookAtToSave = (() => {
        if (lastTarget.current) {
          return lastTarget.current.lookAt.clone()
        }
        if (hasValidTarget.current) {
          return lastKnownTarget.current.clone()
        }
        if (controlsRef.current) {
          return controlsRef.current.target.clone()
        }
        return lastKnownTarget.current.clone()
      })()
      setLastCameraState({
        position: camera.position.clone(),
        lookAt: lookAtToSave
      })
    }
  }, [camera])

  // 目标变化或 resetTrigger 变化时启动动画
  // 使用 useLayoutEffect 确保在浏览器绘制前设置好相机状态
  useLayoutEffect(() => {
    const justBecameVisible = visible && !lastVisible.current
    lastVisible.current = visible

    // 首次渲染时，检查是否有上次保存的相机状态
    if (isFirstRender.current) {
      isFirstRender.current = false

      if (lastCameraState) {
        camera.position.copy(lastCameraState.position)
        if (controlsRef.current) {
          controlsRef.current.target.copy(lastCameraState.lookAt)
          controlsRef.current.update()
          hasValidTarget.current = true
        }
        startPosition.current.copy(lastCameraState.position)
        startTarget.current.copy(lastCameraState.lookAt)
        actualDuration.current = baseDuration
        progress.current = 0
        isAnimating.current = true
        lastTarget.current = { position: target.position.clone(), lookAt: target.lookAt.clone() }
        pendingCallback.current = onAnimationComplete || null
        return
      }

      // 无上次状态，直接设置到目标位置
      camera.position.copy(target.position)
      if (controlsRef.current) {
        controlsRef.current.target.copy(target.lookAt)
        controlsRef.current.update()
        hasValidTarget.current = true
      } else {
        needsInitialTarget.current = true
      }
      lastTarget.current = { position: target.position.clone(), lookAt: target.lookAt.clone() }
      return
    }

    // 不可见时直接设置位置
    if (!visible) {
      camera.position.copy(target.position)
      if (controlsRef.current) {
        controlsRef.current.target.copy(target.lookAt)
        controlsRef.current.update()
        hasValidTarget.current = true
      }
      lastTarget.current = { position: target.position.clone(), lookAt: target.lookAt.clone() }
      isAnimating.current = false
      return
    }

    // 刚变为可见时，直接设置位置
    if (justBecameVisible) {
      camera.position.copy(target.position)
      if (controlsRef.current) {
        controlsRef.current.target.copy(target.lookAt)
        controlsRef.current.update()
        hasValidTarget.current = true
      }
      lastTarget.current = { position: target.position.clone(), lookAt: target.lookAt.clone() }
      return
    }

    // 检查是否是强制重置
    const isForceReset = resetTrigger !== lastResetTrigger.current
    lastResetTrigger.current = resetTrigger

    // 目标未变化时跳过
    if (!isForceReset && lastTarget.current &&
        lastTarget.current.position.equals(target.position) &&
        lastTarget.current.lookAt.equals(target.lookAt)) {
      return
    }

    // 启动动画
    startPosition.current.copy(camera.position)
    if (controlsRef.current) {
      startTarget.current.copy(controlsRef.current.target)
    } else {
      startTarget.current.set(0, 0, 0)
    }
    actualDuration.current = baseDuration
    progress.current = 0
    isAnimating.current = true
    lastTarget.current = { position: target.position.clone(), lookAt: target.lookAt.clone() }
    pendingCallback.current = onAnimationComplete || null
  }, [target.position.x, target.position.y, target.position.z,
      target.lookAt.x, target.lookAt.y, target.lookAt.z, camera, onAnimationComplete, resetTrigger, baseDuration, visible])

  // 每帧更新
  useFrame((_, delta) => {
    // 处理首次渲染时 OrbitControls 未挂载的延迟初始化
    if (needsInitialTarget.current && controlsRef.current) {
      if (isAnimating.current) {
        controlsRef.current.target.copy(startTarget.current)
      } else if (lastTarget.current) {
        controlsRef.current.target.copy(lastTarget.current.lookAt)
      } else {
        controlsRef.current.target.copy(target.lookAt)
      }
      controlsRef.current.update()
      hasValidTarget.current = true
      needsInitialTarget.current = false
    }

    if (!isAnimating.current) return

    progress.current += delta / actualDuration.current
    const t = easeInOutCubic(Math.min(progress.current, 1))

    // 插值相机位置
    camera.position.lerpVectors(startPosition.current, target.position, t)

    // 插值观察目标
    if (controlsRef.current) {
      controlsRef.current.target.lerpVectors(startTarget.current, target.lookAt, t)
      controlsRef.current.update()
      hasValidTarget.current = true
    }

    if (progress.current >= 1) {
      isAnimating.current = false
      if (pendingCallback.current) {
        pendingCallback.current()
        pendingCallback.current = null
      }
    }
  })

  // 计算初始 target（只用于 OrbitControls 挂载时）
  const initialTarget = useMemo(() => {
    if (lastCameraState) {
      return [lastCameraState.lookAt.x, lastCameraState.lookAt.y, lastCameraState.lookAt.z] as [number, number, number]
    }
    return [target.lookAt.x, target.lookAt.y, target.lookAt.z] as [number, number, number]
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // 空依赖，只在首次渲染时计算

  return (
    <OrbitControls
      ref={controlsRef}
      target={initialTarget}
      enabled={enabled && !isAnimating.current}
      enablePan={true}
      enableZoom={true}
      enableRotate={true}
    />
  )
}
