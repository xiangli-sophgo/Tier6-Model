import React, { useRef, useState, useMemo, useEffect } from 'react'
import { ThreeEvent } from '@react-three/fiber'
import { Text, Html } from '@react-three/drei'
import { useSpring } from '@react-spring/three'
import * as THREE from 'three'
import {
  ChipConfig,
  BoardConfig,
  SwitchInstance,
  RackConfig,
  PodConfig,
  RACK_DIMENSIONS,
  BOARD_DIMENSIONS,
  CHIP_DIMENSIONS,
  CHIP_TYPE_NAMES,
  LODLevel,
} from '../../types'
import { sharedMaterials, sharedGeometries, ChipPinData } from './shared'
import { FadingMaterial, FadingBasicMaterial, InstancedPins, InstancedCircuitTraces } from './materials'

// ============================================
// 3D模型组件
// ============================================

// 计算芯片在 Board 上的位置（供 InstancedMesh 使用）
export const getChipPosition = (
  chip: ChipConfig,
  totalChips: number,
  baseY: number
): { x: number; y: number; z: number; dimensions: [number, number, number] } => {
  const baseDimensions = CHIP_DIMENSIONS[chip.type] || [0.06, 0.02, 0.06]
  const dimensions: [number, number, number] = [baseDimensions[0] * 0.9, baseDimensions[1] * 1.2, baseDimensions[2] * 0.9]

  const cols = Math.ceil(Math.sqrt(totalChips))
  const rows = Math.ceil(totalChips / cols)
  const row = chip.position[0]
  const col = chip.position[1]
  const chipsInCurrentRow = row < rows - 1 ? cols : totalChips - (rows - 1) * cols

  const spacing = 0.11
  const rowCenterOffset = (chipsInCurrentRow - 1) / 2
  const x = (col - rowCenterOffset) * spacing
  const z = (row - (rows - 1) / 2) * spacing
  const y = baseY + dimensions[1] / 2

  return { x, y, z, dimensions }
}

// Chip模型 - 高端拟物风格，带有文字标识
// 引脚和电路纹理由 InstancedMesh 统一渲染以提升性能
export const ChipModel: React.FC<{
  chip: ChipConfig
  baseY?: number  // 底板高度
  totalChips?: number  // 总芯片数，用于计算居中
  lodLevel?: LODLevel  // LOD 级别
  onClick?: () => void
  onPointerOver?: () => void
  onPointerOut?: () => void
}> = ({ chip, baseY = 0, totalChips = 8, lodLevel = 'high', onClick, onPointerOver, onPointerOut }) => {
  const [hovered, setHovered] = useState(false)
  const { x, y, z, dimensions } = getChipPosition(chip, totalChips, baseY)

  // 芯片标签文字 - 优先使用配置的label，否则使用类型名称
  const chipLabel = chip.label || 'Chip'
  // 深色金属外壳颜色
  const shellColor = '#1a1a1a'
  const shellColorHover = '#2a2a2a'
  // 顶部标识颜色
  const labelColor = '#4fc3f7'

  return (
    <group position={[x, y, z]}>
      {/* 芯片主体 - 深色哑光外壳 */}
      <mesh
        onClick={onClick}
        onPointerOver={(e) => {
          e.stopPropagation()
          setHovered(true)
          onPointerOver?.()
        }}
        onPointerOut={(e) => {
          e.stopPropagation()
          setHovered(false)
          onPointerOut?.()
        }}
        castShadow
      >
        <boxGeometry args={dimensions} />
        <meshStandardMaterial
          color={hovered ? shellColorHover : shellColor}
          metalness={0.3}
          roughness={0.8}
        />
      </mesh>

      {/* 芯片顶部内嵌区域 - 中等及以上细节 */}
      {lodLevel !== 'low' && (
        <mesh position={[0, dimensions[1] / 2 - 0.001, 0]}>
          <boxGeometry args={[dimensions[0] * 0.92, 0.002, dimensions[2] * 0.92]} />
          <meshStandardMaterial
            color="#0d0d0d"
            metalness={0.2}
            roughness={0.9}
          />
        </mesh>
      )}

      {/* 芯片标识文字 - 仅高细节显示 */}
      {lodLevel === 'high' && (
        <Text
          position={[0, dimensions[1] / 2 + 0.002, 0]}
          fontSize={dimensions[0] * 0.35}
          color={labelColor}
          anchorX="center"
          anchorY="middle"
          rotation={[-Math.PI / 2, 0, 0]}
          material-depthTest={false}
        >
          {chipLabel}
        </Text>
      )}

      {/* 悬停时显示详细信息 - 仅高细节显示 */}
      {lodLevel === 'high' && hovered && (
        <Html center position={[0, 0.06, 0]}>
          <div style={{
            background: 'rgba(0,0,0,0.9)',
            color: '#fff',
            padding: '6px 12px',
            borderRadius: 6,
            fontSize: 12,
            whiteSpace: 'nowrap',
            border: `1px solid ${labelColor}`,
          }}>
            {CHIP_TYPE_NAMES[chip.type]}
            {chip.label && ` - ${chip.label}`}
          </div>
        </Html>
      )}
    </group>
  )
}

// 不同U高度板卡的配色方案
const BOARD_U_COLORS: Record<number, { main: string; mainHover: string; front: string; accent: string }> = {
  1: { main: '#4a5568', mainHover: '#38b2ac', front: '#2d3748', accent: '#63b3ed' },  // 灰蓝色 - 1U交换机/轻量设备
  2: { main: '#2c5282', mainHover: '#38b2ac', front: '#1a365d', accent: '#90cdf4' },  // 深蓝色 - 2U标准服务器
  4: { main: '#553c9a', mainHover: '#38b2ac', front: '#322659', accent: '#b794f4' },  // 紫色 - 4U GPU服务器
}

// Board模型 - 服务器/板卡，根据U高度显示不同样式
export const BoardModel: React.FC<{
  board: BoardConfig
  showChips?: boolean
  interactive?: boolean  // 是否可以交互（高亮和点击）
  targetOpacity?: number  // 目标透明度（会动画过渡）
  lodLevel?: LODLevel  // LOD 级别
  onDoubleClick?: () => void
  onClick?: () => void  // 单击显示详情
  onChipClick?: (chip: ChipConfig) => void  // 芯片点击
}> = ({ board, showChips = false, interactive = true, targetOpacity = 1.0, lodLevel = 'high', onDoubleClick, onClick, onChipClick }) => {
  const groupRef = useRef<THREE.Group>(null)
  const hoveredRef = useRef(false)
  const [, forceRender] = useState(0)
  const canHover = interactive  // 可交互时才能高亮

  // 跟踪是否应该渲染（动画完成后才隐藏）
  const [shouldRender, setShouldRender] = useState(targetOpacity > 0.01)

  useEffect(() => {
    if (targetOpacity > 0.01) {
      setShouldRender(true)
    }
  }, [targetOpacity])

  // 计算所有芯片的位置数据（供 InstancedMesh 使用）- 必须在 early return 之前
  const chipPinData = useMemo((): ChipPinData[] => {
    if (!showChips) return []
    return board.chips.map(chip => {
      const { x, y, z, dimensions } = getChipPosition(chip, board.chips.length, 0.004)
      return {
        position: new THREE.Vector3(x, y, z),
        dimensions
      }
    })
  }, [board.chips, showChips])

  // 使用 spring 监听动画完成
  useSpring({
    opacity: targetOpacity,
    config: { tension: 120, friction: 20 },
    onRest: () => {
      if (targetOpacity < 0.01) {
        setShouldRender(false)
      }
    }
  })

  // 动画完成后才真正隐藏
  if (!shouldRender) return null

  // 根据U高度获取颜色方案
  const uHeight = board.u_height
  const colorScheme = BOARD_U_COLORS[uHeight] || BOARD_U_COLORS[2]

  // 根据U高度计算实际3D尺寸 - 始终使用完整尺寸
  const { uHeight: uSize } = RACK_DIMENSIONS
  const width = BOARD_DIMENSIONS.width
  const height = uHeight * uSize * 0.9  // 留一点间隙
  const depth = BOARD_DIMENSIONS.depth

  // 高亮效果 - 使用ref实现即时响应
  const isHighlighted = canHover && hoveredRef.current

  // 高亮时整体提亮，使用accent颜色作为发光色，与板卡风格统一
  const highlightColor = isHighlighted ? colorScheme.accent : colorScheme.main
  const frontHighlightColor = isHighlighted ? colorScheme.accent : colorScheme.front
  const glowIntensity = isHighlighted ? 0.3 : 0
  const scale = isHighlighted ? 1.01 : 1.0

  return (
    <group>
      {showChips ? (
        // 显示芯片时：拟物化PCB板卡
        <>
          {/* PCB基板 - 多层结构 */}
          {/* 底层 - FR4基材 */}
          <mesh position={[0, -0.002, 0]} castShadow receiveShadow material={sharedMaterials.pcbBase}>
            <boxGeometry args={[width, 0.004, depth]} />
          </mesh>
          {/* 中间层 - 主PCB */}
          <mesh position={[0, 0.001, 0]} castShadow receiveShadow material={sharedMaterials.pcbMiddle}>
            <boxGeometry args={[width - 0.002, 0.004, depth - 0.002]} />
          </mesh>
          {/* 顶层 - 阻焊层(绿油) */}
          <mesh position={[0, 0.0035, 0]} material={sharedMaterials.pcbTop}>
            <boxGeometry args={[width - 0.004, 0.001, depth - 0.004]} />
          </mesh>

          {/* 铜走线 - 仅在高细节模式下显示 */}
          {lodLevel === 'high' && (
            <>
              {/* 横向走线 */}
              {Array.from({ length: 6 }).map((_, i) => {
                const zPos = -depth / 2 + 0.05 + i * (depth / 7)
                const lineWidth = i % 2 === 0 ? 0.004 : 0.002
                return (
                  <mesh key={`trace-h-${i}`} position={[0, 0.0042, zPos]} material={sharedMaterials.copperTrace}>
                    <boxGeometry args={[width - 0.06, 0.0008, lineWidth]} />
                  </mesh>
                )
              })}
              {/* 纵向走线 */}
              {Array.from({ length: 5 }).map((_, i) => {
                const xPos = -width / 2 + 0.06 + i * (width / 6)
                const lineWidth = i % 2 === 0 ? 0.003 : 0.0015
                return (
                  <mesh key={`trace-v-${i}`} position={[xPos, 0.0042, 0]} material={sharedMaterials.copperTrace}>
                    <boxGeometry args={[lineWidth, 0.0008, depth - 0.06]} />
                  </mesh>
                )
              })}
            </>
          )}

          {/* 过孔(Via) - 仅在高细节模式下显示 */}
          {lodLevel === 'high' && Array.from({ length: 8 }).map((_, i) => {
            const viaX = (Math.sin(i * 2.5) * 0.35) * width / 2
            const viaZ = (Math.cos(i * 3.1) * 0.35) * depth / 2
            return (
              <mesh key={`via-${i}`} position={[viaX, 0.0043, viaZ]} rotation={[-Math.PI / 2, 0, 0]} geometry={sharedGeometries.via} material={sharedMaterials.via} />
            )
          })}

          {/* 边缘金手指接口 - 中等及以上细节 */}
          {lodLevel !== 'low' && (
            <mesh position={[0, 0, -depth / 2 + 0.012]} material={sharedMaterials.goldFinger}>
              <boxGeometry args={[width * 0.6, 0.005, 0.018]} />
            </mesh>
          )}

          {/* 安装孔 - 仅在高细节模式下显示 */}
          {lodLevel === 'high' && [[-1, -1], [-1, 1], [1, -1], [1, 1]].map(([dx, dz], i) => (
            <mesh key={`mount-${i}`} position={[dx * (width / 2 - 0.025), 0.0042, dz * (depth / 2 - 0.025)]} rotation={[-Math.PI / 2, 0, 0]} geometry={sharedGeometries.mountHole} material={sharedMaterials.mountHole} />
          ))}

          {/* 使用 InstancedMesh 批量渲染引脚 */}
          <InstancedPins chips={chipPinData} lodLevel={lodLevel} />

          {/* 使用 InstancedMesh 批量渲染电路纹理 */}
          <InstancedCircuitTraces chips={chipPinData} lodLevel={lodLevel} />

          {/* 渲染芯片 - 放在PCB上，居中排布 */}
          {board.chips.map(chip => (
            <ChipModel
              key={chip.id}
              chip={chip}
              baseY={0.004}
              totalChips={board.chips.length}
              lodLevel={lodLevel}
              onClick={() => onChipClick?.(chip)}
            />
          ))}

          {/* 板卡丝印标识 - 高细节模式 */}
          {lodLevel === 'high' && (
            <Text
              position={[width / 2 - 0.06, 0.005, depth / 2 - 0.03]}
              fontSize={0.015}
              color="#c0c0c0"
              anchorX="center"
              anchorY="middle"
              rotation={[-Math.PI / 2, 0, 0]}
              material-depthTest={false}
            >
              {board.label}
            </Text>
          )}
          {/* 版本号丝印 - 高细节模式 */}
          {lodLevel === 'high' && (
            <Text
              position={[-width / 2 + 0.05, 0.005, depth / 2 - 0.03]}
              fontSize={0.008}
              color="#888888"
              anchorX="center"
              anchorY="middle"
              rotation={[-Math.PI / 2, 0, 0]}
              material-depthTest={false}
            >
              REV 1.0
            </Text>
          )}
        </>
      ) : (
        // 不显示芯片时：封闭的服务器盒子
        <>
          {/* 服务器主体 - 金属外壳 */}
          <group ref={groupRef} scale={scale}>
            <mesh
              onClick={canHover ? (e) => {
                e.stopPropagation()
                onClick?.()
              } : undefined}
              onDoubleClick={canHover ? onDoubleClick : undefined}
              onPointerOver={canHover ? (e) => {
                e.stopPropagation()
                hoveredRef.current = true
                forceRender(n => n + 1)
              } : undefined}
              onPointerOut={canHover ? (e) => {
                e.stopPropagation()
                hoveredRef.current = false
                forceRender(n => n + 1)
              } : undefined}
              castShadow={targetOpacity > 0.5}
              receiveShadow={targetOpacity > 0.5}
            >
              <boxGeometry args={[width, height, depth]} />
              <FadingMaterial
                targetOpacity={targetOpacity}
                color={highlightColor}
                emissive={colorScheme.accent}
                emissiveIntensity={glowIntensity}
                metalness={0.7}
                roughness={0.3}
              />
            </mesh>

            {/* 前面板 - 带有指示灯效果 */}
            <mesh position={[0, 0, depth / 2 + 0.001]}>
              <boxGeometry args={[width - 0.02, height - 0.005, 0.002]} />
              <FadingMaterial
                targetOpacity={targetOpacity}
                color={frontHighlightColor}
                emissive={colorScheme.accent}
                emissiveIntensity={glowIntensity}
                metalness={0.5}
                roughness={0.5}
              />
            </mesh>

            {/* U高度标识条 - 左侧彩色条纹，高亮时更亮 */}
            <mesh position={[-width / 2 + 0.008, 0, depth / 2 + 0.002]}>
              <boxGeometry args={[isHighlighted ? 0.016 : 0.012, height - 0.01, 0.001]} />
              <FadingBasicMaterial
                targetOpacity={targetOpacity}
                color={isHighlighted ? '#ffffff' : colorScheme.accent}
              />
            </mesh>

            {/* LED指示灯 - 高亮时更亮 */}
            <mesh position={[-width / 2 + 0.03, height / 2 - 0.015, depth / 2 + 0.003]}>
              <circleGeometry args={[isHighlighted ? 0.008 : 0.006, 16]} />
              <FadingBasicMaterial
                targetOpacity={targetOpacity}
                color={isHighlighted ? '#7fff7f' : '#52c41a'}
              />
            </mesh>
            <mesh position={[-width / 2 + 0.045, height / 2 - 0.015, depth / 2 + 0.003]}>
              <circleGeometry args={[isHighlighted ? 0.008 : 0.006, 16]} />
              <FadingBasicMaterial
                targetOpacity={targetOpacity}
                color={isHighlighted ? '#ffffff' : colorScheme.accent}
              />
            </mesh>

            {/* 板卡标签 - 只在可交互时显示（聚焦到Rack层级或更深），透明度低时隐藏 */}
            {interactive && targetOpacity > 0.3 && (
              <Text
                position={[0, 0, depth / 2 + 0.015]}
                fontSize={0.035}
                color="#ffffff"
                anchorX="center"
                anchorY="middle"
                outlineWidth={0.003}
                outlineColor="#000000"
                renderOrder={1}
                material-depthTest={false}
              >
                {board.label}
              </Text>
            )}
          </group>
        </>
      )}
    </group>
  )
}

// Switch模型 - 网络交换机，与服务器尺寸一致
export const SwitchModel: React.FC<{
  switchData: SwitchInstance
  targetOpacity?: number  // 目标透明度（会动画过渡）
  onClick?: () => void
}> = ({ switchData, targetOpacity = 1.0, onClick }) => {
  const [hovered, setHovered] = useState(false)

  // 跟踪是否应该渲染（动画完成后才隐藏）
  const [shouldRender, setShouldRender] = useState(targetOpacity > 0.01)

  useEffect(() => {
    if (targetOpacity > 0.01) {
      setShouldRender(true)
    }
  }, [targetOpacity])

  // 使用 spring 监听动画完成
  useSpring({
    opacity: targetOpacity,
    config: { tension: 120, friction: 20 },
    onRest: () => {
      if (targetOpacity < 0.01) {
        setShouldRender(false)
      }
    }
  })

  // 动画完成后才真正隐藏
  if (!shouldRender) return null

  // 根据U高度计算实际3D尺寸 - 与BoardModel保持一致
  const { uHeight: uSize } = RACK_DIMENSIONS
  const uHeight = switchData.u_height || 1
  const width = BOARD_DIMENSIONS.width
  const height = uHeight * uSize * 0.9  // 与Board一致
  const depth = BOARD_DIMENSIONS.depth  // 与Board深度一致

  // Switch专用颜色 - 鲜明的深蓝色外壳，与Board明显区分
  const shellColor = '#0a2540'  // 深海蓝
  const shellColorHover = '#0d3a5c'
  const accentColor = '#00d4ff'  // 明亮的青蓝色
  const frontPanelColor = '#061a2e'

  const isHighlighted = hovered
  const glowIntensity = isHighlighted ? 0.4 : 0.1  // 始终有轻微发光

  // 根据U高度计算端口行数和每行端口数
  const portRows = uHeight  // 1U=1行, 2U=2行, 4U=4行
  const portsPerRow = 12  // 每行12个端口
  const portSpacing = (width - 0.12) / portsPerRow  // 端口间距
  const rowSpacing = height / (portRows + 1)  // 行间距

  return (
    <group>
      {/* Switch主体 - 深蓝色金属外壳 */}
      <mesh
        onClick={(e) => {
          e.stopPropagation()
          onClick?.()
        }}
        onPointerOver={(e) => {
          e.stopPropagation()
          setHovered(true)
        }}
        onPointerOut={(e) => {
          e.stopPropagation()
          setHovered(false)
        }}
        castShadow={targetOpacity > 0.5}
        receiveShadow={targetOpacity > 0.5}
      >
        <boxGeometry args={[width, height, depth]} />
        <FadingMaterial
          targetOpacity={targetOpacity}
          color={isHighlighted ? shellColorHover : shellColor}
          emissive={accentColor}
          emissiveIntensity={glowIntensity}
          metalness={0.8}
          roughness={0.2}
        />
      </mesh>

      {/* 前面板 - 更深的颜色 */}
      <mesh position={[0, 0, depth / 2 + 0.001]}>
        <boxGeometry args={[width - 0.02, height - 0.005, 0.002]} />
        <FadingMaterial
          targetOpacity={targetOpacity}
          color={frontPanelColor}
          emissive={accentColor}
          emissiveIntensity={glowIntensity * 0.3}
          metalness={0.6}
          roughness={0.4}
        />
      </mesh>

      {/* 左侧青蓝色标识条 - Switch特有标识，更宽更亮 */}
      <mesh position={[-width / 2 + 0.01, 0, depth / 2 + 0.002]}>
        <boxGeometry args={[isHighlighted ? 0.02 : 0.016, height - 0.008, 0.001]} />
        <FadingBasicMaterial
          targetOpacity={targetOpacity}
          color={isHighlighted ? '#ffffff' : accentColor}
        />
      </mesh>

      {/* 网口区域 - 根据U高度显示多行端口 */}
      {Array.from({ length: portRows }).map((_, rowIdx) => {
        const rowY = -height / 2 + rowSpacing * (rowIdx + 1)
        return (
          <group key={`row-${rowIdx}`}>
            {Array.from({ length: portsPerRow }).map((_, portIdx) => {
              const portX = -width / 2 + 0.06 + portIdx * portSpacing
              const isActive = (rowIdx + portIdx) % 3 !== 0  // 部分端口激活
              return (
                <group key={`port-${rowIdx}-${portIdx}`}>
                  {/* 端口外框 */}
                  <mesh position={[portX, rowY, depth / 2 + 0.002]} geometry={sharedGeometries.portOuter}>
                    <FadingBasicMaterial targetOpacity={targetOpacity} color="#1a1a1a" />
                  </mesh>
                  {/* 端口内部 */}
                  <mesh position={[portX, rowY, depth / 2 + 0.0025]} geometry={sharedGeometries.portInner}>
                    <FadingBasicMaterial targetOpacity={targetOpacity} color="#0a0a0a" />
                  </mesh>
                  {/* 端口LED - 在端口上方 */}
                  <mesh position={[portX, rowY + 0.012, depth / 2 + 0.003]} geometry={sharedGeometries.portLed}>
                    <FadingBasicMaterial
                      targetOpacity={targetOpacity}
                      color={isActive ? (isHighlighted ? '#7fff7f' : '#00ff88') : '#333'}
                    />
                  </mesh>
                </group>
              )
            })}
          </group>
        )
      })}

      {/* 右侧状态LED区域 */}
      <group>
        {/* 电源LED */}
        <mesh position={[width / 2 - 0.025, height / 2 - 0.015, depth / 2 + 0.003]} geometry={sharedGeometries.ledLarge}>
          <FadingBasicMaterial
            targetOpacity={targetOpacity}
            color={isHighlighted ? '#7fff7f' : '#00ff88'}
          />
        </mesh>
        {/* 状态LED */}
        <mesh position={[width / 2 - 0.04, height / 2 - 0.015, depth / 2 + 0.003]} geometry={sharedGeometries.ledLarge}>
          <FadingBasicMaterial
            targetOpacity={targetOpacity}
            color={isHighlighted ? '#ffffff' : accentColor}
          />
        </mesh>
        {/* 活动LED */}
        <mesh position={[width / 2 - 0.055, height / 2 - 0.015, depth / 2 + 0.003]} geometry={sharedGeometries.ledLarge}>
          <FadingBasicMaterial
            targetOpacity={targetOpacity}
            color={isHighlighted ? '#ffff7f' : '#ffa500'}
          />
        </mesh>
      </group>

      {/* 顶部散热孔装饰（2U以上显示） */}
      {uHeight >= 2 && (
        <group position={[0, height / 2 - 0.008, depth / 2 + 0.002]}>
          {Array.from({ length: 6 }).map((_, i) => (
            <mesh key={`vent-${i}`} position={[width / 2 - 0.08 - i * 0.012, 0, 0]}>
              <boxGeometry args={[0.008, 0.004, 0.001]} />
              <FadingBasicMaterial targetOpacity={targetOpacity} color="#0a0a0a" />
            </mesh>
          ))}
        </group>
      )}

    </group>
  )
}


// Pod标签组件 - 支持悬停高亮
export const PodLabel: React.FC<{
  pod: PodConfig
  position: [number, number, number]
  onDoubleClick: () => void
  onHoverChange?: (hovered: boolean) => void
}> = ({ pod, position, onDoubleClick, onHoverChange }) => {
  const [hovered, setHovered] = useState(false)

  const handlePointerOver = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation()
    setHovered(true)
    onHoverChange?.(true)
  }

  const handlePointerOut = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation()
    setHovered(false)
    onHoverChange?.(false)
  }

  return (
    <group
      position={position}
      onDoubleClick={onDoubleClick}
      onPointerOver={handlePointerOver}
      onPointerOut={handlePointerOut}
    >
      {/* 背景板 */}
      <mesh>
        <planeGeometry args={[1.2, 0.4]} />
        <meshBasicMaterial
          color={hovered ? '#7a9fd4' : '#1890ff'}
          transparent
          opacity={hovered ? 1 : 0.9}
        />
      </mesh>
      {/* 文字 */}
      <Text
        position={[0, 0, 0.01]}
        fontSize={0.2}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
        fontWeight="bold"
      >
        {pod.label}
      </Text>
    </group>
  )
}


// ============================================
// 带动画的 Rack 渲染组件
// ============================================

export interface AnimatedRackProps {
  rack: RackConfig
  position: [number, number, number]
  targetOpacity: number
  isHighlighted: boolean
  rackWidth: number
  rackHeight: number
  rackDepth: number
  focusLevel: number
  podId: string
  onNavigateToPod: (podId: string) => void
  onNavigateToRack: (podId: string, rackId: string) => void
  onNodeClick?: (nodeType: 'rack', nodeId: string, label: string, info: Record<string, string | number>) => void
  onHoverChange: (hovered: boolean) => void
}

export const AnimatedRack: React.FC<AnimatedRackProps> = ({
  rack,
  position,
  targetOpacity,
  isHighlighted,
  rackWidth,
  rackHeight,
  rackDepth,
  focusLevel,
  podId,
  onNavigateToPod,
  onNavigateToRack,
  onNodeClick,
  onHoverChange,
}) => {
  // 跟踪是否应该渲染（动画完成后才隐藏）
  const [shouldRender, setShouldRender] = useState(targetOpacity > 0.01)

  // 当目标透明度变化时更新渲染状态
  useEffect(() => {
    if (targetOpacity > 0.01) {
      setShouldRender(true)
    }
  }, [targetOpacity])

  // 使用 spring 监听动画完成
  useSpring({
    opacity: targetOpacity,
    config: { tension: 120, friction: 20 },
    onRest: () => {
      if (targetOpacity < 0.01) {
        setShouldRender(false)
      }
    }
  })

  // 高亮效果参数 - 使用低饱和度高级灰蓝色调
  const rackFrameColor = isHighlighted ? '#2d3748' : '#1a1a1a'
  const rackPillarColor = isHighlighted ? '#3d4758' : '#333333'
  const rackGlowColor = '#4a6080'  // 深灰蓝色
  // emissiveIntensity > 1 配合 toneMapped={false} 触发 Bloom 效果
  const rackGlowIntensity = isHighlighted ? 1.8 : 0
  const rackScale = isHighlighted ? 1.01 : 1.0

  // 动画完成后才真正隐藏
  if (!shouldRender) return null

  return (
    <group position={position} scale={rackScale}>
      {/* 机柜框架 */}
      <group>
        {/* 机柜底座 */}
        <mesh position={[0, -rackHeight / 2 - 0.02, 0]} receiveShadow>
          <boxGeometry args={[rackWidth + 0.04, 0.04, rackDepth + 0.04]} />
          <FadingMaterial
            targetOpacity={targetOpacity}
            color={rackFrameColor}
            emissive={rackGlowColor}
            emissiveIntensity={rackGlowIntensity}
            toneMapped={false}
            metalness={0.6}
            roughness={0.3}
          />
        </mesh>

        {/* 机柜顶部 */}
        <mesh position={[0, rackHeight / 2 + 0.02, 0]} castShadow={targetOpacity > 0.5}>
          <boxGeometry args={[rackWidth + 0.04, 0.04, rackDepth + 0.04]} />
          <FadingMaterial
            targetOpacity={targetOpacity}
            color={rackFrameColor}
            emissive={rackGlowColor}
            emissiveIntensity={rackGlowIntensity}
            toneMapped={false}
            metalness={0.6}
            roughness={0.3}
          />
        </mesh>

        {/* 四个垂直立柱 */}
        {[
          [-rackWidth / 2, 0, -rackDepth / 2],
          [rackWidth / 2, 0, -rackDepth / 2],
          [-rackWidth / 2, 0, rackDepth / 2],
          [rackWidth / 2, 0, rackDepth / 2],
        ].map((pos, i) => (
          <mesh key={`pillar-${i}`} position={pos as [number, number, number]} castShadow={targetOpacity > 0.5}>
            <boxGeometry args={[0.02, rackHeight, 0.02]} />
            <FadingMaterial
              targetOpacity={targetOpacity}
              color={rackPillarColor}
              emissive={rackGlowColor}
              emissiveIntensity={rackGlowIntensity}
              toneMapped={false}
              metalness={0.5}
              roughness={0.4}
            />
          </mesh>
        ))}

        {/* 后面板 */}
        <mesh position={[0, 0, -rackDepth / 2 + 0.005]} receiveShadow>
          <boxGeometry args={[rackWidth - 0.04, rackHeight, 0.01]} />
          <FadingMaterial
            targetOpacity={targetOpacity * 0.7}
            color="#2a2a2a"
            metalness={0.3}
            roughness={0.7}
          />
        </mesh>

        {/* 左侧面板 */}
        <mesh position={[-rackWidth / 2 + 0.005, 0, 0]}>
          <boxGeometry args={[0.01, rackHeight, rackDepth - 0.04]} />
          <FadingMaterial
            targetOpacity={targetOpacity * 0.5}
            color="#2a2a2a"
            metalness={0.3}
            roughness={0.7}
          />
        </mesh>

        {/* 右侧面板 */}
        <mesh position={[rackWidth / 2 - 0.005, 0, 0]}>
          <boxGeometry args={[0.01, rackHeight, rackDepth - 0.04]} />
          <FadingMaterial
            targetOpacity={targetOpacity * 0.5}
            color="#2a2a2a"
            metalness={0.3}
            roughness={0.7}
          />
        </mesh>

        {/* 底部支脚 - 四个角落 */}
        {[
          [-rackWidth / 2 + 0.03, -rackHeight / 2 - 0.04, -rackDepth / 2 + 0.03],
          [rackWidth / 2 - 0.03, -rackHeight / 2 - 0.04, -rackDepth / 2 + 0.03],
          [-rackWidth / 2 + 0.03, -rackHeight / 2 - 0.04, rackDepth / 2 - 0.03],
          [rackWidth / 2 - 0.03, -rackHeight / 2 - 0.04, rackDepth / 2 - 0.03],
        ].map((pos, i) => (
          <mesh key={`foot-${i}`} position={pos as [number, number, number]} geometry={sharedGeometries.rackFoot}>
            <FadingMaterial
              targetOpacity={targetOpacity}
              color="#333333"
              metalness={0.5}
              roughness={0.5}
            />
          </mesh>
        ))}

        {/* 机柜标签 - 透明度低时隐藏 */}
        {targetOpacity > 0.3 && (
          <Text
            position={[0, rackHeight / 2 + 0.25, 0.01]}
            fontSize={0.2}
            color="#ffffff"
            anchorX="center"
            anchorY="middle"
            fontWeight="bold"
            outlineWidth={0.01}
            outlineColor="#000000"
            material-depthTest={false}
          >
            {rack.label}
          </Text>
        )}

        {/* 交互层 - 用于点击和双击，只在顶层和Pod层级有效 */}
        {focusLevel <= 1 && (
          <mesh
            visible={false}
            onClick={(e) => {
              e.stopPropagation()
              onNodeClick?.('rack', rack.id, rack.label, {
                '位置': `(${rack.position[0]}, ${rack.position[1]})`,
                '总U数': rack.total_u,
                '板卡数': rack.boards.length
              })
            }}
            onDoubleClick={() => {
              if (focusLevel === 0) {
                onNavigateToPod(podId)
              } else if (focusLevel === 1) {
                onNavigateToRack(podId, rack.id)
              }
            }}
            onPointerOver={() => onHoverChange(true)}
            onPointerOut={() => onHoverChange(false)}
          >
            <boxGeometry args={[rackWidth, rackHeight, rackDepth]} />
            <meshBasicMaterial transparent opacity={0} />
          </mesh>
        )}
      </group>
    </group>
  )
}
