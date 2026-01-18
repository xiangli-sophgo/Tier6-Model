import React, { useMemo, useRef, useState, useCallback, useEffect } from 'react'
import { Canvas } from '@react-three/fiber'
import { PerspectiveCamera } from '@react-three/drei'
import { EffectComposer, Bloom } from '@react-three/postprocessing'
import { Breadcrumb, Button, Tooltip } from 'antd'
import { ReloadOutlined, QuestionCircleOutlined } from '@ant-design/icons'
import * as THREE from 'three'
import {
  HierarchicalTopology,
  PodConfig,
  RackConfig,
  BoardConfig,
  SwitchInstance,
  ViewState,
  BreadcrumbItem,
  RACK_DIMENSIONS,
  CAMERA_PRESETS,
  KEYBOARD_SHORTCUTS,
} from '../../types'
import { sharedGeometries, sharedBasicMaterials, CameraAnimationTarget, NodePositions } from './shared'
import { CameraController } from './materials'
import { BoardModel, SwitchModel, PodLabel, AnimatedRack } from './models'

// ============================================
// Props 接口定义
// ============================================

interface Scene3DProps {
  topology: HierarchicalTopology | null
  viewState: ViewState
  breadcrumbs: BreadcrumbItem[]
  currentPod: PodConfig | null
  currentRack: RackConfig | null
  currentBoard: BoardConfig | null
  onNavigate: (nodeId: string) => void
  onNavigateToPod: (podId: string) => void
  onNavigateToRack: (podId: string, rackId: string) => void
  onNavigateBack: () => void
  onBreadcrumbClick: (index: number) => void
  canGoBack: boolean
  visible?: boolean  // 是否可见，隐藏时相机直接跳转不执行动画
  // 历史导航
  onNavigateHistoryBack?: () => void
  onNavigateHistoryForward?: () => void
  canGoHistoryBack?: boolean
  canGoHistoryForward?: boolean
  // 节点选择（显示详情）
  onNodeSelect?: (nodeType: 'pod' | 'rack' | 'board' | 'chip' | 'switch', nodeId: string, label: string, info: Record<string, string | number>, subType?: string) => void
}

// ============================================
// 统一场景组件 - 一次性渲染所有层级内容
// ============================================

const UnifiedScene: React.FC<{
  topology: HierarchicalTopology
  focusPath: string[]  // 当前聚焦路径 ['pod_0', 'rack_1', 'board_2']
  onNavigateToPod: (podId: string) => void
  onNavigateToRack: (podId: string, rackId: string) => void
  onNavigateToBoard: (boardId: string) => void
  onNodeClick?: (nodeType: 'pod' | 'rack' | 'board' | 'chip' | 'switch', nodeId: string, label: string, info: Record<string, string | number>, subType?: string) => void
  visible?: boolean  // 是否可见，隐藏时跳过动画
}> = ({ topology, focusPath, onNavigateToPod, onNavigateToRack, onNavigateToBoard, onNodeClick }) => {
  const [hoveredPodId, setHoveredPodId] = useState<string | null>(null)
  const [hoveredRackId, setHoveredRackId] = useState<string | null>(null)

  const rackSpacingX = 1.5
  const rackSpacingZ = 2
  const { uHeight, totalU, width: rackWidth, depth: rackDepth } = RACK_DIMENSIONS
  const rackHeight = totalU * uHeight

  // 计算Pod布局参数
  const { podSpacingX, podSpacingZ, podCols } = useMemo(() => {
    const firstPod = topology.pods[0]
    if (!firstPod) return { podSpacingX: 6, podSpacingZ: 4, podCols: 2 }

    const rackCols = firstPod.grid_size[1]
    const rackRows = firstPod.grid_size[0]
    const podWidth = rackCols * rackSpacingX + 2
    const podDepth = rackRows * rackSpacingZ + 1

    const totalPods = topology.pods.length
    let cols: number
    if (totalPods <= 2) cols = totalPods
    else if (totalPods <= 4) cols = 2
    else if (totalPods <= 6) cols = 3
    else if (totalPods <= 9) cols = 3
    else cols = 4

    return { podSpacingX: podWidth, podSpacingZ: podDepth, podCols: cols }
  }, [topology.pods])

  // 计算所有节点的世界坐标
  const nodePositions = useMemo((): NodePositions => {
    const pods = new Map<string, THREE.Vector3>()
    const racks = new Map<string, THREE.Vector3>()
    const boards = new Map<string, THREE.Vector3>()

    // 计算Pod网格位置
    const getPodGridPosition = (podIndex: number) => {
      const row = Math.floor(podIndex / podCols)
      const col = podIndex % podCols
      return { row, col }
    }

    // 首先计算所有Rack位置以找出中心
    let minX = Infinity, maxX = -Infinity
    let minZ = Infinity, maxZ = -Infinity

    topology.pods.forEach((pod, podIndex) => {
      const { row, col } = getPodGridPosition(podIndex)
      const podOffsetX = col * podSpacingX
      const podOffsetZ = row * podSpacingZ
      pod.racks.forEach(rack => {
        const x = podOffsetX + rack.position[1] * rackSpacingX
        const z = podOffsetZ + rack.position[0] * rackSpacingZ
        minX = Math.min(minX, x)
        maxX = Math.max(maxX, x)
        minZ = Math.min(minZ, z)
        maxZ = Math.max(maxZ, z)
      })
    })

    const centerX = (minX + maxX) / 2
    const centerZ = (minZ + maxZ) / 2

    // 设置所有节点位置
    topology.pods.forEach((pod, podIndex) => {
      const { row, col } = getPodGridPosition(podIndex)
      const podOffsetX = col * podSpacingX
      const podOffsetZ = row * podSpacingZ

      let podSumX = 0, podSumZ = 0, podCount = 0

      pod.racks.forEach(rack => {
        const rackX = podOffsetX + rack.position[1] * rackSpacingX - centerX
        const rackZ = podOffsetZ + rack.position[0] * rackSpacingZ - centerZ
        racks.set(rack.id, new THREE.Vector3(rackX, 0, rackZ))

        podSumX += rackX
        podSumZ += rackZ
        podCount++

        // 计算Board位置
        rack.boards.forEach(board => {
          const boardY = (board.u_position - 1) * uHeight + (board.u_height * uHeight) / 2 - rackHeight / 2
          boards.set(board.id, new THREE.Vector3(rackX, boardY, rackZ))
        })
      })

      // Pod中心
      if (podCount > 0) {
        pods.set(pod.id, new THREE.Vector3(podSumX / podCount, 0, podSumZ / podCount))
      }
    })

    return { pods, racks, boards }
  }, [topology, podSpacingX, podSpacingZ, podCols, uHeight, rackHeight])


  // 获取节点目标透明度 - 非聚焦内容完全隐藏
  const getTargetOpacity = useCallback((nodeId: string, nodeType: 'pod' | 'rack' | 'board' | 'switch'): number => {
    if (focusPath.length === 0) return 1.0 // 顶层全显示

    if (nodeType === 'rack') {
      if (focusPath.length === 1) {
        // 聚焦Pod，只显示该Pod下的Rack，其他Pod的Rack完全隐藏
        const pod = topology.pods.find(p => p.id === focusPath[0])
        const isInPod = pod?.racks.some(r => r.id === nodeId)
        return isInPod ? 1.0 : 0
      }
      if (focusPath.length === 2) {
        // 聚焦Rack，只显示聚焦的Rack
        return focusPath[1] === nodeId ? 1.0 : 0
      }
      if (focusPath.length >= 3) {
        // 聚焦Board，所有Rack都完全隐藏（只显示Board）
        return 0
      }
    }
    if (nodeType === 'board') {
      if (focusPath.length === 1) {
        // 聚焦Pod，只显示该Pod下的Board，其他Pod的Board隐藏
        const pod = topology.pods.find(p => p.id === focusPath[0])
        const isInPod = pod?.racks.some(r => r.boards.some(b => b.id === nodeId))
        return isInPod ? 1.0 : 0
      }
      if (focusPath.length === 2) {
        // 聚焦Rack，只显示该Rack下的Board
        const pod = topology.pods.find(p => p.id === focusPath[0])
        const rack = pod?.racks.find(r => r.id === focusPath[1])
        const isInRack = rack?.boards.some(b => b.id === nodeId)
        return isInRack ? 1.0 : 0
      }
      if (focusPath.length >= 3) {
        // 聚焦Board，只显示聚焦的Board
        return focusPath[2] === nodeId ? 1.0 : 0
      }
    }
    if (nodeType === 'switch') {
      // Switch的nodeId格式为 `${rack.id}/switch`
      const rackId = nodeId.replace('/switch', '')
      if (focusPath.length === 1) {
        // 聚焦Pod，显示该Pod下所有Rack的Switch
        const pod = topology.pods.find(p => p.id === focusPath[0])
        return pod?.racks.some(r => r.id === rackId) ? 1.0 : 0
      }
      if (focusPath.length === 2) {
        // 聚焦Rack，只显示该Rack的Switch
        return focusPath[1] === rackId ? 1.0 : 0
      }
      // Board层级及更深，隐藏所有Switch
      return 0
    }
    return 1.0
  }, [focusPath, topology])

  // 计算所有节点的目标透明度
  const targetOpacities = useMemo(() => {
    const opacities = new Map<string, number>()
    topology.pods.forEach(pod => {
      pod.racks.forEach(rack => {
        opacities.set(rack.id, getTargetOpacity(rack.id, 'rack'))
        rack.boards.forEach(board => {
          opacities.set(board.id, getTargetOpacity(board.id, 'board'))
        })
        // 添加switch的opacity
        const switchId = `${rack.id}/switch`
        opacities.set(switchId, getTargetOpacity(switchId, 'switch'))
      })
    })
    return opacities
  }, [topology, getTargetOpacity])

  // 当前聚焦层级
  const focusLevel = focusPath.length

  return (
    <group>
      {/* 灯光设置 */}
      <ambientLight intensity={0.4} />
      <directionalLight
        position={[10, 15, 10]}
        intensity={1}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
      />
      <directionalLight position={[-5, 10, -5]} intensity={0.3} />
      <pointLight position={[0, 5, 0]} intensity={0.5} />

      {/* 渲染所有Pod */}
      {topology.pods.map(pod => {
        const podCenter = nodePositions.pods.get(pod.id)
        if (!podCenter) return null

        const isPodHighlighted = hoveredPodId === pod.id

        return (
          <group key={pod.id}>
            {/* Pod标签 - 只在顶层或聚焦该Pod时显示 */}
            {(focusLevel === 0 || (focusLevel === 1 && focusPath[0] === pod.id)) && (
              <PodLabel
                pod={pod}
                position={[podCenter.x, rackHeight / 2 + 0.5, podCenter.z]}
                onDoubleClick={() => onNavigateToPod(pod.id)}
                onHoverChange={(hovered) => setHoveredPodId(hovered ? pod.id : null)}
              />
            )}

            {/* 渲染该Pod下的所有Rack - 使用动画组件 */}
            {pod.racks.map(rack => {
              const rackPos = nodePositions.racks.get(rack.id)
              if (!rackPos) return null

              const rackTargetOpacity = targetOpacities.get(rack.id) ?? 1.0
              // 只在顶层和Pod层级时才高亮Rack，在Rack层级及更深时不高亮
              const isRackHighlighted = focusLevel === 0 ? isPodHighlighted : (focusLevel === 1 && hoveredRackId === rack.id)

              return (
                <AnimatedRack
                  key={rack.id}
                  rack={rack}
                  position={[rackPos.x, rackPos.y, rackPos.z]}
                  targetOpacity={rackTargetOpacity}
                  isHighlighted={isRackHighlighted}
                  rackWidth={rackWidth}
                  rackHeight={rackHeight}
                  rackDepth={rackDepth}
                  focusLevel={focusLevel}
                  podId={pod.id}
                  onNavigateToPod={onNavigateToPod}
                  onNavigateToRack={onNavigateToRack}
                  onNodeClick={onNodeClick}
                  onHoverChange={(hovered) => {
                    if (focusLevel === 0) setHoveredPodId(hovered ? pod.id : null)
                    else if (focusLevel === 1) setHoveredRackId(hovered ? rack.id : null)
                  }}
                />
              )
            })}
          </group>
        )
      })}

      {/* 独立渲染所有Board - 不受Rack透明度影响 */}
      {topology.pods.map(pod => (
        <group key={`boards-${pod.id}`}>
          {pod.racks.map(rack => {
            const rackPos = nodePositions.racks.get(rack.id)
            if (!rackPos) return null

            // 是否显示Board详情（聚焦到Rack级别或更深）
            const showBoardDetails = focusLevel >= 2 && focusPath[1] === rack.id

            return rack.boards.map(board => {
              const boardY = (board.u_position - 1) * uHeight + (board.u_height * uHeight) / 2 - rackHeight / 2
              const boardOpacity = targetOpacities.get(board.id) ?? 1.0

              // 是否显示芯片（聚焦到Board级别）
              const showChips = focusLevel >= 3 && focusPath[2] === board.id

              // BoardModel 内部处理动画完成后的隐藏
              return (
                <group key={`${board.id}-${board.label}`} position={[rackPos.x, rackPos.y + boardY, rackPos.z]}>
                  <BoardModel
                    board={board}
                    showChips={showChips}
                    interactive={showBoardDetails}
                    targetOpacity={boardOpacity}
                    onDoubleClick={() => onNavigateToBoard(board.id)}
                    onClick={() => onNodeClick?.('board', board.id, board.label, {
                      'U位置': board.u_position,
                      'U高度': board.u_height,
                      '芯片数': board.chips.length
                    })}
                    onChipClick={(chip) => onNodeClick?.('chip', chip.id, chip.label || chip.type.toUpperCase(), {
                      '类型': chip.type.toUpperCase(),
                      '位置': `(${chip.position[0]}, ${chip.position[1]})`
                    }, chip.type)}
                  />
                </group>
              )
            })
          })}
        </group>
      ))}

      {/* 渲染Rack层级的Switch - 在Pod和Rack层级显示，Board层级隐藏 */}
      {topology.switch_config?.inter_board?.enabled && topology.pods.map(pod => (
        <group key={`switches-${pod.id}`}>
          {pod.racks.map(rack => {
            const rackPos = nodePositions.racks.get(rack.id)
            if (!rackPos) return null

            // 使用统一的targetOpacities获取Switch透明度（支持动画）
            const switchId = `${rack.id}/switch`
            const switchTargetOpacity = targetOpacities.get(switchId) ?? 0
            // 注意：不在这里检查透明度，让SwitchModel内部处理动画完成后的隐藏

            // 获取该Rack下的所有Switch（使用后端计算的u_position）
            const rackSwitches = topology.switches?.filter(
              sw => sw.hierarchy_level === 'inter_board' && sw.parent_id === rack.id
            ) || []

            if (rackSwitches.length === 0) return null

            // 汇总显示：使用第一个Switch的u_position和配置的高度
            const firstSwitch = rackSwitches[0]
            const switchUPosition = firstSwitch.u_position || 1
            const switchUHeight = firstSwitch.u_height || 1  // 使用配置的高度，不累加
            const switchY = (switchUPosition - 1) * uHeight + (switchUHeight * uHeight) / 2 - rackHeight / 2

            // 创建汇总的Switch用于显示
            const summarySwitch: SwitchInstance = {
              id: `${rack.id}/switch_summary`,
              type_id: 'summary',
              layer: 'leaf',
              hierarchy_level: 'inter_board',
              parent_id: rack.id,
              label: `Switch ×${rackSwitches.length}`,
              uplink_ports_used: 0,
              downlink_ports_used: 0,
              inter_ports_used: 0,
              u_height: switchUHeight,
              u_position: switchUPosition
            }

            // 构建每个Switch的详细信息
            const switchInfoObj: Record<string, string | number> = {
              '所属Rack': rack.label,
              'U位置': switchUPosition,
              'U高度': switchUHeight,
            }
            // 添加每个Switch的详情
            rackSwitches.forEach((sw, idx) => {
              switchInfoObj[`[${idx + 1}] ${sw.label}`] = `上行:${sw.uplink_ports_used} 下行:${sw.downlink_ports_used} 互联:${sw.inter_ports_used}`
            })

            return (
              <group key={`${rack.id}/switch`} position={[rackPos.x, rackPos.y + switchY, rackPos.z]}>
                <SwitchModel
                  switchData={summarySwitch}
                  targetOpacity={switchTargetOpacity}
                  onClick={() => onNodeClick?.('switch', rackSwitches[0].id, `${rack.label} Switch ×${rackSwitches.length}`, switchInfoObj)}
                />
              </group>
            )
          })}
        </group>
      ))}

      {/* 地面 */}
      <mesh
        position={[0, -rackHeight / 2 - 0.06, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        receiveShadow
        geometry={sharedGeometries.groundPlane}
        material={sharedBasicMaterials.ground}
      />
      {/* 地面网格线 */}
      <gridHelper
        args={[50, 50, '#bbb', '#ddd']}
        position={[0, -rackHeight / 2 - 0.05, 0]}
      />
    </group>
  )
}


// ============================================
// 导航覆盖层组件
// ============================================

const NavigationOverlay: React.FC<{
  breadcrumbs: BreadcrumbItem[]
  onBreadcrumbClick: (index: number) => void
  onBack: () => void
  canGoBack: boolean
}> = ({ breadcrumbs, onBreadcrumbClick }) => {
  return (
    <div style={{
      position: 'absolute',
      top: 16,
      left: 16,
      zIndex: 100,
      background: 'rgba(255, 255, 255, 0.95)',
      padding: '8px 16px',
      borderRadius: 8,
      boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
    }}>
      <Breadcrumb
        items={breadcrumbs.map((item, index) => ({
          title: (
            <a
              onClick={(e) => {
                e.preventDefault()
                onBreadcrumbClick(index)
              }}
              style={{
                cursor: index < breadcrumbs.length - 1 ? 'pointer' : 'default',
                color: index < breadcrumbs.length - 1 ? '#1890ff' : 'rgba(0, 0, 0, 0.88)',
                fontWeight: index === breadcrumbs.length - 1 ? 500 : 400,
              }}
            >
              {item.label}
            </a>
          ),
        }))}
      />
    </div>
  )
}

// ============================================
// 主Scene3D组件
// ============================================

export const Scene3D: React.FC<Scene3DProps> = ({
  topology,
  viewState,
  breadcrumbs,
  currentPod,
  currentRack,
  currentBoard,
  onNavigate,
  onNavigateToPod,
  onNavigateToRack,
  onNavigateBack,
  onBreadcrumbClick,
  canGoBack,
  visible = true,
  onNodeSelect,
}) => {
  // 用于强制重置相机位置的 key
  const [resetKey, setResetKey] = useState(0)
  // 是否显示快捷键帮助
  const [showKeyboardHelp, setShowKeyboardHelp] = useState(false)
  // 初始相机位置（只用于首次渲染）
  const initialCameraPositionRef = useRef<[number, number, number] | null>(null)

  // 重置视图（相机位置）
  const handleResetView = useCallback(() => {
    setResetKey(k => k + 1)
  }, [])

  // 键盘快捷键处理 - 仅处理3D视图特有的快捷键（R重置视角、?帮助）
  // ESC、Backspace、方向键已在App.tsx中全局处理
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // 如果正在输入框中则忽略
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return
      }

      // R - 重置相机视角 (同时检查 key 和 code 以兼容输入法)
      if (KEYBOARD_SHORTCUTS.resetView.includes(e.code) || e.key === 'r' || e.key === 'R') {
        e.preventDefault()
        setResetKey(k => k + 1)
        return
      }

      // ? - 显示/隐藏快捷键帮助
      if (e.code === 'Slash' && e.shiftKey) {
        e.preventDefault()
        setShowKeyboardHelp(prev => !prev)
        return
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  // 计算所有节点的世界坐标（与 UnifiedScene 保持一致）
  const nodePositions = useMemo(() => {
    if (!topology) return { pods: new Map(), racks: new Map(), boards: new Map() }

    const rackSpacingX = 1.5
    const rackSpacingZ = 2
    const { uHeight, totalU } = RACK_DIMENSIONS
    const rackHeight = totalU * uHeight

    const pods = new Map<string, THREE.Vector3>()
    const racks = new Map<string, THREE.Vector3>()
    const boards = new Map<string, THREE.Vector3>()

    // 计算Pod布局参数
    const firstPod = topology.pods[0]
    let podSpacingX = 6, podSpacingZ = 4, podCols = 2
    if (firstPod) {
      const rackCols = firstPod.grid_size[1]
      const rackRows = firstPod.grid_size[0]
      podSpacingX = rackCols * rackSpacingX + 2
      podSpacingZ = rackRows * rackSpacingZ + 1

      const totalPods = topology.pods.length
      if (totalPods <= 2) podCols = totalPods
      else if (totalPods <= 4) podCols = 2
      else if (totalPods <= 6) podCols = 3
      else if (totalPods <= 9) podCols = 3
      else podCols = 4
    }

    const getPodGridPosition = (podIndex: number) => {
      const row = Math.floor(podIndex / podCols)
      const col = podIndex % podCols
      return { row, col }
    }

    // 首先计算所有Rack位置以找出中心
    let minX = Infinity, maxX = -Infinity
    let minZ = Infinity, maxZ = -Infinity

    topology.pods.forEach((pod, podIndex) => {
      const { row, col } = getPodGridPosition(podIndex)
      const podOffsetX = col * podSpacingX
      const podOffsetZ = row * podSpacingZ
      pod.racks.forEach(rack => {
        const x = podOffsetX + rack.position[1] * rackSpacingX
        const z = podOffsetZ + rack.position[0] * rackSpacingZ
        minX = Math.min(minX, x)
        maxX = Math.max(maxX, x)
        minZ = Math.min(minZ, z)
        maxZ = Math.max(maxZ, z)
      })
    })

    const centerX = (minX + maxX) / 2
    const centerZ = (minZ + maxZ) / 2

    // 设置所有节点位置
    topology.pods.forEach((pod, podIndex) => {
      const { row, col } = getPodGridPosition(podIndex)
      const podOffsetX = col * podSpacingX
      const podOffsetZ = row * podSpacingZ

      let podSumX = 0, podSumZ = 0, podCount = 0

      pod.racks.forEach(rack => {
        const rackX = podOffsetX + rack.position[1] * rackSpacingX - centerX
        const rackZ = podOffsetZ + rack.position[0] * rackSpacingZ - centerZ
        racks.set(rack.id, new THREE.Vector3(rackX, 0, rackZ))

        podSumX += rackX
        podSumZ += rackZ
        podCount++

        // 计算Board位置
        rack.boards.forEach(board => {
          const boardY = (board.u_position - 1) * uHeight + (board.u_height * uHeight) / 2 - rackHeight / 2
          boards.set(board.id, new THREE.Vector3(rackX, boardY, rackZ))
        })
      })

      // Pod中心
      if (podCount > 0) {
        pods.set(pod.id, new THREE.Vector3(podSumX / podCount, 0, podSumZ / podCount))
      }
    })

    return { pods, racks, boards }
  }, [topology])

  // 根据当前视图状态计算相机目标位置和观察点
  const cameraTarget = useMemo((): CameraAnimationTarget => {
    if (viewState.path.length === 0) {
      // 数据中心顶层视图
      const basePreset = CAMERA_PRESETS['pod']
      const lookAt = new THREE.Vector3(0, 0, 0)
      if (topology) {
        const podCount = topology.pods.length
        const racksPerPod = topology.pods[0]?.racks.length || 4
        const scaleFactor = Math.max(1, Math.sqrt(podCount * racksPerPod / 4))
        return {
          position: new THREE.Vector3(basePreset[0] * scaleFactor, basePreset[1] * scaleFactor, basePreset[2] * scaleFactor),
          lookAt
        }
      }
      return { position: new THREE.Vector3(basePreset[0], basePreset[1], basePreset[2]), lookAt }
    }

    if (viewState.path.length === 1 && currentPod && topology) {
      const podCenter = nodePositions.pods.get(currentPod.id)
      if (podCenter) {
        if (topology.pods.length === 1) {
          const basePreset = CAMERA_PRESETS['pod']
          const racksPerPod = topology.pods[0]?.racks.length || 4
          const scaleFactor = Math.max(1, Math.sqrt(racksPerPod / 4))
          return {
            position: new THREE.Vector3(basePreset[0] * scaleFactor, basePreset[1] * scaleFactor, basePreset[2] * scaleFactor),
            lookAt: podCenter.clone()
          }
        }
        const racksCount = currentPod.racks.length
        const distance = 3 + racksCount * 0.5
        return {
          position: new THREE.Vector3(podCenter.x + distance, distance * 0.8, podCenter.z + distance),
          lookAt: podCenter.clone()
        }
      }
    }

    if (viewState.path.length === 2) {
      const rackId = currentRack?.id || viewState.path[1]
      const rackPos = nodePositions.racks.get(rackId)
      if (rackPos) {
        return {
          position: new THREE.Vector3(rackPos.x + 1.0, rackPos.y + 0.8, rackPos.z + 3.0),
          lookAt: new THREE.Vector3(rackPos.x, rackPos.y + 0.10, rackPos.z)
        }
      }
    }

    if (viewState.path.length >= 3) {
      const boardId = currentBoard?.id || viewState.path[2]
      const boardPos = nodePositions.boards.get(boardId)
      if (boardPos) {
        return {
          position: new THREE.Vector3(boardPos.x + 0.5, boardPos.y + 1.0, boardPos.z + 0.8),
          lookAt: new THREE.Vector3(boardPos.x, boardPos.y, boardPos.z)
        }
      }
    }

    // 默认
    const basePreset = CAMERA_PRESETS[viewState.level]
    return { position: new THREE.Vector3(basePreset[0], basePreset[1], basePreset[2]), lookAt: new THREE.Vector3(0, 0, 0) }
  }, [viewState.path, viewState.level, topology, currentPod, currentRack, currentBoard, nodePositions, resetKey])

  // 记录初始相机位置（只在首次计算cameraTarget时设置）
  if (initialCameraPositionRef.current === null) {
    initialCameraPositionRef.current = [cameraTarget.position.x, cameraTarget.position.y, cameraTarget.position.z]
  }

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* 3D Canvas */}
      <Canvas shadows>
        {/* PerspectiveCamera只使用初始位置，后续由CameraController控制 */}
        <PerspectiveCamera
          makeDefault
          position={initialCameraPositionRef.current}
          fov={50}
        />

        {/* 使用 CameraController 实现平滑动画 - 由它控制相机位置 */}
        <CameraController
          target={cameraTarget}
          baseDuration={1.2}
          resetTrigger={resetKey}
          visible={visible}
        />

        <color attach="background" args={['#f0f2f5']} />

        {/* 使用统一场景渲染所有层级，通过相机移动和透明度控制实现层级切换 */}
        {topology && (
          <UnifiedScene
            topology={topology}
            focusPath={viewState.path}
            onNavigateToPod={onNavigateToPod}
            onNavigateToRack={onNavigateToRack}
            onNavigateToBoard={onNavigate}
            onNodeClick={onNodeSelect}
            visible={visible}
          />
        )}

        {/* Bloom 后处理效果 - 实现高级发光 */}
        <EffectComposer>
          <Bloom
            luminanceThreshold={0.9}
            luminanceSmoothing={0.4}
            intensity={0.8}
            mipmapBlur
          />
        </EffectComposer>
      </Canvas>

      {/* 导航覆盖层 */}
      <NavigationOverlay
        breadcrumbs={breadcrumbs}
        onBreadcrumbClick={onBreadcrumbClick}
        onBack={onNavigateBack}
        canGoBack={canGoBack}
      />

      {/* 右上角按钮组 */}
      <div style={{
        position: 'absolute',
        top: 16,
        right: 16,
        display: 'flex',
        gap: 8,
      }}>
        <Tooltip title="快捷键帮助 (?)">
          <Button
            icon={<QuestionCircleOutlined />}
            onClick={() => setShowKeyboardHelp(prev => !prev)}
          />
        </Tooltip>
        <Tooltip title="重置视图 (R)">
          <Button
            icon={<ReloadOutlined />}
            onClick={handleResetView}
          />
        </Tooltip>
      </div>

      {/* 快捷键帮助面板 - 点击空白处关闭 */}
      {showKeyboardHelp && (
        <>
          {/* 透明遮罩层，点击关闭 */}
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              zIndex: 199,
            }}
            onClick={() => setShowKeyboardHelp(false)}
          />
          {/* 帮助面板 */}
          <div style={{
            position: 'absolute',
            top: 60,
            right: 16,
            background: 'rgba(255, 255, 255, 0.98)',
            padding: '16px 20px',
            borderRadius: 8,
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            fontSize: 13,
            zIndex: 200,
            minWidth: 180,
          }}>
            <div style={{ fontWeight: 600, marginBottom: 12, color: '#1890ff' }}>键盘快捷键</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '8px 16px' }}>
              <kbd style={{ background: '#f0f0f0', padding: '2px 6px', borderRadius: 4, fontFamily: 'monospace' }}>Esc</kbd>
              <span>返回上一级</span>
              <kbd style={{ background: '#f0f0f0', padding: '2px 6px', borderRadius: 4, fontFamily: 'monospace' }}>←</kbd>
              <span>历史后退</span>
              <kbd style={{ background: '#f0f0f0', padding: '2px 6px', borderRadius: 4, fontFamily: 'monospace' }}>→</kbd>
              <span>历史前进</span>
              <kbd style={{ background: '#f0f0f0', padding: '2px 6px', borderRadius: 4, fontFamily: 'monospace' }}>R</kbd>
              <span>重置视图</span>
              <kbd style={{ background: '#f0f0f0', padding: '2px 6px', borderRadius: 4, fontFamily: 'monospace' }}>?</kbd>
              <span>显示/隐藏帮助</span>
            </div>
            <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid #eee', color: '#888', fontSize: 11 }}>
              鼠标操作: 左键旋转 / 右键平移 / 滚轮缩放
            </div>
          </div>
        </>
      )}

      {/* 左下角操作提示 */}
      <div style={{
        position: 'absolute',
        bottom: 16,
        left: 16,
        background: 'rgba(0, 0, 0, 0.7)',
        color: '#fff',
        padding: '8px 12px',
        borderRadius: 8,
        fontSize: 12,
      }}>
        {viewState.level === 'pod' && viewState.path.length === 0 && '单击查看详情 | 双击进入内部视图 | 按 ? 查看快捷键'}
        {viewState.level === 'pod' && viewState.path.length === 1 && '单击查看详情 | 双击机柜进入内部视图 | Esc返回'}
        {viewState.level === 'rack' && '单击查看详情 | 双击板卡查看芯片布局 | Esc返回'}
        {(viewState.level === 'board' || viewState.level === 'chip') && 'Esc返回上级 | R重置视图'}
      </div>
    </div>
  )
}
