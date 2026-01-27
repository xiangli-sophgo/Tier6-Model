import React, { useMemo, useRef, useState, useCallback, useEffect } from 'react'
import { Canvas } from '@react-three/fiber'
import { PerspectiveCamera } from '@react-three/drei'
import { EffectComposer, Bloom } from '@react-three/postprocessing'
import { RotateCcw, HelpCircle, LayoutGrid, Network } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import {
  Breadcrumb,
  BreadcrumbItem as ShadcnBreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb'
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
import { sharedGeometries, sharedBasicMaterials, CameraAnimationTarget } from './shared'
import { CameraController } from './materials'
import { BoardModel, SwitchModel, PodLabel, AnimatedRack } from './models'
import { useNodePositions } from '../../hooks/useNodePositions'

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
  // 视图切换
  viewMode?: '3d' | 'topology'
  onViewModeChange?: (mode: '3d' | 'topology') => void
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

  const { uHeight, totalU, width: rackWidth, depth: rackDepth } = RACK_DIMENSIONS
  const rackHeight = totalU * uHeight

  // 使用自定义 hook 计算所有节点的世界坐标
  const nodePositions = useNodePositions(topology)


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
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
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
                    focusLevel={focusLevel}
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
  viewMode?: '3d' | 'topology'
  onViewModeChange?: (mode: '3d' | 'topology') => void
}> = ({ breadcrumbs, onBreadcrumbClick, viewMode, onViewModeChange }) => {
  return (
    <div className="absolute top-4 left-4 z-[100] bg-white/95 px-4 py-2 rounded-lg shadow-md flex items-center gap-3">
      {/* 视图切换器 - 自定义 Segmented */}
      {viewMode && onViewModeChange && (
        <div className="flex rounded-md border border-gray-200 overflow-hidden">
          <button
            onClick={() => onViewModeChange('3d')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-sm transition-colors ${
              viewMode === '3d'
                ? 'bg-blue-500 text-white'
                : 'bg-white text-gray-600 hover:bg-gray-50'
            }`}
          >
            <LayoutGrid className="h-3.5 w-3.5" />
            3D视图
          </button>
          <button
            onClick={() => onViewModeChange('topology')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-sm transition-colors border-l border-gray-200 ${
              viewMode === 'topology'
                ? 'bg-blue-500 text-white'
                : 'bg-white text-gray-600 hover:bg-gray-50'
            }`}
          >
            <Network className="h-3.5 w-3.5" />
            2D视图
          </button>
        </div>
      )}
      {/* 面包屑导航 */}
      <Breadcrumb>
        <BreadcrumbList>
          {breadcrumbs.map((item, index) => (
            <React.Fragment key={index}>
              <ShadcnBreadcrumbItem>
                <BreadcrumbLink
                  href="#"
                  onClick={(e) => {
                    e.preventDefault()
                    onBreadcrumbClick(index)
                  }}
                  className={index < breadcrumbs.length - 1 ? 'text-blue-500 hover:underline' : 'text-gray-900 font-medium cursor-default'}
                >
                  {item.label}
                </BreadcrumbLink>
              </ShadcnBreadcrumbItem>
              {index < breadcrumbs.length - 1 && <BreadcrumbSeparator />}
            </React.Fragment>
          ))}
        </BreadcrumbList>
      </Breadcrumb>
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
  viewMode,
  onViewModeChange,
}) => {
  // 用于强制重置相机位置的 key
  const [resetKey, setResetKey] = useState(0)
  // 是否显示快捷键帮助
  const [showKeyboardHelp, setShowKeyboardHelp] = useState(false)
  // 初始相机位置（只用于首次渲染）
  const initialCameraPositionRef = useRef<[number, number, number] | null>(null)
  // WebGL 上下文状态
  const [webglContextLost, setWebglContextLost] = useState(false)

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

  // 使用自定义 hook 计算所有节点的世界坐标
  const nodePositions = useNodePositions(topology)

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
  }, [viewState.path, viewState.level, topology, currentPod, currentRack, currentBoard, nodePositions])

  // 记录初始相机位置（只在首次计算cameraTarget时设置）
  if (initialCameraPositionRef.current === null) {
    initialCameraPositionRef.current = [cameraTarget.position.x, cameraTarget.position.y, cameraTarget.position.z]
  }

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* WebGL 上下文丢失提示 */}
      {webglContextLost && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          zIndex: 1000,
          background: 'rgba(0, 0, 0, 0.8)',
          color: '#fff',
          padding: '20px 32px',
          borderRadius: 8,
          textAlign: 'center',
        }}>
          <div style={{ fontSize: 16, marginBottom: 12 }}>3D 渲染上下文丢失</div>
          <div style={{ fontSize: 13, color: '#ccc', marginBottom: 16 }}>
            GPU 资源可能被其他应用占用
          </div>
          <Button onClick={() => window.location.reload()}>刷新页面</Button>
        </div>
      )}

      {/* 3D Canvas */}
      <Canvas
        shadows
        onCreated={({ gl }) => {
          // 监听上下文丢失事件
          gl.domElement.addEventListener('webglcontextlost', (event) => {
            event.preventDefault()
            console.error('WebGL 上下文丢失')
            setWebglContextLost(true)
          })

          // 监听上下文恢复事件
          gl.domElement.addEventListener('webglcontextrestored', () => {
            console.log('WebGL 上下文已恢复')
            setWebglContextLost(false)
            setResetKey(k => k + 1) // 触发场景重新渲染
          })
        }}
      >
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
        viewMode={viewMode}
        onViewModeChange={onViewModeChange}
      />

      {/* 右上角按钮组 */}
      <TooltipProvider>
        <div className="absolute top-4 right-4 flex gap-2">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                onClick={() => setShowKeyboardHelp(prev => !prev)}
              >
                <HelpCircle className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>快捷键帮助 (?)</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                onClick={handleResetView}
              >
                <RotateCcw className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>重置视图 (R)</TooltipContent>
          </Tooltip>
        </div>
      </TooltipProvider>

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
