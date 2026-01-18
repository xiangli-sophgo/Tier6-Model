import * as THREE from 'three'
import { PIN_CONFIG, CIRCUIT_TRACE_CONFIG } from '../../types'

// ============================================
// 模块级别的状态缓存（组件卸载后保留）
// ============================================
export let lastCameraState: {
  position: THREE.Vector3
  lookAt: THREE.Vector3
} | null = null

export function setLastCameraState(state: { position: THREE.Vector3; lookAt: THREE.Vector3 } | null) {
  lastCameraState = state
}

// ============================================
// 共享材质和几何体缓存（内存优化）
// ============================================
export const sharedMaterials = {
  // PCB 材质
  pcbBase: new THREE.MeshStandardMaterial({ color: '#1a4d2e', metalness: 0.1, roughness: 0.9 }),
  pcbMiddle: new THREE.MeshStandardMaterial({ color: '#0f3d1f', metalness: 0.15, roughness: 0.85 }),
  pcbTop: new THREE.MeshStandardMaterial({ color: '#0a2f18', metalness: 0.1, roughness: 0.7 }),
  // 铜走线
  copperTrace: new THREE.MeshStandardMaterial({ color: '#c9a227', metalness: 0.6, roughness: 0.4 }),
  // 过孔
  via: new THREE.MeshStandardMaterial({ color: '#b8860b', metalness: 0.7, roughness: 0.3 }),
  // 金手指
  goldFinger: new THREE.MeshStandardMaterial({ color: '#b8923a', metalness: 0.3, roughness: 0.7 }),
  // 安装孔
  mountHole: new THREE.MeshStandardMaterial({ color: '#8a7040', metalness: 0.2, roughness: 0.8 }),
  // 芯片
  chipShell: new THREE.MeshStandardMaterial({ color: '#1a1a1a', metalness: 0.3, roughness: 0.8 }),
  chipShellHover: new THREE.MeshStandardMaterial({ color: '#2a2a2a', metalness: 0.3, roughness: 0.8 }),
  chipTop: new THREE.MeshStandardMaterial({ color: '#0d0d0d', metalness: 0.2, roughness: 0.9 }),
  // 引脚
  pin: new THREE.MeshStandardMaterial({ color: PIN_CONFIG.pinColor, metalness: PIN_CONFIG.pinMetalness, roughness: PIN_CONFIG.pinRoughness }),
  // 电路纹理
  circuitTrace: new THREE.MeshStandardMaterial({ color: CIRCUIT_TRACE_CONFIG.traceColor, metalness: 0.2, roughness: 0.8 }),
}

export const sharedGeometries = {
  // 过孔几何体
  via: new THREE.CylinderGeometry(0.003, 0.003, 0.001, 6),
  // 安装孔几何体
  mountHole: new THREE.CircleGeometry(0.01, 16),
  // LED圆形
  ledSmall: new THREE.CircleGeometry(0.006, 16),
  ledMedium: new THREE.CircleGeometry(0.008, 16),
  ledLarge: new THREE.CircleGeometry(0.006, 12),
  // Switch端口
  portOuter: new THREE.BoxGeometry(0.022, 0.018, 0.001),
  portInner: new THREE.BoxGeometry(0.018, 0.014, 0.001),
  portLed: new THREE.CircleGeometry(0.003, 8),
  // 机柜支脚
  rackFoot: new THREE.CylinderGeometry(0.02, 0.025, 0.04, 8),
  // 地面
  groundPlane: new THREE.PlaneGeometry(50, 50),
}

// 共享的基础材质（不带动态参数）
export const sharedBasicMaterials = {
  // LED颜色
  ledGreen: new THREE.MeshBasicMaterial({ color: '#52c41a' }),
  ledGreenBright: new THREE.MeshBasicMaterial({ color: '#7fff7f' }),
  ledOrange: new THREE.MeshBasicMaterial({ color: '#ffa500' }),
  ledYellow: new THREE.MeshBasicMaterial({ color: '#ffff7f' }),
  // Switch端口
  portFrame: new THREE.MeshBasicMaterial({ color: '#1a1a1a' }),
  portInner: new THREE.MeshBasicMaterial({ color: '#0a0a0a' }),
  portLedActive: new THREE.MeshBasicMaterial({ color: '#00ff88' }),
  portLedInactive: new THREE.MeshBasicMaterial({ color: '#333' }),
  // 机柜
  rackBackPanel: new THREE.MeshStandardMaterial({ color: '#2a2a2a', metalness: 0.3, roughness: 0.7 }),
  rackFoot: new THREE.MeshStandardMaterial({ color: '#333333', metalness: 0.5, roughness: 0.5 }),
  // 地面
  ground: new THREE.MeshStandardMaterial({ color: '#e8e8e8' }),
}

// ============================================
// 动画工具函数
// ============================================

// 缓动函数：先加速后减速
export function easeInOutCubic(t: number): number {
  return t < 0.5
    ? 4 * t * t * t
    : 1 - Math.pow(-2 * t + 2, 3) / 2
}

// 向量近似相等比较（容差）
export function vectorNearlyEquals(a: THREE.Vector3, b: THREE.Vector3, tolerance: number = 0.01): boolean {
  return Math.abs(a.x - b.x) < tolerance && Math.abs(a.y - b.y) < tolerance && Math.abs(a.z - b.z) < tolerance
}

// ============================================
// 类型定义
// ============================================

export interface CameraAnimationTarget {
  position: THREE.Vector3
  lookAt: THREE.Vector3
}

export interface ChipPinData {
  position: THREE.Vector3
  dimensions: [number, number, number]
}

export interface NodePositions {
  pods: Map<string, THREE.Vector3>      // Pod中心位置
  racks: Map<string, THREE.Vector3>     // Rack位置
  boards: Map<string, THREE.Vector3>    // Board世界坐标
}

// 3D视图节点详情
export interface Scene3DNodeDetail {
  id: string
  type: 'pod' | 'rack' | 'board' | 'chip' | 'switch'
  label: string
  subType?: string  // 如 chip 的 npu/cpu
  info: Record<string, string | number>  // 额外信息
  connections: { label: string; bandwidth?: number }[]
}
