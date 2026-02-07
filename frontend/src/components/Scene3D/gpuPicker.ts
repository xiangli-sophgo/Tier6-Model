import * as THREE from 'three'
import { createContext, useContext } from 'react'

// ============================================
// GPU Picking - 基于离屏渲染的物体拾取
// 用唯一颜色 ID 渲染每个可交互物体，通过读取鼠标位置像素确定命中目标
// ============================================

export interface PickableObjectInfo {
  type: 'pod' | 'rack' | 'board' | 'chip' | 'switch'
  id: string
  label: string
  info: Record<string, string | number>
  subType?: string
}

// 用于 Picking 场景的简化材质（无光照，仅颜色）
const pickingMaterial = new THREE.MeshBasicMaterial({
  side: THREE.DoubleSide,
})

export class GPUPicker {
  private renderTarget: THREE.WebGLRenderTarget
  private pickingScene: THREE.Scene
  private pixelBuffer: Uint8Array
  private idCounter: number = 1  // 从 1 开始，0 表示无命中
  private idToInfo: Map<number, PickableObjectInfo> = new Map()
  private idToMesh: Map<number, THREE.Mesh> = new Map()
  private objectToId: Map<string, number> = new Map()  // objectKey -> id

  constructor() {
    // 1x1 渲染目标 - 只需读取鼠标位置的 1 个像素
    this.renderTarget = new THREE.WebGLRenderTarget(1, 1, {
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
    })
    this.pickingScene = new THREE.Scene()
    this.pickingScene.background = new THREE.Color(0x000000)
    this.pixelBuffer = new Uint8Array(4)
  }

  // 将 ID 编码为 RGB 颜色 (支持 16M+ 个物体)
  private idToColor(id: number): THREE.Color {
    const r = ((id >> 16) & 0xff) / 255
    const g = ((id >> 8) & 0xff) / 255
    const b = (id & 0xff) / 255
    return new THREE.Color(r, g, b)
  }

  // 从 RGB 像素值解码 ID
  private colorToId(r: number, g: number, b: number): number {
    return (r << 16) | (g << 8) | b
  }

  // 注册可交互物体 - 创建简化的 Picking 几何体
  register(
    objectKey: string,
    info: PickableObjectInfo,
    geometry: THREE.BufferGeometry,
    worldMatrix: THREE.Matrix4
  ): number {
    // 如果已注册，先注销
    const existingId = this.objectToId.get(objectKey)
    if (existingId !== undefined) {
      this.unregister(existingId)
    }

    const id = this.idCounter++
    const color = this.idToColor(id)

    const material = pickingMaterial.clone()
    material.color = color
    const mesh = new THREE.Mesh(geometry, material)
    mesh.applyMatrix4(worldMatrix)
    mesh.matrixAutoUpdate = false

    this.pickingScene.add(mesh)
    this.idToInfo.set(id, info)
    this.idToMesh.set(id, mesh)
    this.objectToId.set(objectKey, id)

    return id
  }

  // 更新已注册物体的世界矩阵
  updateTransform(id: number, worldMatrix: THREE.Matrix4): void {
    const mesh = this.idToMesh.get(id)
    if (mesh) {
      mesh.matrix.copy(worldMatrix)
      mesh.matrixWorldNeedsUpdate = true
    }
  }

  // 注销物体
  unregister(id: number): void {
    const mesh = this.idToMesh.get(id)
    if (mesh) {
      this.pickingScene.remove(mesh)
      if (mesh.material instanceof THREE.Material) {
        mesh.material.dispose()
      }
    }
    this.idToMesh.delete(id)
    const info = this.idToInfo.get(id)
    if (info) {
      this.objectToId.delete(`${info.type}:${info.id}`)
    }
    this.idToInfo.delete(id)
  }

  // 通过 objectKey 注销
  unregisterByKey(objectKey: string): void {
    const id = this.objectToId.get(objectKey)
    if (id !== undefined) {
      this.unregister(id)
      this.objectToId.delete(objectKey)
    }
  }

  // 用于 pick 的临时相机（避免修改原始相机）
  private pickingCamera = new THREE.PerspectiveCamera()

  // 执行 Pick - 通过修改相机投影矩阵，只渲染鼠标位置 1 像素到 1x1 RT
  pick(
    mouse: THREE.Vector2,  // NDC 坐标 (-1 ~ 1)
    camera: THREE.Camera,
    renderer: THREE.WebGLRenderer,
    canvasWidth: number,
    canvasHeight: number,
  ): PickableObjectInfo | null {
    if (this.idToInfo.size === 0) return null

    // 将 NDC 坐标转换为像素坐标
    const pixelX = ((mouse.x + 1) / 2) * canvasWidth
    const pixelY = ((-mouse.y + 1) / 2) * canvasHeight

    // 复制原始相机的矩阵和投影
    this.pickingCamera.matrixWorld.copy(camera.matrixWorld)
    this.pickingCamera.matrixWorldInverse.copy(camera.matrixWorldInverse)
    this.pickingCamera.projectionMatrix.copy(camera.projectionMatrix)

    // 修改投影矩阵：偏移+缩放使 1x1 RT 只覆盖鼠标位置的 1 像素
    // 利用 projectionMatrix 的 [2][0], [2][1] 偏移 + [0][0], [1][1] 缩放
    const projMatrix = this.pickingCamera.projectionMatrix
    const elements = projMatrix.elements
    elements[0] *= canvasWidth      // 缩放 X: 整个画布宽度映射到 1 像素
    elements[4] *= canvasWidth
    elements[8] = elements[8] * canvasWidth + canvasWidth - 2 * pixelX   // 偏移 X
    elements[12] *= canvasWidth
    elements[1] *= canvasHeight     // 缩放 Y
    elements[5] *= canvasHeight
    elements[9] = elements[9] * canvasHeight + 2 * pixelY - canvasHeight // 偏移 Y
    elements[13] *= canvasHeight

    // 保存渲染器状态
    const currentRenderTarget = renderer.getRenderTarget()

    // 渲染到 1x1 RT
    renderer.setRenderTarget(this.renderTarget)
    renderer.clear()
    renderer.render(this.pickingScene, this.pickingCamera)

    // 读取 1 像素
    renderer.readRenderTargetPixels(this.renderTarget, 0, 0, 1, 1, this.pixelBuffer)

    // 恢复渲染器状态
    renderer.setRenderTarget(currentRenderTarget)

    // 解码 ID
    const id = this.colorToId(this.pixelBuffer[0], this.pixelBuffer[1], this.pixelBuffer[2])
    if (id === 0) return null

    return this.idToInfo.get(id) ?? null
  }

  // 清理所有资源
  dispose(): void {
    this.renderTarget.dispose()
    this.idToMesh.forEach(mesh => {
      if (mesh.material instanceof THREE.Material) {
        mesh.material.dispose()
      }
    })
    this.idToMesh.clear()
    this.idToInfo.clear()
    this.objectToId.clear()
  }

  // 获取已注册物体数量
  get registeredCount(): number {
    return this.idToInfo.size
  }
}

// ============================================
// GPU Picker React Context
// ============================================

export interface GPUPickerContextValue {
  picker: GPUPicker
}

export const GPUPickerContext = createContext<GPUPickerContextValue | null>(null)
export const useGPUPicker = () => useContext(GPUPickerContext)
