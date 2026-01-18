/**
 * LLM 推理模拟系统 - 事件优先级队列
 *
 * 基于最小堆实现的事件队列，按时间戳排序
 */

import { SimEvent } from './types'

/**
 * 事件优先级队列
 * 使用二叉堆实现，支持 O(log n) 的插入和取出
 */
export class EventQueue {
  private heap: SimEvent[] = []

  /** 获取队列长度 */
  get length(): number {
    return this.heap.length
  }

  /** 队列是否为空 */
  isEmpty(): boolean {
    return this.heap.length === 0
  }

  /** 添加事件 */
  enqueue(event: SimEvent): void {
    this.heap.push(event)
    this.bubbleUp(this.heap.length - 1)
  }

  /** 批量添加事件 */
  enqueueBatch(events: SimEvent[]): void {
    for (const event of events) {
      this.enqueue(event)
    }
  }

  /** 取出最早的事件 */
  dequeue(): SimEvent | undefined {
    if (this.heap.length === 0) return undefined
    if (this.heap.length === 1) return this.heap.pop()

    const result = this.heap[0]
    this.heap[0] = this.heap.pop()!
    this.bubbleDown(0)
    return result
  }

  /** 查看最早的事件（不取出） */
  peek(): SimEvent | undefined {
    return this.heap[0]
  }

  /** 清空队列 */
  clear(): void {
    this.heap = []
  }

  /** 获取所有事件（按时间排序） */
  toSortedArray(): SimEvent[] {
    return [...this.heap].sort((a, b) => a.timestamp - b.timestamp)
  }

  /** 上浮操作 */
  private bubbleUp(index: number): void {
    while (index > 0) {
      const parentIndex = Math.floor((index - 1) / 2)
      if (this.heap[parentIndex].timestamp <= this.heap[index].timestamp) {
        break
      }
      this.swap(index, parentIndex)
      index = parentIndex
    }
  }

  /** 下沉操作 */
  private bubbleDown(index: number): void {
    const length = this.heap.length
    while (true) {
      const leftChild = 2 * index + 1
      const rightChild = 2 * index + 2
      let smallest = index

      if (
        leftChild < length &&
        this.heap[leftChild].timestamp < this.heap[smallest].timestamp
      ) {
        smallest = leftChild
      }

      if (
        rightChild < length &&
        this.heap[rightChild].timestamp < this.heap[smallest].timestamp
      ) {
        smallest = rightChild
      }

      if (smallest === index) break

      this.swap(index, smallest)
      index = smallest
    }
  }

  /** 交换两个元素 */
  private swap(i: number, j: number): void {
    ;[this.heap[i], this.heap[j]] = [this.heap[j], this.heap[i]]
  }
}

/**
 * 任务依赖图
 * 用于管理任务之间的依赖关系，支持拓扑排序
 */
export class TaskDependencyGraph<T extends { id: string; dependencies: string[] }> {
  private tasks: Map<string, T> = new Map()
  private dependents: Map<string, Set<string>> = new Map()
  private inDegree: Map<string, number> = new Map()

  /** 添加任务 */
  addTask(task: T): void {
    this.tasks.set(task.id, task)
    this.inDegree.set(task.id, task.dependencies.length)

    // 建立反向依赖关系
    for (const dep of task.dependencies) {
      if (!this.dependents.has(dep)) {
        this.dependents.set(dep, new Set())
      }
      this.dependents.get(dep)!.add(task.id)
    }
  }

  /** 获取可执行的任务（入度为0） */
  getReadyTasks(): T[] {
    const ready: T[] = []
    for (const [id, degree] of this.inDegree) {
      if (degree === 0) {
        const task = this.tasks.get(id)
        if (task) ready.push(task)
      }
    }
    return ready
  }

  /** 标记任务完成，更新依赖 */
  completeTask(taskId: string): T[] {
    const newlyReady: T[] = []
    const dependentIds = this.dependents.get(taskId)

    if (dependentIds) {
      for (const depId of dependentIds) {
        const newDegree = (this.inDegree.get(depId) ?? 1) - 1
        this.inDegree.set(depId, newDegree)
        if (newDegree === 0) {
          const task = this.tasks.get(depId)
          if (task) newlyReady.push(task)
        }
      }
    }

    // 移除已完成的任务
    this.tasks.delete(taskId)
    this.inDegree.delete(taskId)
    this.dependents.delete(taskId)

    return newlyReady
  }

  /** 获取任务 */
  getTask(taskId: string): T | undefined {
    return this.tasks.get(taskId)
  }

  /** 是否还有未完成的任务 */
  hasRemainingTasks(): boolean {
    return this.tasks.size > 0
  }

  /** 获取剩余任务数 */
  getRemainingCount(): number {
    return this.tasks.size
  }
}
