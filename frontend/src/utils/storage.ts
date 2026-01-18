/**
 * IndexedDB 存储模块
 *
 * 替代后端 JSON 文件存储，实现配置的本地持久化
 */

import { ManualConnectionConfig, ManualConnection, GlobalSwitchConfig } from '../types';

// 数据库名称和版本
const DB_NAME = 'Tier6TopologyDB';
const DB_VERSION = 1;

// 存储名称
const STORES = {
  CONFIGS: 'savedConfigs',
  MANUAL_CONNECTIONS: 'manualConnections',
};

// 保存的配置接口
export interface SavedConfig {
  name: string;
  description?: string;
  pod_count: number;
  racks_per_pod: number;
  board_configs: {
    u1: { count: number; chips: { npu: number; cpu: number } };
    u2: { count: number; chips: { npu: number; cpu: number } };
    u4: { count: number; chips: { npu: number; cpu: number } };
  };
  rack_config?: {
    total_u: number;
    boards: Array<{
      id: string;
      name: string;
      u_height: number;
      count: number;
      chips: Array<{ name: string; count: number }>;
    }>;
  };
  switch_config?: GlobalSwitchConfig;
  manual_connections?: ManualConnectionConfig;
  created_at?: string;
  updated_at?: string;
}

/**
 * 打开数据库连接
 */
function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => {
      reject(new Error('无法打开数据库'));
    };

    request.onsuccess = () => {
      resolve(request.result);
    };

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;

      // 创建配置存储
      if (!db.objectStoreNames.contains(STORES.CONFIGS)) {
        db.createObjectStore(STORES.CONFIGS, { keyPath: 'name' });
      }

      // 创建手动连接存储
      if (!db.objectStoreNames.contains(STORES.MANUAL_CONNECTIONS)) {
        db.createObjectStore(STORES.MANUAL_CONNECTIONS, { keyPath: 'id' });
      }
    };
  });
}

// ============================================
// 配置存储 API
// ============================================

/**
 * 获取所有保存的配置列表
 */
export async function listConfigs(): Promise<SavedConfig[]> {
  const db = await openDB();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORES.CONFIGS, 'readonly');
    const store = transaction.objectStore(STORES.CONFIGS);
    const request = store.getAll();

    request.onsuccess = () => {
      const configs = request.result as SavedConfig[];
      // 按更新时间倒序
      configs.sort((a, b) => {
        const timeA = a.updated_at ?? a.created_at ?? '';
        const timeB = b.updated_at ?? b.created_at ?? '';
        return timeB.localeCompare(timeA);
      });
      resolve(configs);
    };

    request.onerror = () => {
      reject(new Error('读取配置列表失败'));
    };

    transaction.oncomplete = () => {
      db.close();
    };
  });
}

/**
 * 获取指定名称的配置
 */
export async function getConfig(name: string): Promise<SavedConfig | null> {
  const db = await openDB();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORES.CONFIGS, 'readonly');
    const store = transaction.objectStore(STORES.CONFIGS);
    const request = store.get(name);

    request.onsuccess = () => {
      resolve(request.result ?? null);
    };

    request.onerror = () => {
      reject(new Error(`读取配置 '${name}' 失败`));
    };

    transaction.oncomplete = () => {
      db.close();
    };
  });
}

/**
 * 保存配置
 */
export async function saveConfig(config: SavedConfig): Promise<SavedConfig> {
  const db = await openDB();
  const now = new Date().toISOString();

  // 检查是否存在，保留创建时间
  const existing = await getConfig(config.name);
  if (existing) {
    config.created_at = existing.created_at;
  } else {
    config.created_at = now;
  }
  config.updated_at = now;

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORES.CONFIGS, 'readwrite');
    const store = transaction.objectStore(STORES.CONFIGS);
    const request = store.put(config);

    request.onsuccess = () => {
      resolve(config);
    };

    request.onerror = () => {
      reject(new Error(`保存配置 '${config.name}' 失败`));
    };

    transaction.oncomplete = () => {
      db.close();
    };
  });
}

/**
 * 删除配置
 */
export async function deleteConfig(name: string): Promise<void> {
  const db = await openDB();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORES.CONFIGS, 'readwrite');
    const store = transaction.objectStore(STORES.CONFIGS);
    const request = store.delete(name);

    request.onsuccess = () => {
      resolve();
    };

    request.onerror = () => {
      reject(new Error(`删除配置 '${name}' 失败`));
    };

    transaction.oncomplete = () => {
      db.close();
    };
  });
}

// ============================================
// 手动连接存储 API
// ============================================

const MANUAL_CONNECTIONS_KEY = '_manual_connections';

/**
 * 获取手动连接配置
 */
export async function getManualConnections(): Promise<ManualConnectionConfig> {
  const db = await openDB();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORES.MANUAL_CONNECTIONS, 'readonly');
    const store = transaction.objectStore(STORES.MANUAL_CONNECTIONS);
    const request = store.get(MANUAL_CONNECTIONS_KEY);

    request.onsuccess = () => {
      const result = request.result;
      if (result) {
        resolve(result.config as ManualConnectionConfig);
      } else {
        resolve({
          enabled: false,
          mode: 'append',
          connections: [],
        });
      }
    };

    request.onerror = () => {
      reject(new Error('读取手动连接配置失败'));
    };

    transaction.oncomplete = () => {
      db.close();
    };
  });
}

/**
 * 保存手动连接配置
 */
export async function saveManualConnections(config: ManualConnectionConfig): Promise<ManualConnectionConfig> {
  const db = await openDB();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORES.MANUAL_CONNECTIONS, 'readwrite');
    const store = transaction.objectStore(STORES.MANUAL_CONNECTIONS);
    const request = store.put({ id: MANUAL_CONNECTIONS_KEY, config });

    request.onsuccess = () => {
      resolve(config);
    };

    request.onerror = () => {
      reject(new Error('保存手动连接配置失败'));
    };

    transaction.oncomplete = () => {
      db.close();
    };
  });
}

/**
 * 添加单个手动连接
 */
export async function addManualConnection(connection: ManualConnection): Promise<ManualConnectionConfig> {
  const config = await getManualConnections();

  // 检查是否已存在相同连接
  const exists = config.connections.some(
    c => c.source === connection.source && c.target === connection.target
  );
  if (exists) {
    throw new Error('该连接已存在');
  }

  // 添加创建时间
  if (!connection.created_at) {
    connection.created_at = new Date().toISOString();
  }

  config.connections.push(connection);
  return saveManualConnections(config);
}

/**
 * 删除单个手动连接
 */
export async function deleteManualConnection(connectionId: string): Promise<void> {
  const config = await getManualConnections();
  const originalCount = config.connections.length;
  config.connections = config.connections.filter(c => c.id !== connectionId);

  if (config.connections.length === originalCount) {
    throw new Error(`连接 '${connectionId}' 不存在`);
  }

  await saveManualConnections(config);
}

/**
 * 清空手动连接（可按层级清空）
 */
export async function clearManualConnections(hierarchyLevel?: string): Promise<void> {
  const config = await getManualConnections();

  if (hierarchyLevel) {
    config.connections = config.connections.filter(c => c.hierarchy_level !== hierarchyLevel);
  } else {
    config.connections = [];
  }

  await saveManualConnections(config);
}

// ============================================
// 静态配置（原后端常量）
// ============================================

/**
 * 获取支持的Chip类型
 */
export function getChipTypes(): { types: { id: string; name: string; color: string }[] } {
  return {
    types: [
      { id: 'npu', name: 'NPU', color: '#eb2f96' },
      { id: 'cpu', name: 'CPU', color: '#1890ff' },
    ],
  };
}

/**
 * 获取Rack物理尺寸配置
 */
export function getRackDimensions(): {
  width: number;
  depth: number;
  u_height: number;
  total_u: number;
  full_height: number;
} {
  return {
    width: 0.6,
    depth: 1.0,
    u_height: 0.0445,
    total_u: 42,
    full_height: 42 * 0.0445,
  };
}

/**
 * 获取各层级连接的默认带宽和延迟配置
 *
 * 单位说明:
 * - bandwidth: GB/s (字节/秒，非 Gbps)
 * - latency: us (微秒)
 *
 * 参考值来源 (presets.ts):
 * - NVLink 4.0 (H100): 900 GB/s, 1 us
 * - NVLink 3.0 (A100): 600 GB/s, 1 us
 * - InfiniBand NDR: 50 GB/s (400 Gbps), 2 us
 * - PCIe 4.0: 64 GB/s, 5 us
 */
export function getLevelConnectionDefaults(): {
  datacenter: { bandwidth: number; latency: number };
  pod: { bandwidth: number; latency: number };
  rack: { bandwidth: number; latency: number };
  board: { bandwidth: number; latency: number };
} {
  return {
    // Pod 间: 跨 Pod 通信，通常走数据中心网络
    datacenter: { bandwidth: 50.0, latency: 50.0 },
    // Rack 间: 同 Pod 内不同 Rack，通常走 InfiniBand
    pod: { bandwidth: 50.0, latency: 5.0 },
    // Board 间: 同 Rack 内不同节点/Board，走 InfiniBand
    rack: { bandwidth: 50.0, latency: 2.0 },
    // Chip 间: 同 Board 内芯片互联，走 NVLink
    board: { bandwidth: 900.0, latency: 1.0 },
  };
}

// ============================================
// 缓存清理 API
// ============================================

/**
 * 清除所有缓存数据（IndexedDB + localStorage）
 */
export async function clearAllCache(): Promise<void> {
  // 1. 删除 IndexedDB 数据库
  return new Promise((resolve, reject) => {
    const request = indexedDB.deleteDatabase(DB_NAME);

    request.onsuccess = () => {
      // 2. 清除 localStorage 中的相关缓存
      const keysToRemove = [
        'tier6_topology_config_cache',
        'tier6_sider_width_cache',
      ];
      keysToRemove.forEach(key => localStorage.removeItem(key));

      resolve();
    };

    request.onerror = () => {
      reject(new Error('清除缓存失败'));
    };

    request.onblocked = () => {
      reject(new Error('数据库被占用，请关闭其他标签页后重试'));
    };
  });
}
