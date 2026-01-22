/**
 * 左侧导航栏组件
 * 使用全局视角模式：点击菜单项调用 setViewMode 而不是路由导航
 */

import React, { useState } from 'react'
import { Layout, Menu } from 'antd'
import { useWorkbench } from '@/contexts/WorkbenchContext'
import {
  DashboardOutlined,
  ApartmentOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  PartitionOutlined,
} from '@ant-design/icons'
import type { MenuProps } from 'antd'

const { Sider } = Layout

type MenuItem = Required<MenuProps>['items'][number]

// 菜单项配置 - 使用 viewMode 值作为 key
const menuItems: MenuItem[] = [
  {
    key: 'dashboard',
    icon: <DashboardOutlined />,
    label: '概览',
  },
  {
    key: 'topology',
    icon: <ApartmentOutlined />,
    label: '互联拓扑',
  },
  {
    key: 'deployment',
    icon: <ThunderboltOutlined />,
    label: '部署分析',
  },
  {
    key: 'results',
    icon: <DatabaseOutlined />,
    label: '结果管理',
  },
  {
    key: 'knowledge',
    icon: <PartitionOutlined />,
    label: '知识网络',
  },
]

export const Sidebar: React.FC = () => {
  const { ui } = useWorkbench()
  const [collapsed, setCollapsed] = useState(false)

  const handleMenuClick: MenuProps['onClick'] = (e) => {
    const mode = e.key as 'dashboard' | 'topology' | 'deployment' | 'results' | 'knowledge' | '3d'
    ui.setViewMode(mode)
  }

  return (
    <Sider
      collapsible
      collapsed={collapsed}
      onCollapse={setCollapsed}
      trigger={null}
      width={180}
      collapsedWidth={80}
      style={{
        overflow: 'hidden',
        height: '100vh',
        position: 'sticky',
        top: 0,
        left: 0,
        display: 'flex',
        flexDirection: 'column',
        background: '#f0f0f0',
      }}
      theme="light"
    >
      {/* Logo 区域 */}
      <div
        style={{
          height: 64,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderBottom: '1px solid #f0f0f0',
          flexShrink: 0,
          background: '#f0f0f0',
        }}
      >
        <div
          style={{
            width: 32,
            height: 32,
            borderRadius: 8,
            background: 'linear-gradient(135deg, #5E6AD2 0%, #7C3AED 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#fff',
            fontSize: 12,
            fontWeight: 700,
          }}
        >
          T6+
        </div>
      </div>

      {/* 导航菜单 */}
      <div style={{ flex: 1, overflow: 'auto', background: '#f0f0f0' }}>
        <Menu
          mode="inline"
          selectedKeys={[ui.viewMode]}
          items={menuItems}
          onClick={handleMenuClick}
          style={{ borderRight: 0, background: '#f0f0f0' }}
        />
      </div>

      {/* 底部固定区域：版本号 + 折叠按钮 */}
      <div style={{ flexShrink: 0, background: '#f0f0f0' }}>
        {/* 版本号区域 */}
        <div
          style={{
            height: 1300,
            borderTop: '1px solid #f0f0f0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: collapsed ? '0 8px' : '0 16px',
            background: '#f0f0f0',
          }}
        >
          {!collapsed ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: '50%',
                  background: '#10b981',
                  boxShadow: '0 0 6px rgba(16, 185, 129, 0.5)',
                }}
              />
              <span style={{ fontSize: 12, color: '#999' }}>v{__APP_VERSION__}</span>
            </div>
          ) : (
            <div
              style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                background: '#10b981',
                boxShadow: '0 0 6px rgba(16, 185, 129, 0.5)',
              }}
              title={`v${__APP_VERSION__}`}
            />
          )}
        </div>
      </div>
    </Sider>
  )
}
