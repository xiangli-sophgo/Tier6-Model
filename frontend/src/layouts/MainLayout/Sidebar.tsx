/**
 * 左侧导航栏组件
 */

import React, { useState } from 'react'
import { Layout, Menu, Button } from 'antd'
import { useNavigate, useLocation } from 'react-router-dom'
import {
  DashboardOutlined,
  ApartmentOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  BarChartOutlined,
  PartitionOutlined,
  LeftOutlined,
  RightOutlined,
} from '@ant-design/icons'
import type { MenuProps } from 'antd'

const { Sider } = Layout

type MenuItem = Required<MenuProps>['items'][number]

// 菜单项配置
const menuItems: MenuItem[] = [
  {
    key: '/',
    icon: <DashboardOutlined />,
    label: '首页',
  },
  {
    key: '/topology',
    icon: <ApartmentOutlined />,
    label: '互联拓扑',
  },
  {
    key: '/deployment',
    icon: <ThunderboltOutlined />,
    label: '部署分析',
  },
  {
    key: '/results',
    icon: <DatabaseOutlined />,
    label: '结果汇总',
  },
  {
    key: '/analysis',
    icon: <BarChartOutlined />,
    label: '结果分析',
  },
  {
    key: '/knowledge',
    icon: <PartitionOutlined />,
    label: '知识网络',
  },
]

export const Sidebar: React.FC = () => {
  const navigate = useNavigate()
  const location = useLocation()
  const [collapsed, setCollapsed] = useState(false)

  // 获取当前激活的菜单项
  const getSelectedKeys = () => {
    const path = location.pathname
    // 如果是详情页（如 /analysis/xxx），则高亮父级菜单
    if (path.startsWith('/analysis/')) return ['/analysis']
    if (path.startsWith('/results/')) return ['/results']
    return [path]
  }

  const handleMenuClick: MenuProps['onClick'] = (e) => {
    navigate(e.key)
  }

  return (
    <Sider
      collapsible
      collapsed={collapsed}
      onCollapse={setCollapsed}
      trigger={null}
      width={150}
      collapsedWidth={80}
      style={{
        overflow: 'hidden',
        height: '100vh',
        position: 'sticky',
        top: 0,
        left: 0,
        display: 'flex',
        flexDirection: 'column',
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
      <div style={{ flex: 1, overflow: 'auto' }}>
        <Menu
          mode="inline"
          selectedKeys={getSelectedKeys()}
          items={menuItems}
          onClick={handleMenuClick}
          style={{ borderRight: 0 }}
        />
      </div>

      {/* 底部固定区域：版本号 + 折叠按钮 */}
      <div style={{ flexShrink: 0 }}>
        {/* 版本号区域 */}
        <div
          style={{
            height: 1250,
            borderTop: '1px solid #f0f0f0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: collapsed ? '0 8px' : '0 16px',
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
