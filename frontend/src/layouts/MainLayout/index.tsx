/**
 * 主布局组件
 * 包含左侧导航和内容区域
 */

import React from 'react'
import { Layout } from 'antd'
import { Outlet } from 'react-router-dom'
import { Sidebar } from './Sidebar'

const { Content } = Layout

export const MainLayout: React.FC = () => {
  return (
    <Layout style={{ height: '100vh' }}>
      <Sidebar />
      <Layout style={{ height: '100vh', overflow: 'hidden' }}>
        <Content style={{ height: '100%', overflow: 'hidden', background: '#f0f2f5' }}>
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  )
}
