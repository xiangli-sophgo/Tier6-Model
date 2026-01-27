/**
 * 左侧导航栏组件
 * 使用全局视角模式：点击菜单项调用 setViewMode 而不是路由导航
 */

import React, { useState } from 'react'
import { useWorkbench } from '@/contexts/WorkbenchContext'
import { LayoutDashboard, Network, Zap, Database, GitBranch } from 'lucide-react'
import { cn } from '@/lib/utils'

type ViewMode = 'dashboard' | 'topology' | 'deployment' | 'results' | 'knowledge' | '3d'

interface MenuItem {
  key: ViewMode
  icon: React.ReactNode
  label: string
}

// 菜单项配置 - 使用 viewMode 值作为 key
const menuItems: MenuItem[] = [
  {
    key: 'dashboard',
    icon: <LayoutDashboard className="h-5 w-5" />,
    label: '概览',
  },
  {
    key: 'topology',
    icon: <Network className="h-5 w-5" />,
    label: '互联拓扑',
  },
  {
    key: 'deployment',
    icon: <Zap className="h-5 w-5" />,
    label: '部署分析',
  },
  {
    key: 'results',
    icon: <Database className="h-5 w-5" />,
    label: '结果管理',
  },
  {
    key: 'knowledge',
    icon: <GitBranch className="h-5 w-5" />,
    label: '知识网络',
  },
]

export const Sidebar: React.FC = () => {
  const { ui } = useWorkbench()
  const [collapsed] = useState(false)

  const handleMenuClick = (mode: ViewMode) => {
    ui.setViewMode(mode)
  }

  return (
    <aside
      className={cn(
        "flex flex-col h-screen sticky top-0 left-0 bg-gray-100 transition-all duration-300",
        collapsed ? "w-20" : "w-[180px]"
      )}
    >
      {/* Logo 区域 */}
      <div className="h-16 flex items-center justify-center border-b border-gray-200 flex-shrink-0">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#5E6AD2] to-[#7C3AED] flex items-center justify-center text-white text-xs font-bold">
          T6+
        </div>
      </div>

      {/* 导航菜单 */}
      <nav className="flex-1 overflow-auto py-4">
        {menuItems.map((item) => (
          <button
            key={item.key}
            onClick={() => handleMenuClick(item.key)}
            className={cn(
              "w-full flex items-center gap-3 px-4 py-3 transition-colors",
              "hover:bg-white/50",
              ui.viewMode === item.key
                ? "bg-white text-primary font-medium shadow-sm"
                : "text-gray-700"
            )}
          >
            <span className="flex-shrink-0">{item.icon}</span>
            {!collapsed && <span className="text-sm">{item.label}</span>}
          </button>
        ))}
      </nav>

      {/* 底部固定区域：版本号 */}
      <div className="flex-shrink-0 border-t border-gray-200">
        <div className={cn(
          "h-12 flex items-center justify-center",
          collapsed ? "px-2" : "px-4"
        )}>
          {!collapsed ? (
            <div className="flex items-center gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-success shadow-[0_0_6px_rgba(16,185,129,0.5)]" />
              <span className="text-xs text-gray-500">v{__APP_VERSION__}</span>
            </div>
          ) : (
            <div
              className="w-1.5 h-1.5 rounded-full bg-success shadow-[0_0_6px_rgba(16,185,129,0.5)]"
              title={`v${__APP_VERSION__}`}
            />
          )}
        </div>
      </div>
    </aside>
  )
}
