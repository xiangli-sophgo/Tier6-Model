/**
 * 左侧导航栏组件
 * 使用全局视角模式：点击菜单项调用 setViewMode 而不是路由导航
 * 支持折叠/展开功能
 */

import React from 'react'
import { useWorkbench } from '@/contexts/WorkbenchContext'
import { LayoutDashboard, Network, Zap, Database, GitBranch, ChevronLeft, ChevronRight, Palette } from 'lucide-react'
import { cn } from '@/lib/utils'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'

type ViewMode = 'dashboard' | 'topology' | 'deployment' | 'results' | 'knowledge' | '3d' | 'playground'

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
  {
    key: 'playground',
    icon: <Palette className="h-5 w-5" />,
    label: 'Card展示',
  },
]

export const Sidebar: React.FC = () => {
  const { ui } = useWorkbench()
  const collapsed = ui.sidebarCollapsed

  const handleMenuClick = (mode: ViewMode) => {
    ui.setViewMode(mode)
  }

  return (
    <TooltipProvider delayDuration={100}>
      <aside
        className={cn(
          "flex flex-col h-screen fixed top-0 left-0 z-50 bg-gradient-to-b from-blue-50 to-white overflow-hidden",
          collapsed ? "w-16" : "w-[180px]"
        )}
        style={{
          boxShadow: '4px 0 16px rgba(37, 99, 235, 0.12), 2px 0 8px rgba(37, 99, 235, 0.08)',
          transition: 'width 320ms cubic-bezier(0.4, 0, 0.2, 1)'
        }}
      >
        {/* Logo 区域 */}
        <div className="h-16 flex items-center justify-center flex-shrink-0">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-600 to-blue-400 flex items-center justify-center text-white text-xs font-bold shadow-md hover:shadow-lg transition-shadow">
            T6+
          </div>
        </div>

        {/* 导航菜单 */}
        <nav className="flex-1 overflow-auto py-4 px-2">
          {menuItems.map((item) => {
            const button = (
              <button
                key={item.key}
                onClick={() => handleMenuClick(item.key)}
                className={cn(
                  "w-full flex items-center gap-3 py-2.5 rounded-lg",
                  collapsed ? "justify-center px-3" : "justify-start px-6",
                  ui.viewMode === item.key
                    ? "bg-blue-100 text-blue-700 font-medium shadow-sm"
                    : "text-text-secondary hover:bg-blue-50 hover:text-blue-600"
                )}
                style={{
                  transition: 'background-color 200ms ease, color 200ms ease, box-shadow 200ms ease'
                }}
              >
                <span className="flex-shrink-0">{item.icon}</span>
                <span
                  className="text-sm"
                  style={{
                    overflow: 'hidden',
                    whiteSpace: 'nowrap',
                    transition: 'width 320ms cubic-bezier(0.4, 0, 0.2, 1), opacity 320ms cubic-bezier(0.4, 0, 0.2, 1)',
                    width: collapsed ? '0px' : 'auto',
                    opacity: collapsed ? 0 : 1
                  }}
                >
                  {item.label}
                </span>
              </button>
            )

            // 折叠时显示 Tooltip
            if (collapsed) {
              return (
                <Tooltip key={item.key}>
                  <TooltipTrigger asChild>
                    {button}
                  </TooltipTrigger>
                  <TooltipContent side="right" sideOffset={8}>
                    {item.label}
                  </TooltipContent>
                </Tooltip>
              )
            }

            return button
          })}
        </nav>

        {/* 底部固定区域：折叠按钮 + 版本号 */}
        <div className="flex-shrink-0 border-t border-blue-100">
          {/* 折叠/展开按钮 */}
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={ui.toggleSidebar}
                className={cn(
                  "w-full flex items-center justify-center h-10 text-text-muted hover:text-blue-600 hover:bg-blue-50",
                  collapsed && "px-0"
                )}
                style={{
                  transition: 'color 200ms ease, background-color 200ms ease'
                }}
              >
                {collapsed ? (
                  <ChevronRight className="h-4 w-4" />
                ) : (
                  <ChevronLeft className="h-4 w-4" />
                )}
              </button>
            </TooltipTrigger>
            <TooltipContent side="right" sideOffset={8}>
              {collapsed ? '展开侧边栏' : '收起侧边栏'}
            </TooltipContent>
          </Tooltip>

          {/* 版本号 */}
          <div
            className={cn(
              "h-10 flex items-center justify-center border-t border-blue-50",
              collapsed ? "px-2" : "px-4"
            )}
            style={{
              transition: 'padding 320ms cubic-bezier(0.4, 0, 0.2, 1)'
            }}
          >
            {!collapsed ? (
              <div
                className="flex items-center gap-2"
                style={{
                  transition: 'opacity 320ms cubic-bezier(0.4, 0, 0.2, 1)',
                  opacity: 1
                }}
              >
                <div className="w-1.5 h-1.5 rounded-full bg-green-500 shadow-[0_0_6px_rgba(34,197,94,0.5)]" />
                <span className="text-xs text-text-muted">v{__APP_VERSION__}</span>
              </div>
            ) : (
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="w-1.5 h-1.5 rounded-full bg-green-500 shadow-[0_0_6px_rgba(34,197,94,0.5)]" />
                </TooltipTrigger>
                <TooltipContent side="right" sideOffset={8}>
                  v{__APP_VERSION__}
                </TooltipContent>
              </Tooltip>
            )}
          </div>
        </div>
      </aside>
    </TooltipProvider>
  )
}
