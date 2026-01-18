import React from 'react'
import ReactDOM from 'react-dom/client'
import { ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import App from './App'
import './index.css'

// Ant Design 主题配置 - 使用 CSS 变量中定义的设计系统
const theme = {
  token: {
    // 主色
    colorPrimary: '#4F6BED',
    colorPrimaryHover: '#3D56D9',
    colorPrimaryBg: 'rgba(79, 107, 237, 0.08)',
    colorPrimaryBgHover: 'rgba(79, 107, 237, 0.12)',

    // 成功/警告/错误
    colorSuccess: '#10b981',
    colorWarning: '#f59e0b',
    colorError: '#ef4444',

    // 文字
    colorText: '#1A1A1A',
    colorTextSecondary: '#666666',
    colorTextTertiary: '#999999',
    colorTextDisabled: '#CCCCCC',

    // 背景
    colorBgContainer: '#FFFFFF',
    colorBgElevated: '#FFFFFF',
    colorBgLayout: '#F7F7F7',
    colorBgSpotlight: '#EFEFEF',

    // 边框
    colorBorder: '#E5E5E5',
    colorBorderSecondary: 'rgba(0, 0, 0, 0.05)',

    // 圆角
    borderRadius: 8,
    borderRadiusSM: 6,
    borderRadiusLG: 12,

    // 字体
    fontFamily: "'Inter', 'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', -apple-system, sans-serif",
    fontSize: 14,
    fontSizeSM: 13,
    fontSizeLG: 16,

    // 阴影
    boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08), 0 1px 2px rgba(0, 0, 0, 0.04)',
    boxShadowSecondary: '0 4px 12px rgba(0, 0, 0, 0.08), 0 2px 4px rgba(0, 0, 0, 0.04)',

    // 控件
    controlHeight: 34,
    controlHeightSM: 28,
    controlHeightLG: 40,
  },
  components: {
    Button: {
      fontWeight: 500,
      primaryShadow: '0 2px 8px rgba(79, 107, 237, 0.25)',
    },
    Card: {
      headerFontSize: 15,
      headerFontSizeSM: 14,
    },
    Collapse: {
      headerBg: 'transparent',
      contentBg: 'transparent',
    },
    Input: {
      activeBorderColor: '#4F6BED',
      hoverBorderColor: '#D4D4D4',
    },
    Select: {
      optionSelectedBg: 'rgba(79, 107, 237, 0.08)',
    },
    Tabs: {
      inkBarColor: '#4F6BED',
      itemSelectedColor: '#4F6BED',
    },
    Tag: {
      fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
    },
    Modal: {
      borderRadiusLG: 16,
    },
    Tooltip: {
      colorBgSpotlight: '#1A1A1A',
    },
  },
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ConfigProvider theme={theme} locale={zhCN}>
      <App />
    </ConfigProvider>
  </React.StrictMode>,
)
