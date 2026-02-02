import * as React from 'react'
import { Input } from './input'
import { cn } from '@/lib/utils'

export interface NumberInputProps {
  value: number | undefined
  onChange: (value: number | undefined) => void
  min?: number
  max?: number
  step?: number
  className?: string
  disabled?: boolean
  suffix?: string
  placeholder?: string
}

/**
 * 数字输入组件
 * - 支持键盘输入和滚轮调节
 * - 聚焦时滚轮可改变数值，同时阻止页面滚动
 * - 未聚焦时滚轮正常滚动页面
 */
export const NumberInput: React.FC<NumberInputProps> = ({
  value,
  onChange,
  min = 0,
  max = Number.MAX_SAFE_INTEGER,
  step = 1,
  className = '',
  disabled = false,
  suffix,
  placeholder,
}) => {
  const inputRef = React.useRef<HTMLInputElement>(null)
  const [isFocused, setIsFocused] = React.useState(false)

  // 聚焦时：滚轮改变数值，阻止页面滚动
  const handleWheel = (e: React.WheelEvent<HTMLInputElement>) => {
    // 只有聚焦时才处理滚轮
    if (!isFocused || disabled) return

    // 阻止事件冒泡和默认行为，防止页面滚动
    e.stopPropagation()
    e.preventDefault()

    // 手动处理数值变化
    const currentValue = value ?? 0
    const delta = e.deltaY > 0 ? -step : step
    const newValue = Math.min(max, Math.max(min, currentValue + delta))

    // 处理浮点数精度问题
    const decimals = (step.toString().split('.')[1] || '').length
    const roundedValue = Number(newValue.toFixed(decimals))

    if (roundedValue !== value) {
      onChange(roundedValue)
    }
  }

  // 使用原生事件监听来正确阻止滚动
  React.useEffect(() => {
    const input = inputRef.current
    if (!input) return

    const preventScroll = (e: WheelEvent) => {
      if (isFocused) {
        e.preventDefault()
      }
    }

    input.addEventListener('wheel', preventScroll, { passive: false })
    return () => input.removeEventListener('wheel', preventScroll)
  }, [isFocused])

  return (
    <div className="flex items-center">
      <Input
        ref={inputRef}
        type="number"
        value={value ?? ''}
        onChange={(e) => {
          const v = e.target.value === '' ? undefined : parseFloat(e.target.value)
          if (v === undefined || (!isNaN(v) && v >= min && v <= max)) {
            onChange(v)
          }
        }}
        onWheel={handleWheel}
        onFocus={() => setIsFocused(true)}
        onBlur={() => setIsFocused(false)}
        min={min}
        max={max}
        step={step}
        className={cn(className)}
        disabled={disabled}
        placeholder={placeholder}
      />
      {suffix && <span className="ml-1 text-xs text-gray-500">{suffix}</span>}
    </div>
  )
}
