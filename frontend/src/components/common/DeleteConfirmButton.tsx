/**
 * DeleteConfirmButton - 带确认弹窗的删除按钮
 *
 * 通用组件: 点击后弹出 AlertDialog 确认，确认后执行 onConfirm 回调。
 */

import React, { useState } from 'react'
import { Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from '@/components/ui/alert-dialog'

interface DeleteConfirmButtonProps {
  /** 要删除的对象名称，显示在确认弹窗中 */
  name: string
  /** 对象类型描述，如 "芯片预设"、"模型预设" */
  label: string
  /** 确认删除后的回调 */
  onConfirm: () => void | Promise<void>
  /** 是否禁用 */
  disabled?: boolean
}

export const DeleteConfirmButton: React.FC<DeleteConfirmButtonProps> = ({
  name, label, onConfirm, disabled,
}) => {
  const [open, setOpen] = useState(false)

  const handleConfirm = async () => {
    await onConfirm()
    setOpen(false)
  }

  return (
    <>
      <Button
        variant="outline"
        size="sm"
        onClick={() => setOpen(true)}
        disabled={disabled}
        className="text-red-500 hover:text-red-700 hover:bg-red-50"
      >
        <Trash2 className="h-3.5 w-3.5 mr-1" />{label}
      </Button>
      <AlertDialog open={open} onOpenChange={setOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>确认删除</AlertDialogTitle>
            <AlertDialogDescription>
              确定要删除{label} "{name}" 吗？此操作不可撤销。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction onClick={handleConfirm} className="bg-red-600 hover:bg-red-700">
              删除
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}
