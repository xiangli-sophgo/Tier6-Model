import * as React from "react"

import { cn } from "@/lib/utils"
import { formControlStyles } from "./form-control-styles"

export interface TextareaProps
  extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => {
    return (
      <textarea
        className={cn(
          formControlStyles,
          // Textarea 特有：最小高度
          "min-h-[60px]",
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Textarea.displayName = "Textarea"

export { Textarea }
