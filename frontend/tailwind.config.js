/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#4F6BED',
          hover: '#3D56D9',
          bg: 'rgba(79, 107, 237, 0.08)',
        },
        success: '#10b981',
        warning: '#f59e0b',
        error: '#ef4444',
        border: {
          DEFAULT: '#E5E5E5',
          subtle: 'rgba(0, 0, 0, 0.05)',
          strong: '#D4D4D4',
        },
        text: {
          primary: '#1A1A1A',
          secondary: '#666666',
          muted: '#999999',
        },
        bg: {
          base: '#F7F7F7',
          elevated: '#FFFFFF',
          surface: '#EFEFEF',
          inset: '#E8E8E8',
          hover: 'rgba(0, 0, 0, 0.03)',
        }
      },
      borderRadius: {
        DEFAULT: '8px',
        lg: '12px',
        sm: '6px',
      },
      fontFamily: {
        sans: ["'Inter'", "'Noto Sans SC'", '-apple-system', 'sans-serif'],
        mono: ["'JetBrains Mono'", "'SF Mono'", 'monospace'],
        display: ["'Plus Jakarta Sans'", "'Noto Sans SC'", 'sans-serif'],
      },
      boxShadow: {
        xs: '0 1px 2px rgba(0, 0, 0, 0.05)',
        sm: '0 1px 3px rgba(0, 0, 0, 0.08), 0 1px 2px rgba(0, 0, 0, 0.04)',
        md: '0 4px 12px rgba(0, 0, 0, 0.08), 0 2px 4px rgba(0, 0, 0, 0.04)',
        focus: '0 0 0 3px rgba(79, 107, 237, 0.2)',
      },
      animation: {
        pulse: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spring-bounce': 'spring-bounce 0.6s ease-out',
        'connection-pulse': 'connection-pulse 2s ease-in-out infinite',
        'accordion-down': 'accordion-down 0.2s ease-out',
        'accordion-up': 'accordion-up 0.2s ease-out',
        'collapsible-down': 'collapsible-down 0.15s ease-out',
        'collapsible-up': 'collapsible-up 0.15s ease-out',
      },
      keyframes: {
        pulse: {
          '0%, 100%': {
            opacity: '1',
            transform: 'scale(1)',
          },
          '50%': {
            opacity: '0.6',
            transform: 'scale(1.2)',
          },
        },
        'spring-bounce': {
          '0%': { transform: 'scale(1)' },
          '25%': { transform: 'scale(1.05)' },
          '50%': { transform: 'scale(0.98)' },
          '75%': { transform: 'scale(1.02)' },
          '100%': { transform: 'scale(1)' },
        },
        'connection-pulse': {
          '0%, 100%': {
            'stroke-opacity': '1',
            'stroke-width': '2',
          },
          '50%': {
            'stroke-opacity': '0.7',
            'stroke-width': '3',
          },
        },
        'accordion-down': {
          from: { height: '0' },
          to: { height: 'var(--radix-accordion-content-height)' },
        },
        'accordion-up': {
          from: { height: 'var(--radix-accordion-content-height)' },
          to: { height: '0' },
        },
        'collapsible-down': {
          from: { height: '0', opacity: '0' },
          to: { height: 'var(--radix-collapsible-content-height)', opacity: '1' },
        },
        'collapsible-up': {
          from: { height: 'var(--radix-collapsible-content-height)', opacity: '1' },
          to: { height: '0', opacity: '0' },
        },
      },
    },
  },
  plugins: [],
}
