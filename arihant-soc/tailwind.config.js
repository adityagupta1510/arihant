/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'cyber-bg-start': '#05070D',
        'cyber-bg-end': '#0A0F1F',
        'cyber-primary': '#00E0FF',
        'cyber-secondary': '#00FFA3',
        'cyber-critical': '#FF3B3B',
        'cyber-warning': '#FFB020',
        'cyber-card': 'rgba(15, 23, 42, 0.6)',
        'cyber-border': 'rgba(0, 224, 255, 0.1)',
        'cyber-surface': 'rgba(15, 23, 42, 0.8)',
      },
      fontFamily: {
        'mono': ['JetBrains Mono', 'Fira Code', 'monospace'],
        'sans': ['Inter', 'system-ui', 'sans-serif'],
      },
      fontSize: {
        'display': ['72px', { lineHeight: '1.1', letterSpacing: '-0.02em' }],
        'h1': ['48px', { lineHeight: '1.2', letterSpacing: '-0.02em' }],
        'h2': ['36px', { lineHeight: '1.3', letterSpacing: '-0.01em' }],
        'h3': ['24px', { lineHeight: '1.4' }],
        'h4': ['20px', { lineHeight: '1.5' }],
        'body-lg': ['18px', { lineHeight: '1.6' }],
        'body': ['16px', { lineHeight: '1.6' }],
        'body-sm': ['14px', { lineHeight: '1.5' }],
        'caption': ['12px', { lineHeight: '1.4' }],
      },
      spacing: {
        '18': '4.5rem',
        '22': '5.5rem',
        '30': '7.5rem',
        '88': '22rem',
        '128': '32rem',
      },
      maxWidth: {
        '8xl': '1440px',
      },
      animation: {
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
        'float': 'float 6s ease-in-out infinite',
        'scan': 'scan 3s linear infinite',
        'data-flow': 'data-flow 20s linear infinite',
        'blink': 'blink 1.5s ease-in-out infinite',
        'spin-slow': 'spin 8s linear infinite',
        'ping-slow': 'ping 2s cubic-bezier(0, 0, 0.2, 1) infinite',
        'bounce-subtle': 'bounce-subtle 2s ease-in-out infinite',
        'gradient-shift': 'gradient-shift 8s ease infinite',
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { boxShadow: '0 0 20px rgba(0, 224, 255, 0.3)' },
          '50%': { boxShadow: '0 0 50px rgba(0, 224, 255, 0.6)' },
        },
        'float': {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-20px)' },
        },
        'scan': {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        },
        'data-flow': {
          '0%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
          '100%': { backgroundPosition: '0% 50%' },
        },
        'blink': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.3' },
        },
        'bounce-subtle': {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-5px)' },
        },
        'gradient-shift': {
          '0%, 100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        },
      },
      backdropBlur: {
        'xs': '2px',
        '3xl': '64px',
      },
      boxShadow: {
        'cyber': '0 0 30px rgba(0, 224, 255, 0.15)',
        'cyber-hover': '0 0 60px rgba(0, 224, 255, 0.25)',
        'cyber-intense': '0 0 80px rgba(0, 224, 255, 0.4)',
        'critical': '0 0 40px rgba(255, 59, 59, 0.4)',
        'success': '0 0 40px rgba(0, 255, 163, 0.3)',
        'warning': '0 0 40px rgba(255, 176, 32, 0.3)',
        'inner-glow': 'inset 0 0 30px rgba(0, 224, 255, 0.1)',
        'card': '0 8px 32px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.02) inset',
        'card-hover': '0 16px 48px rgba(0, 0, 0, 0.5), 0 0 60px rgba(0, 224, 255, 0.1)',
      },
      borderRadius: {
        '4xl': '2rem',
      },
      transitionDuration: {
        '400': '400ms',
      },
      zIndex: {
        '60': '60',
        '70': '70',
        '80': '80',
        '90': '90',
        '100': '100',
      },
    },
  },
  plugins: [],
}
