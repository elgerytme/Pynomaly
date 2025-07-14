/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/htmx_app/templates/**/*.html",
    "./src/htmx_app/static/js/**/*.js",
    "./src/htmx_app/**/*.py",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
          950: '#172554',
        },
        secondary: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
          950: '#020617',
        },
        success: {
          50: '#f0fdf4',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80',
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
          800: '#166534',
          900: '#14532d',
          950: '#052e16',
        },
        warning: {
          50: '#fffbeb',
          100: '#fef3c7',
          200: '#fde68a',
          300: '#fcd34d',
          400: '#fbbf24',
          500: '#f59e0b',
          600: '#d97706',
          700: '#b45309',
          800: '#92400e',
          900: '#78350f',
          950: '#451a03',
        },
        danger: {
          50: '#fef2f2',
          100: '#fee2e2',
          200: '#fecaca',
          300: '#fca5a5',
          400: '#f87171',
          500: '#ef4444',
          600: '#dc2626',
          700: '#b91c1c',
          800: '#991b1b',
          900: '#7f1d1d',
          950: '#450a0a',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '104': '26rem',
        '120': '30rem',
      },
      borderRadius: {
        '4xl': '2rem',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
        'pulse-slow': 'pulse 3s infinite',
        'bounce-light': 'bounce 2s infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(100%)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideDown: {
          '0%': { transform: 'translateY(-100%)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
      boxShadow: {
        'soft': '0 2px 15px 0 rgba(0, 0, 0, 0.05)',
        'medium': '0 4px 25px 0 rgba(0, 0, 0, 0.1)',
        'hard': '0 10px 50px 0 rgba(0, 0, 0, 0.15)',
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms')({
      strategy: 'class', // only generate classes
    }),
    require('@tailwindcss/typography'),
    require('@tailwindcss/aspect-ratio'),
    
    // Custom plugin for component classes
    function({ addComponents, theme }) {
      addComponents({
        '.btn': {
          padding: `${theme('spacing.2')} ${theme('spacing.4')}`,
          borderRadius: theme('borderRadius.md'),
          fontWeight: theme('fontWeight.medium'),
          transition: 'all 0.2s ease-in-out',
          cursor: 'pointer',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          '&:disabled': {
            opacity: '0.5',
            cursor: 'not-allowed',
          },
        },
        '.btn-sm': {
          padding: `${theme('spacing.1')} ${theme('spacing.3')}`,
          fontSize: theme('fontSize.sm'),
        },
        '.btn-lg': {
          padding: `${theme('spacing.3')} ${theme('spacing.6')}`,
          fontSize: theme('fontSize.lg'),
        },
        '.btn-primary': {
          backgroundColor: theme('colors.primary.600'),
          color: theme('colors.white'),
          '&:hover': {
            backgroundColor: theme('colors.primary.700'),
          },
          '&:focus': {
            outline: 'none',
            boxShadow: `0 0 0 3px ${theme('colors.primary.200')}`,
          },
        },
        '.btn-secondary': {
          backgroundColor: theme('colors.secondary.200'),
          color: theme('colors.secondary.900'),
          '&:hover': {
            backgroundColor: theme('colors.secondary.300'),
          },
          '&:focus': {
            outline: 'none',
            boxShadow: `0 0 0 3px ${theme('colors.secondary.300')}`,
          },
        },
        '.btn-outline': {
          backgroundColor: 'transparent',
          borderWidth: '1px',
          borderColor: theme('colors.primary.600'),
          color: theme('colors.primary.600'),
          '&:hover': {
            backgroundColor: theme('colors.primary.600'),
            color: theme('colors.white'),
          },
        },
        '.card': {
          backgroundColor: theme('colors.white'),
          borderRadius: theme('borderRadius.lg'),
          boxShadow: theme('boxShadow.soft'),
          overflow: 'hidden',
        },
        '.card-header': {
          padding: `${theme('spacing.4')} ${theme('spacing.6')}`,
          backgroundColor: theme('colors.gray.50'),
          borderBottomWidth: '1px',
          borderBottomColor: theme('colors.gray.200'),
        },
        '.card-body': {
          padding: `${theme('spacing.4')} ${theme('spacing.6')}`,
        },
        '.card-footer': {
          padding: `${theme('spacing.4')} ${theme('spacing.6')}`,
          backgroundColor: theme('colors.gray.50'),
          borderTopWidth: '1px',
          borderTopColor: theme('colors.gray.200'),
        },
        '.form-input': {
          borderWidth: '1px',
          borderColor: theme('colors.gray.300'),
          borderRadius: theme('borderRadius.md'),
          padding: `${theme('spacing.2')} ${theme('spacing.3')}`,
          '&:focus': {
            outline: 'none',
            borderColor: theme('colors.primary.500'),
            boxShadow: `0 0 0 3px ${theme('colors.primary.200')}`,
          },
        },
        '.alert': {
          padding: theme('spacing.4'),
          borderRadius: theme('borderRadius.md'),
          borderWidth: '1px',
        },
        '.alert-success': {
          backgroundColor: theme('colors.success.50'),
          borderColor: theme('colors.success.200'),
          color: theme('colors.success.800'),
        },
        '.alert-warning': {
          backgroundColor: theme('colors.warning.50'),
          borderColor: theme('colors.warning.200'),
          color: theme('colors.warning.800'),
        },
        '.alert-danger': {
          backgroundColor: theme('colors.danger.50'),
          borderColor: theme('colors.danger.200'),
          color: theme('colors.danger.800'),
        },
        '.alert-info': {
          backgroundColor: theme('colors.blue.50'),
          borderColor: theme('colors.blue.200'),
          color: theme('colors.blue.800'),
        },
      })
    }
  ],
}