/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pynomaly/presentation/web/templates/**/*.html",
    "./src/pynomaly/presentation/web/assets/js/**/*.js",
    "./src/pynomaly/presentation/web/static/js/**/*.js",
  ],
  theme: {
    extend: {
      colors: {
        'pynomaly': {
          50: '#eff6ff',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        },
        'anomaly': {
          50: '#fef2f2',
          500: '#ef4444',
          600: '#dc2626',
          700: '#b91c1c',
        }
      },
      fontFamily: {
        'sans': ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
    require('@tailwindcss/aspect-ratio'),
  ],
}