/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pynomaly/presentation/web/templates/**/*.html",
    "./src/pynomaly/presentation/web/static/**/*.js",
    "./src/pynomaly/presentation/web/components/**/*.html",
    "./src/pynomaly/presentation/web/assets/**/*.js",
    "./tests/ui/**/*.py",
    "./docs/**/*.md"
  ],
  darkMode: 'class', // Enable dark mode support
  theme: {
    extend: {
      // Pynomaly Brand Colors
      colors: {
        // Primary Palette - Anomaly Detection Theme
        primary: {
          50: '#f0f9ff',   // Lightest blue
          100: '#e0f2fe',  // Very light blue
          200: '#bae6fd',  // Light blue
          300: '#7dd3fc',  // Medium light blue
          400: '#38bdf8',  // Medium blue
          500: '#0ea5e9',  // Primary blue (brand color)
          600: '#0284c7',  // Darker blue
          700: '#0369a1',  // Dark blue
          800: '#075985',  // Very dark blue
          900: '#0c4a6e',  // Darkest blue
          950: '#082f49'   // Ultra dark blue
        },
        
        // Secondary Palette - Success/Normal State
        secondary: {
          50: '#f0fdf4',   // Lightest green
          100: '#dcfce7',  // Very light green
          200: '#bbf7d0',  // Light green
          300: '#86efac',  // Medium light green
          400: '#4ade80',  // Medium green
          500: '#22c55e',  // Primary green
          600: '#16a34a',  // Darker green
          700: '#15803d',  // Dark green
          800: '#166534',  // Very dark green
          900: '#14532d',  // Darkest green
          950: '#052e16'   // Ultra dark green
        },
        
        // Accent Palette - Anomaly/Alert State
        accent: {
          50: '#fef2f2',   // Lightest red
          100: '#fee2e2',  // Very light red
          200: '#fecaca',  // Light red
          300: '#fca5a5',  // Medium light red
          400: '#f87171',  // Medium red
          500: '#ef4444',  // Primary red (alert color)
          600: '#dc2626',  // Darker red
          700: '#b91c1c',  // Dark red
          800: '#991b1b',  // Very dark red
          900: '#7f1d1d',  // Darkest red
          950: '#450a0a'   // Ultra dark red
        },
        
        // Chart and Visualization Colors
        chart: {
          normal: '#22c55e',     // Green for normal data
          anomaly: '#ef4444',    // Red for anomalies
          threshold: '#f59e0b',  // Yellow for thresholds
          prediction: '#8b5cf6', // Purple for predictions
          confidence: '#06b6d4'  // Teal for confidence intervals
        }
      },
      
      // Typography
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Monaco', 'monospace']
      }
    }
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography')
  ]
}
