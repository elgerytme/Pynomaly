# D3.js Chart Components Library

## Overview

A comprehensive, production-ready D3.js chart library for the Pynomaly anomaly detection platform. This library provides interactive, accessible, and responsive data visualization components with real-time updates, advanced interactions, and comprehensive accessibility support.

## Features

### 🎯 **Core Capabilities**

- **Real-time Data Updates** - Streaming data support with smooth animations
- **Interactive Visualizations** - Zoom, pan, brush selection, and tooltips
- **Responsive Design** - Adapts to all screen sizes and devices
- **Accessibility First** - WCAG 2.1 AA compliant with screen reader support
- **Theme Support** - Light/dark themes with high contrast options
- **Performance Optimized** - Efficient rendering for large datasets

### 📊 **Chart Types**

#### TimeSeriesChart

Interactive line charts for time-based anomaly detection data.

- Anomaly markers with confidence indicators
- Confidence bands for uncertainty visualization
- Real-time data streaming capabilities
- Interactive cursor tracking
- Keyboard navigation support

#### ScatterPlotChart

2D scatter plots for visualizing anomalies in feature space.

- Brush selection for data exploration
- Zoom and pan interactions
- Color and size encoding for multiple dimensions
- Interactive legends
- Point clustering visualization

#### HeatmapChart

Grid-based visualizations for correlation matrices and density maps.

- Interactive cell selection
- Multiple color schemes
- Value labels and tooltips
- Responsive grid layouts
- Export capabilities

## Quick Start

### Installation

```javascript
// Include D3.js dependency
<script src="https://d3js.org/d3.v7.min.js"></script>

// Include chart library
<script src="/static/js/src/components/d3-chart-library.js"></script>

// Include styles
<link rel="stylesheet" href="/static/css/d3-charts.css">
```

### Basic Usage

```javascript
// Create a time series chart
const chart = new TimeSeriesChart('#chart-container', {
    title: 'Anomaly Detection Timeline',
    description: 'Real-time anomaly detection results',
    width: 800,
    height: 400,
    showAnomalies: true,
    showConfidenceBands: true,
    animated: true,
    responsive: true
});

// Set initial data
chart.setData(timeSeriesData);

// Add real-time data points
chart.addDataPoint({
    timestamp: new Date(),
    value: 42.5,
    isAnomaly: false,
    confidence: 0.85
});

// Listen for events
document.getElementById('chart-container').addEventListener('anomaly-selected', (event) => {
    const { data } = event.detail;
    console.log('Anomaly selected:', data);
});
```

## Chart Components

### TimeSeriesChart

Perfect for monitoring anomaly detection over time with real-time updates.

#### Constructor Options

```javascript
const options = {
    // Basic configuration
    width: 800,                    // Chart width in pixels
    height: 400,                   // Chart height in pixels
    margin: {                      // Chart margins
        top: 20,
        right: 30,
        bottom: 40,
        left: 50
    },

    // Data accessors
    xAccessor: d => d.timestamp,   // Time accessor function
    yAccessor: d => d.value,       // Value accessor function
    anomalyAccessor: d => d.isAnomaly,      // Anomaly flag accessor
    confidenceAccessor: d => d.confidence,  // Confidence accessor

    // Visual options
    showAnomalies: true,           // Show anomaly markers
    showConfidenceBands: false,    // Show confidence intervals
    interpolation: d3.curveMonotoneX, // Line interpolation

    // Behavior
    animated: true,                // Enable animations
    responsive: true,              // Responsive behavior
    accessibility: true            // Accessibility features
};

const chart = new TimeSeriesChart('#container', options);
```

#### Data Format

```javascript
const timeSeriesData = [
    {
        timestamp: new Date('2023-01-01T00:00:00Z'),
        value: 42.5,
        isAnomaly: false,
        confidence: 0.85
    },
    {
        timestamp: new Date('2023-01-01T00:01:00Z'),
        value: 87.2,
        isAnomaly: true,
        confidence: 0.92
    }
    // ... more data points
];
```

#### Methods

```javascript
// Set new data
chart.setData(newData);

// Add single data point (for streaming)
chart.addDataPoint(point, maxPoints = 1000);

// Update chart theme
chart.updateTheme(newTheme);

// Resize chart
chart.resize();

// Clean up resources
chart.destroy();
```

#### Events

```javascript
// Anomaly marker clicked
container.addEventListener('anomaly-selected', (event) => {
    const { data, chart } = event.detail;
    // Handle anomaly selection
});
```

### ScatterPlotChart

Ideal for visualizing anomalies in 2D feature space with interactive exploration.

#### Constructor Options

```javascript
const options = {
    // Data accessors
    xAccessor: d => d.x,           // X coordinate accessor
    yAccessor: d => d.y,           // Y coordinate accessor
    colorAccessor: d => d.anomalyScore,  // Color encoding accessor
    sizeAccessor: d => d.confidence,     // Size encoding accessor
    anomalyAccessor: d => d.isAnomaly,   // Anomaly flag accessor

    // Labels
    xLabel: 'Feature 1',           // X-axis label
    yLabel: 'Feature 2',           // Y-axis label

    // Interactions
    enableBrushing: true,          // Enable brush selection
    enableZoom: true,              // Enable zoom/pan
    showDensity: false,            // Show density background

    // Visual options
    animated: true,
    responsive: true,
    accessibility: true
};

const chart = new ScatterPlotChart('#container', options);
```

#### Data Format

```javascript
const scatterData = [
    {
        x: 15.2,
        y: 42.8,
        anomalyScore: 0.12,
        confidence: 0.89,
        isAnomaly: false
    },
    {
        x: 87.5,
        y: 23.1,
        anomalyScore: 0.94,
        confidence: 0.95,
        isAnomaly: true
    }
    // ... more data points
];
```

#### Events

```javascript
// Single point selected
container.addEventListener('point-selected', (event) => {
    const { data, chart } = event.detail;
    // Handle point selection
});

// Multiple points selected via brushing
container.addEventListener('points-selected', (event) => {
    const { data, chart } = event.detail;
    // Handle brush selection
});
```

### HeatmapChart

Excellent for correlation matrices and density visualization.

#### Constructor Options

```javascript
const options = {
    // Data accessors
    xAccessor: d => d.x,           // X category accessor
    yAccessor: d => d.y,           // Y category accessor
    valueAccessor: d => d.value,   // Value accessor

    // Visual options
    gridSize: 20,                  // Grid cell size
    colorScheme: d3.interpolateViridis, // Color scheme
    showLabels: true,              // Show value labels

    // Behavior
    enableZoom: false,             // Zoom behavior
    animated: true,
    responsive: true,
    accessibility: true
};

const chart = new HeatmapChart('#container', options);
```

#### Data Format

```javascript
const heatmapData = [
    {
        x: 'Feature A',
        y: 'Feature B',
        value: 0.75
    },
    {
        x: 'Feature A',
        y: 'Feature C',
        value: 0.32
    }
    // ... more data points
];
```

#### Events

```javascript
// Heatmap cell selected
container.addEventListener('cell-selected', (event) => {
    const { data, chart } = event.detail;
    // Handle cell selection
});
```

## Advanced Features

### Real-Time Data Streaming

```javascript
// Set up real-time updates
const updateInterval = setInterval(() => {
    const newPoint = {
        timestamp: new Date(),
        value: generateRandomValue(),
        isAnomaly: detectAnomaly(),
        confidence: calculateConfidence()
    };

    // Add to chart (keeps last 1000 points)
    chart.addDataPoint(newPoint, 1000);
}, 2000);

// Clean up
clearInterval(updateInterval);
```

### Theme Management

```javascript
// Initialize chart library
const chartLibrary = new D3ChartLibrary();

// Switch themes
chartLibrary.currentTheme = 'dark';
chartLibrary.updateAllChartsTheme();

// Listen for system theme changes
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    const theme = e.matches ? 'dark' : 'light';
    chartLibrary.currentTheme = theme;
    chartLibrary.updateAllChartsTheme();
});
```

### Accessibility Features

```javascript
// Enable comprehensive accessibility
const chart = new TimeSeriesChart('#container', {
    accessibility: true,
    title: 'Anomaly Detection Results',
    description: 'Time series chart showing 100 data points with 3 detected anomalies'
});

// Announce updates to screen readers
chart.announceToScreenReader('Chart updated with new data');

// Add keyboard navigation
document.addEventListener('keydown', (event) => {
    if (event.key === 'Tab' && event.target.closest('.chart-container')) {
        // Handle chart keyboard navigation
    }
});
```

### Performance Optimization

```javascript
// Optimize for large datasets
const chart = new TimeSeriesChart('#container', {
    animated: false,              // Disable animations for performance
    responsive: true,
    // Use efficient data structures
    xAccessor: d => d[0],         // Use array indices for better performance
    yAccessor: d => d[1]
});

// Batch data updates
const dataBuffer = [];
const flushInterval = setInterval(() => {
    if (dataBuffer.length > 0) {
        chart.setData([...chart.data, ...dataBuffer]);
        dataBuffer.length = 0;
    }
}, 100);
```

## Styling and Customization

### CSS Variables

```css
:root {
    --chart-background: #ffffff;
    --chart-border: #e2e8f0;
    --chart-grid: #f1f5f9;
    --primary-color: #0ea5e9;
    --secondary-color: #22c55e;
    --danger-color: #dc2626;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
}

/* Dark theme */
[data-theme="dark"] {
    --chart-background: #1e293b;
    --chart-border: #374151;
    --chart-grid: #4b5563;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
}
```

### Custom Styling

```css
/* Customize chart containers */
.chart-container {
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* Style anomaly markers */
.chart-container .anomaly-marker {
    fill: #ff4444;
    stroke: #ffffff;
    stroke-width: 3;
}

/* Custom tooltip styling */
.chart-tooltip {
    background: rgba(0, 0, 0, 0.95);
    color: white;
    border-radius: 8px;
    font-size: 14px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}
```

## Browser Support

- **Chrome** 90+
- **Firefox** 88+
- **Safari** 14+
- **Edge** 90+
- **Mobile browsers** (iOS Safari, Chrome Mobile)

## Performance Considerations

### Memory Management

- Charts automatically clean up event listeners on destroy
- Use `chart.destroy()` when removing charts from DOM
- Large datasets are automatically throttled for smooth performance

### Animation Performance

- Animations use `requestAnimationFrame` for optimal performance
- Automatic animation reduction for users with `prefers-reduced-motion`
- Configurable animation duration and easing

### Data Optimization

- Use efficient data structures (arrays vs objects)
- Implement data pagination for very large datasets
- Consider data aggregation for time series with many points

## Accessibility Compliance

### WCAG 2.1 AA Features

- **Color Contrast** - All visual elements meet 4.5:1 contrast ratio
- **Keyboard Navigation** - Full keyboard control for all interactions
- **Screen Reader Support** - Comprehensive ARIA labels and live regions
- **Focus Management** - Clear focus indicators and logical tab order
- **Alternative Content** - Text alternatives for all visual information

### Implementation

```javascript
// Accessibility-first chart creation
const chart = new TimeSeriesChart('#container', {
    accessibility: true,
    title: 'Monthly Anomaly Detection Results',
    description: 'Line chart showing anomaly detection over 12 months with 15 detected anomalies',
    // Chart will automatically include ARIA labels, descriptions, and keyboard navigation
});

// Screen reader announcements
chart.announceToScreenReader('New anomaly detected with 94% confidence');
```

## Testing

### Unit Testing

```javascript
// Example test
describe('TimeSeriesChart', () => {
    test('should render with data', () => {
        const container = document.createElement('div');
        const chart = new TimeSeriesChart(container, {});
        chart.setData(mockData);

        expect(container.querySelector('svg')).toBeTruthy();
        expect(container.querySelectorAll('.anomaly-marker')).toHaveLength(3);
    });
});
```

### Accessibility Testing

```javascript
// Automated accessibility testing with axe-core
import { toHaveNoViolations } from 'jest-axe';

test('chart should be accessible', async () => {
    const container = document.createElement('div');
    const chart = new TimeSeriesChart(container, { accessibility: true });
    chart.setData(mockData);

    const results = await axe(container);
    expect(results).toHaveNoViolations();
});
```

## Contributing

### Development Setup

```bash
# Install dependencies
npm install d3

# Run tests
npm test

# Build for production
npm run build
```

### Code Style

- Follow ESLint configuration
- Use TypeScript for type safety
- Include comprehensive JSDoc comments
- Write tests for all new features

## Examples

See the comprehensive demo at `/d3-charts-demo.html` for interactive examples of all chart types with real-time data, accessibility features, and customization options.

## License

Part of the Pynomaly anomaly detection platform. See main project license for details.

## Support

For issues, feature requests, or questions:

- Create an issue in the main Pynomaly repository
- Check the demo page for usage examples
- Review the accessibility guidelines documentation
