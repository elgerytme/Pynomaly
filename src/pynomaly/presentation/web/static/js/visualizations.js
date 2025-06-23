// Visualization functions using D3.js and Apache ECharts

// Global chart instances
let timelineChart;
let rateChart;
let scatterChart;
let heatmapChart;
let dashboardChart;

// Initialize charts on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    
    // Auto-refresh every 30 seconds
    setInterval(refreshCharts, 30000);
});

// Initialize all charts
function initializeCharts() {
    createTimelineChart();
    createRateChart();
    createScatterChart();
    createHeatmapChart();
    createDashboard();
}

// D3.js Timeline Chart
function createTimelineChart() {
    const container = d3.select('#timeline-chart');
    container.selectAll('*').remove();
    
    const margin = {top: 20, right: 30, bottom: 40, left: 60};
    const width = container.node().getBoundingClientRect().width - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;
    
    const svg = container
        .append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Parse dates
    const data = detectionTimeline.map(d => ({
        ...d,
        timestamp: new Date(d.timestamp)
    }));
    
    // Scales
    const x = d3.scaleTime()
        .domain(d3.extent(data, d => d.timestamp))
        .range([0, width]);
    
    const y = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.anomalies)])
        .nice()
        .range([height, 0]);
    
    // Line generator
    const line = d3.line()
        .x(d => x(d.timestamp))
        .y(d => y(d.anomalies))
        .curve(d3.curveMonotoneX);
    
    // Add axes
    svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x).tickFormat(d3.timeFormat('%m/%d %H:%M')));
    
    svg.append('g')
        .call(d3.axisLeft(y));
    
    // Add axis labels
    svg.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('y', 0 - margin.left)
        .attr('x', 0 - (height / 2))
        .attr('dy', '1em')
        .style('text-anchor', 'middle')
        .text('Number of Anomalies');
    
    // Add line
    svg.append('path')
        .datum(data)
        .attr('fill', 'none')
        .attr('stroke', '#3B82F6')
        .attr('stroke-width', 2)
        .attr('d', line);
    
    // Add dots
    svg.selectAll('.dot')
        .data(data)
        .enter().append('circle')
        .attr('class', 'dot')
        .attr('cx', d => x(d.timestamp))
        .attr('cy', d => y(d.anomalies))
        .attr('r', 4)
        .attr('fill', '#3B82F6')
        .on('mouseover', function(event, d) {
            const tooltip = d3.select('body').append('div')
                .attr('class', 'd3-tooltip')
                .style('opacity', 0);
            
            tooltip.transition()
                .duration(200)
                .style('opacity', .9);
            
            tooltip.html(`Time: ${d.timestamp.toLocaleString()}<br/>Anomalies: ${d.anomalies}`)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 28) + 'px');
        })
        .on('mouseout', function() {
            d3.selectAll('.d3-tooltip').remove();
        });
}

// ECharts Rate Chart
function createRateChart() {
    const chartDom = document.getElementById('rate-chart');
    rateChart = echarts.init(chartDom);
    
    // Group data by detector
    const detectorRates = {};
    anomalyRates.forEach(item => {
        if (!detectorRates[item.detector_id]) {
            detectorRates[item.detector_id] = [];
        }
        detectorRates[item.detector_id].push(item.rate);
    });
    
    const detectors = Object.keys(detectorRates);
    const avgRates = detectors.map(d => {
        const rates = detectorRates[d];
        return (rates.reduce((a, b) => a + b, 0) / rates.length * 100).toFixed(2);
    });
    
    const option = {
        title: {
            show: false
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            },
            formatter: '{b}: {c}%'
        },
        xAxis: {
            type: 'category',
            data: detectors.map(d => `Detector ${d.slice(-4)}`),
            axisLabel: {
                rotate: 45
            }
        },
        yAxis: {
            type: 'value',
            name: 'Anomaly Rate (%)',
            axisLabel: {
                formatter: '{value}%'
            }
        },
        series: [{
            data: avgRates,
            type: 'bar',
            itemStyle: {
                color: function(params) {
                    const value = params.value;
                    if (value > 20) return '#EF4444';
                    if (value > 10) return '#F59E0B';
                    return '#10B981';
                }
            }
        }]
    };
    
    rateChart.setOption(option);
    
    // Resize handler
    window.addEventListener('resize', function() {
        rateChart.resize();
    });
}

// D3.js Scatter Plot
function createScatterChart() {
    const container = d3.select('#scatter-chart');
    container.selectAll('*').remove();
    
    // Generate sample data for demonstration
    const data = generateScatterData();
    
    const margin = {top: 20, right: 30, bottom: 40, left: 60};
    const width = container.node().getBoundingClientRect().width - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;
    
    const svg = container
        .append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Scales
    const x = d3.scaleLinear()
        .domain([0, 1])
        .range([0, width]);
    
    const y = d3.scaleLinear()
        .domain([0, 100])
        .range([height, 0]);
    
    // Color scale
    const color = d3.scaleOrdinal()
        .domain(['normal', 'anomaly'])
        .range(['#10B981', '#EF4444']);
    
    // Add axes
    svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x));
    
    svg.append('g')
        .call(d3.axisLeft(y));
    
    // Add axis labels
    svg.append('text')
        .attr('text-anchor', 'end')
        .attr('x', width)
        .attr('y', height + margin.bottom)
        .text('Anomaly Score');
    
    svg.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('y', 0 - margin.left)
        .attr('x', 0 - (height / 2))
        .attr('dy', '1em')
        .style('text-anchor', 'middle')
        .text('Sample Index');
    
    // Add dots
    svg.selectAll('.dot')
        .data(data)
        .enter().append('circle')
        .attr('class', 'dot')
        .attr('r', 3)
        .attr('cx', d => x(d.score))
        .attr('cy', d => y(d.index))
        .style('fill', d => color(d.type))
        .style('opacity', 0.7)
        .on('mouseover', function(event, d) {
            d3.select(this).attr('r', 5);
            
            const tooltip = d3.select('body').append('div')
                .attr('class', 'd3-tooltip')
                .style('opacity', 0);
            
            tooltip.transition()
                .duration(200)
                .style('opacity', .9);
            
            tooltip.html(`Score: ${d.score.toFixed(3)}<br/>Type: ${d.type}`)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 28) + 'px');
        })
        .on('mouseout', function() {
            d3.select(this).attr('r', 3);
            d3.selectAll('.d3-tooltip').remove();
        });
    
    // Add legend
    const legend = svg.selectAll('.legend')
        .data(color.domain())
        .enter().append('g')
        .attr('class', 'legend')
        .attr('transform', (d, i) => `translate(0,${i * 20})`);
    
    legend.append('rect')
        .attr('x', width - 18)
        .attr('width', 18)
        .attr('height', 18)
        .style('fill', color);
    
    legend.append('text')
        .attr('x', width - 24)
        .attr('y', 9)
        .attr('dy', '.35em')
        .style('text-anchor', 'end')
        .text(d => d);
}

// ECharts Heatmap
function createHeatmapChart() {
    const chartDom = document.getElementById('heatmap-chart');
    heatmapChart = echarts.init(chartDom);
    
    // Generate heatmap data
    const hours = Array.from({length: 24}, (_, i) => `${i}:00`);
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    
    const data = [];
    for (let i = 0; i < days.length; i++) {
        for (let j = 0; j < hours.length; j++) {
            // Simulate anomaly frequency (higher during business hours)
            const value = Math.random() * 10 + (j >= 9 && j <= 17 ? 5 : 0);
            data.push([j, i, value.toFixed(2)]);
        }
    }
    
    const option = {
        tooltip: {
            position: 'top',
            formatter: function(params) {
                return `${days[params.value[1]]} ${hours[params.value[0]]}<br/>Anomalies: ${params.value[2]}`;
            }
        },
        grid: {
            height: '70%',
            top: '10%'
        },
        xAxis: {
            type: 'category',
            data: hours,
            splitArea: {
                show: true
            }
        },
        yAxis: {
            type: 'category',
            data: days,
            splitArea: {
                show: true
            }
        },
        visualMap: {
            min: 0,
            max: 20,
            calculable: true,
            orient: 'horizontal',
            left: 'center',
            bottom: '5%',
            inRange: {
                color: ['#D1FAE5', '#34D399', '#10B981', '#059669', '#047857']
            }
        },
        series: [{
            name: 'Anomaly Frequency',
            type: 'heatmap',
            data: data,
            label: {
                show: false
            },
            emphasis: {
                itemStyle: {
                    shadowBlur: 10,
                    shadowColor: 'rgba(0, 0, 0, 0.5)'
                }
            }
        }]
    };
    
    heatmapChart.setOption(option);
    
    window.addEventListener('resize', function() {
        heatmapChart.resize();
    });
}

// ECharts Dashboard
function createDashboard() {
    const chartDom = document.getElementById('dashboard-chart');
    dashboardChart = echarts.init(chartDom);
    
    const option = {
        title: {
            text: 'Anomaly Detection Performance Metrics',
            left: 'center'
        },
        grid: [
            {left: '5%', top: '15%', width: '40%', height: '35%'},
            {right: '5%', top: '15%', width: '40%', height: '35%'},
            {left: '5%', bottom: '5%', width: '40%', height: '35%'},
            {right: '5%', bottom: '5%', width: '40%', height: '35%'}
        ],
        xAxis: [
            {gridIndex: 0, type: 'category', data: ['Precision', 'Recall', 'F1-Score']},
            {gridIndex: 1, type: 'value'},
            {gridIndex: 2, type: 'category', data: ['IForest', 'LOF', 'OCSVM', 'AutoEncoder']},
            {gridIndex: 3, type: 'time'}
        ],
        yAxis: [
            {gridIndex: 0, type: 'value', max: 1},
            {gridIndex: 1, type: 'value'},
            {gridIndex: 2, type: 'value'},
            {gridIndex: 3, type: 'value'}
        ],
        series: [
            // Performance Metrics
            {
                type: 'bar',
                xAxisIndex: 0,
                yAxisIndex: 0,
                data: [0.92, 0.88, 0.90],
                itemStyle: {color: '#3B82F6'}
            },
            // ROC Curve
            {
                type: 'line',
                xAxisIndex: 1,
                yAxisIndex: 1,
                data: generateROCData(),
                smooth: true,
                itemStyle: {color: '#10B981'}
            },
            // Algorithm Comparison
            {
                type: 'bar',
                xAxisIndex: 2,
                yAxisIndex: 2,
                data: [450, 380, 420, 510],
                itemStyle: {color: '#F59E0B'}
            },
            // Real-time Monitoring
            {
                type: 'line',
                xAxisIndex: 3,
                yAxisIndex: 3,
                data: generateTimeSeriesData(),
                smooth: true,
                areaStyle: {
                    opacity: 0.3
                },
                itemStyle: {color: '#EF4444'}
            }
        ]
    };
    
    dashboardChart.setOption(option);
    
    window.addEventListener('resize', function() {
        dashboardChart.resize();
    });
}

// Helper functions
function generateScatterData() {
    const data = [];
    for (let i = 0; i < 100; i++) {
        const score = Math.random();
        const isAnomaly = score > 0.8;
        data.push({
            index: i,
            score: score,
            type: isAnomaly ? 'anomaly' : 'normal'
        });
    }
    return data;
}

function generateROCData() {
    const data = [];
    for (let i = 0; i <= 100; i++) {
        const x = i / 100;
        const y = Math.pow(x, 0.3);
        data.push([x, y]);
    }
    return data;
}

function generateTimeSeriesData() {
    const data = [];
    const now = new Date();
    for (let i = 0; i < 100; i++) {
        const time = new Date(now.getTime() - (100 - i) * 60 * 1000);
        const value = Math.random() * 20 + 10 + Math.sin(i / 10) * 5;
        data.push([time, value]);
    }
    return data;
}

// Update time range
function updateTimeRange(range) {
    // Update button states
    document.querySelectorAll('.viz-control-button').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.range === range) {
            btn.classList.add('active');
        }
    });
    
    // Refresh charts with new data
    refreshCharts();
}

// Refresh all charts
function refreshCharts() {
    // In a real application, this would fetch new data from the server
    createTimelineChart();
    
    if (rateChart) rateChart.resize();
    if (heatmapChart) heatmapChart.resize();
    if (dashboardChart) dashboardChart.resize();
    
    createScatterChart();
}