/**
 * Advanced Visualizations JavaScript
 * Handles complex D3.js and ECharts visualizations
 */

// Global state for visualization management
window.AdvancedViz = {
    charts: {},
    interactions: {},
    animations: {},
    
    // Initialize all advanced visualizations
    init() {
        this.initAnomalyHeatmap();
        this.initTimeSeriesExplorer();
        this.initFeatureCorrelationMatrix();
        this.initInteractiveScatterplot();
        this.initAnomalyDistribution();
        this.initDetectorComparison();
        this.setupInteractions();
    },
    
    // Anomaly Heatmap with D3.js
    initAnomalyHeatmap() {
        const container = d3.select("#anomaly-heatmap");
        if (container.empty()) return;
        
        const margin = {top: 80, right: 80, bottom: 100, left: 100};
        const width = 800 - margin.left - margin.right;
        const height = 600 - margin.top - margin.bottom;
        
        // Clear existing
        container.selectAll("*").remove();
        
        const svg = container
            .append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom);
            
        const g = svg.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);
            
        // Generate sample heatmap data
        const features = Array.from({length: 20}, (_, i) => `Feature_${i+1}`);
        const samples = Array.from({length: 50}, (_, i) => `Sample_${i+1}`);
        const data = [];
        
        for (let i = 0; i < features.length; i++) {
            for (let j = 0; j < samples.length; j++) {
                data.push({
                    feature: features[i],
                    sample: samples[j],
                    value: Math.random(),
                    anomaly: Math.random() > 0.9
                });
            }
        }
        
        // Scales
        const xScale = d3.scaleBand()
            .domain(samples)
            .range([0, width])
            .padding(0.01);
            
        const yScale = d3.scaleBand()
            .domain(features)
            .range([0, height])
            .padding(0.01);
            
        const colorScale = d3.scaleSequential(d3.interpolateRdYlBu)
            .domain([0, 1]);
            
        // Add zoom and pan
        const zoom = d3.zoom()
            .scaleExtent([1, 8])
            .on("zoom", (event) => {
                g.attr("transform", event.transform);
            });
            
        svg.call(zoom);
        
        // Tooltip
        const tooltip = d3.select("body").append("div")
            .attr("class", "tooltip")
            .style("opacity", 0)
            .style("position", "absolute")
            .style("background", "rgba(0, 0, 0, 0.8)")
            .style("color", "white")
            .style("padding", "8px")
            .style("border-radius", "4px")
            .style("font-size", "12px")
            .style("pointer-events", "none");
        
        // Draw heatmap cells
        g.selectAll(".cell")
            .data(data)
            .enter().append("rect")
            .attr("class", "cell")
            .attr("x", d => xScale(d.sample))
            .attr("y", d => yScale(d.feature))
            .attr("width", xScale.bandwidth())
            .attr("height", yScale.bandwidth())
            .attr("fill", d => d.anomaly ? "#ff4757" : colorScale(d.value))
            .attr("stroke", d => d.anomaly ? "#2f3542" : "none")
            .attr("stroke-width", d => d.anomaly ? 2 : 0)
            .on("mouseover", (event, d) => {
                tooltip.transition().duration(200).style("opacity", .9);
                tooltip.html(`
                    <strong>${d.feature}</strong><br/>
                    Sample: ${d.sample}<br/>
                    Value: ${d.value.toFixed(3)}<br/>
                    ${d.anomaly ? '<span style="color: #ff4757;">⚠️ Anomaly</span>' : 'Normal'}
                `)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 28) + "px");
            })
            .on("mouseout", () => {
                tooltip.transition().duration(500).style("opacity", 0);
            })
            .on("click", (event, d) => {
                this.showAnomalyDetails(d);
            });
        
        // Add axes
        g.append("g")
            .attr("class", "x axis")
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(xScale))
            .selectAll("text")
            .style("text-anchor", "end")
            .attr("dx", "-.8em")
            .attr("dy", ".15em")
            .attr("transform", "rotate(-65)");
            
        g.append("g")
            .attr("class", "y axis")
            .call(d3.axisLeft(yScale));
        
        // Add title
        svg.append("text")
            .attr("x", (width + margin.left + margin.right) / 2)
            .attr("y", margin.top / 2)
            .attr("text-anchor", "middle")
            .style("font-size", "16px")
            .style("font-weight", "bold")
            .text("Feature-Sample Anomaly Heatmap");
            
        this.charts.heatmap = { svg, g, xScale, yScale, data };
    },
    
    // Time Series Explorer with brush and zoom
    initTimeSeriesExplorer() {
        const container = d3.select("#time-series-explorer");
        if (container.empty()) return;
        
        const margin = {top: 20, right: 20, bottom: 100, left: 50};
        const margin2 = {top: 430, right: 20, bottom: 30, left: 50};
        const width = 900 - margin.left - margin.right;
        const height = 500 - margin.top - margin.bottom;
        const height2 = 500 - margin2.top - margin2.bottom;
        
        container.selectAll("*").remove();
        
        const svg = container.append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom);
        
        // Generate time series data
        const parseTime = d3.timeParse("%Y-%m-%d %H:%M");
        const formatTime = d3.timeFormat("%Y-%m-%d %H:%M");
        
        const data = Array.from({length: 1000}, (_, i) => {
            const date = new Date(Date.now() - (1000 - i) * 60 * 60 * 1000);
            const value = Math.sin(i / 50) * 10 + Math.random() * 5;
            const anomaly = Math.random() > 0.95;
            return {
                date,
                value: anomaly ? value * 2 : value,
                anomaly
            };
        });
        
        // Scales
        const xScale = d3.scaleTime()
            .domain(d3.extent(data, d => d.date))
            .range([0, width]);
            
        const yScale = d3.scaleLinear()
            .domain(d3.extent(data, d => d.value))
            .range([height, 0]);
            
        const xScale2 = d3.scaleTime()
            .domain(xScale.domain())
            .range([0, width]);
            
        const yScale2 = d3.scaleLinear()
            .domain(yScale.domain())
            .range([height2, 0]);
        
        // Line generators
        const line = d3.line()
            .x(d => xScale(d.date))
            .y(d => yScale(d.value))
            .curve(d3.curveMonotoneX);
            
        const line2 = d3.line()
            .x(d => xScale2(d.date))
            .y(d => yScale2(d.value))
            .curve(d3.curveMonotoneX);
        
        // Clip path
        svg.append("defs").append("clipPath")
            .attr("id", "clip")
            .append("rect")
            .attr("width", width)
            .attr("height", height);
        
        // Focus area
        const focus = svg.append("g")
            .attr("class", "focus")
            .attr("transform", `translate(${margin.left},${margin.top})`);
            
        // Context area
        const context = svg.append("g")
            .attr("class", "context")
            .attr("transform", `translate(${margin2.left},${margin2.top})`);
        
        // Brush
        const brush = d3.brushX()
            .extent([[0, 0], [width, height2]])
            .on("brush end", brushed);
        
        // Zoom
        const zoom = d3.zoom()
            .scaleExtent([1, Infinity])
            .translateExtent([[0, 0], [width, height]])
            .extent([[0, 0], [width, height]])
            .on("zoom", zoomed);
        
        // Add data
        focus.append("path")
            .datum(data)
            .attr("class", "line")
            .attr("clip-path", "url(#clip)")
            .style("fill", "none")
            .style("stroke", "#3B82F6")
            .style("stroke-width", 2)
            .attr("d", line);
            
        // Add anomaly points
        focus.selectAll(".anomaly-point")
            .data(data.filter(d => d.anomaly))
            .enter().append("circle")
            .attr("class", "anomaly-point")
            .attr("clip-path", "url(#clip)")
            .attr("cx", d => xScale(d.date))
            .attr("cy", d => yScale(d.value))
            .attr("r", 4)
            .style("fill", "#EF4444")
            .style("stroke", "#B91C1C")
            .style("stroke-width", 2);
        
        // Context line
        context.append("path")
            .datum(data)
            .attr("class", "line")
            .style("fill", "none")
            .style("stroke", "#6B7280")
            .style("stroke-width", 1)
            .attr("d", line2);
        
        // Add brush
        context.append("g")
            .attr("class", "brush")
            .call(brush)
            .call(brush.move, xScale.range());
        
        // Add axes
        focus.append("g")
            .attr("class", "axis axis--x")
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(xScale));
            
        focus.append("g")
            .attr("class", "axis axis--y")
            .call(d3.axisLeft(yScale));
            
        context.append("g")
            .attr("class", "axis axis--x")
            .attr("transform", `translate(0,${height2})`)
            .call(d3.axisBottom(xScale2));
        
        // Add zoom rect
        svg.append("rect")
            .attr("class", "zoom")
            .attr("width", width)
            .attr("height", height)
            .attr("transform", `translate(${margin.left},${margin.top})`)
            .style("fill", "none")
            .style("pointer-events", "all")
            .call(zoom);
        
        function brushed(event) {
            if (event.sourceEvent && event.sourceEvent.type === "zoom") return;
            const s = event.selection || xScale2.range();
            xScale.domain(s.map(xScale2.invert, xScale2));
            focus.select(".line").attr("d", line);
            focus.selectAll(".anomaly-point")
                .attr("cx", d => xScale(d.date))
                .attr("cy", d => yScale(d.value));
            focus.select(".axis--x").call(d3.axisBottom(xScale));
            svg.select(".zoom").call(zoom.transform, d3.zoomIdentity
                .scale(width / (s[1] - s[0]))
                .translate(-s[0], 0));
        }
        
        function zoomed(event) {
            if (event.sourceEvent && event.sourceEvent.type === "brush") return;
            const t = event.transform;
            xScale.domain(t.rescaleX(xScale2).domain());
            focus.select(".line").attr("d", line);
            focus.selectAll(".anomaly-point")
                .attr("cx", d => xScale(d.date))
                .attr("cy", d => yScale(d.value));
            focus.select(".axis--x").call(d3.axisBottom(xScale));
            context.select(".brush").call(brush.move, xScale.range().map(t.invertX, t));
        }
        
        this.charts.timeSeries = { svg, focus, context, data };
    },
    
    // Feature Correlation Matrix
    initFeatureCorrelationMatrix() {
        const container = document.getElementById('feature-correlation-matrix');
        if (!container) return;
        
        // Generate correlation matrix data
        const features = Array.from({length: 15}, (_, i) => `Feature_${i+1}`);
        const correlationData = [];
        
        for (let i = 0; i < features.length; i++) {
            for (let j = 0; j < features.length; j++) {
                const correlation = i === j ? 1 : (Math.random() - 0.5) * 2;
                correlationData.push([i, j, correlation]);
            }
        }
        
        const option = {
            title: {
                text: 'Feature Correlation Matrix',
                left: 'center',
                textStyle: {
                    fontSize: 16,
                    fontWeight: 'bold'
                }
            },
            tooltip: {
                trigger: 'item',
                formatter: function(params) {
                    return `${features[params.data[0]]} ↔ ${features[params.data[1]]}<br/>Correlation: ${params.data[2].toFixed(3)}`;
                }
            },
            animation: true,
            grid: {
                height: '70%',
                top: '10%'
            },
            xAxis: {
                type: 'category',
                data: features,
                splitArea: {
                    show: true
                },
                axisLabel: {
                    rotate: 45,
                    fontSize: 10
                }
            },
            yAxis: {
                type: 'category',
                data: features,
                splitArea: {
                    show: true
                },
                axisLabel: {
                    fontSize: 10
                }
            },
            visualMap: {
                min: -1,
                max: 1,
                calculable: true,
                orient: 'horizontal',
                left: 'center',
                bottom: '5%',
                inRange: {
                    color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffcc', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
                }
            },
            series: [{
                name: 'Correlation',
                type: 'heatmap',
                data: correlationData,
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
        
        const chart = echarts.init(container);
        chart.setOption(option);
        
        // Make responsive
        window.addEventListener('resize', () => chart.resize());
        
        this.charts.correlation = chart;
    },
    
    // Interactive Scatterplot with clustering
    initInteractiveScatterplot() {
        const container = document.getElementById('interactive-scatterplot');
        if (!container) return;
        
        // Generate scatter data with clusters
        const clusters = 5;
        const pointsPerCluster = 100;
        const scatterData = [];
        const anomalyData = [];
        
        for (let c = 0; c < clusters; c++) {
            const centerX = Math.random() * 100;
            const centerY = Math.random() * 100;
            
            for (let i = 0; i < pointsPerCluster; i++) {
                const x = centerX + (Math.random() - 0.5) * 20;
                const y = centerY + (Math.random() - 0.5) * 20;
                const isAnomaly = Math.random() > 0.95;
                
                const point = [x, y, c];
                
                if (isAnomaly) {
                    anomalyData.push(point);
                } else {
                    scatterData.push(point);
                }
            }
        }
        
        const option = {
            title: {
                text: 'Anomaly Detection Scatterplot',
                left: 'center',
                textStyle: {
                    fontSize: 16,
                    fontWeight: 'bold'
                }
            },
            tooltip: {
                trigger: 'item',
                formatter: function(params) {
                    const type = params.seriesName === 'Normal' ? 'Normal Point' : 'Anomaly';
                    return `${type}<br/>X: ${params.data[0].toFixed(2)}<br/>Y: ${params.data[1].toFixed(2)}<br/>Cluster: ${params.data[2]}`;
                }
            },
            legend: {
                bottom: 10,
                data: ['Normal', 'Anomalies']
            },
            brush: {
                toolbox: ['rect', 'polygon', 'lineX', 'lineY', 'keep', 'clear'],
                xAxisIndex: 0,
                yAxisIndex: 0
            },
            toolbox: {
                feature: {
                    brush: {
                        type: ['rect', 'polygon', 'lineX', 'lineY', 'keep', 'clear']
                    },
                    saveAsImage: {},
                    dataZoom: {},
                    restore: {}
                }
            },
            xAxis: {
                type: 'value',
                scale: true,
                axisLabel: {
                    formatter: '{value}'
                },
                splitLine: {
                    show: false
                }
            },
            yAxis: {
                type: 'value',
                scale: true,
                axisLabel: {
                    formatter: '{value}'
                },
                splitLine: {
                    show: false
                }
            },
            dataZoom: [
                {
                    type: 'inside',
                    xAxisIndex: 0,
                    yAxisIndex: 0
                },
                {
                    type: 'slider',
                    xAxisIndex: 0,
                    bottom: 50
                },
                {
                    type: 'slider',
                    yAxisIndex: 0,
                    right: 20
                }
            ],
            series: [
                {
                    name: 'Normal',
                    type: 'scatter',
                    data: scatterData,
                    symbolSize: function(data) {
                        return 6;
                    },
                    itemStyle: {
                        color: function(params) {
                            const colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];
                            return colors[params.data[2] % colors.length];
                        },
                        opacity: 0.7
                    },
                    emphasis: {
                        itemStyle: {
                            borderColor: '#333',
                            borderWidth: 1
                        }
                    }
                },
                {
                    name: 'Anomalies',
                    type: 'scatter',
                    data: anomalyData,
                    symbolSize: 10,
                    itemStyle: {
                        color: '#DC2626',
                        borderColor: '#991B1B',
                        borderWidth: 2
                    },
                    emphasis: {
                        itemStyle: {
                            borderWidth: 3,
                            shadowBlur: 10,
                            shadowOffsetX: 0,
                            shadowOffsetY: 0,
                            shadowColor: 'rgba(220, 38, 38, 0.5)'
                        }
                    }
                }
            ]
        };
        
        const chart = echarts.init(container);
        chart.setOption(option);
        
        // Handle brush selection
        chart.on('brushSelected', function(params) {
            const selected = params.batch[0].selected;
            if (selected && selected.length > 0) {
                const selectedData = selected[0].dataIndex;
                console.log('Selected points:', selectedData);
                // Handle selected points
            }
        });
        
        window.addEventListener('resize', () => chart.resize());
        
        this.charts.scatterplot = chart;
    },
    
    // Anomaly Distribution Chart
    initAnomalyDistribution() {
        const container = document.getElementById('anomaly-distribution');
        if (!container) return;
        
        // Generate distribution data
        const detectors = ['IsolationForest', 'LocalOutlierFactor', 'OneClassSVM', 'ECOD', 'COPOD'];
        const distributionData = detectors.map(detector => {
            const scores = Array.from({length: 1000}, () => Math.random()).sort();
            const binSize = 0.1;
            const bins = [];
            
            for (let i = 0; i <= 1; i += binSize) {
                const count = scores.filter(s => s >= i && s < i + binSize).length;
                bins.push([i, count]);
            }
            
            return {
                name: detector,
                type: 'line',
                data: bins,
                smooth: true,
                symbol: 'none',
                lineStyle: {
                    width: 2
                },
                areaStyle: {
                    opacity: 0.3
                }
            };
        });
        
        const option = {
            title: {
                text: 'Anomaly Score Distributions by Detector',
                left: 'center',
                textStyle: {
                    fontSize: 16,
                    fontWeight: 'bold'
                }
            },
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'cross'
                },
                formatter: function(params) {
                    let result = `Score Range: ${params[0].data[0].toFixed(1)} - ${(params[0].data[0] + 0.1).toFixed(1)}<br/>`;
                    params.forEach(param => {
                        result += `${param.seriesName}: ${param.data[1]} samples<br/>`;
                    });
                    return result;
                }
            },
            legend: {
                bottom: 10,
                data: detectors
            },
            grid: {
                left: '3%',
                right: '4%',
                bottom: '15%',
                containLabel: true
            },
            toolbox: {
                feature: {
                    saveAsImage: {},
                    dataZoom: {},
                    restore: {},
                    magicType: {
                        type: ['line', 'bar']
                    }
                }
            },
            xAxis: {
                type: 'value',
                name: 'Anomaly Score',
                nameLocation: 'center',
                nameGap: 30,
                min: 0,
                max: 1,
                splitLine: {
                    show: false
                }
            },
            yAxis: {
                type: 'value',
                name: 'Sample Count',
                nameLocation: 'center',
                nameGap: 50
            },
            dataZoom: [
                {
                    type: 'inside'
                },
                {
                    type: 'slider',
                    bottom: 50
                }
            ],
            series: distributionData
        };
        
        const chart = echarts.init(container);
        chart.setOption(option);
        
        window.addEventListener('resize', () => chart.resize());
        
        this.charts.distribution = chart;
    },
    
    // Detector Performance Comparison
    initDetectorComparison() {
        const container = document.getElementById('detector-comparison');
        if (!container) return;
        
        const detectors = ['IsolationForest', 'LocalOutlierFactor', 'OneClassSVM', 'ECOD', 'COPOD', 'ABOD', 'KNN'];
        const metrics = ['Precision', 'Recall', 'F1-Score', 'AUC', 'Training Time', 'Prediction Time'];
        
        // Generate performance data
        const performanceData = detectors.map(detector => {
            return metrics.map(metric => {
                let value;
                if (metric.includes('Time')) {
                    value = Math.random() * 1000; // milliseconds
                } else {
                    value = 0.6 + Math.random() * 0.4; // 0.6-1.0 for quality metrics
                }
                return value;
            });
        });
        
        const option = {
            title: {
                text: 'Detector Performance Comparison',
                left: 'center',
                textStyle: {
                    fontSize: 16,
                    fontWeight: 'bold'
                }
            },
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'shadow'
                },
                formatter: function(params) {
                    let result = `${params[0].axisValue}<br/>`;
                    params.forEach(param => {
                        const value = param.value;
                        const formatted = param.axisValue.includes('Time') ? 
                            `${value.toFixed(1)}ms` : value.toFixed(3);
                        result += `${param.seriesName}: ${formatted}<br/>`;
                    });
                    return result;
                }
            },
            legend: {
                bottom: 10,
                data: detectors,
                type: 'scroll'
            },
            grid: {
                left: '3%',
                right: '4%',
                bottom: '20%',
                containLabel: true
            },
            toolbox: {
                feature: {
                    saveAsImage: {},
                    restore: {},
                    magicType: {
                        type: ['line', 'bar']
                    }
                }
            },
            xAxis: {
                type: 'category',
                data: metrics,
                axisLabel: {
                    rotate: 45
                }
            },
            yAxis: {
                type: 'value',
                name: 'Performance Score / Time (ms)',
                nameLocation: 'center',
                nameGap: 50
            },
            series: detectors.map((detector, index) => ({
                name: detector,
                type: 'bar',
                data: performanceData[index],
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowOffsetX: 0,
                        shadowOffsetY: 0,
                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }
                }
            }))
        };
        
        const chart = echarts.init(container);
        chart.setOption(option);
        
        window.addEventListener('resize', () => chart.resize());
        
        this.charts.comparison = chart;
    },
    
    // Setup interactive features
    setupInteractions() {
        // Chart linking - when one chart is brushed, update others
        this.setupChartLinking();
        
        // Real-time data updates
        this.setupRealTimeUpdates();
        
        // Export functionality
        this.setupExportFeatures();
        
        // Animation controls
        this.setupAnimationControls();
    },
    
    setupChartLinking() {
        // Implementation for linking chart interactions
        console.log('Chart linking setup complete');
    },
    
    setupRealTimeUpdates() {
        // Simulate real-time updates every 5 seconds
        setInterval(() => {
            this.updateChartsWithNewData();
        }, 5000);
    },
    
    updateChartsWithNewData() {
        // Update time series with new data point
        if (this.charts.timeSeries) {
            const data = this.charts.timeSeries.data;
            const newPoint = {
                date: new Date(),
                value: Math.sin(data.length / 50) * 10 + Math.random() * 5,
                anomaly: Math.random() > 0.95
            };
            
            data.push(newPoint);
            if (data.length > 1000) {
                data.shift(); // Remove oldest point
            }
            
            // Redraw time series
            this.initTimeSeriesExplorer();
        }
    },
    
    setupExportFeatures() {
        // Add export buttons to each chart
        document.querySelectorAll('.chart-container').forEach(container => {
            const exportBtn = container.querySelector('.export-chart');
            if (exportBtn) {
                exportBtn.addEventListener('click', (e) => {
                    const chartId = container.id;
                    this.exportChart(chartId);
                });
            }
        });
    },
    
    exportChart(chartId) {
        const chart = this.charts[chartId];
        if (!chart) return;
        
        if (chart.getDataURL) {
            // ECharts export
            const url = chart.getDataURL({
                pixelRatio: 2,
                backgroundColor: '#fff'
            });
            
            const link = document.createElement('a');
            link.download = `${chartId}-chart.png`;
            link.href = url;
            link.click();
        } else if (chart.svg) {
            // D3.js export
            this.exportD3Chart(chart.svg, chartId);
        }
    },
    
    exportD3Chart(svg, filename) {
        const svgData = new XMLSerializer().serializeToString(svg.node());
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();
        
        img.onload = function() {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            
            const link = document.createElement('a');
            link.download = `${filename}-chart.png`;
            link.href = canvas.toDataURL();
            link.click();
        };
        
        img.src = 'data:image/svg+xml;base64,' + btoa(svgData);
    },
    
    setupAnimationControls() {
        // Add animation control panel
        const controlPanel = document.getElementById('animation-controls');
        if (!controlPanel) return;
        
        // Play/pause button
        const playBtn = controlPanel.querySelector('#play-animation');
        if (playBtn) {
            playBtn.addEventListener('click', () => {
                this.toggleAnimation();
            });
        }
        
        // Speed control
        const speedSlider = controlPanel.querySelector('#animation-speed');
        if (speedSlider) {
            speedSlider.addEventListener('input', (e) => {
                this.setAnimationSpeed(e.target.value);
            });
        }
    },
    
    toggleAnimation() {
        this.animations.playing = !this.animations.playing;
        
        if (this.animations.playing) {
            this.startAnimations();
        } else {
            this.stopAnimations();
        }
    },
    
    startAnimations() {
        // Start chart animations
        Object.values(this.charts).forEach(chart => {
            if (chart.setOption) {
                // Enable ECharts animations
                chart.setOption({
                    animation: true,
                    animationDuration: 1000,
                    animationEasing: 'cubicOut'
                });
            }
        });
    },
    
    stopAnimations() {
        // Stop chart animations
        Object.values(this.charts).forEach(chart => {
            if (chart.setOption) {
                chart.setOption({
                    animation: false
                });
            }
        });
    },
    
    setAnimationSpeed(speed) {
        const duration = Math.max(100, 2000 - (speed * 19));
        
        Object.values(this.charts).forEach(chart => {
            if (chart.setOption) {
                chart.setOption({
                    animationDuration: duration
                });
            }
        });
    },
    
    showAnomalyDetails(data) {
        // Show detailed anomaly information in modal
        const modal = document.getElementById('anomaly-detail-modal');
        if (!modal) return;
        
        const content = modal.querySelector('.modal-content');
        content.innerHTML = `
            <h3 class="text-lg font-bold mb-4">Anomaly Details</h3>
            <div class="space-y-2">
                <p><strong>Feature:</strong> ${data.feature}</p>
                <p><strong>Sample:</strong> ${data.sample}</p>
                <p><strong>Value:</strong> ${data.value.toFixed(4)}</p>
                <p><strong>Status:</strong> ${data.anomaly ? '⚠️ Anomaly' : '✅ Normal'}</p>
            </div>
            <div class="mt-6 flex justify-end">
                <button onclick="this.closest('.modal').style.display='none'" 
                        class="bg-gray-500 text-white px-4 py-2 rounded hover:bg-gray-600">
                    Close
                </button>
            </div>
        `;
        
        modal.style.display = 'block';
    },
    
    // Cleanup method
    destroy() {
        // Clean up all charts and event listeners
        Object.values(this.charts).forEach(chart => {
            if (chart.dispose) {
                chart.dispose();
            }
        });
        
        this.charts = {};
        this.interactions = {};
        this.animations = {};
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (typeof d3 !== 'undefined' && typeof echarts !== 'undefined') {
        window.AdvancedViz.init();
    } else {
        console.warn('D3.js or ECharts not loaded. Advanced visualizations disabled.');
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (window.AdvancedViz) {
        window.AdvancedViz.destroy();
    }
});
