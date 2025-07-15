/**
 * ECharts Dashboard Components
 * Provides advanced interactive dashboards using Apache ECharts
 */

class EChartsDashboard {
  constructor() {
    this.charts = new Map();
    this.themes = {
      light: {
        backgroundColor: '#ffffff',
        textStyle: {
          color: '#333333'
        },
        title: {
          textStyle: {
            color: '#333333'
          }
        }
      },
      dark: {
        backgroundColor: '#1f2937',
        textStyle: {
          color: '#ffffff'
        },
        title: {
          textStyle: {
            color: '#ffffff'
          }
        }
      }
    };
    this.currentTheme = 'light';
  }

  /**
   * Initialize ECharts library
   */
  async init() {
    if (typeof echarts === 'undefined') {
      console.warn('ECharts library not loaded');
      return false;
    }
    
    // Register custom themes
    echarts.registerTheme('light', this.themes.light);
    echarts.registerTheme('dark', this.themes.dark);
    
    return true;
  }

  /**
   * Creates an anomaly detection dashboard
   */
  createAnomalyDashboard(containerId, data, config = {}) {
    const container = document.getElementById(containerId);
    if (!container || typeof echarts === 'undefined') return null;

    const chart = echarts.init(container, this.currentTheme);
    
    const option = {
      title: {
        text: 'Anomaly Detection Dashboard',
        left: 'center',
        textStyle: {
          fontSize: 18,
          fontWeight: 'bold'
        }
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross',
          label: {
            backgroundColor: '#6a7985'
          }
        },
        formatter: function(params) {
          let result = `<strong>${params[0].axisValue}</strong><br/>`;
          params.forEach(param => {
            result += `${param.marker} ${param.seriesName}: ${param.value}<br/>`;
          });
          return result;
        }
      },
      legend: {
        data: ['Normal Data', 'Anomalies', 'Anomaly Score'],
        top: '10%'
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      toolbox: {
        feature: {
          saveAsImage: {
            title: 'Save as Image'
          },
          dataZoom: {
            title: {
              zoom: 'Zoom',
              back: 'Reset Zoom'
            }
          },
          brush: {
            title: {
              rect: 'Rectangle Select',
              polygon: 'Polygon Select',
              lineX: 'Horizontal Select',
              lineY: 'Vertical Select',
              keep: 'Keep Previous',
              clear: 'Clear Selection'
            }
          }
        }
      },
      xAxis: [
        {
          type: 'category',
          data: data.timestamps,
          axisPointer: {
            type: 'shadow'
          }
        }
      ],
      yAxis: [
        {
          type: 'value',
          name: 'Value',
          position: 'left'
        },
        {
          type: 'value',
          name: 'Anomaly Score',
          position: 'right',
          max: 1.0,
          axisLabel: {
            formatter: '{value}'
          }
        }
      ],
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100
        },
        {
          start: 0,
          end: 100,
          handleIcon: 'M10.7,11.9H9.3c-0.4,0.2-0.8,0.1-1.1-0.1C7.8,11.4,7.7,11,7.7,10.5v-3c0-0.4,0.2-0.8,0.5-1.1c0.3-0.2,0.7-0.3,1.1-0.1h1.4'
        }
      ],
      series: [
        {
          name: 'Normal Data',
          type: 'scatter',
          data: data.normal.map((value, index) => [index, value]),
          symbolSize: 6,
          itemStyle: {
            color: '#2563eb'
          },
          emphasis: {
            itemStyle: {
              color: '#1d4ed8'
            }
          }
        },
        {
          name: 'Anomalies',
          type: 'scatter',
          data: data.anomalies.map((value, index) => [index, value]),
          symbolSize: 10,
          itemStyle: {
            color: '#dc2626'
          },
          emphasis: {
            itemStyle: {
              color: '#b91c1c'
            }
          }
        },
        {
          name: 'Anomaly Score',
          type: 'line',
          yAxisIndex: 1,
          data: data.scores,
          smooth: true,
          lineStyle: {
            color: '#f59e0b',
            width: 2
          },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              {
                offset: 0,
                color: 'rgba(245, 158, 11, 0.3)'
              },
              {
                offset: 1,
                color: 'rgba(245, 158, 11, 0.1)'
              }
            ])
          }
        }
      ],
      brush: {
        toolbox: ['rect', 'polygon', 'lineX', 'lineY', 'keep', 'clear'],
        xAxisIndex: 0
      }
    };

    chart.setOption(option);

    // Add event listeners for interactivity
    chart.on('click', (params) => {
      if (params.componentType === 'series') {
        const customEvent = new CustomEvent('anomalyPointClicked', {
          detail: {
            seriesName: params.seriesName,
            dataIndex: params.dataIndex,
            value: params.value,
            data: params.data
          }
        });
        document.dispatchEvent(customEvent);
      }
    });

    chart.on('brushSelected', (params) => {
      const customEvent = new CustomEvent('dataSelected', {
        detail: {
          brushComponent: params.brushComponent,
          areas: params.areas
        }
      });
      document.dispatchEvent(customEvent);
    });

    // Handle window resize
    window.addEventListener('resize', () => {
      chart.resize();
    });

    this.charts.set(containerId, chart);
    return chart;
  }

  /**
   * Creates a real-time monitoring gauge
   */
  createMonitoringGauge(containerId, value, config = {}) {
    const container = document.getElementById(containerId);
    if (!container || typeof echarts === 'undefined') return null;

    const chart = echarts.init(container, this.currentTheme);
    
    const option = {
      title: {
        text: config.title || 'System Health',
        left: 'center',
        top: '10%'
      },
      series: [
        {
          name: 'Health Score',
          type: 'gauge',
          startAngle: 180,
          endAngle: 0,
          center: ['50%', '75%'],
          radius: '80%',
          min: 0,
          max: 100,
          splitNumber: 8,
          axisLine: {
            lineStyle: {
              width: 6,
              color: [
                [0.25, '#dc2626'],
                [0.5, '#f59e0b'],
                [0.75, '#10b981'],
                [1, '#059669']
              ]
            }
          },
          pointer: {
            icon: 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
            length: '12%',
            width: 20,
            offsetCenter: [0, '-60%'],
            itemStyle: {
              color: 'auto'
            }
          },
          axisTick: {
            length: 12,
            lineStyle: {
              color: 'auto',
              width: 2
            }
          },
          splitLine: {
            length: 20,
            lineStyle: {
              color: 'auto',
              width: 5
            }
          },
          axisLabel: {
            color: '#464646',
            fontSize: 12,
            distance: -60,
            formatter: function (value) {
              if (value === 87.5) {
                return 'Good';
              } else if (value === 62.5) {
                return 'OK';
              } else if (value === 37.5) {
                return 'Poor';
              } else if (value === 12.5) {
                return 'Bad';
              }
              return '';
            }
          },
          title: {
            offsetCenter: [0, '-20%'],
            fontSize: 20
          },
          detail: {
            fontSize: 30,
            offsetCenter: [0, '-35%'],
            valueAnimation: true,
            formatter: function (value) {
              return Math.round(value) + '%';
            },
            color: 'auto'
          },
          data: [
            {
              value: value,
              name: 'Health Score'
            }
          ]
        }
      ]
    };

    chart.setOption(option);
    this.charts.set(containerId, chart);
    return chart;
  }

  /**
   * Creates a feature correlation heatmap
   */
  createCorrelationHeatmap(containerId, data, config = {}) {
    const container = document.getElementById(containerId);
    if (!container || typeof echarts === 'undefined') return null;

    const chart = echarts.init(container, this.currentTheme);
    
    const hours = data.features;
    const days = data.features;
    
    // Convert correlation matrix to ECharts format
    const heatmapData = [];
    for (let i = 0; i < data.correlations.length; i++) {
      for (let j = 0; j < data.correlations[i].length; j++) {
        heatmapData.push([i, j, data.correlations[i][j]]);
      }
    }

    const option = {
      title: {
        text: 'Feature Correlation Matrix',
        left: 'center'
      },
      tooltip: {
        position: 'top',
        formatter: function (params) {
          return `${hours[params.data[0]]} × ${days[params.data[1]]}<br/>Correlation: ${params.data[2].toFixed(3)}`;
        }
      },
      grid: {
        height: '50%',
        top: '10%'
      },
      xAxis: {
        type: 'category',
        data: hours,
        splitArea: {
          show: true
        },
        axisLabel: {
          rotate: 45
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
        min: -1,
        max: 1,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '15%',
        inRange: {
          color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
        }
      },
      series: [
        {
          name: 'Correlation',
          type: 'heatmap',
          data: heatmapData,
          label: {
            show: true,
            formatter: function (params) {
              return params.data[2].toFixed(2);
            }
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0, 0, 0, 0.5)'
            }
          }
        }
      ]
    };

    chart.setOption(option);
    this.charts.set(containerId, chart);
    return chart;
  }

  /**
   * Creates a multi-detector comparison chart
   */
  createDetectorComparison(containerId, data, config = {}) {
    const container = document.getElementById(containerId);
    if (!container || typeof echarts === 'undefined') return null;

    const chart = echarts.init(container, this.currentTheme);
    
    const option = {
      title: {
        text: 'Detector Performance Comparison',
        left: 'center'
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow'
        }
      },
      legend: {
        data: ['Precision', 'Recall', 'F1-Score', 'AUC'],
        top: '10%'
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: data.detectors
      },
      yAxis: {
        type: 'value',
        max: 1.0,
        axisLabel: {
          formatter: '{value}'
        }
      },
      series: [
        {
          name: 'Precision',
          type: 'bar',
          data: data.precision,
          itemStyle: {
            color: '#2563eb'
          }
        },
        {
          name: 'Recall',
          type: 'bar',
          data: data.recall,
          itemStyle: {
            color: '#10b981'
          }
        },
        {
          name: 'F1-Score',
          type: 'bar',
          data: data.f1_score,
          itemStyle: {
            color: '#f59e0b'
          }
        },
        {
          name: 'AUC',
          type: 'line',
          data: data.auc,
          itemStyle: {
            color: '#dc2626'
          },
          lineStyle: {
            width: 3
          },
          symbol: 'circle',
          symbolSize: 8
        }
      ]
    };

    chart.setOption(option);
    this.charts.set(containerId, chart);
    return chart;
  }

  /**
   * Creates a real-time streaming chart
   */
  createStreamingChart(containerId, config = {}) {
    const container = document.getElementById(containerId);
    if (!container || typeof echarts === 'undefined') return null;

    const chart = echarts.init(container, this.currentTheme);
    
    let data = [];
    let anomalyData = [];
    let now = new Date();
    
    const option = {
      title: {
        text: 'Real-time Anomaly Detection',
        left: 'center'
      },
      tooltip: {
        trigger: 'axis',
        formatter: function (params) {
          params = params[0];
          var date = new Date(params.name);
          return date.getHours() + ':' + date.getMinutes() + ':' + date.getSeconds() + ' : ' + params.value[1];
        },
        axisPointer: {
          animation: false
        }
      },
      xAxis: {
        type: 'time',
        splitLine: {
          show: false
        }
      },
      yAxis: {
        type: 'value',
        boundaryGap: [0, '100%'],
        splitLine: {
          show: false
        }
      },
      series: [
        {
          name: 'Data Stream',
          type: 'line',
          showSymbol: false,
          hoverAnimation: false,
          data: data,
          smooth: true,
          lineStyle: {
            color: '#2563eb'
          }
        },
        {
          name: 'Anomalies',
          type: 'scatter',
          data: anomalyData,
          symbolSize: 10,
          itemStyle: {
            color: '#dc2626'
          }
        }
      ]
    };

    chart.setOption(option);
    
    // Store for real-time updates
    this.charts.set(containerId, {
      chart,
      data,
      anomalyData,
      option
    });
    
    return chart;
  }

  /**
   * Updates streaming chart with new data
   */
  updateStreamingChart(containerId, newValue, isAnomaly = false) {
    const chartObj = this.charts.get(containerId);
    if (!chartObj) return;

    const { chart, data, anomalyData, option } = chartObj;
    const now = new Date();
    
    data.push({
      name: now.toString(),
      value: [now, newValue]
    });
    
    if (isAnomaly) {
      anomalyData.push({
        name: now.toString(),
        value: [now, newValue]
      });
    }
    
    // Keep only last 50 points
    if (data.length > 50) {
      data.shift();
    }
    if (anomalyData.length > 20) {
      anomalyData.shift();
    }
    
    chart.setOption({
      series: [
        {
          data: data
        },
        {
          data: anomalyData
        }
      ]
    });
  }

  /**
   * Updates chart theme
   */
  setTheme(theme) {
    this.currentTheme = theme;
    
    this.charts.forEach((chart, containerId) => {
      if (chart.dispose) {
        chart.dispose();
        const container = document.getElementById(containerId);
        if (container) {
          this.charts.set(containerId, echarts.init(container, theme));
        }
      }
    });
  }

  /**
   * Exports chart as image
   */
  exportChart(containerId, format = 'png') {
    const chartObj = this.charts.get(containerId);
    const chart = chartObj?.chart || chartObj;
    
    if (!chart) return null;
    
    return chart.getDataURL({
      type: format,
      pixelRatio: 2,
      backgroundColor: '#fff'
    });
  }

  /**
   * Resizes all charts
   */
  resizeCharts() {
    this.charts.forEach((chartObj) => {
      const chart = chartObj.chart || chartObj;
      if (chart && chart.resize) {
        chart.resize();
      }
    });
  }

  /**
   * Destroys chart and cleans up resources
   */
  destroyChart(containerId) {
    const chartObj = this.charts.get(containerId);
    if (chartObj) {
      const chart = chartObj.chart || chartObj;
      if (chart && chart.dispose) {
        chart.dispose();
      }
      this.charts.delete(containerId);
    }
  }

  /**
   * Destroys all charts
   */
  destroyAll() {
    this.charts.forEach((chartObj, containerId) => {
      this.destroyChart(containerId);
    });
  }
}

// Export for use in other modules
window.EChartsDashboard = EChartsDashboard;

// Auto-initialize if DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', async () => {
    window.echartsManager = new EChartsDashboard();
    await window.echartsManager.init();
  });
} else {
  window.echartsManager = new EChartsDashboard();
  window.echartsManager.init();
}