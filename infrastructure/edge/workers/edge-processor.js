/**
 * CloudFlare Worker for Edge Computing
 * Handles real-time data processing at the edge
 */

addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
});

class EdgeProcessor {
  constructor() {
    this.cache = caches.default;
    this.analytics = new EdgeAnalytics();
  }

  async handleRequest(request) {
    const url = new URL(request.url);
    const path = url.pathname;

    // Route to appropriate handler
    if (path.startsWith('/edge/analytics')) {
      return this.handleAnalytics(request);
    } else if (path.startsWith('/edge/cache')) {
      return this.handleCaching(request);
    } else if (path.startsWith('/edge/transform')) {
      return this.handleDataTransform(request);
    } else if (path.startsWith('/edge/ml-inference')) {
      return this.handleMLInference(request);
    }

    return new Response('Not Found', { status: 404 });
  }

  async handleAnalytics(request) {
    try {
      const data = await request.json();
      
      // Process analytics data at edge
      const processedData = {
        timestamp: Date.now(),
        user_agent: request.headers.get('User-Agent'),
        country: request.cf.country,
        region: request.cf.region,
        city: request.cf.city,
        datacenter: request.cf.colo,
        ...data
      };

      // Store in edge analytics
      await this.analytics.record(processedData);

      // Forward to origin if needed
      if (data.forward_to_origin) {
        const originRequest = new Request(ORIGIN_ANALYTICS_ENDPOINT, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${ANALYTICS_API_KEY}`
          },
          body: JSON.stringify(processedData)
        });
        
        fetch(originRequest); // Fire and forget
      }

      return new Response(JSON.stringify({
        status: 'success',
        processed_at: processedData.timestamp,
        location: `${processedData.city}, ${processedData.country}`
      }), {
        headers: { 'Content-Type': 'application/json' }
      });

    } catch (error) {
      return new Response(JSON.stringify({
        error: 'Analytics processing failed',
        message: error.message
      }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      });
    }
  }

  async handleCaching(request) {
    const url = new URL(request.url);
    const cacheKey = new Request(url.toString(), request);
    
    // Check cache first
    let response = await this.cache.match(cacheKey);
    
    if (!response) {
      // Cache miss - fetch from origin
      response = await fetch(request);
      
      if (response.status === 200) {
        // Cache successful responses
        const cacheHeaders = new Headers(response.headers);
        cacheHeaders.set('Cache-Control', 'public, max-age=3600');
        
        response = new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers: cacheHeaders
        });
        
        // Store in cache
        event.waitUntil(this.cache.put(cacheKey, response.clone()));
      }
    }

    // Add edge cache headers
    const edgeResponse = new Response(response.body, response);
    edgeResponse.headers.set('X-Edge-Cache', response.headers.has('cf-cache-status') ? 'HIT' : 'MISS');
    edgeResponse.headers.set('X-Edge-Location', request.cf.colo);
    
    return edgeResponse;
  }

  async handleDataTransform(request) {
    try {
      const data = await request.json();
      const transformType = new URL(request.url).searchParams.get('type');

      let transformedData;
      
      switch (transformType) {
        case 'normalize':
          transformedData = this.normalizeData(data);
          break;
        case 'aggregate':
          transformedData = this.aggregateData(data);
          break;
        case 'filter':
          transformedData = this.filterData(data, request);
          break;
        case 'enrich':
          transformedData = await this.enrichData(data, request);
          break;
        default:
          throw new Error(`Unknown transform type: ${transformType}`);
      }

      return new Response(JSON.stringify({
        status: 'success',
        original_count: Array.isArray(data) ? data.length : 1,
        transformed_count: Array.isArray(transformedData) ? transformedData.length : 1,
        data: transformedData,
        processed_at: Date.now(),
        edge_location: request.cf.colo
      }), {
        headers: { 'Content-Type': 'application/json' }
      });

    } catch (error) {
      return new Response(JSON.stringify({
        error: 'Data transformation failed',
        message: error.message
      }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      });
    }
  }

  async handleMLInference(request) {
    try {
      const data = await request.json();
      const modelType = new URL(request.url).searchParams.get('model');

      // Simple edge ML inference for common models
      let prediction;
      
      switch (modelType) {
        case 'anomaly_detection':
          prediction = this.detectAnomalies(data);
          break;
        case 'classification':
          prediction = this.classifyData(data);
          break;
        case 'recommendation':
          prediction = await this.generateRecommendations(data);
          break;
        default:
          // For complex models, proxy to origin
          return this.proxyToOriginML(request);
      }

      return new Response(JSON.stringify({
        status: 'success',
        model: modelType,
        prediction: prediction,
        confidence: prediction.confidence || 0.85,
        processed_at: Date.now(),
        edge_location: request.cf.colo
      }), {
        headers: { 'Content-Type': 'application/json' }
      });

    } catch (error) {
      return new Response(JSON.stringify({
        error: 'ML inference failed',
        message: error.message
      }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      });
    }
  }

  normalizeData(data) {
    if (!Array.isArray(data)) return data;
    
    return data.map(item => {
      const normalized = {};
      for (const [key, value] of Object.entries(item)) {
        if (typeof value === 'string') {
          normalized[key] = value.toLowerCase().trim();
        } else if (typeof value === 'number') {
          normalized[key] = Math.round(value * 100) / 100;
        } else {
          normalized[key] = value;
        }
      }
      return normalized;
    });
  }

  aggregateData(data) {
    if (!Array.isArray(data)) return data;
    
    const aggregated = data.reduce((acc, item) => {
      const key = item.category || 'default';
      if (!acc[key]) {
        acc[key] = { count: 0, total: 0, items: [] };
      }
      acc[key].count++;
      acc[key].total += item.value || 0;
      acc[key].items.push(item);
      return acc;
    }, {});

    // Calculate averages
    Object.keys(aggregated).forEach(key => {
      aggregated[key].average = aggregated[key].total / aggregated[key].count;
    });

    return aggregated;
  }

  filterData(data, request) {
    const url = new URL(request.url);
    const filters = {};
    
    // Extract filter parameters
    url.searchParams.forEach((value, key) => {
      if (key.startsWith('filter_')) {
        filters[key.replace('filter_', '')] = value;
      }
    });

    if (!Array.isArray(data)) return data;
    
    return data.filter(item => {
      return Object.entries(filters).every(([key, value]) => {
        if (!item.hasOwnProperty(key)) return true;
        return String(item[key]).toLowerCase().includes(value.toLowerCase());
      });
    });
  }

  async enrichData(data, request) {
    // Add geo-location and device information
    const enrichment = {
      location: {
        country: request.cf.country,
        region: request.cf.region,
        city: request.cf.city,
        timezone: request.cf.timezone,
        latitude: request.cf.latitude,
        longitude: request.cf.longitude
      },
      network: {
        asn: request.cf.asn,
        datacenter: request.cf.colo
      },
      request_info: {
        user_agent: request.headers.get('User-Agent'),
        accept_language: request.headers.get('Accept-Language'),
        timestamp: Date.now()
      }
    };

    if (Array.isArray(data)) {
      return data.map(item => ({ ...item, ...enrichment }));
    } else {
      return { ...data, ...enrichment };
    }
  }

  detectAnomalies(data) {
    // Simple anomaly detection using statistical methods
    if (!Array.isArray(data) || data.length < 3) {
      return { anomalies: [], confidence: 0.5 };
    }

    const values = data.map(item => item.value || 0);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const stdDev = Math.sqrt(values.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / values.length);
    
    const threshold = 2; // 2 standard deviations
    const anomalies = data.filter((item, index) => {
      const value = item.value || 0;
      return Math.abs(value - mean) > threshold * stdDev;
    });

    return {
      anomalies: anomalies,
      total_points: data.length,
      anomaly_count: anomalies.length,
      confidence: anomalies.length > 0 ? 0.8 : 0.3,
      statistics: { mean, stdDev, threshold }
    };
  }

  classifyData(data) {
    // Simple rule-based classification
    const features = data.features || {};
    let category = 'unknown';
    let confidence = 0.5;

    // Example classification rules
    if (features.value > 100) {
      category = 'high';
      confidence = 0.8;
    } else if (features.value > 50) {
      category = 'medium';
      confidence = 0.7;
    } else {
      category = 'low';
      confidence = 0.9;
    }

    return {
      category: category,
      confidence: confidence,
      features_analyzed: Object.keys(features).length
    };
  }

  async generateRecommendations(data) {
    // Simple collaborative filtering
    const userPreferences = data.preferences || {};
    const context = data.context || {};
    
    const recommendations = [
      { id: 'rec_1', score: 0.85, reason: 'Based on similar users' },
      { id: 'rec_2', score: 0.72, reason: 'Popular in your region' },
      { id: 'rec_3', score: 0.68, reason: 'Trending now' }
    ];

    // Filter based on context
    const filtered = recommendations.filter(rec => rec.score > 0.7);

    return {
      recommendations: filtered,
      total_generated: recommendations.length,
      total_filtered: filtered.length,
      confidence: 0.75
    };
  }

  async proxyToOriginML(request) {
    const originUrl = ORIGIN_ML_ENDPOINT + request.url.split('/edge')[1];
    const originRequest = new Request(originUrl, {
      method: request.method,
      headers: request.headers,
      body: request.body
    });

    const response = await fetch(originRequest);
    
    // Add edge processing headers
    const edgeResponse = new Response(response.body, response);
    edgeResponse.headers.set('X-Processed-At', 'origin');
    edgeResponse.headers.set('X-Edge-Location', request.cf.colo);
    
    return edgeResponse;
  }
}

class EdgeAnalytics {
  constructor() {
    this.buffer = [];
    this.maxBufferSize = 1000;
  }

  async record(data) {
    this.buffer.push({
      ...data,
      edge_timestamp: Date.now()
    });

    // Flush buffer if it gets too large
    if (this.buffer.length >= this.maxBufferSize) {
      await this.flush();
    }
  }

  async flush() {
    if (this.buffer.length === 0) return;

    try {
      const batchData = this.buffer.splice(0, this.maxBufferSize);
      
      const request = new Request(ANALYTICS_BATCH_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${ANALYTICS_API_KEY}`
        },
        body: JSON.stringify({
          batch: batchData,
          batch_size: batchData.length,
          flushed_at: Date.now()
        })
      });

      // Fire and forget
      fetch(request);
    } catch (error) {
      console.error('Failed to flush analytics:', error);
    }
  }
}

async function handleRequest(request) {
  const processor = new EdgeProcessor();
  return processor.handleRequest(request);
}

// Environment variables (set in CloudFlare Workers dashboard)
const ORIGIN_ANALYTICS_ENDPOINT = 'https://api.mlops.com/analytics/batch';
const ORIGIN_ML_ENDPOINT = 'https://api.mlops.com/ml';
const ANALYTICS_BATCH_ENDPOINT = 'https://api.mlops.com/analytics/batch';
const ANALYTICS_API_KEY = 'your-analytics-api-key';