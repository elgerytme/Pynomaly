/**
 * Pynomaly API Client Example - JavaScript
 */

class PynomaliClient {
    constructor(baseUrl = 'https://api.pynomaly.com') {
        this.baseUrl = baseUrl;
        this.accessToken = null;
    }
    
    async login(username, password) {
        const response = await fetch(`${this.baseUrl}/api/v1/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password })
        });
        
        if (!response.ok) {
            throw new Error(`Login failed: ${response.statusText}`);
        }
        
        const tokenData = await response.json();
        this.accessToken = tokenData.access_token;
        
        return tokenData;
    }
    
    async detectAnomalies(data, algorithm = 'isolation_forest', parameters = { contamination: 0.1 }) {
        const response = await fetch(`${this.baseUrl}/api/v1/detection/detect`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.accessToken}`
            },
            body: JSON.stringify({
                data,
                algorithm,
                parameters
            })
        });
        
        if (!response.ok) {
            throw new Error(`Detection failed: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    async trainModel(trainingData, algorithm, parameters = null, modelName = null) {
        const payload = {
            training_data: trainingData,
            algorithm
        };
        
        if (parameters) payload.parameters = parameters;
        if (modelName) payload.model_name = modelName;
        
        const response = await fetch(`${this.baseUrl}/api/v1/detection/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.accessToken}`
            },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            throw new Error(`Training failed: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    async getHealth() {
        const response = await fetch(`${this.baseUrl}/api/v1/health`);
        
        if (!response.ok) {
            throw new Error(`Health check failed: ${response.statusText}`);
        }
        
        return await response.json();
    }
}

// Example usage
(async () => {
    const client = new PynomaliClient();
    
    try {
        // Login
        await client.login('your_username', 'your_password');
        
        // Detect anomalies
        const result = await client.detectAnomalies([1.0, 2.0, 3.0, 100.0, 4.0, 5.0]);
        console.log('Detected anomalies:', result.anomalies);
        
        // Check health
        const health = await client.getHealth();
        console.log('System status:', health.status);
        
    } catch (error) {
        console.error('Error:', error.message);
    }
})();
