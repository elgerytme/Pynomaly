document.addEventListener('DOMContentLoaded', () => {
    const logPanel = document.querySelector('.log-panel');
    logPanel.innerHTML = `
        <h2>Live Request Log</h2>
        <div class="log-controls">
            <button onclick="clearLogs()">Clear</button>
            <button onclick="toggleAutoScroll()">Auto Scroll: ON</button>
        </div>
        <div id='log-content' class='log-container'></div>
    `;

    let autoScroll = true;
    
    // Connect to WebSocket for live logs
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/explorer/logs`;
    const logSocket = new WebSocket(wsUrl);
    
    logSocket.onopen = (event) => {
        console.log('WebSocket connected for live logs');
        addLogEntry({
            timestamp: Date.now() / 1000,
            level: 'INFO',
            message: 'Connected to live log stream',
            method: 'WS',
            path: '/ws/explorer/logs',
            status: 200
        });
    };
    
    logSocket.onmessage = (event) => {
        try {
            const logData = JSON.parse(event.data);
            addLogEntry(logData);
        } catch (e) {
            console.error('Failed to parse log message:', e);
        }
    };
    
    logSocket.onclose = (event) => {
        console.log('WebSocket disconnected');
        addLogEntry({
            timestamp: Date.now() / 1000,
            level: 'WARN',
            message: 'Disconnected from live log stream',
            method: 'WS',
            path: '/ws/explorer/logs',
            status: 'CLOSED'
        });
        
        // Attempt to reconnect after 5 seconds
        setTimeout(() => {
            console.log('Attempting to reconnect...');
            location.reload();
        }, 5000);
    };
    
    logSocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        addLogEntry({
            timestamp: Date.now() / 1000,
            level: 'ERROR',
            message: 'WebSocket connection error',
            method: 'WS',
            path: '/ws/explorer/logs',
            status: 'ERROR'
        });
    };
    
    function addLogEntry(logData) {
        const logContent = document.getElementById('log-content');
        const timestamp = new Date(logData.timestamp * 1000).toLocaleTimeString();
        
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry log-${logData.level?.toLowerCase() || 'info'}`;
        logEntry.innerHTML = `
            <span class="timestamp">${timestamp}</span>
            <span class="level">${logData.level || 'INFO'}</span>
            <span class="method">${logData.method || 'N/A'}</span>
            <span class="path">${logData.path || 'N/A'}</span>
            <span class="status status-${Math.floor(logData.status / 100)}xx">${logData.status || 'N/A'}</span>
            <span class="duration">${logData.duration || 'N/A'}</span>
            <span class="message">${logData.message || 'No message'}</span>
        `;
        
        logContent.appendChild(logEntry);
        
        // Keep only the last 100 log entries
        while (logContent.children.length > 100) {
            logContent.removeChild(logContent.firstChild);
        }
        
        // Auto-scroll to bottom if enabled
        if (autoScroll) {
            logContent.scrollTop = logContent.scrollHeight;
        }
    }
    
    window.clearLogs = function() {
        document.getElementById('log-content').innerHTML = '';
    };
    
    window.toggleAutoScroll = function() {
        autoScroll = !autoScroll;
        const button = document.querySelector('.log-controls button:nth-child(2)');
        button.textContent = `Auto Scroll: ${autoScroll ? 'ON' : 'OFF'}`;
    };
});
