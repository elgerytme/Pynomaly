document.addEventListener('DOMContentLoaded', () => {
    const logPanel = document.querySelector('.log-panel');
    logPanel.innerHTML = `
        <h2>Live Logs</h2>
        <div id='log-content'></div>
    `;

    // Connect to WebSocket for live logs
    const logSocket = new WebSocket('ws://localhost:8000/logs');
    logSocket.onmessage = (event) => {
        const logContent = document.getElementById('log-content');
        logContent.innerText += `${event.data}\n`;
    };
});
