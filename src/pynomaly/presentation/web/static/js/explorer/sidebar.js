document.addEventListener('DOMContentLoaded', () => {
    // Fetch OpenAPI JSON to populate sidebar
    fetch('/api/v1/openapi.json')
        .then(response => response.json())
        .then(data => {
            // Process OpenAPI data to extract endpoints by tag
            const paths = data.paths || {};
            const tags = {};

            for (const path in paths) {
                const methods = paths[path];
                
                for (const method in methods) {
                    const endpoint = methods[method];
                    (endpoint.tags || []).forEach(tag => {
                        if (!tags[tag]) tags[tag] = [];
                        tags[tag].push({ path, method, summary: endpoint.summary });
                    });
                }
            }

            // Render sidebar
            const sidebar = document.querySelector('.sidebar');
            sidebar.innerHTML = `
                <h2>API Endpoints</h2>
                ${Object.keys(tags).map(tag => `
                    <div class="tag-group">
                        <h3 class="tag-header" onclick="toggleTag('${tag}')">${tag} (${tags[tag].length})</h3>
                        <ul class="endpoint-list" id="tag-${tag}">
                            ${tags[tag].map(ep => `
                                <li class="endpoint-item" onclick="selectEndpoint('${ep.path}', '${ep.method}')">
                                    <span class="method method-${ep.method}">${ep.method.toUpperCase()}</span>
                                    <span class="path">${ep.path}</span>
                                    ${ep.summary ? `<div class="summary">${ep.summary}</div>` : ''}
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                `).join('')}
            `;
        })
        .catch(error => {
            console.warn('Could not load OpenAPI spec:', error);
            const sidebar = document.querySelector('.sidebar');
            sidebar.innerHTML = `
                <h2>API Endpoints</h2>
                <p>Unable to load API endpoints. OpenAPI specification not available.</p>
                <div class="manual-endpoints">
                    <h3>Common Endpoints</h3>
                    <ul class="endpoint-list">
                        <li class="endpoint-item" onclick="selectEndpoint('/health', 'get')">
                            <span class="method method-get">GET</span>
                            <span class="path">/health</span>
                        </li>
                        <li class="endpoint-item" onclick="selectEndpoint('/datasets', 'get')">
                            <span class="method method-get">GET</span>
                            <span class="path">/datasets</span>
                        </li>
                        <li class="endpoint-item" onclick="selectEndpoint('/detectors', 'get')">
                            <span class="method method-get">GET</span>
                            <span class="path">/detectors</span>
                        </li>
                    </ul>
                </div>
            `;
        });
});

// Toggle tag visibility
window.toggleTag = function(tag) {
    const tagElement = document.getElementById(`tag-${tag}`);
    tagElement.style.display = tagElement.style.display === 'none' ? 'block' : 'none';
};
});
