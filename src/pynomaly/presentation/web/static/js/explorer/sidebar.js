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
            sidebar.innerHTML = Object.keys(tags).map(tag => `
                <div>
                    <h3>${tag}</h3>
                    <ul>
                        ${tags[tag].map(ep => `<li>${ep.method.toUpperCase()} - ${ep.path}</li>`).join('')}
                    </ul>
                </div>
            `).join('');
        });
});
