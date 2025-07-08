document.addEventListener('htmx:configRequest', (event) => {
    // Add authentication header if available
    const token = localStorage.getItem('auth_token');
    if (token) {
        event.detail.headers['Authorization'] = `Bearer ${token}`;
    }
});

let openApiSpec = null;

// Fetch OpenAPI spec for form generation
fetch('/api/v1/openapi.json')
    .then(response => response.json())
    .then(data => {
        openApiSpec = data;
    })
    .catch(err => console.warn('OpenAPI spec not available:', err));

function generateFormFromSchema(endpoint, method, schema) {
    if (!schema || !schema.properties) {
        return '<p>No parameters required</p>';
    }

    const fields = Object.entries(schema.properties).map(([name, prop]) => {
        const type = prop.type === 'integer' ? 'number' : (prop.type === 'string' ? 'text' : 'text');
        const required = schema.required && schema.required.includes(name) ? 'required' : '';
        
        return `
            <div class="field">
                <label for="${name}">${name}${required ? ' *' : ''}:</label>
                <input type="${type}" name="${name}" ${required} 
                       placeholder="${prop.description || ''}" />
                ${prop.description ? `<small>${prop.description}</small>` : ''}
            </div>
        `;
    }).join('');

    return fields;
}

function generateCurlCommand(endpoint, method, formData) {
    const baseUrl = window.location.origin;
    const url = `${baseUrl}${endpoint}`;
    
    let curl = `curl -X ${method.toUpperCase()} "${url}"`;
    
    // Add headers
    curl += ` -H "Content-Type: application/json"`;
    const token = localStorage.getItem('auth_token');
    if (token) {
        curl += ` -H "Authorization: Bearer ${token}"`;
    }
    
    // Add data if POST/PUT/PATCH
    if (['post', 'put', 'patch'].includes(method.toLowerCase()) && formData) {
        const jsonData = {};
        for (const [key, value] of formData.entries()) {
            jsonData[key] = value;
        }
        curl += ` -d '${JSON.stringify(jsonData)}'`;
    }
    
    return curl;
}

function createEndpointForm(path, method, endpointSpec) {
    const requestBody = endpointSpec.requestBody;
    const schema = requestBody?.content?.['application/json']?.schema;
    
    return `
        <div class="endpoint-form">
            <h3>${method.toUpperCase()} ${path}</h3>
            <p>${endpointSpec.summary || 'No description available'}</p>
            
            <form onsubmit="submitEndpointForm(event, '${path}', '${method}')">
                ${generateFormFromSchema(path, method, schema)}
                <button type="submit">Send Request</button>
            </form>
            
            <div class="curl-section">
                <h4>cURL Command:</h4>
                <pre id="curl-${path.replace(/\//g, '-')}-${method}"></pre>
            </div>
            
            <div class="response-section">
                <h4>Response:</h4>
                <pre id="response-${path.replace(/\//g, '-')}-${method}"></pre>
            </div>
        </div>
    `;
}

function submitEndpointForm(event, path, method) {
    event.preventDefault();
    
    const form = event.target;
    const formData = new FormData(form);
    const curlId = `curl-${path.replace(/\//g, '-')}-${method}`;
    const responseId = `response-${path.replace(/\//g, '-')}-${method}`;
    
    // Generate and display cURL command
    const curlCommand = generateCurlCommand(path, method, formData);
    document.getElementById(curlId).textContent = curlCommand;
    
    // Prepare request data
    const requestData = {};
    for (const [key, value] of formData.entries()) {
        requestData[key] = value;
    }
    
    // Send request
    const url = `/api/v1${path}`;
    const options = {
        method: method.toUpperCase(),
        headers: {
            'Content-Type': 'application/json',
        },
    };
    
    const token = localStorage.getItem('auth_token');
    if (token) {
        options.headers['Authorization'] = `Bearer ${token}`;
    }
    
    if (['POST', 'PUT', 'PATCH'].includes(method.toUpperCase())) {
        options.body = JSON.stringify(requestData);
    }
    
    fetch(url, options)
        .then(response => {
            return response.json().then(data => ({ status: response.status, data }));
        })
        .then(({ status, data }) => {
            const responseEl = document.getElementById(responseId);
            responseEl.innerHTML = `
                <div class="status">Status: ${status}</div>
                <div class="json">${JSON.stringify(data, null, 2)}</div>
            `;
        })
        .catch(error => {
            const responseEl = document.getElementById(responseId);
            responseEl.innerHTML = `<div class="error">Error: ${error.message}</div>`;
        });
}

// Handle endpoint selection from sidebar
window.selectEndpoint = function(path, method) {
    if (!openApiSpec) {
        alert('OpenAPI specification not loaded yet');
        return;
    }
    
    const endpointSpec = openApiSpec.paths[path][method];
    const contentArea = document.querySelector('.content');
    
    contentArea.innerHTML = createEndpointForm(path, method, endpointSpec);
};

document.addEventListener('DOMContentLoaded', () => {
    const contentArea = document.querySelector('.content');
    
    contentArea.innerHTML = `
        <div class="welcome">
            <h1>API Explorer</h1>
            <p>Select an endpoint from the sidebar to start exploring the API.</p>
            
            <div class="auth-section">
                <h3>Authentication</h3>
                <form onsubmit="authenticate(event)">
                    <input type="text" id="username" placeholder="Username" required />
                    <input type="password" id="password" placeholder="Password" required />
                    <button type="submit">Login</button>
                </form>
                <div id="auth-status"></div>
            </div>
        </div>
    `;
});

function authenticate(event) {
    event.preventDefault();
    
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    
    fetch('/api/v1/auth/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password })
    })
    .then(response => response.json())
    .then(data => {
        if (data.access_token) {
            localStorage.setItem('auth_token', data.access_token);
            document.getElementById('auth-status').innerHTML = 
                '<div class="success">Authenticated successfully!</div>';
        } else {
            document.getElementById('auth-status').innerHTML = 
                '<div class="error">Authentication failed</div>';
        }
    })
    .catch(error => {
        document.getElementById('auth-status').innerHTML = 
            `<div class="error">Error: ${error.message}</div>`;
    });
}
