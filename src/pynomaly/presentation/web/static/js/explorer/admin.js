document.addEventListener('DOMContentLoaded', () => {
    const adminPanel = document.querySelector('.admin-panel');
    adminPanel.innerHTML = `
        <h2>Admin Panel</h2>
        
        <div class="admin-tabs">
            <button class="tab-button active" onclick="showTab('users')">Users</button>
            <button class="tab-button" onclick="showTab('datasets')">Datasets</button>
            <button class="tab-button" onclick="showTab('detectors')">Detectors</button>
        </div>
        
        <!-- Users Tab -->
        <div id="tab-users" class="tab-content active">
            <h3>User Management</h3>
            
            <div class="crud-section">
                <h4>Create User</h4>
                <form hx-post='/htmx/admin/create-user' hx-target='#user-response' hx-swap='innerHTML'>
                    <input type='text' name='username' placeholder='Username' required />
                    <input type='email' name='email' placeholder='Email' required />
                    <select name='role'>
                        <option value='user'>User</option>
                        <option value='admin'>Admin</option>
                        <option value='viewer'>Viewer</option>
                    </select>
                    <button type='submit'>Create User</button>
                </form>
                <div id='user-response'></div>
            </div>
            
            <div class="crud-section">
                <h4>User List</h4>
                <div hx-get='/htmx/admin/list-users' hx-trigger='load' hx-target='this' hx-swap='innerHTML'>
                    Loading users...
                </div>
            </div>
        </div>
        
        <!-- Datasets Tab -->
        <div id="tab-datasets" class="tab-content">
            <h3>Dataset Management</h3>
            
            <div class="crud-section">
                <h4>Upload Dataset</h4>
                <form hx-post='/htmx/admin/upload-dataset' hx-target='#dataset-response' hx-swap='innerHTML' hx-encoding='multipart/form-data'>
                    <input type='text' name='name' placeholder='Dataset Name' required />
                    <input type='file' name='file' accept='.csv,.json,.parquet' required />
                    <textarea name='description' placeholder='Description'></textarea>
                    <button type='submit'>Upload Dataset</button>
                </form>
                <div id='dataset-response'></div>
            </div>
            
            <div class="crud-section">
                <h4>Dataset List</h4>
                <div hx-get='/htmx/admin/list-datasets' hx-trigger='load' hx-target='this' hx-swap='innerHTML'>
                    Loading datasets...
                </div>
            </div>
        </div>
        
        <!-- Detectors Tab -->
        <div id="tab-detectors" class="tab-content">
            <h3>Detector Management</h3>
            
            <div class="crud-section">
                <h4>Create Detector</h4>
                <form hx-post='/htmx/admin/create-detector' hx-target='#detector-response' hx-swap='innerHTML'>
                    <input type='text' name='name' placeholder='Detector Name' required />
                    <select name='algorithm'>
                        <option value='IsolationForest'>Isolation Forest</option>
                        <option value='LocalOutlierFactor'>Local Outlier Factor</option>
                        <option value='OneClassSVM'>One-Class SVM</option>
                        <option value='ECOD'>ECOD</option>
                    </select>
                    <input type='number' name='contamination' placeholder='Contamination (0.1)' step='0.01' min='0' max='1' value='0.1' />
                    <textarea name='description' placeholder='Description'></textarea>
                    <button type='submit'>Create Detector</button>
                </form>
                <div id='detector-response'></div>
            </div>
            
            <div class="crud-section">
                <h4>Detector List</h4>
                <div hx-get='/htmx/admin/list-detectors' hx-trigger='load' hx-target='this' hx-swap='innerHTML'>
                    Loading detectors...
                </div>
            </div>
        </div>
    `;
});

// Tab functionality
window.showTab = function(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(`tab-${tabName}`).classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
};
