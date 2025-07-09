-- Pynomaly Database Initialization Script
-- Creates necessary tables and indexes for development

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS pynomaly_core;
CREATE SCHEMA IF NOT EXISTS pynomaly_audit;
CREATE SCHEMA IF NOT EXISTS pynomaly_metrics;

-- Set search path
SET search_path TO pynomaly_core, public;

-- Create core tables
CREATE TABLE IF NOT EXISTS datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    file_path VARCHAR(500),
    file_size BIGINT,
    columns_count INTEGER,
    rows_count BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS detectors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    algorithm VARCHAR(100) NOT NULL,
    parameters JSONB,
    status VARCHAR(50) DEFAULT 'created',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID,
    dataset_id UUID REFERENCES datasets(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS detection_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    detector_id UUID REFERENCES detectors(id) ON DELETE CASCADE,
    dataset_id UUID REFERENCES datasets(id) ON DELETE CASCADE,
    anomaly_scores JSONB,
    outlier_labels JSONB,
    metrics JSONB,
    execution_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID
);

CREATE TABLE IF NOT EXISTS training_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    detector_id UUID REFERENCES detectors(id) ON DELETE CASCADE,
    status VARCHAR(50) DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID
);

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE
);

-- Create audit tables
CREATE TABLE IF NOT EXISTS pynomaly_audit.audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create metrics tables
CREATE TABLE IF NOT EXISTS pynomaly_metrics.performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(255) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    tags JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_datasets_created_at ON datasets(created_at);
CREATE INDEX IF NOT EXISTS idx_datasets_created_by ON datasets(created_by);
CREATE INDEX IF NOT EXISTS idx_detectors_algorithm ON detectors(algorithm);
CREATE INDEX IF NOT EXISTS idx_detectors_status ON detectors(status);
CREATE INDEX IF NOT EXISTS idx_detectors_dataset_id ON detectors(dataset_id);
CREATE INDEX IF NOT EXISTS idx_detection_results_detector_id ON detection_results(detector_id);
CREATE INDEX IF NOT EXISTS idx_detection_results_created_at ON detection_results(created_at);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);
CREATE INDEX IF NOT EXISTS idx_training_jobs_detector_id ON training_jobs(detector_id);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);
CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON pynomaly_audit.audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON pynomaly_audit.audit_log(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON pynomaly_audit.audit_log(action);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_name ON pynomaly_metrics.performance_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON pynomaly_metrics.performance_metrics(timestamp);

-- Create JSONB indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_datasets_metadata ON datasets USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_detectors_parameters ON detectors USING GIN (parameters);
CREATE INDEX IF NOT EXISTS idx_detection_results_metrics ON detection_results USING GIN (metrics);
CREATE INDEX IF NOT EXISTS idx_training_jobs_metrics ON training_jobs USING GIN (metrics);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_tags ON pynomaly_metrics.performance_metrics USING GIN (tags);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_datasets_updated_at BEFORE UPDATE ON datasets FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_detectors_updated_at BEFORE UPDATE ON detectors FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default admin user (password: admin123)
INSERT INTO users (username, email, password_hash, is_admin) 
VALUES ('admin', 'admin@pynomaly.dev', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewDJhwcHpOdvnpOu', TRUE)
ON CONFLICT (username) DO NOTHING;

-- Create sample dataset for testing
INSERT INTO datasets (name, description, file_path, file_size, columns_count, rows_count, created_by)
VALUES (
    'Sample Dataset',
    'Sample anomaly detection dataset for testing',
    '/tmp/sample_dataset.csv',
    1024,
    10,
    1000,
    (SELECT id FROM users WHERE username = 'admin')
) ON CONFLICT DO NOTHING;

-- Grant permissions
GRANT USAGE ON SCHEMA pynomaly_core TO pynomaly;
GRANT USAGE ON SCHEMA pynomaly_audit TO pynomaly;
GRANT USAGE ON SCHEMA pynomaly_metrics TO pynomaly;

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA pynomaly_core TO pynomaly;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA pynomaly_audit TO pynomaly;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA pynomaly_metrics TO pynomaly;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA pynomaly_core TO pynomaly;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA pynomaly_audit TO pynomaly;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA pynomaly_metrics TO pynomaly;

-- Create view for user-friendly dataset information
CREATE OR REPLACE VIEW dataset_summary AS
SELECT 
    d.id,
    d.name,
    d.description,
    d.columns_count,
    d.rows_count,
    d.file_size,
    d.created_at,
    u.username as created_by_username,
    COUNT(det.id) as detector_count
FROM datasets d
LEFT JOIN users u ON d.created_by = u.id
LEFT JOIN detectors det ON d.id = det.dataset_id
GROUP BY d.id, d.name, d.description, d.columns_count, d.rows_count, d.file_size, d.created_at, u.username;

-- Create view for training job status
CREATE OR REPLACE VIEW training_job_status AS
SELECT 
    tj.id,
    tj.status,
    tj.started_at,
    tj.completed_at,
    tj.error_message,
    d.name as detector_name,
    d.algorithm,
    ds.name as dataset_name,
    u.username as created_by_username
FROM training_jobs tj
JOIN detectors d ON tj.detector_id = d.id
JOIN datasets ds ON d.dataset_id = ds.id
LEFT JOIN users u ON tj.created_by = u.id;

COMMIT;
