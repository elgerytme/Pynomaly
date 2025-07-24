-- Initialize Anomaly Detection database
-- This script runs when PostgreSQL container starts for the first time

-- Create additional databases if needed
CREATE DATABASE anomaly_detection_test;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create application schemas
CREATE SCHEMA IF NOT EXISTS anomaly_detection;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Create application user (if using different user than postgres)
-- Uncomment and modify as needed
-- CREATE USER anomaly_app WITH ENCRYPTED PASSWORD 'your_secure_password';
-- GRANT USAGE ON SCHEMA anomaly_detection TO anomaly_app;
-- GRANT CREATE ON SCHEMA anomaly_detection TO anomaly_app;
-- GRANT USAGE ON SCHEMA monitoring TO anomaly_app;
-- GRANT CREATE ON SCHEMA monitoring TO anomaly_app;

-- Set default search path
-- ALTER USER anomaly_app SET search_path = anomaly_detection, monitoring, public;

-- Performance tuning settings (adjust based on your needs)
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Logging configuration
ALTER SYSTEM SET log_statement = 'mod';
ALTER SYSTEM SET log_duration = 'on';
ALTER SYSTEM SET log_min_duration_statement = 1000;

-- Connection settings
ALTER SYSTEM SET max_prepared_transactions = 100;

-- Restart required for some settings to take effect
SELECT pg_reload_conf();