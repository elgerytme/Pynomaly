-- SQL script for initializing the test database
CREATE TABLE IF NOT EXISTS dummy_data (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    value INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO dummy_data (name, value) VALUES
('Sample A', 123),
('Sample B', 456),
('Sample C', 789);

-- Add more initialization logic as required
