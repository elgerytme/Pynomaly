#!/bin/bash
set -e

echo "Starting Pynomaly API..."

# Wait for database
echo "Waiting for database..."
while ! nc -z postgres 5432; do
  sleep 1
done
echo "Database is ready!"

# Wait for Redis
echo "Waiting for Redis..."
while ! nc -z redis 6379; do
  sleep 1
done
echo "Redis is ready!"

# Run database migrations
echo "Running database migrations..."
python -m alembic upgrade head

# Initialize monitoring
echo "Initializing monitoring..."
python scripts/initialize_monitoring.py

# Start the application
echo "Starting application..."
exec "$@"