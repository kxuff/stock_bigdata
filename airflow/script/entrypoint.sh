#!/usr/bin/env bash
# Airflow entrypoint script

# Wait for database to be ready
echo "Waiting for database to be ready..."
sleep 10
echo "Proceeding with Airflow initialization..."

# Install additional packages
echo "Installing Python packages..."
pip install --no-cache-dir \
    apache-airflow-providers-apache-spark \
    apache-airflow-providers-postgres \
    pyspark==3.5.0 \
    pyiceberg[s3fs] \
    requests \
    pandas

# Initialize the database if it's the webserver
if [ "$1" = "webserver" ]; then
    echo "Initializing Airflow database..."
    airflow db init
    echo "Creating admin user..."
    airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com \
        --password admin || true
fi

# Execute the command
echo "Starting Airflow component: $1"
exec airflow "$@"
