#!/bin/sh

echo "Starting MinIO initialization..."

# Wait for MinIO to be ready
echo "Waiting for MinIO to be ready..."
until mc alias set myminio http://minio:9000 admin password > /dev/null 2>&1; do
    echo "MinIO is not ready yet. Waiting..."
    sleep 3
done

echo "MinIO is ready!"

# Configure MinIO client
echo "Configuring MinIO client..."
mc alias set myminio http://minio:9000 admin password

# Create lakehouse buckets
echo "Creating lakehouse buckets..."
mc mb myminio/bronze --ignore-existing
mc mb myminio/silver --ignore-existing
mc mb myminio/ml --ignore-existing

# Set buckets to public for read/write access in this local stack
echo "Setting bucket permissions to public..."
mc anonymous set public myminio/bronze
mc anonymous set public myminio/silver
mc anonymous set public myminio/ml

# Verify bucket creation and permissions
echo "Verifying bucket setup..."
mc ls myminio/

echo "MinIO initialization complete!"
echo "Buckets have been created at s3a://bronze/, s3a://silver/, and s3a://ml/"
echo "Bucket permissions set to public for read/write access"

# Keep container running to ensure initialization completes
echo "Initialization container will exit now that setup is complete"
