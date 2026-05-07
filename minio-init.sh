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

# Create the lakehouse bucket
echo "Creating lakehouse bucket..."
mc mb myminio/lakehouse --ignore-existing

# Set bucket to public for read/write access
echo "Setting bucket permissions to public..."
mc anonymous set public myminio/lakehouse

# Verify bucket creation and permissions
echo "Verifying bucket setup..."
mc ls myminio/

echo "MinIO initialization complete!"
echo "Bucket 'lakehouse' has been created at s3a://lakehouse/"
echo "Bucket permissions set to public for read/write access"

# Keep container running to ensure initialization completes
echo "Initialization container will exit now that setup is complete"
