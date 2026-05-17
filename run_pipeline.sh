#!/bin/bash

set -e

PIDS=""

cleanup() {
    log_warn "Stopping submitted Spark jobs..."
    for pid in $PIDS; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
}

trap cleanup INT TERM EXIT

echo "=========================================="
echo "🚀 Starting BigData Pipeline"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Step 1: Check Docker containers
log_info "Step 1: Checking Docker containers..."
docker-compose ps | grep -E "spark-master|broker|minio|nessie" || {
    log_error "Required containers not running!"
    exit 1
}

# Step 2: Clean up old checkpoints and tables
log_warn "Step 2: Cleaning up old checkpoints..."
docker-compose exec -T minio bash -c "mc alias set minio http://localhost:9000 admin password 2>/dev/null; mc rm --force -r minio/bronze/ 2>/dev/null || true" || true
docker-compose exec -T minio bash -c "mc rm --force -r minio/silver/ 2>/dev/null || true" || true

log_info "Creating fresh tables..."
# docker-compose exec -T spark-master spark-sql << EOF
# -- Drop old tables if exist
# DROP TABLE IF EXISTS nessie.bronze.stock_market;
# DROP TABLE IF EXISTS nessie.bronze.stock_news;
# DROP TABLE IF EXISTS nessie.silver.stock_market;
# DROP TABLE IF EXISTS nessie.silver.stock_news;
# DROP NAMESPACE IF EXISTS nessie.bronze CASCADE;
# DROP NAMESPACE IF EXISTS nessie.silver CASCADE;
# EOF

# Step 3: Run kafka_to_bronze
log_info "Step 3: Starting kafka_to_bronze job..."
docker-compose exec -T spark-master spark-submit \
  --master spark://spark-master:7077 \
  --conf spark.executor.memory=1g \
  --conf spark.executor.cores=1 \
  --conf spark.cores.max=1 \
  //opt/spark-apps/kafka_to_bronze.py &

KAFKA_BRONZE_PID=$!
PIDS="$PIDS $KAFKA_BRONZE_PID"

# Wait for kafka_to_bronze to create tables
sleep 10

# Step 4: Run bronze_to_silver
log_info "Step 4: Starting bronze_to_silver job..."
docker-compose exec -T spark-master spark-submit \
  --master spark://spark-master:7077 \
  --conf spark.executor.memory=1g \
  --conf spark.executor.cores=1 \
  --conf spark.cores.max=1 \
  //opt/spark-apps/bronze_to_silver.py &

BRONZE_SILVER_PID=$!
PIDS="$PIDS $BRONZE_SILVER_PID"

# Step 5: Run silver_to_ml_features
log_info "Step 5: Starting silver_to_ml_features job..."
docker-compose exec -T spark-master spark-submit \
  --master spark://spark-master:7077 \
  --conf spark.executor.memory=3500m \
  --conf spark.executor.cores=2 \
  --conf spark.cores.max=2 \
  //opt/spark-apps/silver_to_ml_features.py &

SILVER_ML_PID=$!
PIDS="$PIDS $SILVER_ML_PID"

log_info "All jobs started! Waiting for completion..."
log_warn "Press Ctrl+C to stop all jobs"

# Wait for all background jobs
wait
trap - INT TERM EXIT

log_info "All jobs completed successfully!"
echo "=========================================="
