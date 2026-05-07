#!/bin/bash

SPARK_MASTER="spark://spark-master:7077"
APP_FILE="//opt/spark-apps/test_spark.py"


PACKAGES_STR=$(IFS=,; echo "${PACKAGES[*]}")

docker-compose exec -it spark-master //opt/spark/bin/spark-submit \
    --master $SPARK_MASTER \
    $APP_FILE