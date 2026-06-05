import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession.builder.appName("TestSpark").getOrCreate()
    data = [("Alice", 34), ("Bob", 45), ("Charlie", 29)]
    df = spark.createDataFrame(data, ["Name", "Age"])
    df.show()
    spark.stop()    
