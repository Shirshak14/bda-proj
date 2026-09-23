"""
spark_utils.py
--------------
Small helper module so every script in this project creates its
SparkSession the same way instead of repeating boilerplate.
"""

from pyspark.sql import SparkSession


def get_spark(app_name: str = "EcommerceBigDataProject") -> SparkSession:
    """
    Create (or fetch, if already running) a local SparkSession.

    local[*]  -> use all cores available on this machine.
    We also bump the shuffle partitions down from Spark's default of 200,
    because on a laptop with a ~1M-row dataset, 200 tiny partitions just
    adds scheduling overhead. 8 is a reasonable number for local testing;
    on a real cluster with more data you would leave the default or tune
    it based on cluster size.
    """
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
