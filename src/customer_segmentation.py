"""
customer_segmentation.py
-------------------------
Step G: Customer segmentation using RFM (Recency, Frequency, Monetary)
features computed in Spark SQL, followed by K-Means clustering using
PySpark MLlib.

RFM definitions used:
- Recency  : days between the customer's most recent invoice and one
             day after the dataset's overall latest invoice date.
             (Lower recency = purchased more recently = better.)
- Frequency: number of DISTINCT invoices (orders) placed by the customer.
- Monetary : total amount spent by the customer (sum of TotalPrice).

Cluster labels are NOT hard-coded. After K-Means runs, we look at the
actual mean Monetary/Frequency/Recency of each cluster and rank the
clusters accordingly, so the "High-value" label always goes to
whichever cluster the DATA says spends the most -- not a guess.
"""

import os
import sys

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import functions as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR
from preprocessing import save_cleaned_data
from spark_utils import get_spark


def compute_rfm(spark, sales_df):
    sales_df.createOrReplaceTempView("sales")

    reference_date_row = spark.sql("SELECT MAX(InvoiceDate) AS max_date FROM sales").collect()[0]
    reference_date = reference_date_row["max_date"]

    rfm_df = spark.sql(
        f"""
        SELECT
            CustomerID,
            DATEDIFF(TIMESTAMP('{reference_date}'), MAX(InvoiceDate)) AS Recency,
            COUNT(DISTINCT InvoiceNo) AS Frequency,
            ROUND(SUM(TotalPrice), 2) AS Monetary
        FROM sales
        GROUP BY CustomerID
        """
    )
    # Recency of 0 means "last purchase was on the very last day in the
    # dataset" -- add 1 so log-style downstream reasoning never hits 0.
    rfm_df = rfm_df.withColumn("Recency", F.col("Recency") + F.lit(1))
    return rfm_df


def run_kmeans_segmentation(rfm_df, k: int = 4, seed: int = 42):
    """
    K-Means on standardized RFM features.

    k=4 is a reasonable default for a college project (maps loosely to
    "high value / regular / occasional / low-activity"), but the elbow
    method (see the commented block below) is the correct way to justify
    k for your report -- run it once, screenshot the elbow plot, and
    state the k you chose based on it.
    """
    assembler = VectorAssembler(
        inputCols=["Recency", "Frequency", "Monetary"], outputCol="rfm_features_raw"
    )
    assembled = assembler.transform(rfm_df)

    scaler = StandardScaler(
        inputCol="rfm_features_raw", outputCol="rfm_features", withMean=True, withStd=True
    )
    scaler_model = scaler.fit(assembled)
    scaled = scaler_model.transform(assembled)

    kmeans = KMeans(featuresCol="rfm_features", predictionCol="cluster", k=k, seed=seed)
    model = kmeans.fit(scaled)
    clustered = model.transform(scaled)

    return clustered, model


def _elbow_method_snippet():
    """
    Not called automatically (keeps the default run fast). Uncomment
    and run manually once to justify your choice of k with a real plot:

        costs = []
        for k in range(2, 9):
            km = KMeans(featuresCol="rfm_features", k=k, seed=42)
            m = km.fit(scaled)
            costs.append(m.summary.trainingCost)
        # then plot k (2..8) vs costs with matplotlib to find the "elbow"
    """
    pass


def label_clusters(clustered_df):
    """
    Rank clusters by their actual mean Monetary value (descending) and
    assign human-readable labels in that order. If the data only really
    supports 2-3 meaningfully different clusters, duplicate/similar
    labels are fine -- we do NOT force 4 distinct stories onto the data.
    """
    cluster_profile = (
        clustered_df.groupBy("cluster")
        .agg(
            F.round(F.avg("Recency"), 1).alias("avg_recency"),
            F.round(F.avg("Frequency"), 1).alias("avg_frequency"),
            F.round(F.avg("Monetary"), 1).alias("avg_monetary"),
            F.count("*").alias("num_customers"),
        )
        .orderBy(F.desc("avg_monetary"))
        .toPandas()
    )

    label_options = [
        "High-value customers",
        "Regular customers",
        "Occasional customers",
        "Low-activity customers",
    ]
    cluster_profile["segment_label"] = [
        label_options[i] if i < len(label_options) else f"Segment {i+1}"
        for i in range(len(cluster_profile))
    ]

    cluster_to_label = dict(zip(cluster_profile["cluster"], cluster_profile["segment_label"]))
    return cluster_profile, cluster_to_label


if __name__ == "__main__":
    spark = get_spark("CustomerSegmentation")
    sales_df, _ = save_cleaned_data(spark)

    rfm_df = compute_rfm(spark, sales_df)
    rfm_df.cache()
    print("[segmentation] RFM table cached in Spark (no Hadoop/winutils write required)")

    clustered, model = run_kmeans_segmentation(rfm_df, k=4)
    cluster_profile, cluster_to_label = label_clusters(clustered)

    print("\n=== Cluster profile (ranked by avg Monetary) ===")
    print(cluster_profile.to_string(index=False))

    mapping_expr = F.create_map([F.lit(x) for pair in cluster_to_label.items() for x in pair])
    clustered = clustered.withColumn("SegmentLabel", mapping_expr[F.col("cluster")])

    clustered.select(
        "CustomerID", "Recency", "Frequency", "Monetary", "cluster", "SegmentLabel"
    ).cache()
    print("[segmentation] Customer segments cached in Spark (no Hadoop/winutils write required)")

    cluster_profile.to_csv(os.path.join(RESULTS_DIR, "segmentation__cluster_profile.csv"), index=False)

    spark.stop()
