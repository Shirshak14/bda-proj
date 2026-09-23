"""
recommendation.py
------------------
Step H: Recommendation system using PySpark MLlib's ALS
(Alternating Least Squares) collaborative filtering.

Why ALS fits this dataset: Online Retail II has real user-item
interaction data (CustomerID x StockCode, with Quantity purchased),
but NO explicit star ratings. That is exactly the "implicit feedback"
case ALS is designed for (implicitPrefs=True) -- we treat total
quantity purchased of a product by a customer as a confidence signal
that the customer likes that product, following the standard
Hu/Koren/Volinsky implicit-ALS approach used in production recommender
systems.

StockCode is alphanumeric (e.g. "85123A"), so we use StringIndexer to
map it (and CustomerID, for safety) to the numeric IDs ALS requires,
then map back to the original StockCode/Description for the final
human-readable output.
"""

import os
import sys

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import IndexToString, StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR
from preprocessing import save_cleaned_data
from spark_utils import get_spark


def build_interaction_table(sales_df):
    """
    One row per (CustomerID, StockCode) with total quantity purchased
    -- this becomes our implicit "rating"/confidence value for ALS.
    """
    interactions = (
        sales_df.groupBy("CustomerID", "StockCode")
        .agg(F.sum("Quantity").alias("total_qty"))
        .filter(F.col("total_qty") > 0)
    )
    return interactions


def train_als_model(interactions_df, seed: int = 42):
    stock_indexer = StringIndexer(
        inputCol="StockCode", outputCol="StockCodeIndex", handleInvalid="skip"
    )
    indexer_model = stock_indexer.fit(interactions_df)
    indexed = indexer_model.transform(interactions_df)

    train_df, test_df = indexed.randomSplit([0.8, 0.2], seed=seed)

    als = ALS(
        userCol="CustomerID",
        itemCol="StockCodeIndex",
        ratingCol="total_qty",
        implicitPrefs=True,
        rank=10,
        maxIter=10,
        regParam=0.1,
        coldStartStrategy="drop",
        seed=seed,
    )
    model = als.fit(train_df)

    predictions = model.transform(test_df)
    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="total_qty", predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    print(f"[recommendation] ALS RMSE on held-out interactions: {rmse:.4f}")
    print(
        "[recommendation] Note: RMSE is a secondary sanity-check metric here, not the "
        "main goal -- with implicit feedback the real target is ranking quality "
        "(would the recommended products plausibly interest the customer), which "
        "is best judged by inspecting the actual recommendations below."
    )

    return model, indexer_model, rmse


def get_recommendations_for_customer(spark, model, indexer_model, sales_df, customer_id: int, n: int = 5):
    labels = indexer_model.labels  # index position -> original StockCode string
    index_to_stock = IndexToString(
        inputCol="StockCodeIndex", outputCol="StockCode", labels=labels
    )

    user_df = spark.createDataFrame([(customer_id,)], ["CustomerID"])
    recs = model.recommendForUserSubset(user_df, n)

    if recs.count() == 0:
        return None

    exploded = recs.select(
        "CustomerID", F.explode("recommendations").alias("rec")
    ).select(
        "CustomerID",
        F.col("rec.StockCodeIndex").alias("StockCodeIndex"),
        F.col("rec.rating").alias("score"),
    )

    exploded = index_to_stock.transform(exploded)

    product_lookup = sales_df.select("StockCode", "Description").dropDuplicates(["StockCode"])
    result = exploded.join(product_lookup, on="StockCode", how="left").orderBy(F.desc("score"))
    return result


if __name__ == "__main__":
    spark = get_spark("Recommendation")
    sales_df, _ = save_cleaned_data(spark)

    interactions = build_interaction_table(sales_df)
    print(f"[recommendation] Built {interactions.count()} (customer, product) interaction rows")

    model, indexer_model, rmse = train_als_model(interactions)

    # Demo: recommend for whichever customer bought the most distinct products
    sample_customer_row = (
        interactions.groupBy("CustomerID").count().orderBy(F.desc("count")).first()
    )
    sample_customer_id = sample_customer_row["CustomerID"]

    print(f"\n=== Sample recommendations for Customer ID: {sample_customer_id} ===")
    recs = get_recommendations_for_customer(spark, model, indexer_model, sales_df, sample_customer_id, n=5)
    if recs is not None:
        recs_pdf = recs.toPandas()
        print(recs_pdf.to_string(index=False))
        recs_pdf.to_csv(
            os.path.join(RESULTS_DIR, f"recommendations__customer_{sample_customer_id}.csv"),
            index=False,
        )
    else:
        print("No recommendations could be generated for this customer (cold-start).")

    spark.stop()
