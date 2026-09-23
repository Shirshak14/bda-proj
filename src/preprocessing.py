"""
preprocessing.py
-----------------
Step B: Data cleaning and preprocessing, done entirely with PySpark
DataFrame operations and Spark SQL -- no pandas here.

Cleaning rules applied (documented so you can justify each one in your
report / viva):

1. Drop rows with a NULL Customer ID.
   -> Without a customer id we cannot do customer-level behaviour
      analysis, RFM, segmentation, or recommendations at all, so these
      rows are unusable for THIS project (they are still valid rows for
      e.g. pure product-level analysis, just not for us).
2. Cast InvoiceDate to a proper timestamp type.
3. Remove exact duplicate rows.
4. Split out cancellations: any Invoice starting with 'C' is a
   cancellation/return (documented in the official UCI variable list).
   We keep cancellations in a separate DataFrame (useful for a
   "returns" insight) but exclude them from revenue/behaviour metrics,
   since a cancellation has a negative Quantity and would distort totals.
5. Remove rows with Quantity <= 0 or Price <= 0 from the "clean sales"
   DataFrame (after cancellations are already split out, these are
   typically data-entry adjustments / free samples, not real sales).
6. Add a computed TotalPrice = Quantity * Price column, since the raw
   dataset has no single "amount spent" column.
"""

import os
import sys

from pyspark.sql import functions as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_ingestion import load_raw_dataframe
from spark_utils import get_spark


def _standardize_columns(df):
    """
    Handle the fact that column names differ slightly between dataset
    mirrors (e.g. 'CustomerID' vs 'Customer ID', 'UnitPrice' vs 'Price').
    We rename everything to one consistent internal schema.
    """
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "").replace("_", "")
        if key == "customerid":
            rename_map[col] = "CustomerID"
        elif key == "invoiceno" or key == "invoice":
            rename_map[col] = "InvoiceNo"
        elif key == "stockcode":
            rename_map[col] = "StockCode"
        elif key == "description":
            rename_map[col] = "Description"
        elif key == "quantity":
            rename_map[col] = "Quantity"
        elif key == "invoicedate":
            rename_map[col] = "InvoiceDate"
        elif key in ("unitprice", "price"):
            rename_map[col] = "UnitPrice"
        elif key == "country":
            rename_map[col] = "Country"

    for old, new in rename_map.items():
        df = df.withColumnRenamed(old, new)
    return df


def clean_data(spark):
    raw_df = load_raw_dataframe(spark)
    df = _standardize_columns(raw_df)

    required_cols = [
        "InvoiceNo", "StockCode", "Description", "Quantity",
        "InvoiceDate", "UnitPrice", "CustomerID", "Country",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns missing after standardization: {missing}. "
            f"Actual columns found: {df.columns}"
        )

    total_rows = df.count()

    # 1. Drop rows with null CustomerID
    df = df.filter(F.col("CustomerID").isNotNull())

    # 2. Cast InvoiceDate to timestamp (handles both string and native types)
    df = df.withColumn("InvoiceDate", F.to_timestamp(F.col("InvoiceDate")))

    # 3. Drop exact duplicates
    df = df.dropDuplicates()

    # Cast numeric columns defensively
    df = df.withColumn("Quantity", F.col("Quantity").cast("int"))
    df = df.withColumn("UnitPrice", F.col("UnitPrice").cast("double"))
    df = df.withColumn("CustomerID", F.col("CustomerID").cast("int"))

    # 4. Split cancellations vs normal sales
    df = df.withColumn("IsCancellation", F.col("InvoiceNo").startswith("C"))
    cancellations_df = df.filter(F.col("IsCancellation") == True)  # noqa: E712
    sales_df = df.filter(F.col("IsCancellation") == False)  # noqa: E712

    # 5. Remove non-positive quantity/price from the clean sales set
    sales_df = sales_df.filter((F.col("Quantity") > 0) & (F.col("UnitPrice") > 0))

    # 6. Computed revenue column
    sales_df = sales_df.withColumn("TotalPrice", F.round(F.col("Quantity") * F.col("UnitPrice"), 2))

    # Extract date parts used later for trend analysis
    sales_df = (
        sales_df.withColumn("InvoiceYear", F.year("InvoiceDate"))
        .withColumn("InvoiceMonth", F.month("InvoiceDate"))
        .withColumn("InvoiceHour", F.hour("InvoiceDate"))
        .withColumn("InvoiceYearMonth", F.date_format("InvoiceDate", "yyyy-MM"))
    )

    clean_rows = sales_df.count()
    print(f"[preprocessing] Raw rows: {total_rows}")
    print(f"[preprocessing] Rows after removing nulls/dupes/cancellations/invalid: {clean_rows}")
    print(f"[preprocessing] Cancellation rows kept separately: {cancellations_df.count()}")

    return sales_df, cancellations_df


def save_cleaned_data(spark):
    sales_df, cancellations_df = clean_data(spark)
    # Windows local Spark can require Hadoop winutils.exe for filesystem writes.
    # Keep the cleaned DataFrame in Spark memory instead; the pipeline consumes it directly.
    sales_df.cache()
    print("[preprocessing] Cleaned sales data cached in Spark (no Hadoop/winutils write required)")
    return sales_df, cancellations_df


if __name__ == "__main__":
    spark = get_spark("Preprocessing")
    save_cleaned_data(spark)
    spark.stop()
