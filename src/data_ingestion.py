"""
data_ingestion.py
------------------
Step A of the pipeline: E-commerce data ingestion.

The Online Retail II dataset is distributed by UCI as a single .xlsx
file with TWO sheets ("Year 2009-2010" and "Year 2010-2011"). Spark
cannot read .xlsx natively without extra jars, so we use pandas ONLY
for this one-time format conversion (xlsx -> csv), exactly as allowed
by the assignment ("Pandas only where appropriate"). Every analysis
step after this uses PySpark, not pandas.

If you haven't downloaded the real dataset yet, this script will fall
back to a small schema-identical DEMO file so you can test that the
whole pipeline runs end-to-end. The demo file is clearly not the real
dataset -- see README.md "Dataset setup" for the real download link.
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import RAW_EXCEL_PATH, RAW_CSV_PATH, DEMO_CSV_PATH
from spark_utils import get_spark


def ensure_csv_available() -> str:
    """
    Returns the path to a CSV file ready to be read by Spark.
    Priority: already-converted CSV > real xlsx (converted now) > demo CSV.
    """
    if os.path.exists(RAW_CSV_PATH):
        print(f"[ingestion] Found existing CSV at {RAW_CSV_PATH}")
        return RAW_CSV_PATH

    if os.path.exists(RAW_EXCEL_PATH):
        print(f"[ingestion] Converting real dataset {RAW_EXCEL_PATH} -> CSV (one-time, pandas)")
        sheet_names = ["Year 2009-2010", "Year 2010-2011"]
        frames = []
        xls = pd.ExcelFile(RAW_EXCEL_PATH)
        for sheet in xls.sheet_names:
            if sheet not in sheet_names:
                print(f"[ingestion] Note: unexpected sheet name '{sheet}', reading it anyway")
            df = pd.read_excel(RAW_EXCEL_PATH, sheet_name=sheet)
            frames.append(df)
        full_df = pd.concat(frames, ignore_index=True)
        full_df.to_csv(RAW_CSV_PATH, index=False)
        print(f"[ingestion] Wrote combined CSV with {len(full_df)} rows to {RAW_CSV_PATH}")
        return RAW_CSV_PATH

    print(
        "[ingestion] WARNING: Real dataset not found at "
        f"{RAW_EXCEL_PATH} or {RAW_CSV_PATH}.\n"
        "[ingestion] Falling back to the small DEMO sample so you can test "
        "the pipeline. This is NOT the real dataset -- see README.md "
        "'Dataset setup' before generating your final report/screenshots."
    )
    if not os.path.exists(DEMO_CSV_PATH):
        raise FileNotFoundError(
            f"No dataset found at all. Expected the real file at {RAW_EXCEL_PATH} "
            f"or the demo file at {DEMO_CSV_PATH}. Run make_demo_data.py first "
            "if you just want to test the pipeline."
        )
    return DEMO_CSV_PATH


def load_raw_dataframe(spark):
    """Load the raw transactions into a Spark DataFrame with schema inference."""
    csv_path = ensure_csv_available()

    df = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv(csv_path)
    )
    return df


if __name__ == "__main__":
    spark = get_spark("DataIngestion")
    df = load_raw_dataframe(spark)

    print("\n=== Inferred Schema ===")
    df.printSchema()

    row_count = df.count()
    print(f"\n=== Total rows ingested: {row_count} ===")
    print("\n=== Sample rows ===")
    df.show(5, truncate=False)

    spark.stop()
