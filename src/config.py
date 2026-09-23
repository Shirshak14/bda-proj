"""
config.py
---------
Central place for file paths used across the project, so every script
agrees on where raw data, processed data, and outputs live.
"""

import os

# Project root = one level above this src/ folder
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# The real dataset, once downloaded, should be placed here.
# See README.md section "Dataset setup" for exact download instructions.
RAW_EXCEL_PATH = os.path.join(DATA_RAW_DIR, "online_retail_II.xlsx")
RAW_CSV_PATH = os.path.join(DATA_RAW_DIR, "online_retail_II.csv")

# Small schema-identical demo file (NOT the real dataset) used only so
# the pipeline can be test-run before you've downloaded the real ~7MB file.
DEMO_CSV_PATH = os.path.join(DATA_RAW_DIR, "demo_sample_data.csv")

CLEANED_PARQUET_PATH = os.path.join(DATA_PROCESSED_DIR, "cleaned_transactions.parquet")
RFM_PARQUET_PATH = os.path.join(DATA_PROCESSED_DIR, "rfm_table.parquet")
SEGMENTED_PARQUET_PATH = os.path.join(DATA_PROCESSED_DIR, "customer_segments.parquet")

for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, OUTPUT_DIR, PLOTS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)
