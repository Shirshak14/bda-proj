"""
make_demo_data.py
------------------
Generates a SMALL, clearly-synthetic CSV with the exact same 8 columns
as the real Online Retail II dataset (Invoice, StockCode, Description,
Quantity, InvoiceDate, Price, Customer ID, Country).

WHY THIS EXISTS: this is a college project running on a laptop with no
internet access baked into the automated build step, so this file lets
you confirm every script in src/ actually runs end-to-end BEFORE you
download the real ~7 MB / 1,067,371-row dataset.

DO NOT use this file's output for your actual report, screenshots, or
results -- it is randomly generated and the numbers mean nothing. Once
you've downloaded the real dataset (see README.md "Dataset setup"),
data_ingestion.py will automatically prefer the real file over this one.
"""

import os
import random
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DEMO_CSV_PATH

random.seed(42)

PRODUCTS = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER"),
    ("71053", "WHITE METAL LANTERN"),
    ("84406B", "CREAM CUPID HEARTS COAT HANGER"),
    ("21730", "GLASS STAR FROSTED T-LIGHT HOLDER"),
    ("22752", "SET 7 BABUSHKA NESTING BOXES"),
    ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE"),
    ("22633", "HAND WARMER UNION JACK"),
    ("21212", "PACK OF 72 RETROSPOT CAKE CASES"),
    ("20725", "LUNCH BAG RED RETROSPOT"),
    ("22423", "REGENCY CAKESTAND 3 TIER"),
    ("47566", "PARTY BUNTING"),
    ("85099B", "JUMBO BAG RED RETROSPOT"),
]

COUNTRIES = ["United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands"]

N_CUSTOMERS = 400
N_TRANSACTIONS = 8000

start_date = datetime(2010, 1, 1)
rows = []
invoice_counter = 536000

for i in range(N_TRANSACTIONS):
    invoice_counter += 1
    invoice_no = str(invoice_counter)
    is_cancellation = random.random() < 0.02
    if is_cancellation:
        invoice_no = "C" + invoice_no

    customer_id = random.randint(12000, 12000 + N_CUSTOMERS)
    country = random.choice(COUNTRIES)
    n_items = random.randint(1, 5)
    invoice_date = start_date + timedelta(
        days=random.randint(0, 500), hours=random.randint(8, 19), minutes=random.randint(0, 59)
    )

    for _ in range(n_items):
        stock_code, description = random.choice(PRODUCTS)
        quantity = random.randint(1, 20)
        if is_cancellation:
            quantity = -quantity
        price = round(random.uniform(0.5, 15.0), 2)

        rows.append(
            {
                "Invoice": invoice_no,
                "StockCode": stock_code,
                "Description": description,
                "Quantity": quantity,
                "InvoiceDate": invoice_date.strftime("%Y-%m-%d %H:%M:%S"),
                "Price": price,
                "Customer ID": customer_id,
                "Country": country,
            }
        )

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(DEMO_CSV_PATH), exist_ok=True)
df.to_csv(DEMO_CSV_PATH, index=False)
print(f"Demo (synthetic, NOT real) dataset written: {DEMO_CSV_PATH}")
print(f"Rows: {len(df)}, Customers: {df['Customer ID'].nunique()}")
