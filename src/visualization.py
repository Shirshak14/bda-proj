"""
visualization.py
-----------------
Step I: Data visualization.

Design choice: Spark does the heavy aggregation (millions of rows ->
small summarized tables), and only those already-small pandas
DataFrames (saved as CSV by analysis.py / customer_segmentation.py /
recommendation.py) are plotted with matplotlib/seaborn. This is the
standard, correct big-data pattern: never pull the raw multi-million
row dataset into pandas just to plot it.

Run this AFTER run_project.py (or analysis.py, customer_segmentation.py,
recommendation.py individually) so the CSVs in output/results exist.
"""

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")  # safe for headless / script execution
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import PLOTS_DIR, RESULTS_DIR

sns.set_theme(style="whitegrid")


def _save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[visualization] Saved {path}")


def plot_top_products(results_dir=RESULTS_DIR):
    path = os.path.join(results_dir, "products__top_products_by_revenue.csv")
    if not os.path.exists(path):
        print(f"[visualization] Skipping top products chart, {path} not found")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=df, y="Description", x="revenue", ax=ax, palette="viridis")
    ax.set_title("Top 10 Products by Revenue")
    ax.set_xlabel("Revenue")
    ax.set_ylabel("Product")
    _save(fig, "01_top_products_by_revenue.png")


def plot_revenue_by_country(results_dir=RESULTS_DIR):
    """
    Stands in for "revenue by category" (this dataset has no separate
    category column -- see README/report for why Country is used as
    the secondary business dimension instead).
    """
    path = os.path.join(results_dir, "patterns__country_revenue.csv")
    if not os.path.exists(path):
        print(f"[visualization] Skipping revenue-by-country chart, {path} not found")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=df, y="Country", x="revenue", ax=ax, palette="magma")
    ax.set_title("Revenue by Country (Top 10)")
    ax.set_xlabel("Revenue")
    ax.set_ylabel("Country")
    _save(fig, "02_revenue_by_country.png")


def plot_spending_distribution(results_dir=RESULTS_DIR):
    path = os.path.join(results_dir, "behaviour__spending_distribution.csv")
    if not os.path.exists(path):
        print(f"[visualization] Skipping spending distribution chart, {path} not found")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["total_spent"], bins=40, kde=True, ax=ax, color="steelblue")
    ax.set_title("Customer Spending Distribution")
    ax.set_xlabel("Total Spent per Customer")
    _save(fig, "03_customer_spending_distribution.png")


def plot_purchase_trend(results_dir=RESULTS_DIR):
    path = os.path.join(results_dir, "patterns__monthly_trend.csv")
    if not os.path.exists(path):
        print(f"[visualization] Skipping purchase trend chart, {path} not found")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df, x="InvoiceYearMonth", y="revenue", marker="o", ax=ax)
    ax.set_title("Monthly Revenue Trend")
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Revenue")
    plt.xticks(rotation=45)
    _save(fig, "04_monthly_revenue_trend.png")


def plot_segment_distribution(results_dir=RESULTS_DIR):
    path = os.path.join(results_dir, "segmentation__cluster_profile.csv")
    if not os.path.exists(path):
        print(f"[visualization] Skipping segment distribution chart, {path} not found")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        df["num_customers"],
        labels=df["segment_label"],
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("pastel"),
    )
    ax.set_title("Customer Segment Distribution")
    _save(fig, "05_customer_segment_distribution.png")


def plot_recommendation_example(results_dir=RESULTS_DIR):
    matches = glob.glob(os.path.join(results_dir, "recommendations__customer_*.csv"))
    if not matches:
        print("[visualization] Skipping recommendation chart, no recommendation CSV found")
        return
    df = pd.read_csv(matches[0])
    fig, ax = plt.subplots(figsize=(8, 5))
    label_col = "Description" if "Description" in df.columns else "StockCode"
    sns.barplot(data=df, y=label_col, x="score", ax=ax, palette="crest")
    ax.set_title("Sample ALS Recommendation Scores")
    ax.set_xlabel("ALS Recommendation Score")
    _save(fig, "06_sample_recommendations.png")


if __name__ == "__main__":
    plot_top_products()
    plot_revenue_by_country()
    plot_spending_distribution()
    plot_purchase_trend()
    plot_segment_distribution()
    plot_recommendation_example()
    print("\n[visualization] Done. Check output/plots/")
