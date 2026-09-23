"""
run_project.py
---------------
Single entry point that runs the whole pipeline end-to-end:

  ingestion -> preprocessing -> behaviour/product/pattern analysis
  -> RFM + K-Means segmentation -> ALS recommendation -> visualization

Usage:
    python run_project.py

Expected total runtime on a laptop:
  - With the demo (synthetic) dataset: under a minute.
  - With the real ~1M-row Online Retail II dataset: a few minutes,
    depending on your machine (this is local[*] Spark, not a cluster).
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from analysis import (  # noqa: E402
    run_customer_behaviour_analysis,
    run_product_category_analysis,
    run_purchase_pattern_analysis,
    save_all_results,
)
from customer_segmentation import (  # noqa: E402
    compute_rfm,
    label_clusters,
    run_kmeans_segmentation,
)
from preprocessing import save_cleaned_data  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from recommendation import (  # noqa: E402
    build_interaction_table,
    get_recommendations_for_customer,
    train_als_model,
)
from spark_utils import get_spark  # noqa: E402
from config import RESULTS_DIR  # noqa: E402


def main():
    spark = get_spark("EcommerceBigDataProject_FullRun")

    print("\n" + "=" * 70)
    print("STEP 1/6: DATA INGESTION + CLEANING")
    print("=" * 70)
    sales_df, cancellations_df = save_cleaned_data(spark)
    sales_df.cache()

    print("\n" + "=" * 70)
    print("STEP 2/6: CUSTOMER BEHAVIOUR / PRODUCT / PURCHASE PATTERN ANALYSIS")
    print("=" * 70)
    behaviour = run_customer_behaviour_analysis(spark, sales_df)
    products = run_product_category_analysis(spark, sales_df)
    patterns = run_purchase_pattern_analysis(spark, sales_df)
    save_all_results({"behaviour": behaviour, "products": products, "patterns": patterns})
    print(behaviour["summary"].to_string(index=False))

    print("\n" + "=" * 70)
    print("STEP 3/6: RFM + K-MEANS CUSTOMER SEGMENTATION")
    print("=" * 70)
    rfm_df = compute_rfm(spark, sales_df)
    rfm_df.cache()
    clustered, kmeans_model = run_kmeans_segmentation(rfm_df, k=4)
    cluster_profile, cluster_to_label = label_clusters(clustered)
    print(cluster_profile.to_string(index=False))

    mapping_expr = F.create_map([F.lit(x) for pair in cluster_to_label.items() for x in pair])
    clustered = clustered.withColumn("SegmentLabel", mapping_expr[F.col("cluster")])
    clustered.select(
        "CustomerID", "Recency", "Frequency", "Monetary", "cluster", "SegmentLabel"
    ).cache()
    cluster_profile.to_csv(os.path.join(RESULTS_DIR, "segmentation__cluster_profile.csv"), index=False)

    print("\n" + "=" * 70)
    print("STEP 4/6: ALS RECOMMENDATION SYSTEM")
    print("=" * 70)
    interactions = build_interaction_table(sales_df)
    als_model, indexer_model, rmse = train_als_model(interactions)

    sample_customer_row = (
        interactions.groupBy("CustomerID").count().orderBy(F.desc("count")).first()
    )
    sample_customer_id = sample_customer_row["CustomerID"]
    print(f"Generating sample recommendations for Customer ID: {sample_customer_id}")
    recs = get_recommendations_for_customer(
        spark, als_model, indexer_model, sales_df, sample_customer_id, n=5
    )
    if recs is not None:
        recs_pdf = recs.toPandas()
        print(recs_pdf.to_string(index=False))
        recs_pdf.to_csv(
            os.path.join(RESULTS_DIR, f"recommendations__customer_{sample_customer_id}.csv"),
            index=False,
        )

    print("\n" + "=" * 70)
    print("STEP 5/6: VISUALIZATION")
    print("=" * 70)
    # Imported here (after results exist) and run as a plain function call
    import visualization

    visualization.plot_top_products()
    visualization.plot_revenue_by_country()
    visualization.plot_spending_distribution()
    visualization.plot_purchase_trend()
    visualization.plot_segment_distribution()
    visualization.plot_recommendation_example()

    print("\n" + "=" * 70)
    print("STEP 6/6: DONE")
    print("=" * 70)
    print("All results saved under output/results/, all plots under output/plots/")

    # Generate a browser-ready dashboard from the completed results.
    import dashboard
    dashboard.build_dashboard()

    spark.stop()


if __name__ == "__main__":
    main()
