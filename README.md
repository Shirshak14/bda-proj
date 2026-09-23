# 🛒 E-Commerce Customer Behaviour Analysis & Recommendation System

A Big Data Analytics mini-project built using **Apache PySpark** to analyze large-scale e-commerce transaction data, understand customer purchasing behaviour, segment customers using **RFM + K-Means**, and generate product recommendations using **ALS Collaborative Filtering**.

## 🚀 Dashboard

The project includes a browser-based analytics dashboard containing:

- 📊 Key business metrics
- 👥 Customer segmentation
- 🛍️ Top products
- 🌍 Revenue by country
- 📈 Monthly revenue trends
- 🤖 ALS product recommendations
- 📋 Detailed analysis tables

### 🌐 Live Dashboard

**[Open E-Commerce Analytics Dashboard](./output/dashboard.html)**

> If viewing the project through GitHub Pages, the dashboard will open directly in the browser.

---

## 🎯 Project Aim

To design and implement a Big Data pipeline using **PySpark** that processes real-world e-commerce transaction data, analyzes customer behaviour, performs customer segmentation, and provides personalized product recommendations.

---

## 📌 Objectives

- Process a large real-world e-commerce dataset using PySpark.
- Clean and preprocess transaction data.
- Analyze customer purchasing behaviour.
- Analyze product sales and purchasing patterns.
- Perform RFM (Recency, Frequency, Monetary) analysis.
- Segment customers using K-Means clustering.
- Build a product recommendation system using ALS.
- Generate business insights through visualizations.
- Provide an interactive browser-based dashboard.

---

## 📂 Dataset

The project uses the **Online Retail II** dataset from the UCI Machine Learning Repository.

**Dataset:** Online Retail II  
**Source:** UCI Machine Learning Repository

The dataset contains more than **1 million transaction records** from a UK-based online retailer.

### Dataset Features

- Invoice
- StockCode
- Description
- Quantity
- InvoiceDate
- Price
- Customer ID
- Country

The original dataset contains:

**1,067,371 transaction records**

After cleaning:

**779,425 valid transaction records**

---

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Apache PySpark | Big Data processing |
| Spark SQL | Data analysis |
| Spark MLlib | Machine Learning |
| K-Means | Customer segmentation |
| ALS | Product recommendation |
| Pandas | XLSX → CSV conversion and small aggregated results |
| Matplotlib | Visualization |
| Seaborn | Visualization |
| HTML/CSS | Dashboard |
| VS Code | Development |

---

## 🏗️ Project Architecture

```text
Online Retail II Dataset
          │
          ▼
   Data Ingestion
          │
          ▼
   Data Cleaning
          │
          ▼
   PySpark DataFrames
          │
     ┌────┼───────────────┐
     ▼    ▼               ▼
Behaviour  RFM          ALS
Analysis   Analysis     Recommendation
     │      │               │
     │      ▼               │
     │   K-Means             │
     │   Segmentation        │
     └──────┼────────────────┘
            ▼
       Visualizations
            │
            ▼
        Dashboard