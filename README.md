# 🛒 E-Commerce Customer Behaviour Analysis & Recommendation System

A Big Data Analytics mini-project built using **Apache PySpark** to analyze large-scale e-commerce transaction data, understand customer purchasing behaviour, segment customers using **RFM + K-Means**, and generate product recommendations using **ALS Collaborative Filtering**.

---

## 🚀 Live Dashboard

### 🌐 [Open the E-Commerce Analytics Dashboard](https://shirshak14.github.io/bda-proj/)

The dashboard provides a browser-based view of the project's analysis and results.

It includes:

- 📊 Key business metrics
- 👥 Customer segmentation
- 🛍️ Top products
- 🌍 Revenue by country
- 📈 Monthly revenue trends
- 🤖 ALS product recommendations
- 📋 Detailed analysis tables
- 📊 Generated project visualizations

---

## 🎯 Project Aim

To design and implement a Big Data Analytics pipeline using **Apache PySpark** that processes real-world e-commerce transaction data, analyzes customer behaviour, performs customer segmentation, and generates personalized product recommendations.

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
- Present the results through a browser-based dashboard.

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

### Dataset Processing

The original dataset contains:

**1,067,371 transaction records**

After data cleaning:

**779,425 valid transaction records**

Cancellation records were handled separately during preprocessing.

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
| Pandas | Data processing and aggregated results |
| Matplotlib | Data visualization |
| Seaborn | Data visualization |
| HTML/CSS | Dashboard |
| Jupyter Notebook | Analysis and experimentation |
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
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
   Behaviour Analysis   RFM Analysis    ALS Model
          │                │                │
          │                ▼                │
          │          K-Means Clustering     │
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                    Data Visualizations
                           │
                           ▼
                       Dashboard
