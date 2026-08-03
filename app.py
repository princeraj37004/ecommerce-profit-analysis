import streamlit as st
import pandas as pd

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="E-Commerce Profit Analysis",
    page_icon="📊",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("e-commerce.csv")

dataset = load_data()

# =========================
# TITLE
# =========================
st.title("🛒 E-Commerce Sales & Profit Analysis")

st.write(
    "Interactive dashboard for analyzing e-commerce sales, "
    "profitability, customers, and business performance."
)

# =========================
# KPI METRICS
# =========================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📦 Total Orders",
        f"{len(dataset):,}"
    )

with col2:
    st.metric(
        "💰 Total Profit",
        f"{dataset['Profit_Amount'].sum():,.2f}"
    )

with col3:
    st.metric(
        "📈 Average Profit",
        f"{dataset['Profit_Amount'].mean():,.2f}"
    )

with col4:
    return_rate = (
        dataset["Returned"].eq("Yes").mean() * 100
    )

    st.metric(
        "🔄 Return Rate",
        f"{return_rate:.2f}%"
    )

st.divider()

# =========================
# PRODUCT CATEGORY PROFIT
# =========================
st.subheader("📦 Profit by Product Category")

category_profit = (
    dataset.groupby("Product_Category")["Profit_Amount"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(category_profit)

# =========================
# CUSTOMER SEGMENT PROFIT
# =========================
st.subheader("👥 Profit by Customer Segment")

segment_profit = (
    dataset.groupby("Customer_Segment")["Profit_Amount"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(segment_profit)

# =========================
# MONTHLY PROFIT TREND
# =========================
st.subheader("📅 Monthly Profit Trend")

monthly_profit = (
    dataset.groupby("Month")["Profit_Amount"]
    .sum()
)

st.line_chart(monthly_profit)

# =========================
# DATA PREVIEW
# =========================
st.subheader("📋 Dataset Preview")

st.dataframe(
    dataset.head(10),
    use_container_width=True
)
