import streamlit as st
import pandas as pd
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="E-Commerce Profit Analysis",
    page_icon="🛒",
    layout="wide"
)

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv("e-commerce.csv")


# ==========================================
# LOAD ML MODEL
# ==========================================

@st.cache_resource
def load_model():
    model = joblib.load("tuned_random_forest.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor


dataset = load_data()

model, preprocessor = load_model()


# ==========================================
# TITLE
# ==========================================

st.title("🛒 E-Commerce Sales & Profit Analysis")

st.write(
    "Interactive dashboard for analyzing e-commerce sales, "
    "profitability, customers, and business performance."
)


# ==========================================
# KPI METRICS
# ==========================================

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
        dataset['Returned'].eq('Yes').mean() * 100
    )

    st.metric(
        "🔄 Return Rate",
        f"{return_rate:.2f}%"
    )


st.divider()


# ==========================================
# PROFIT BY PRODUCT CATEGORY
# ==========================================

st.subheader("📦 Profit by Product Category")

category_profit = (
    dataset
    .groupby('Product_Category')['Profit_Amount']
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(category_profit)


# ==========================================
# CUSTOMER SEGMENT PROFIT
# ==========================================

st.subheader("👥 Profit by Customer Segment")

segment_profit = (
    dataset
    .groupby('Customer_Segment')['Profit_Amount']
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(segment_profit)


# ==========================================
# MONTHLY PROFIT
# ==========================================

st.subheader("📅 Monthly Profit Trend")

monthly_profit = (
    dataset
    .groupby('Month')['Profit_Amount']
    .sum()
)

st.line_chart(monthly_profit)


# ==========================================
# ML PROFIT PREDICTION
# ==========================================

st.divider()

st.header("🤖 Profit Prediction")

st.write(
    "Enter order and customer details to predict the expected profit."
)


# ------------------------------------------
# Create input form using an existing row
# ------------------------------------------

with st.form("profit_prediction_form"):

    st.subheader("📝 Order Details")

    col1, col2, col3 = st.columns(3)

    with col1:

        order_id = st.text_input(
            "Order ID",
            value=str(dataset["Order_ID"].iloc[0])
        )

        customer_id = st.text_input(
            "Customer ID",
            value=str(dataset["Customer_ID"].iloc[0])
        )

        order_date = st.date_input(
            "Order Date"
        )

        customer_age = st.number_input(
            "Customer Age",
            min_value=1,
            max_value=100,
            value=30
        )

        customer_gender = st.selectbox(
            "Customer Gender",
            dataset["Customer_Gender"].dropna().unique()
        )

        country = st.selectbox(
            "Country",
            dataset["Country"].dropna().unique()
        )

        city = st.selectbox(
            "City",
            dataset["City"].dropna().unique()
        )

    with col2:

        customer_segment = st.selectbox(
            "Customer Segment",
            dataset["Customer_Segment"].dropna().unique()
        )

        product_id = st.text_input(
            "Product ID",
            value=str(dataset["Product_ID"].iloc[0])
        )

        product_category = st.selectbox(
            "Product Category",
            dataset["Product_Category"].dropna().unique()
        )

        product_subcategory = st.selectbox(
            "Product Subcategory",
            dataset["Product_Subcategory"].dropna().unique()
        )

        brand = st.selectbox(
            "Brand",
            dataset["Brand"].dropna().unique()
        )

        unit_price = st.number_input(
            "Unit Price",
            min_value=0.0,
            value=100.0
        )

        quantity = st.number_input(
            "Quantity",
            min_value=1,
            value=1
        )

    with col3:

        discount_percent = st.number_input(
            "Discount Percent",
            min_value=0.0,
            max_value=100.0,
            value=10.0
        )

        discount_amount = st.number_input(
            "Discount Amount",
            min_value=0.0,
            value=0.0
        )

        coupon_used = st.selectbox(
            "Coupon Used",
            dataset["Coupon_Used"].dropna().unique()
        )

        shipping_cost = st.number_input(
            "Shipping Cost",
            min_value=0.0,
            value=50.0
        )

        tax_amount = st.number_input(
            "Tax Amount",
            min_value=0.0,
            value=20.0
        )

        payment_method = st.selectbox(
            "Payment Method",
            dataset["Payment_Method"].dropna().unique()
        )

        device_type = st.selectbox(
            "Device Type",
            dataset["Device_Type"].dropna().unique()
        )


    predict_button = st.form_submit_button(
        "🔮 Predict Profit"
    )


# ==========================================
# PREDICTION
# ==========================================

if predict_button:

    # Create input DataFrame

    input_data = pd.DataFrame({

        "Order_ID": [order_id],

        "Customer_ID": [customer_id],

        "Order_Date": [str(order_date)],

        "Year": [order_date.year],

        "Month": [order_date.month],

        "Day": [order_date.day],

        "Day_Of_Week": [order_date.strftime("%A")],

        "Quarter": [f"Q{((order_date.month - 1) // 3) + 1}"],

        "Customer_Age": [customer_age],

        "Customer_Gender": [customer_gender],

        "Country": [country],

        "City": [city],

        "Customer_Segment": [customer_segment],

        "Product_ID": [product_id],

        "Product_Category": [product_category],

        "Product_Subcategory": [product_subcategory],

        "Brand": [brand],

        "Unit_Price": [unit_price],

        "Quantity": [quantity],

        "Discount_Percent": [discount_percent],

        "Discount_Amount": [discount_amount],

        "Coupon_Used": [coupon_used],

        "Shipping_Cost": [shipping_cost],

        "Tax_Amount": [tax_amount],

        "Payment_Method": [payment_method],

        "Device_Type": [device_type]

    })


    try:

        # Transform input

        transformed_data = preprocessor.transform(
            input_data
        )


        # Predict

        prediction = model.predict(
            transformed_data
        )


        # Display result

        st.success(
            f"💰 Predicted Profit: ₹{prediction[0]:,.2f}"
        )


    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )
