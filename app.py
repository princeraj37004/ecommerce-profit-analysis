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


# Load dataset and model
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
        dataset["Returned"].eq("Yes").mean() * 100
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
    .groupby("Product_Category")["Profit_Amount"]
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
    .groupby("Customer_Segment")["Profit_Amount"]
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
    .groupby("Month")["Profit_Amount"]
    .sum()
)


st.line_chart(monthly_profit)


# ==========================================
# ML PROFIT PREDICTION
# ==========================================

st.divider()

st.header("🤖 Profit Prediction")

st.write(
    "Enter customer and order details to predict "
    "the expected profit using the Tuned Random Forest model."
)


# ==========================================
# PROFIT PREDICTION FORM
# ==========================================

with st.form("profit_prediction_form"):

    col1, col2, col3 = st.columns(3)


    # ======================================
    # COLUMN 1
    # ======================================

    with col1:

        year = st.number_input(
            "Year",
            min_value=2020,
            max_value=2035,
            value=2025
        )


        month = st.number_input(
            "Month",
            min_value=1,
            max_value=12,
            value=1
        )


        day = st.number_input(
            "Day",
            min_value=1,
            max_value=31,
            value=1
        )


        day_of_week = st.selectbox(
            "Day Of Week",
            dataset["Day_Of_Week"]
            .dropna()
            .unique()
        )


        quarter = st.selectbox(
            "Quarter",
            dataset["Quarter"]
            .dropna()
            .unique()
        )


        customer_age = st.number_input(
            "Customer Age",
            min_value=1,
            max_value=100,
            value=30
        )


        customer_gender = st.selectbox(
            "Customer Gender",
            dataset["Customer_Gender"]
            .dropna()
            .unique()
        )


        customer_segment = st.selectbox(
            "Customer Segment",
            dataset["Customer_Segment"]
            .dropna()
            .unique()
        )


        product_category = st.selectbox(
            "Product Category",
            dataset["Product_Category"]
            .dropna()
            .unique()
        )


        product_subcategory = st.selectbox(
            "Product Subcategory",
            dataset["Product_Subcategory"]
            .dropna()
            .unique()
        )


        brand = st.selectbox(
            "Brand",
            dataset["Brand"]
            .dropna()
            .unique()
        )


    # ======================================
    # COLUMN 2
    # ======================================

    with col2:

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
            dataset["Coupon_Used"]
            .dropna()
            .unique()
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
            dataset["Payment_Method"]
            .dropna()
            .unique()
        )


        device_type = st.selectbox(
            "Device Type",
            dataset["Device_Type"]
            .dropna()
            .unique()
        )


        traffic_source = st.selectbox(
            "Traffic Source",
            dataset["Traffic_Source"]
            .dropna()
            .unique()
        )


    # ======================================
    # COLUMN 3
    # ======================================

    with col3:

        membership_status = st.selectbox(
            "Membership Status",
            dataset["Membership_Status"]
            .dropna()
            .unique()
        )


        shipping_method = st.selectbox(
            "Shipping Method",
            dataset["Shipping_Method"]
            .dropna()
            .unique()
        )


        warehouse_region = st.selectbox(
            "Warehouse Region",
            dataset["Warehouse_Region"]
            .dropna()
            .unique()
        )


        delivery_days = st.number_input(
            "Delivery Days",
            min_value=0,
            value=5
        )


        order_status = st.selectbox(
            "Order Status",
            dataset["Order_Status"]
            .dropna()
            .unique()
        )


        returned = st.selectbox(
            "Returned",
            dataset["Returned"]
            .dropna()
            .unique()
        )


        review_rating = st.number_input(
            "Review Rating",
            min_value=0.0,
            max_value=5.0,
            value=4.0
        )


        customer_lifetime_value = st.number_input(
            "Customer Lifetime Value",
            min_value=0.0,
            value=1000.0
        )


        season = st.selectbox(
            "Season",
            dataset["Season"]
            .dropna()
            .unique()
        )


        holiday_season = st.selectbox(
            "Holiday Season",
            dataset["Holiday_Season"]
            .dropna()
            .unique()
        )


        high_value_order = st.selectbox(
            "High Value Order",
            dataset["High_Value_Order"]
            .dropna()
            .unique()
        )


    # ======================================
    # PREDICT BUTTON
    # ======================================

    predict_button = st.form_submit_button(
        "🔮 Predict Profit"
    )


# ==========================================
# PREDICTION
# ==========================================

if predict_button:

    # ======================================
    # CREATE INPUT DATA
    # ======================================

    input_data = pd.DataFrame({

        "Year": [year],

        "Month": [month],

        "Day": [day],

        "Day_Of_Week": [day_of_week],

        "Quarter": [quarter],

        "Customer_Age": [customer_age],

        "Customer_Gender": [customer_gender],

        "Customer_Segment": [customer_segment],

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

        "Device_Type": [device_type],

        "Traffic_Source": [traffic_source],

        "Membership_Status": [membership_status],

        "Shipping_Method": [shipping_method],

        "Warehouse_Region": [warehouse_region],

        "Delivery_Days": [delivery_days],

        "Order_Status": [order_status],

        "Returned": [returned],

        "Review_Rating": [review_rating],

        "Customer_Lifetime_Value": [customer_lifetime_value],

        "Season": [season],

        "Holiday_Season": [holiday_season],

        "High_Value_Order": [high_value_order]

    })


    # ======================================
    # PREDICTION
    # ======================================

    try:

        # Transform input using saved preprocessor
        transformed_data = preprocessor.transform(
            input_data
        )


        # Predict profit
        prediction = model.predict(
            transformed_data
        )


        # Display result
        st.success(
            f"💰 Predicted Profit: ₹{prediction[0]:,.2f}"
        )


    except Exception as e:
        # ==========================================
# MODEL PERFORMANCE
# ==========================================

st.divider()

st.header("📊 Model Performance Comparison")

model_results = pd.DataFrame({
    "Model": [
        "Clean Decision Tree",
        "Random Forest",
        "Tuned Random Forest"
    ],
    "MAE": [
        28.368530,
        25.456367,
        26.037192
    ],
    "MSE": [
        4487.709443,
        3560.009619,
        3437.715239
    ],
    "RMSE": [
        66.990368,
        59.665816,
        58.632033
    ],
    "R² Score": [
        0.600543,
        0.683118,
        0.694004
    ]
})

# Display comparison table
st.dataframe(
    model_results,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# BEST MODEL
# ==========================================

st.subheader("🏆 Best Model")

st.success(
    "Tuned Random Forest is the best performing model "
    "with an R² Score of 69.40% and RMSE of 58.63."
)


# ==========================================
# R² SCORE CHART
# ==========================================

st.subheader("📈 R² Score Comparison")

r2_chart = model_results.set_index("Model")["R² Score"]

st.bar_chart(r2_chart)

        st.error(
            f"Prediction Error: {e}"
        )
