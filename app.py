# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from insights import load_cleaned, top_bottom_products, store_monthly_series, load_model, forecast_company, suggest_for_store, inventory_plan_per_product
import os

st.set_page_config(page_title="Zudio Sales Insights", layout="wide")
st.title("🛍️ Zudio — Sales Analytics & Recommendations")

# Load cleaned data (must run data_prep.py first)
try:
    df = load_cleaned()
except FileNotFoundError:
    st.error("cleaned_sales.csv not found. Run data_prep.py first.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
view = st.sidebar.radio("View level", ["All Data", "State Wise", "City Wise", "Store Wise"])

state_list = sorted(df["State"].dropna().unique().tolist())
state_selected = None
city_selected = None
store_selected = None

if view == "State Wise":
    state_selected = st.sidebar.selectbox("Select state", ["All"] + state_list)
elif view == "City Wise":
    state_selected = st.sidebar.selectbox("Select state", ["All"] + state_list)
    if state_selected and state_selected != "All":
        city_list = sorted(df[df["State"] == state_selected]["City"].dropna().unique().tolist())
    else:
        city_list = sorted(df["City"].dropna().unique().tolist())
    city_selected = st.sidebar.selectbox("Select city", ["All"] + city_list)
elif view == "Store Wise":
    state_selected = st.sidebar.selectbox("Select state", ["All"] + state_list)
    if state_selected and state_selected != "All":
        city_list = sorted(df[df["State"] == state_selected]["City"].dropna().unique().tolist())
    else:
        city_list = sorted(df["City"].dropna().unique().tolist())
    city_selected = st.sidebar.selectbox("Select city", ["All"] + city_list)
    # store list filtered by state+city
    store_list = df.copy()
    if state_selected and state_selected != "All":
        store_list = store_list[store_list["State"] == state_selected]
    if city_selected and city_selected != "All":
        store_list = store_list[store_list["City"] == city_selected]
    store_names = sorted(store_list["StoreName"].dropna().unique().tolist())
    store_selected = st.sidebar.selectbox("Select store", ["All"] + store_names)

# Apply filters to dataframe
filtered = df.copy()
if view == "State Wise" and state_selected and state_selected != "All":
    filtered = filtered[filtered["State"] == state_selected]
elif view == "City Wise":
    if state_selected and state_selected != "All":
        filtered = filtered[filtered["State"] == state_selected]
    if city_selected and city_selected != "All":
        filtered = filtered[filtered["City"] == city_selected]
elif view == "Store Wise":
    if state_selected and state_selected != "All":
        filtered = filtered[filtered["State"] == state_selected]
    if city_selected and city_selected != "All":
        filtered = filtered[filtered["City"] == city_selected]
    if store_selected and store_selected != "All":
        filtered = filtered[filtered["StoreName"] == store_selected]

st.markdown(f"### Showing: **{view}** " + (f" - {state_selected}" if state_selected else ""))
st.dataframe(filtered.head(25))

# View Insights button
if st.button("🔍 View Insights"):
    st.subheader("Overview Metrics")
    total_rev = filtered["TotalAmount"].sum()
    total_qty = filtered["Quantity"].sum()
    st.metric("Total Revenue", f"₹{total_rev:,.0f}")
    st.metric("Total Quantity Sold", f"{int(total_qty):,}")

    # monthly trend
    monthly = filtered.groupby(filtered["Date"].dt.to_period("M"))["TotalAmount"].sum().reset_index()
    monthly["Date"] = monthly["Date"].astype(str)
    if monthly.shape[0] > 0:
        fig = px.line(monthly, x="Date", y="TotalAmount", title="Monthly Sales Trend")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No monthly data to plot.")

    # top 10 products
    top_products = filtered.groupby("ProductName")["TotalAmount"].sum().reset_index().sort_values("TotalAmount", ascending=False).head(10)
    if not top_products.empty:
        fig2 = px.bar(top_products, x="ProductName", y="TotalAmount", title="Top 10 Products")
        st.plotly_chart(fig2, use_container_width=True)
        st.table(top_products)

    # If store selected, show suggestions
    if view == "Store Wise" and store_selected and store_selected != "All":
        # map store_selected back to StoreID
        store_row = df[df["StoreName"] == store_selected].iloc[0]
        store_id = store_row["StoreID"]
        st.subheader(f"Suggestions for {store_selected}")
        suggestions = suggest_for_store(df, store_id)
        for s in suggestions:
            st.write("- ", s)

    # Company-level forecast (if model exists)
    if os.path.exists("models/sales_forecast_model.pkl"):
        model = load_model()
        months = st.slider("Forecast months", 3, 12, 6)
        fc = forecast_company(model, steps=months)
        st.subheader("Company-level Forecast")
        st.line_chart(fc["forecast"])
        st.write(fc)
        # inventory plan example
        safety = st.slider("Safety stock %", 0, 50, 20) / 100.0
        if st.button("Generate Inventory Plan"):
            plan = inventory_plan_per_product(filtered, fc, safety_stock_pct=safety)
            st.write(plan.head(50))
            csv = plan.to_csv(index=False).encode("utf-8")
            st.download_button("Download Inventory Plan", data=csv, file_name="inventory_plan.csv", mime="text/csv")
    else:
        st.info("Forecast model not found. Run analysis_model.py to train SARIMA model.")
