# insights.py
import pandas as pd
import joblib
import os
import numpy as np

CLEANED_FILE = "cleaned_sales.csv"
MODEL_FILE = os.path.join("models", "sales_forecast_model.pkl")

def load_cleaned():
    if not os.path.exists(CLEANED_FILE):
        raise FileNotFoundError(f"{CLEANED_FILE} missing. Run data_prep.py")
    df = pd.read_csv(CLEANED_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    # ensure these names exist
    if "StoreName" not in df.columns:
        df["StoreName"] = df["StoreID"].apply(lambda x: f"Store_{x}")
    if "ProductName" not in df.columns:
        df["ProductName"] = df["ProductID"].apply(lambda x: f"Product_{x}")
    return df

def top_bottom_products(df, store_id=None, top_n=10):
    d = df.copy()
    if store_id:
        d = d[d["StoreID"] == str(store_id)]
    prod_sales = d.groupby(["ProductID"])["TotalAmount"].sum().sort_values(ascending=False)
    top = prod_sales.head(top_n)
    bottom = prod_sales.tail(top_n)
    return top, bottom

def store_monthly_series(df, store_id=None):
    d = df.copy()
    if store_id:
        d = d[d["StoreID"] == str(store_id)]
    monthly = d.groupby(pd.Grouper(key="Date", freq="M"))["TotalAmount"].sum()
    monthly = monthly.asfreq("M").fillna(0)
    return monthly

def load_model(path=MODEL_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError("Forecast model not found. Run analysis_model.py to train model.")
    return joblib.load(path)

def forecast_company(model, steps=6):
    fc = model.get_forecast(steps=steps)
    ci = fc.conf_int()
    res = pd.DataFrame({
        "forecast": fc.predicted_mean,
        "lower": ci.iloc[:,0],
        "upper": ci.iloc[:,1]
    })
    # index to timestamp (monthly)
    try:
        res.index = res.index.to_period("M").to_timestamp()
    except Exception:
        pass
    return res

def inventory_plan_per_product(df, forecast_df, safety_stock_pct=0.2, months=1):
    """
    Simple heuristic:
    - Use recent product revenue share to split company forecast to product level
    - Convert revenue -> qty using avg price per product
    """
    recent_cutoff = df["Date"].max() - pd.DateOffset(months=3)
    recent = df[df["Date"] >= recent_cutoff]
    prod_rev = recent.groupby("ProductID")["TotalAmount"].sum()
    prod_price = df.groupby("ProductID")["Price"].mean()
    total_rev = prod_rev.sum() if len(prod_rev)>0 else 1.0
    rows = []
    for prod, rev in prod_rev.items():
        share = rev / total_rev
        avg_price = prod_price.get(prod, np.nan)
        for ts, f in forecast_df["forecast"].items():
            est_rev = f * share
            est_qty = 0 if pd.isna(avg_price) or avg_price == 0 else int(round(est_rev / avg_price))
            qty_with_safety = int(np.ceil(est_qty * (1 + safety_stock_pct)))
            rows.append({"ProductID": prod, "Month": pd.to_datetime(ts).strftime("%Y-%m"), "Est_Qty": est_qty, "Qty_with_Safety": qty_with_safety})
    plan = pd.DataFrame(rows)
    return plan

def suggest_for_store(df, store_id):
    df_store = df[df["StoreID"] == str(store_id)]
    suggestions = []
    if df_store.empty:
        return ["No data for this store."]
    monthly = store_monthly_series(df, store_id)
    # simple decline detection
    if len(monthly) >= 2 and monthly.iloc[-1] < monthly.iloc[-2]:
        suggestions.append("Recent month sales dropped vs previous month → consider short-term promotions/discounts.")
    # low avg sales
    avg = monthly.mean() if len(monthly)>0 else 0
    if avg < monthly.max() * 0.4:
        suggestions.append("Sales are inconsistent; improve merchandising/visibility for top items.")
    # top product suggestion
    top = df_store.groupby("ProductName")["TotalAmount"].sum().sort_values(ascending=False)
    if not top.empty:
        best = top.index[0]
        suggestions.append(f"Increase stock & display for top product: {best}. Consider bundles with complementary items.")
    # worst product
    worst = top.index[-1] if len(top)>1 else None
    if worst:
        suggestions.append(f"Consider discount or delist for low-performing product: {worst}.")
    return suggestions

# Demo runner
def main_demo():
    df = load_cleaned()
    print("Top company products:")
    top, bottom = top_bottom_products(df, top_n=5)
    print(top.head())
    # forecast if model exists
    if os.path.exists(MODEL_FILE):
        model = load_model()
        fc = forecast_company(model, steps=6)
        print("\nForecast (next 6 months):")
        print(fc)
        plan = inventory_plan_per_product(df, fc, safety_stock_pct=0.2)
        plan.to_csv("inventory_plan_detailed.csv", index=False)
        print("Saved inventory_plan_detailed.csv")
    else:
        print("Model not found, skip forecasting demo.")
    # sample suggestion for one store
    sample_store = df["StoreID"].iloc[0]
    print("\nSuggestions for store:", sample_store)
    print("\n".join(suggest_for_store(df, sample_store)))

if __name__ == "__main__":
    main_demo()
