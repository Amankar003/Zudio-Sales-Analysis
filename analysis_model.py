# analysis_model.py
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX

CLEANED_FILE = "cleaned_sales.csv"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "sales_forecast_model.pkl")

def basic_eda(df):
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nNull counts:\n", df.isnull().sum())
    # Top stores
    top_stores = df.groupby("StoreID")["TotalAmount"].sum().sort_values(ascending=False).head(10)
    print("\nTop 10 stores by revenue:\n", top_stores)
    # Top products
    top_products = df.groupby("ProductID")["TotalAmount"].sum().sort_values(ascending=False).head(10)
    print("\nTop 10 products by revenue:\n", top_products)
    # Category totals if present
    if "Category" in df.columns:
        print("\nRevenue by Category:")
        print(df.groupby("Category")["TotalAmount"].sum().sort_values(ascending=False))

def monthly_series(df):
    df["Date"] = pd.to_datetime(df["Date"])
    monthly = df.groupby(pd.Grouper(key="Date", freq="M"))["TotalAmount"].sum()
    monthly = monthly.asfreq("M").fillna(0)
    return monthly

def train_sarima(monthly, save_path=MODEL_FILE):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # simple train/test
    if len(monthly) < 24:
        print("Warning: less than 24 months of data — SARIMA may not be ideal.")
    train = monthly[:-3] if len(monthly) > 3 else monthly
    print("Training SARIMA on", len(train), "points")
    model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12),
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    joblib.dump(res, save_path)
    print("Model saved to:", save_path)
    return res

def plot_monthly(monthly, out="monthly_sales.png"):
    plt.figure(figsize=(10,5))
    monthly.plot(marker="o")
    plt.title("Monthly Total Sales")
    plt.ylabel("TotalAmount")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print("Saved plot:", out)

def main():
    if not os.path.exists(CLEANED_FILE):
        raise FileNotFoundError(f"{CLEANED_FILE} not found. Run data_prep.py first.")
    df = pd.read_csv(CLEANED_FILE)
    basic_eda(df)
    monthly = monthly_series(df)
    plot_monthly(monthly)
    model = train_sarima(monthly)
    print("AIC:", getattr(model, "aic", None))

if __name__ == "__main__":
    main()
