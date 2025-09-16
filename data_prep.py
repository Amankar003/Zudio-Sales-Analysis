# data_prep.py
import pandas as pd
import os

SALES_FILE = "zudio_sales.csv"
STORES_FILE = "zudio_stores.csv"
PRODUCTS_FILE = "zudio_products.csv"
CLEANED_FILE = "cleaned_sales.csv"

def safe_read(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Please place it in project root.")
    return pd.read_csv(path)

def normalize_columns(df):
    df.columns = df.columns.str.strip()
    return df

def ensure_cols(df, col_defaults: dict):
    # Add missing columns with default values
    for c, v in col_defaults.items():
        if c not in df.columns:
            df[c] = v
    return df

def load_and_clean():
    sales = safe_read(SALES_FILE)
    stores = safe_read(STORES_FILE)
    products = safe_read(PRODUCTS_FILE)

    sales = normalize_columns(sales)
    stores = normalize_columns(stores)
    products = normalize_columns(products)

    # Standardize expected column names in sales
    # Common possibilities handled by fallback mapping
    # Accept: Price or BasePrice or UnitPrice
    price_col = None
    for c in ["Price", "UnitPrice", "BasePrice"]:
        if c in sales.columns:
            price_col = c
            break
    if price_col is None:
        # maybe price only in product file
        price_col = None

    # Ensure key columns exist with safe defaults
    sales = ensure_cols(sales, {
        "SaleID": pd.NA,
        "StoreID": pd.NA,
        "ProductID": pd.NA,
        "Date": pd.NA,
        "Quantity": 0,
        "Discount": 0.0
    })
    stores = ensure_cols(stores, {
        "StoreID": pd.NA,
        "City": pd.NA,
        "State": pd.NA,
        "StoreName": pd.NA
    })
    products = ensure_cols(products, {
        "ProductID": pd.NA,
        "ProductName": pd.NA,
        "Category": pd.NA,
        "SubCategory": pd.NA,
        "Brand": pd.NA,
        "BasePrice": 0.0
    })

    # If sales has no Price column but products has BasePrice, merge price from products for calculation
    if "Price" not in sales.columns:
        if "BasePrice" in products.columns:
            # ensure ProductID types match
            sales["ProductID"] = sales["ProductID"].astype(str)
            products["ProductID"] = products["ProductID"].astype(str)
            sales = sales.merge(products[["ProductID", "BasePrice"]], on="ProductID", how="left")
            sales = sales.rename(columns={"BasePrice": "Price"})
        else:
            sales["Price"] = 0.0
    else:
        sales = sales.rename(columns={price_col: "Price"}) if price_col and price_col != "Price" else sales

    # Compute TotalAmount if not present
    if "TotalAmount" not in sales.columns and "Price" in sales.columns:
        # coerce numeric
        sales["Quantity"] = pd.to_numeric(sales["Quantity"], errors="coerce").fillna(0).astype(int)
        sales["Price"] = pd.to_numeric(sales["Price"], errors="coerce").fillna(0.0)
        sales["Discount"] = pd.to_numeric(sales.get("Discount", 0.0), errors="coerce").fillna(0.0)
        sales["TotalAmount"] = sales["Quantity"] * sales["Price"] * (1 - sales["Discount"])
    else:
        sales["TotalAmount"] = pd.to_numeric(sales["TotalAmount"], errors="coerce").fillna(0.0)

    # Parse Date
    sales["Date"] = pd.to_datetime(sales["Date"], errors="coerce")

    # Drop rows without critical info
    sales = sales.dropna(subset=["Date", "StoreID", "ProductID"])

    # Convert IDs to str
    sales["StoreID"] = sales["StoreID"].astype(str)
    sales["ProductID"] = sales["ProductID"].astype(str)
    stores["StoreID"] = stores["StoreID"].astype(str)
    products["ProductID"] = products["ProductID"].astype(str)

    # Create fallback names if not present
    if "StoreName" not in stores.columns or stores["StoreName"].isna().all():
        stores["StoreName"] = stores["StoreID"].apply(lambda x: f"Store_{x}")
    if "ProductName" not in products.columns or products["ProductName"].isna().all():
        products["ProductName"] = products["ProductID"].apply(lambda x: f"Product_{x}")

    # Merge everything
    merged = sales.merge(stores, on="StoreID", how="left", suffixes=("", "_store"))
    merged = merged.merge(products, on="ProductID", how="left", suffixes=("", "_prod"))

    # Fill missing city/state with unknown
    merged["City"] = merged.get("City", pd.Series()).fillna("Unknown")
    merged["State"] = merged.get("State", pd.Series()).fillna("Unknown")

    # Reset index and save
    merged = merged.drop_duplicates().reset_index(drop=True)
    merged.to_csv(CLEANED_FILE, index=False)
    print(f"Cleaned data saved to {CLEANED_FILE} (rows: {len(merged)})")
    return merged

if __name__ == "__main__":
    load_and_clean()
