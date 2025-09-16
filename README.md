# 🛍️ Retail Sales Analytics & Forecasting Dashboard  

This project focuses on **Retail Sales Analytics & Forecasting** for Zudio retail stores.  
It provides detailed **sales insights, product-wise analysis, inventory planning, and forecasting models** to help businesses make **data-driven decisions**.  

---

## 🚀 Features  
- 📊 **Sales Performance Dashboard** – Analyze sales by store, product, and month  
- 🔮 **Forecasting Models** – Predict future sales using Machine Learning & Time Series techniques  
- 📦 **Inventory Planning** – Optimize stock levels with demand forecasts  
- 📈 **Visual Insights** – Graphs, charts, and KPIs for better decision-making  
- ⚡ **Automated Data Pipeline** – Cleans and prepares raw CSV data  
- 🛠️ **Interactive Dashboard** – Run via `app.py` for real-time analysis  

---

## 🛠️ Tech Stack  
- **Programming Language**: Python  
- **Libraries & Tools**:  
  - `Pandas`, `NumPy` → Data Cleaning & Processing  
  - `Matplotlib`, `Seaborn`, `Plotly` → Visualizations  
  - `Scikit-learn` / `Statsmodels` → Forecasting & ML Models  
  - `Streamlit` / `Flask` → Interactive Dashboard  
- **Data**: CSV files (`zudio_sales.csv`, `zudio_products.csv`, `zudio_stores.csv`)  

---

## 📂 Project Structure  

Retail-Sales-Analytics/
│-- app.py # Main dashboard application
│-- analysis_model.py # ML/Forecasting model scripts
│-- data_prep.py # Data cleaning & preprocessing
│-- insights.py # Sales insights & visualization functions
│-- models/ # Saved trained models
│-- cleaned_sales.csv # Preprocessed sales dataset
│-- inventory_plan_detailed.csv # Suggested inventory planning
│-- zudio_sales.csv # Raw sales dataset
│-- zudio_products.csv # Product details
│-- zudio_stores.csv # Store information
│-- monthly_sales.png # Sample visualization
│-- requirements.txt # Dependencies
│-- README.md # Documentation


---

## ⚙️ Installation & Setup  

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/Amankar003/Zudio-Sales-Analysis.git
   cd Zudio-Sales-Analysis

python -m venv venv
source venv/bin/activate    # For Linux/Mac
venv\Scripts\activate       # For Windows

pip install -r requirements.txt

streamlit run app.py

## 📊  Workflow

Load raw sales, product, and store data

Clean & preprocess datasets (data_prep.py)

Generate sales insights & KPIs (insights.py)

Train forecasting models (analysis_model.py)

Save results & visualizations

Run interactive dashboard (app.py)
