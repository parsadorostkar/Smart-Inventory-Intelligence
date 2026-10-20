import pandas as pd 
from load_data import df_combined

print("Combined dataset shape:", df_combined.shape)
print(df_combined.head())

is_cancelled = df_combined["InvoiceNo"].astype(str).str.startswith("C")
print(is_cancelled.value_counts())

is_positive_quantity = df_combined["Quantity"] > 0
print(is_positive_quantity.value_counts())

valid_sales_mask = (
    ~is_cancelled
    & is_positive_quantity
    & (df_combined["UnitPrice"] > 0)
)

df_sales = df_combined.loc[valid_sales_mask].copy()

print("Valid sales shape:", df_sales.shape)
print("Unique products:", df_sales["StockCode"].nunique())
print("Date range:", df_sales["InvoiceDate"].min(), "to", df_sales["InvoiceDate"].max())
print("Remaining non-positive prices:", (df_sales["UnitPrice"] <= 0).sum())


df_sales["Date"] = df_sales["InvoiceDate"].dt.floor("D")

daily_demand = (
    df_sales
    .groupby(["Date", "StockCode"])["Quantity"]
    .sum()
    .reset_index(name="Demand")
)

print("Daily demand shape:", daily_demand.shape)
print(daily_demand.head(10))

product_days = daily_demand.groupby("StockCode")["Date"].agg(
    first_sale="min",
    last_sale="max",
    active_days="nunique"
)

print("Number of products:", len(product_days))
print("\nProducts with the fewest observed sales days:")
print(product_days.sort_values("active_days").head(10))

print("\nObserved sales days per product:")
print(product_days["active_days"].describe())

daily_demand = daily_demand.sort_values(["StockCode", "Date"])

full_daily = (
    daily_demand
    .set_index("Date")
    .groupby("StockCode")["Demand"]
    .resample("D")
    .sum()
    .reset_index()
)

print("Rows before:", len(daily_demand))
print("Rows after:", len(full_daily))
print("Zero-demand days:", (full_daily["Demand"] == 0).sum())
print(full_daily.head(10))

print("Missing values:")
print(full_daily.isna().sum())

print("\nNegative demand rows:", (full_daily["Demand"] < 0).sum())
print("Zero demand percentage:", round((full_daily["Demand"] == 0).mean() * 100, 2), "%")
print("Unique products:", full_daily["StockCode"].nunique())
print("Date range:", full_daily["Date"].min(), "to", full_daily["Date"].max())
print("Duplicate product-date pairs:", full_daily.duplicated(["StockCode", "Date"]).sum())

full_daily.to_csv("daily_demand.csv", index=False)

