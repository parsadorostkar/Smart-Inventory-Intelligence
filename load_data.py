
from pathlib import Path
import pandas as pd

file_path = (
    Path.home()
    / "Downloads"
    / "online+retail+ii"
    / "Online_Retail_II.xlsx"
)

# Load both time periods
df_09_10 = pd.read_excel(
    file_path,
    sheet_name="Year 2009-2010"
)

df_10_11 = pd.read_excel(
    file_path,
    sheet_name="Year 2010-2011"
)

# Standardize column names
df_09_10 = df_09_10.rename(
    columns={
        "Invoice": "InvoiceNo",
        "Price": "UnitPrice",
        "Customer ID": "CustomerID",
    }
)

df_10_11 = df_10_11.rename(
    columns={
        "Invoice": "InvoiceNo",
        "Price": "UnitPrice",
        "Customer ID": "CustomerID",
    }
)

# Combine without removing any records
df = pd.concat(
    [df_09_10, df_10_11],
    ignore_index=True
)

print("Combined shape:", df.shape)
print("Date range:")
print(df["InvoiceDate"].min())
print(df["InvoiceDate"].max())

print("\nRows per source sheet:")
print({
    "2009-2010": len(df_09_10),
    "2010-2011": len(df_10_11),
})

print("\nExact duplicate rows:", df.duplicated().sum())

print("\nRows in overlapping date range:")
overlap = df[
    (df["InvoiceDate"] >= "2010-12-01")
    & (df["InvoiceDate"] <= "2010-12-09 23:59:59")
]
print(len(overlap))

print("\nColumn names:")
print(df.columns.tolist())


# Track the source of each record
df_09_10 = df_09_10.copy()
df_10_11 = df_10_11.copy()

# 1. Duplicates within each sheet
print("Duplicates within 2009-2010:",
      df_09_10.duplicated().sum())

print("Duplicates within 2010-2011:",
      df_10_11.duplicated().sum())

# 2. Isolate the overlapping period
start = pd.Timestamp("2010-12-01")
end = pd.Timestamp("2010-12-10")

overlap_09_10 = df_09_10[
    (df_09_10["InvoiceDate"] >= start)
    & (df_09_10["InvoiceDate"] < end)
].copy()

overlap_10_11 = df_10_11[
    (df_10_11["InvoiceDate"] >= start)
    & (df_10_11["InvoiceDate"] < end)
].copy()

print("\nOverlap rows in first sheet:", len(overlap_09_10))
print("Overlap rows in second sheet:", len(overlap_10_11))

# 3. Compare exact rows across sheets
hash_09_10 = pd.util.hash_pandas_object(
    overlap_09_10, index=False
)
hash_10_11 = pd.util.hash_pandas_object(
    overlap_10_11, index=False
)

counts_09_10 = hash_09_10.value_counts()
counts_10_11 = hash_10_11.value_counts()

common_hashes = counts_09_10.index.intersection(
    counts_10_11.index
)

shared_occurrences = sum(
    min(counts_09_10[h], counts_10_11[h])
    for h in common_hashes
)

print("\nDistinct matching row hashes:", len(common_hashes))
print("Matching row occurrences across sheets:", shared_occurrences)


# Keep the first sheet in full.
# From the second sheet, keep only dates after Dec 9, 2010.
cutoff = pd.Timestamp("2010-12-10")

df_combined = pd.concat(
    [
        df_09_10,
        df_10_11[df_10_11["InvoiceDate"] >= cutoff],
    ],
    ignore_index=True,
)

print("Shape after removing temporal overlap:", df_combined.shape)
print("Date range:", df_combined["InvoiceDate"].min(),
      "to", df_combined["InvoiceDate"].max())

print("\nExact duplicate rows remaining:",
      df_combined.duplicated().sum())

# Audit cancellations, returns, and unusual prices
invoice_text = df_combined["InvoiceNo"].astype(str)

print("\nCancelled invoice rows:",
      invoice_text.str.startswith("C").sum())

print("Negative quantity rows:",
      (df_combined["Quantity"] < 0).sum())

print("Zero quantity rows:",
      (df_combined["Quantity"] == 0).sum())

print("Zero or negative unit prices:",
      (df_combined["UnitPrice"] <= 0).sum())

print("\nQuantity percentiles:")
print(df_combined["Quantity"].quantile(
    [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
))

print("\nUnit price percentiles:")
print(df_combined["UnitPrice"].quantile(
    [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
))


# 1. Relationship between cancellations and negative quantities
is_cancelled = df_combined["InvoiceNo"].astype(str).str.startswith("C")
is_negative_qty = df_combined["Quantity"] < 0

print("Cancellation / quantity cross-check:")
print(pd.crosstab(
    is_cancelled,
    is_negative_qty,
    rownames=["Cancelled invoice"],
    colnames=["Negative quantity"],
))

# 2. Inspect cancelled invoice examples
print("\nCancelled invoice examples:")
print(
    df_combined.loc[
        is_cancelled,
        ["InvoiceNo", "StockCode", "Description",
         "Quantity", "InvoiceDate", "UnitPrice"]
    ].head(15).to_string(index=False)
)

# 3. Inspect remaining exact duplicates
duplicate_rows = df_combined[
    df_combined.duplicated(keep=False)
]

print("\nRows belonging to duplicate groups:",
      len(duplicate_rows))

print("\nExample duplicate groups:")
print(
    duplicate_rows.sort_values(
        ["InvoiceNo", "StockCode", "InvoiceDate"]
    ).head(20).to_string(index=False)
)

# 4. Inspect zero/negative prices
unusual_price = df_combined["UnitPrice"] <= 0

print("\nRows with zero or negative prices:",
      unusual_price.sum())

print("\nDescriptions among zero/negative-price rows:")
print(
    df_combined.loc[unusual_price, "Description"]
    .value_counts(dropna=False)
    .head(20)
)

print("\nExamples of zero/negative-price rows:")
print(
    df_combined.loc[
        unusual_price,
        ["InvoiceNo", "StockCode", "Description",
         "Quantity", "UnitPrice"]
    ].head(20).to_string(index=False)
)

# 5. Inspect the largest positive quantities
print("\nLargest positive quantity rows:")
print(
    df_combined.nlargest(10, "Quantity")[
        ["InvoiceNo", "StockCode", "Description",
         "Quantity", "InvoiceDate", "UnitPrice"]
    ].to_string(index=False)
)
