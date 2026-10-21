from pathlib import Path

import pandas as pd


FILE_PATH = (
    Path.home()
    / "Downloads"
    / "online+retail+ii"
    / "Online_Retail_II.xlsx"
)


def load_data():
    """Load, standardize, and combine the two source worksheets."""

    df_09_10 = pd.read_excel(
        FILE_PATH,
        sheet_name="Year 2009-2010",
    )

    df_10_11 = pd.read_excel(
        FILE_PATH,
        sheet_name="Year 2010-2011",
    )

    rename_columns = {
        "Invoice": "InvoiceNo",
        "Price": "UnitPrice",
        "Customer ID": "CustomerID",
    }

    df_09_10 = df_09_10.rename(columns=rename_columns)
    df_10_11 = df_10_11.rename(columns=rename_columns)

    cutoff = pd.Timestamp("2010-12-10")

    df_combined = pd.concat(
        [
            df_09_10,
            df_10_11.loc[df_10_11["InvoiceDate"] >= cutoff],
        ],
        ignore_index=True,
    )

    return df_combined


def inspect_data(df_combined):
    """Print diagnostic reports about the combined dataset."""

    print("Combined shape:", df_combined.shape)
    print("Date range:")
    print(df_combined["InvoiceDate"].min())
    print(df_combined["InvoiceDate"].max())

    print("\nExact duplicate rows:", df_combined.duplicated().sum())

    invoice_text = df_combined["InvoiceNo"].astype(str)
    is_cancelled = invoice_text.str.startswith("C")
    is_negative_qty = df_combined["Quantity"] < 0
    unusual_price = df_combined["UnitPrice"] <= 0

    print("\nCancelled invoice rows:", is_cancelled.sum())
    print("Negative quantity rows:", is_negative_qty.sum())
    print("Zero quantity rows:", (df_combined["Quantity"] == 0).sum())
    print("Zero or negative unit prices:", unusual_price.sum())

    print("\nQuantity percentiles:")
    print(df_combined["Quantity"].quantile(
        [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
    ))

    print("\nUnit price percentiles:")
    print(df_combined["UnitPrice"].quantile(
        [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
    ))


if __name__ == "__main__":
    data = load_data()
    inspect_data(data)
