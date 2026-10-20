
import pandas as pd
from pathlib import Path

file_path = (
    Path.home()
    / "Downloads"
    / "online+retail+ii"
    / "Online_Retail_II.xlsx"
)

excel = pd.ExcelFile(file_path)

print("Available sheets:")
print(excel.sheet_names)

for sheet in excel.sheet_names:
    data = pd.read_excel(file_path, sheet_name=sheet)
    print(f"\nSheet: {sheet}")
    print("Shape:", data.shape)
    print("Start:", data["InvoiceDate"].min())
    print("End:", data["InvoiceDate"].max())
