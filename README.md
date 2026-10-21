\# Smart Inventory Intelligence



A Python-based inventory analytics project that evaluates demand forecasting and inventory replenishment policies using historical retail transaction data.



The project focuses on intermittent demand, inventory simulation, and rolling-origin backtesting to compare inventory policies under different service-level targets.



\## Key Features



\- Historical retail transaction data preparation

\- Daily demand aggregation by product

\- Demand forecasting and inventory policy evaluation

\- Inventory simulation with lead times, replenishment, and backorders

\- Rolling-origin backtesting across multiple evaluation windows

\- Comparison of service levels, unmet demand, and average on-hand inventory



\## Project Structure



| File | Description |

|---|---|

| `analyze\_demand\_cleaned.py` | Main analysis, inventory simulation, and backtesting script |

| `load\_data.py` | Loads and combines the source Excel worksheets |

| `prepare\_data.py` | Filters sales records and generates daily demand data |

| `inspect\_data.py` | Optional utility for inspecting the source dataset |

| `inventory\_engine.py` | Standalone inventory calculation functions; not currently integrated into the main workflow |

| `requirements.txt` | Python dependencies |

| `rolling\_backtest\_by\_fold.csv` | Backtest results by evaluation window |

| `rolling\_backtest\_summary.csv` | Aggregated backtest results |

| `.gitignore` | Excludes generated data and temporary files |



\## Requirements



\- Python 3.10 or newer recommended

\- pandas

\- NumPy

\- scikit-learn

\- openpyxl



Install the dependencies:



```bash

pip install -r requirements.txt

```



\## Dataset



This project uses the Online Retail II transaction dataset.



The source Excel file is expected at:



```text

\~/Downloads/online+retail+ii/Online\_Retail\_II.xlsx

```



The workbook must contain the worksheets `Year 2009-2010` and `Year 2010-2011`.



The dataset is not included in this repository. Obtain the dataset separately and place it at the expected path, or update the path in `load\_data.py`.



\## Workflow



\### 1. Inspect the source dataset (optional)



```bash

python inspect\_data.py

```



\### 2. Prepare daily demand data



```bash

python prepare\_data.py

```



This step loads the source workbook, filters cancelled invoices, non-positive quantities, and non-positive unit prices, and aggregates valid sales quantities by product and day.



It generates:



```text

daily\_demand.csv

```



The preparation process also removes the overlapping period between the two source worksheets by retaining the first worksheet in full and keeping records from the second worksheet dated December 10, 2010, onward.



\### 3. Run the main analysis



```bash

python analyze\_demand\_cleaned.py

```



The main script reads `daily\_demand.csv` and generates model data and backtesting outputs.



Generated files include:



\- `model\_data.csv`

\- `rolling\_backtest\_by\_fold.csv`

\- `rolling\_backtest\_summary.csv`



\## Backtesting Methodology



The rolling-origin backtest evaluates inventory policies over five complete 28-day evaluation windows. An additional incomplete two-day window is skipped.



Policy parameters are estimated using historical data available before each evaluation window. Products must meet the minimum historical-data requirements to be eligible for evaluation.



The reported policies are compared using:



\- Same-day fill rate

\- Unmet demand

\- Average on-hand inventory



The evaluation windows are simulated independently rather than as one continuous inventory timeline. The initial inventory is standardized using the Q95 target to improve comparability across policies.



Missing product-day records are treated as zero recorded demand. This is a modeling assumption and may not always represent true demand when records are missing or incomplete.
