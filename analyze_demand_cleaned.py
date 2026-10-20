import pandas as pd 
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from pathlib import Path
import math

df = pd.read_csv(
    "daily_demand.csv",
    parse_dates=["Date"],
    dtype={"StockCode": "string"}
)

print("Dataset shape:", df.shape)
print("Total units sold:", df["Demand"].sum())
print("Zero-demand percentage:", round((df["Demand"] == 0).mean() * 100, 2), "%")


product_stats = df.groupby("StockCode").agg(
    total_demand=("Demand", "sum"),
    avg_daily_demand=("Demand", "mean"),
    observed_days=("Date", "count"),
    selling_days=("Demand", lambda x: (x > 0).sum())
)

product_stats["selling_day_ratio"] = (
    product_stats["selling_days"] / product_stats["observed_days"]
)

print("\nProducts with highest total demand:")
print(product_stats.sort_values("total_demand", ascending=False).head(10))

print("\nProducts with most selling days:")
print(product_stats.sort_values("selling_days", ascending=False).head(10))


daily_total = df.groupby("Date")["Demand"].sum()

print("\nTotal daily demand statistics:")
print(daily_total.describe())


print("Highest-demand dates:")
print(daily_total.sort_values(ascending=False).head(10))


print("\nHighest product-day demand:")
print(
    df.nlargest(10, "Demand")[
        ["Date", "StockCode", "Demand"]
    ].to_string(index=False)
)

positive_demand = df.loc[df["Demand"] > 0, "Demand"]

print("\nPositive-demand percentiles:")
print(positive_demand.quantile([0.50, 0.90, 0.95, 0.99, 0.999]))

monthly_demand = (
    df.groupby(pd.Grouper(key="Date", freq="MS"))["Demand"]
    .sum()
)

print("Monthly total demand:")
print(monthly_demand)

print("\nMonthly demand statistics:")
print(monthly_demand.describe())


df["DayOfWeek"] = df["Date"].dt.dayofweek

weekday_demand = (
    df.groupby("DayOfWeek")["Demand"]
    .agg(["mean", "median", "sum", "count"])
)

weekday_demand.index = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

print(weekday_demand)


# ==========================================
# 1. Feature Engineering
# ==========================================

df = df.sort_values(["StockCode", "Date"]).copy()

df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month

grouped_demand = df.groupby("StockCode")["Demand"]

# Recent demand
df["Demand_Lag_1"] = grouped_demand.shift(1)
df["Demand_Lag_7"] = grouped_demand.shift(7)

df["Demand_Rolling_Mean_7"] = (
    df.groupby("StockCode")["Demand"]
    .transform(lambda s: s.shift(1).rolling(7).mean())
)

# Longer-term demand
df["Demand_Lag_14"] = grouped_demand.shift(14)
df["Demand_Lag_28"] = grouped_demand.shift(28)

df["Demand_Rolling_Mean_28"] = (
    df.groupby("StockCode")["Demand"]
    .transform(lambda s: s.shift(1).rolling(28).mean())
)

df["Demand_Rolling_Max_28"] = (
    df.groupby("StockCode")["Demand"]
    .transform(lambda s: s.shift(1).rolling(28).max())
)

df["Positive_Demand_Rate_28"] = (
    df.groupby("StockCode")["Demand"]
    .transform(lambda s: s.shift(1).gt(0).rolling(28).mean())
)

# ==========================================
# 2. Prepare Modeling Dataset
# ==========================================

feature_columns = [
    "Demand_Lag_1",
    "Demand_Lag_7",
    "Demand_Lag_14",
    "Demand_Lag_28",
    "Demand_Rolling_Mean_7",
    "Demand_Rolling_Mean_28",
    "Demand_Rolling_Max_28",
    "Positive_Demand_Rate_28",
    "DayOfWeek",
    "Month"
]

required_features = feature_columns.copy()

model_data = df.dropna(subset=required_features).copy()

model_data.to_csv("model_data.csv", index=False)

print("\nModeling dataset:")
print("Original rows:", len(df))
print("Rows ready for modeling:", len(model_data))
print("Rows removed:", len(df) - len(model_data))
print("Missing feature values:")
print(model_data[required_features].isna().sum())

# ==========================================
# 3. Chronological Train/Test Split
# ==========================================

model_data = model_data.sort_values(
    ["Date", "StockCode"]
).copy()

dates = sorted(model_data["Date"].unique())
split_index = int(len(dates) * 0.8)
cutoff_date = dates[split_index]

train = model_data[model_data["Date"] < cutoff_date].copy()
test = model_data[model_data["Date"] >= cutoff_date].copy()

print("\nTime-based split:")
print("Cutoff date:", cutoff_date)
print("Train rows:", len(train))
print("Test rows:", len(test))
print("Train dates:", train["Date"].min(), "to", train["Date"].max())
print("Test dates:", test["Date"].min(), "to", test["Date"].max())

# ==========================================
# 4. Baseline Evaluation
# ==========================================

y_true = test["Demand"].to_numpy()

baseline_predictions = {
    "Lag_1": test["Demand_Lag_1"].to_numpy(),
    "Lag_7": test["Demand_Lag_7"].to_numpy(),
    "Rolling_Mean_7": test["Demand_Rolling_Mean_7"].to_numpy()
}

results = []

for name, y_pred_baseline in baseline_predictions.items():
    errors = y_true - y_pred_baseline

    results.append({
        "Model": name,
        "MAE": np.mean(np.abs(errors)),
        "RMSE": np.sqrt(np.mean(errors ** 2)),
        "WAPE (%)": (
            np.sum(np.abs(errors)) / np.sum(y_true) * 100
        )
    })

# ==========================================
# 5. Train HistGradientBoosting
# ==========================================

X_train = train[feature_columns]
y_train = train["Demand"]

X_test = test[feature_columns]
y_test = test["Demand"]

model = HistGradientBoostingRegressor(
    learning_rate=0.1,
    max_iter=100,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=42
)

model.fit(X_train, y_train)


def forecast_next_days(model, product_history, days=8):
    history = (
        product_history[["Date", "Demand"]]
        .sort_values("Date")
        .copy()
        .reset_index(drop=True)
    )

    if days < 1:
        raise ValueError("days must be at least 1")

    if len(history) < 28:
        raise ValueError("At least 28 days of history are required")

    if history[["Date", "Demand"]].isna().any().any():
        raise ValueError("History contains missing values")

    if not history["Date"].is_monotonic_increasing:
        raise ValueError("History dates must be chronological")

    if history["Date"].duplicated().any():
        raise ValueError("History contains duplicate dates")

    if not history["Date"].diff().dropna().eq(
        pd.Timedelta(days=1)
    ).all():
        raise ValueError("History must contain consecutive daily records")

    predictions = []

    for _ in range(days):
        demand = history["Demand"].to_numpy()
        next_date = history["Date"].iloc[-1] + pd.Timedelta(days=1)

        features = pd.DataFrame([{
            "Demand_Lag_1": demand[-1],
            "Demand_Lag_7": demand[-7],
            "Demand_Lag_14": demand[-14],
            "Demand_Lag_28": demand[-28],
            "Demand_Rolling_Mean_7": demand[-7:].mean(),
            "Demand_Rolling_Mean_28": demand[-28:].mean(),
            "Demand_Rolling_Max_28": demand[-28:].max(),
            "Positive_Demand_Rate_28": (demand[-28:] > 0).mean(),
            "DayOfWeek": next_date.dayofweek,
            "Month": next_date.month
        }], columns=feature_columns)

        prediction = max(
            0.0,
            float(model.predict(features)[0])
        )

        predictions.append({
            "Date": next_date,
            "Predicted_Demand": prediction
        })

       
        history = pd.concat([
            history,
            pd.DataFrame({
                "Date": [next_date],
                "Demand": [prediction]
            })
        ], ignore_index=True)

    return pd.DataFrame(predictions)


def croston_forecast(demand, alpha=0.1, horizon=7):
    """
    Basic Croston method for intermittent demand.
    Returns the total forecast for the requested horizon.
    """
    values = pd.Series(demand).astype(float).to_numpy()

    if len(values) == 0:
        return 0.0

    positive_indices = [
        i for i, value in enumerate(values) if value > 0
    ]

    if len(positive_indices) == 0:
        return 0.0

    first_index = positive_indices[0]

    demand_size = values[first_index]
    interval = first_index + 1
    elapsed = 0

    for i in range(first_index + 1, len(values)):
        elapsed += 1

        if values[i] > 0:
            demand_size = (
                alpha * values[i]
                + (1 - alpha) * demand_size
            )

            interval = (
                alpha * elapsed
                + (1 - alpha) * interval
            )

            elapsed = 0

    daily_forecast = demand_size / interval

    return daily_forecast * horizon



def backtest_inventory_forecasts(
    df,
    model_data,
    feature_columns,
    product_stats,
    max_products=30
):
    origin_dates = pd.date_range(
        start="2011-07-20",
        end="2011-12-02",
        freq="7D"
    )

    evaluation_start = origin_dates.min()

    # Select products using historical data only
    historical_df = df[df["Date"] < evaluation_start]

    historical_stats = historical_df.groupby("StockCode").agg(
        total_demand=("Demand", "sum"),
        observed_days=("Date", "count"),
        selling_days=("Demand", lambda x: (x > 0).sum())
    )

    eligible = historical_stats[
        (historical_stats["observed_days"] >= 60)
        & (historical_stats["selling_days"] >= 5)
    ]

    ranked_ids = eligible.sort_values("total_demand").index

    if len(ranked_ids) == 0:
        raise ValueError("No eligible products found.")

    positions = np.linspace(
        0,
        len(ranked_ids) - 1,
        num=min(max_products, len(ranked_ids))
    ).round().astype(int)

    selected_products = ranked_ids[positions].unique()

    results = []

    for origin_date in origin_dates:

        # Train only on data available at the forecast origin
        train_fold = model_data[
            model_data["Date"] <= origin_date
        ]

        if train_fold.empty:
            continue

        print(
            f"\nTraining model through {origin_date.date()} "
            f"({len(train_fold):,} rows)"
        )

        fold_model = HistGradientBoostingRegressor(
            learning_rate=0.1,
            max_iter=100,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=42
        )

        fold_model.fit(
            train_fold[feature_columns],
            train_fold["Demand"]
        )

        expected_dates = list(pd.date_range(
            start=origin_date + pd.Timedelta(days=1),
            periods=7,
            freq="D"
        ))

        for product_id in selected_products:

            product_data = (
                df.loc[
                    df["StockCode"] == product_id,
                    ["Date", "Demand"]
                ]
                .sort_values("Date")
                .reset_index(drop=True)
            )

            history = product_data[
                product_data["Date"] <= origin_date
            ].copy()

            future = product_data[
                (product_data["Date"] > origin_date)
                & (
                    product_data["Date"]
                    <= origin_date + pd.Timedelta(days=7)
                )
            ].copy()

            if len(history) < 28 or len(future) != 7:
                continue

            # Verify that all seven future dates are present
            if future["Date"].tolist() != expected_dates:
                continue

            # Demand diagnostics using historical data only
            demand = history["Demand"].to_numpy()
            positive_demand = demand[demand > 0]

            observed_days = len(demand)
            positive_days = len(positive_demand)

            zero_rate = (
                1 - positive_days / observed_days
                if observed_days > 0
                else np.nan
            )

            adi = (
                observed_days / positive_days
                if positive_days > 0
                else np.inf
            )

            if positive_days > 1 and positive_demand.mean() > 0:
                cv_squared = (
                    positive_demand.std(ddof=1)
                    / positive_demand.mean()
                ) ** 2
            else:
                cv_squared = np.nan

            # Demand pattern classification
            if np.isnan(cv_squared) or not np.isfinite(adi):
                demand_pattern = "Insufficient_Data"

            elif adi < 1.32:
                if cv_squared < 0.49:
                    demand_pattern = "Smooth"
                else:
                    demand_pattern = "Erratic"

            else:
                if cv_squared < 0.49:
                    demand_pattern = "Intermittent"
                else:
                    demand_pattern = "Lumpy"

            # Generate forecasts
            forecast = forecast_next_days(
                model=fold_model,
                product_history=history,
                days=7
            )

            baseline_history = history.tail(7)

            baseline_forecast_7d = float(
                baseline_history["Demand"].mean() * 7
            )

            croston_forecast_7d = croston_forecast(
                demand=history["Demand"],
                alpha=0.1,
                horizon=7
            )

            actual_total = float(future["Demand"].sum())

            predicted_total = float(
                forecast["Predicted_Demand"].sum()
            )

            # Hybrid selection rule
            if zero_rate >= 0.70:
                hybrid_forecast_7d = croston_forecast_7d
                hybrid_model = "Croston"
            else:
                hybrid_forecast_7d = predicted_total
                hybrid_model = "HistGradientBoosting"

            # Save results
            results.append({
                "Product": product_id,
                "Origin_Date": origin_date,
                "Actual_7d": actual_total,
                "Forecast_7d": predicted_total,
                "Baseline_Forecast_7d": baseline_forecast_7d,
                "Croston_Forecast_7d": croston_forecast_7d,
                "Hybrid_Forecast_7d": hybrid_forecast_7d,
                "Hybrid_Model": hybrid_model,
                "Forecast_Error_7d": actual_total - predicted_total,
                "Zero_Rate": zero_rate,
                "ADI": adi,
                "CV2": cv_squared,
                "Demand_Pattern": demand_pattern
            })

    results_df = pd.DataFrame(results)



    if results_df.empty:
        raise ValueError(
            "No valid forecast windows were generated."
        )

    print("\nHybrid model selection counts:")
    print(results_df["Hybrid_Model"].value_counts())

    return results_df


    results_df["Baseline_Error"] = (
        results_df["Actual_7d"]
        - results_df["Baseline_Forecast_7d"]
    )                      
    results_df["Croston_Error"] = (
        results_df["Actual_7d"]
        - results_df["Croston_Forecast_7d"]
    )

    results_df["Hybrid_Error"] = (
        results_df["Actual_7d"]
        - results_df["Hybrid_Forecast_7d"]
    )
    
    results_df["Baseline_Abs_Error"] = (
        results_df["Baseline_Error"].abs()
    )   

    results_df["Croston_Abs_Error"] = (
        results_df["Croston_Error"].abs()
        )

    results_df["Hybrid_Abs_Error"] = (
        results_df["Hybrid_Error"].abs()
    )   

    comparison = pd.DataFrame({
        "Model": [
            "HistGradientBoosting",
            "Rolling_Mean_Baseline",
            "Croston",
            "Hybrid"
        ],
        "MAE": [
            (results_df["Actual_7d"] - results_df["Forecast_7d"]).abs().mean(),
            results_df["Baseline_Abs_Error"].mean(),
            results_df["Croston_Abs_Error"].mean(),
            results_df["Hybrid_Abs_Error"].mean()
        ],
        "WAPE (%)": [
            (results_df["Actual_7d"] - results_df["Forecast_7d"]).abs().sum()
            / results_df["Actual_7d"].sum() * 100,

            results_df["Baseline_Abs_Error"].sum()
            / results_df["Actual_7d"].sum() * 100,

            results_df["Croston_Abs_Error"].sum()
            / results_df["Actual_7d"].sum() * 100,

            results_df["Hybrid_Abs_Error"].sum()
            / results_df["Actual_7d"].sum() * 100
        ]
    })

    print("\nForecasting model comparison:")
    print(comparison.round(2).to_string(index=False))

    print("\nHybrid model selection counts:")
    print(results_df["Hybrid_Model"].value_counts())

    print("\nDemand pattern distribution:")
    print(results_df["Demand_Pattern"].value_counts())

    print("\nPattern distribution by product:")
    print(
        results_df.groupby(
            ["Product", "Demand_Pattern"]
        ).size().unstack(fill_value=0).to_string()
    )

    print(
        results_df[
            ["Product", "Origin_Date", "Zero_Rate", "ADI", "CV2"]
        ].head(20).round(3).to_string(index=False)
    )
    
    if results_df.empty:
        raise ValueError(
            "No valid forecast windows were generated."
        )

    errors = results_df["Forecast_Error_7d"]

    actual = results_df["Actual_7d"].to_numpy()

    model_pred = results_df["Forecast_7d"].to_numpy()
    baseline_pred = results_df["Baseline_Forecast_7d"].to_numpy()
    croston_pred = results_df["Croston_Forecast_7d"].to_numpy()


    def evaluate_forecast(actual, predicted):
        error = actual - predicted

        mae = np.mean(np.abs(error))
        wape = (
        np.sum(np.abs(error)) / np.sum(actual) * 100
        if np.sum(actual) > 0 else np.nan
        )

        return mae, wape


    print("\nModel comparison on identical forecast windows")

    
    actual = results_df["Actual_7d"].to_numpy()

    model_pred = results_df["Forecast_7d"].to_numpy()
    baseline_pred = results_df["Baseline_Forecast_7d"].to_numpy()
    croston_pred = results_df["Croston_Forecast_7d"].to_numpy()

    def evaluate_forecast(actual, predicted):
        error = actual - predicted

        mae = np.mean(np.abs(error))
        total_actual = np.sum(actual)

        wape = (
            np.sum(np.abs(error)) / total_actual * 100
            if total_actual > 0 else np.nan
        )

        return mae, wape

    print("\nModel comparison on identical forecast windows")

    for name, predicted in [
        ("HistGradientBoosting", model_pred),
        ("7-day Mean Baseline", baseline_pred),
        ("Croston", croston_pred)
    ]:
        mae, wape = evaluate_forecast(actual, predicted)

        print(
            f"{name}: "
            f"MAE={mae:.2f}, "
            f"WAPE={wape:.2f}%"
        )

    
    # Product-level comparison: exploratory analysis
    product_comparison = []

    for product_id, group in results_df.groupby("Product"):
        actual_product = group["Actual_7d"].to_numpy()

        model_predictions = {
            "HistGradientBoosting": group["Forecast_7d"].to_numpy(),
            "Rolling_Mean_Baseline": group["Baseline_Forecast_7d"].to_numpy(),
            "Croston": group["Croston_Forecast_7d"].to_numpy()
        }

        row = {
            "Product": product_id,
            "Windows": len(group),
            "Actual_Demand": actual_product.sum()
        }

        for name, predicted in model_predictions.items():
            abs_error = np.abs(actual_product - predicted)

            row[f"{name}_MAE"] = abs_error.mean()
            row[f"{name}_WAPE"] = (
                abs_error.sum() / actual_product.sum() * 100
                if actual_product.sum() > 0 else np.nan
            )

        product_comparison.append(row)

    product_comparison_df = pd.DataFrame(product_comparison)

    product_comparison_df["Best_MAE_Model"] = (
        product_comparison_df[
            [
                "HistGradientBoosting_MAE",
                "Rolling_Mean_Baseline_MAE",
                "Croston_MAE"
            ]
        ]
        .idxmin(axis=1)
        .str.replace("_MAE", "", regex=False)
    )

    print(
        product_comparison_df.sort_values(
            "Actual_Demand", ascending=False
        ).round(2).to_string(
            index=False,
            col_space=12
        )
    )

    print("\nBest model by product (based on MAE):")
    print(
        product_comparison_df["Best_MAE_Model"]
        .value_counts()
        .to_string()
    )

    errors = results_df["Forecast_Error_7d"]


    safety_stock_estimate = max(
        0.0,
        float(errors.quantile(0.95))
    )

    print("\nRolling-origin backtest summary")
    print("Valid forecast windows:", len(results_df))
    print("Products evaluated:", results_df["Product"].nunique())
    print("Forecast origins:", results_df["Origin_Date"].nunique())
    print(
        "7-day MAE:",
        round(errors.abs().mean(), 2)
    )
    print(
        "95th percentile of underforecast error:",
        round(safety_stock_estimate, 2)
    )
    

    underforecast_errors = results_df.copy()
    underforecast_errors["Underforecast_Error"] = (
        underforecast_errors["Forecast_Error_7d"].clip(lower=0)
    )

    product_error_summary = (
        underforecast_errors.groupby("Product")
        .agg(
            Windows=("Underforecast_Error", "count"),
            Mean_Underforecast=("Underforecast_Error", "mean"),
            Safety_Stock=("Underforecast_Error", lambda x: x.quantile(0.95))
        )
        .reset_index()
    )

    print("\nPer-product underforecast summary:")
    print(
        product_error_summary.sort_values(
            "Safety_Stock", ascending=False
        ).round(2).to_string(index=False)
    )

    return results_df, safety_stock_estimate, product_error_summary


y_pred = np.maximum(model.predict(X_test), 0)

errors = y_test.to_numpy() - y_pred

results.append({
    "Model": "HistGradientBoosting",
    "MAE": np.mean(np.abs(errors)),
    "RMSE": np.sqrt(np.mean(errors ** 2)),
    "WAPE (%)": (
        np.sum(np.abs(errors)) / np.sum(y_test.to_numpy()) * 100
    )
})

# ==========================================
# 6. Compare All Models
# ==========================================

results_df = pd.DataFrame(results).sort_values("MAE")

print("\nFinal model comparison:")
print(results_df.round(4).to_string(index=False))


# ==========================================
# 7. Error Analysis by Demand Group
# ==========================================

error_data = test[["Demand", "Demand_Rolling_Mean_7"]].copy()

error_data["Prediction"] = y_pred
error_data["Actual"] = y_test.to_numpy()

error_data["Model_Error"] = np.abs(
    error_data["Actual"] - error_data["Prediction"]
)

error_data["Baseline_Error"] = np.abs(
    error_data["Actual"] - error_data["Demand_Rolling_Mean_7"]
)

error_data["Demand_Group"] = pd.cut(
    error_data["Actual"],
    bins=[-0.1, 0, 10, 100, 1000, np.inf],
    labels=["Zero", "1-10", "11-100", "101-1000", "1000+"]
)

group_results = error_data.groupby(
    "Demand_Group", observed=False
).agg(
    Rows=("Actual", "size"),
    Total_Demand=("Actual", "sum"),
    Model_MAE=("Model_Error", "mean"),
    Baseline_MAE=("Baseline_Error", "mean"),
    Model_Total_Error=("Model_Error", "sum"),
    Baseline_Total_Error=("Baseline_Error", "sum")
).reset_index()

print("\nError analysis by demand group:")
print(group_results.round(3).to_string(index=False))


# ==========================================
# 8. Inspect Largest Forecast Errors
# ==========================================

error_data["Absolute_Error"] = error_data["Model_Error"]

error_data["Date"] = test["Date"].to_numpy()
error_data["StockCode"] = test["StockCode"].to_numpy()

error_data["Demand_Lag_7"] = test["Demand_Lag_7"].to_numpy()
error_data["Demand_Rolling_Max_28"] = (
    test["Demand_Rolling_Max_28"].to_numpy()
)
error_data["Positive_Demand_Rate_28"] = (
    test["Positive_Demand_Rate_28"].to_numpy()
)

largest_errors = error_data.nlargest(
    15, "Absolute_Error"
)

print("\nTop 15 largest forecast errors:")

print(
    largest_errors[
        [
            "Date",
            "StockCode",
            "Actual",
            "Prediction",
            "Absolute_Error",
            "Demand_Lag_7",
            "Demand_Rolling_Max_28",
            "Positive_Demand_Rate_28"
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.2f}")
)

# ==========================================
# 9. Inspect Demand Around Largest Errors
# ==========================================

top_cases = largest_errors.head(5)[
    ["Date", "StockCode"]
].drop_duplicates()

for _, case in top_cases.iterrows():
    product = case["StockCode"]
    target_date = case["Date"]

    history = df[
        (df["StockCode"] == product)
        & (
            df["Date"].between(
                target_date - pd.Timedelta(days=14),
                target_date + pd.Timedelta(days=3)
            )
        )
    ][["Date", "Demand"]].sort_values("Date")

    print(f"\nProduct: {product} | Target date: {target_date.date()}")
    print(history.to_string(index=False))



product_id = df.groupby("StockCode").size().idxmax()

product_history = df.loc[
    df["StockCode"] == product_id,
    ["Date", "Demand"]
]

forecast = forecast_next_days(
    model=model,
    product_history=product_history,
    days=8
)

print("\nSelected product:", product_id)
print(
    forecast.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)
print("Total predicted demand:", round(
    forecast["Predicted_Demand"].sum(), 2
))

backtest_results = backtest_inventory_forecasts(
    df, model_data, feature_columns, product_stats
)


# Compare all forecasting methods on identical backtest windows

forecast_columns = {
    "HistGradientBoosting": "Forecast_7d",
    "Croston": "Croston_Forecast_7d",
    "7-Day Mean Baseline": "Baseline_Forecast_7d",
    "Hybrid": "Hybrid_Forecast_7d",
}

evaluation = backtest_results.dropna(
    subset=["Actual_7d", *forecast_columns.values()]
).copy()

actual = evaluation["Actual_7d"]
total_actual = actual.abs().sum()

metrics = []

for model_name, forecast_column in forecast_columns.items():
    predictions = evaluation[forecast_column]
    errors = predictions - actual

    metrics.append({
        "Model": model_name,
        "MAE": errors.abs().mean(),
        "WAPE (%)": (
            errors.abs().sum() / total_actual * 100
            if total_actual > 0 else float("nan")
        ),
        "Bias": errors.mean(),
        "Total_Bias": errors.sum(),
    })

metrics_df = (
    pd.DataFrame(metrics)
    .sort_values("MAE")
    .reset_index(drop=True)
)

print("\nForecast comparison on identical backtest windows:")
print(metrics_df.round(3).to_string(index=False))


# Evaluate all forecasting methods by product demand level

product_demand = (
    evaluation.groupby("Product")["Actual_7d"]
    .sum()
)

evaluation["Demand_Level"] = evaluation["Product"].map(
    lambda product: (
        "Low"
        if product_demand[product] < 100
        else "Medium"
        if product_demand[product] < 1000
        else "High"
    )
)

group_results = []

for level, group in evaluation.groupby("Demand_Level"):
    actual = group["Actual_7d"]
    denominator = actual.abs().sum()

    for model_name, forecast_column in forecast_columns.items():
        errors = group[forecast_column] - actual

        group_results.append({
            "Demand_Level": level,
            "Model": model_name,
            "Windows": len(group),
            "MAE": errors.abs().mean(),
            "WAPE (%)": (
                errors.abs().sum() / denominator * 100
                if denominator > 0 else float("nan")
            ),
            "Bias": errors.mean(),
        })

group_results_df = pd.DataFrame(group_results)

print("\nPerformance by demand level:")
print(
    group_results_df
    .sort_values(["Demand_Level", "MAE"])
    .round(3)
    .to_string(index=False)
)


print("\nSample backtest results:")

sample_results = backtest_results.head(10).copy()
sample_results["Actual_7d"] = sample_results["Actual_7d"].round(2)
sample_results["Forecast_7d"] = sample_results["Forecast_7d"].round(2)
sample_results["Forecast_Error_7d"] = sample_results["Forecast_Error_7d"].round(2)

print(sample_results.to_string(index=False))

backtest_results["Model_Abs_Error"] = (
    backtest_results["Actual_7d"]
    - backtest_results["Forecast_7d"]
).abs()

backtest_results["Baseline_Abs_Error"] = (
    backtest_results["Actual_7d"]
    - backtest_results["Baseline_Forecast_7d"]
).abs()

print("\nModel vs 7-day mean baseline:")
print(
    "Model MAE:",
    round(backtest_results["Model_Abs_Error"].mean(), 2)
)
print(
    "Baseline MAE:",
    round(backtest_results["Baseline_Abs_Error"].mean(), 2)
)

backtest_results["Abs_Error_7d"] = (
    backtest_results["Forecast_Error_7d"].abs()
)

backtest_results["Underforecast_Error"] = (
    backtest_results["Forecast_Error_7d"].clip(lower=0)
)

print("\nOverall error diagnostics:")
print(
    backtest_results[
        ["Actual_7d", "Forecast_7d", "Forecast_Error_7d", "Abs_Error_7d"]
    ].describe().round(2)
)

print("\nProducts with the largest mean absolute errors:")
product_mae = (
    backtest_results.groupby("Product")
    .agg(
        Windows=("Abs_Error_7d", "count"),
        Mean_Absolute_Error=("Abs_Error_7d", "mean"),
        Max_Absolute_Error=("Abs_Error_7d", "max"),
        Total_Actual=("Actual_7d", "sum"),
        Total_Forecast=("Forecast_7d", "sum"),
    )
    .sort_values("Mean_Absolute_Error", ascending=False)
)

print(product_mae.head(10).round(2).to_string())


product_id = "84077"

product_errors = backtest_results[
    backtest_results["Product"].astype(str) == product_id
].copy()

product_errors["Abs_Error_7d"] = (
    product_errors["Forecast_Error_7d"].abs()
)

product_errors = product_errors.sort_values(
    "Abs_Error_7d", ascending=False
)

print(f"\nLargest forecast errors for product {product_id}:")
print(
    product_errors[
        [
            "Origin_Date",
            "Actual_7d",
            "Forecast_7d",
            "Forecast_Error_7d",
            "Abs_Error_7d",
        ]
        ].head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


product_id = "84077"

product_history = (
    df[df["StockCode"].astype(str) == product_id]
    .sort_values("Date")
    .copy()
)

inspection_start = pd.Timestamp("2011-09-01")
inspection_end = pd.Timestamp("2011-11-10")

period = product_history[
    product_history["Date"].between(
        inspection_start, inspection_end
    )
][["Date", "Demand"]].copy()

period["Rolling_Mean_7"] = (
    period["Demand"].shift(1).rolling(7).mean()
)

period["Rolling_Max_7"] = (
    period["Demand"].shift(1).rolling(7).max()
)

print("\nProduct 84077: daily demand around the October spike")
print(
    period[
        (period["Demand"] > 100)
        | (period["Date"].between(
            "2011-10-20", "2011-11-02"
        ))
        ].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# Relative error analysis by product
product_relative_error = (
    backtest_results.groupby("Product")
    .apply(
        lambda group: pd.Series({
            "Windows": len(group),
            "Total_Actual": group["Actual_7d"].sum(),
            "Total_Absolute_Error": (
                group["Actual_7d"] - group["Forecast_7d"]
            ).abs().sum(),
            "MAE": (
                group["Actual_7d"] - group["Forecast_7d"]
            ).abs().mean(),
            "WAPE_pct": (
                (
                    group["Actual_7d"] - group["Forecast_7d"]
                ).abs().sum()
                / group["Actual_7d"].sum() * 100
                if group["Actual_7d"].sum() > 0
                else float("nan")
            ),
            "Bias": group["Forecast_Error_7d"].mean()
        })
    )
    .sort_values("WAPE_pct", ascending=False)
)

print("\nProducts with the highest relative error (WAPE):")
print(product_relative_error.round(2).to_string())


# Performance by product demand level
product_demand_levels = (
    backtest_results.groupby("Product")
    .agg(
        Total_Actual=("Actual_7d", "sum"),
        Total_Absolute_Error=(
            "Forecast_Error_7d",
            lambda x: x.abs().sum()
        ),
        Mean_Bias=("Forecast_Error_7d", "mean"),
        Windows=("Actual_7d", "count")
    )
)

product_demand_levels["WAPE_pct"] = (
    product_demand_levels["Total_Absolute_Error"]
    / product_demand_levels["Total_Actual"].replace(0, float("nan"))
    * 100
)

product_demand_levels["Demand_Level"] = pd.qcut(
    product_demand_levels["Total_Actual"].rank(method="first"),
    q=3,
    labels=["Low", "Medium", "High"]
)


# Aggregate WAPE by demand level
group_error = backtest_results.copy()

group_error["Abs_Error"] = (
    group_error["Actual_7d"] - group_error["Forecast_7d"]
).abs()

group_error["Demand_Level"] = (
    group_error["Product"]
    .map(product_demand_levels["Demand_Level"])
)

aggregate_wape = (
    group_error.groupby("Demand_Level", observed=True)
    .agg(
        Total_Actual=("Actual_7d", "sum"),
        Total_Absolute_Error=("Abs_Error", "sum"),
        Products=("Product", "nunique")
    )
)

aggregate_wape["WAPE_pct"] = (
    aggregate_wape["Total_Absolute_Error"]
    / aggregate_wape["Total_Actual"].replace(0, float("nan"))
    * 100
)

print("\nAggregate WAPE by demand level:")
print(aggregate_wape.round(2).to_string())

print("\nPerformance by product demand level:")
print(
    product_demand_levels.groupby("Demand_Level", observed=True)
    .agg(
        Products=("Total_Actual", "count"),
        Total_Actual=("Total_Actual", "sum"),
        Mean_WAPE=("WAPE_pct", "mean"),
        Median_WAPE=("WAPE_pct", "median"),
        Mean_Bias=("Mean_Bias", "mean")
    )
    .round(2)
    .to_string()
)


# Compare model vs baseline by demand level
comparison = backtest_results.copy()

comparison["Model_Abs_Error"] = (
    comparison["Actual_7d"] - comparison["Forecast_7d"]
).abs()

comparison["Baseline_Abs_Error"] = (
    comparison["Actual_7d"] - comparison["Baseline_Forecast_7d"]
).abs()

comparison["Demand_Level"] = comparison["Product"].map(
    product_demand_levels["Demand_Level"]
)

group_comparison = comparison.groupby(
    "Demand_Level", observed=True
).agg(
    Products=("Product", "nunique"),
    Actual_Demand=("Actual_7d", "sum"),
    Model_Error=("Model_Abs_Error", "sum"),
    Baseline_Error=("Baseline_Abs_Error", "sum")
)

group_comparison["Model_WAPE_pct"] = (
    group_comparison["Model_Error"]
    / group_comparison["Actual_Demand"].replace(0, float("nan"))
    * 100
)

group_comparison["Baseline_WAPE_pct"] = (
    group_comparison["Baseline_Error"]
    / group_comparison["Actual_Demand"].replace(0, float("nan"))
    * 100
)

group_comparison["Improvement_pct"] = (
    (
        group_comparison["Baseline_WAPE_pct"]
        - group_comparison["Model_WAPE_pct"]
    )
    / group_comparison["Baseline_WAPE_pct"]
    * 100
)

print("\nModel vs baseline by demand level:")
print(group_comparison.round(2).to_string())


# Product-level model vs baseline comparison
product_comparison = (
    comparison.groupby("Product")
    .agg(
        Windows=("Actual_7d", "count"),
        Actual_Demand=("Actual_7d", "sum"),
        Model_Error=("Model_Abs_Error", "sum"),
        Baseline_Error=("Baseline_Abs_Error", "sum")
    )
)

product_comparison["Model_WAPE_pct"] = (
    product_comparison["Model_Error"]
    / product_comparison["Actual_Demand"].replace(0, float("nan"))
    * 100
)

product_comparison["Baseline_WAPE_pct"] = (
    product_comparison["Baseline_Error"]
    / product_comparison["Actual_Demand"].replace(0, float("nan"))
    * 100
)

product_comparison["Improvement_pct"] = (
    (
        product_comparison["Baseline_WAPE_pct"]
        - product_comparison["Model_WAPE_pct"]
    )
    / product_comparison["Baseline_WAPE_pct"].replace(0, float("nan"))
    * 100
)

print("\nProducts where ML underperforms the baseline:")
print(
    product_comparison.sort_values("Improvement_pct")
    .round(2)
    .to_string()
)

# Intermittent demand diagnostics
demand_pattern = (
    df.groupby("StockCode")["Demand"]
    .agg(
        Mean_Daily_Demand="mean",
        Std_Daily_Demand="std",
        Zero_Demand_Days=lambda x: (x == 0).mean() * 100,
        Positive_Demand_Days=lambda x: (x > 0).sum()
    )
)

demand_pattern["Demand_CV"] = (
    demand_pattern["Std_Daily_Demand"]
    / demand_pattern["Mean_Daily_Demand"].replace(0, float("nan"))
)

print("\nDemand intermittency diagnostics:")
print(
    demand_pattern.sort_values(
        "Zero_Demand_Days", ascending=False
    ).head(15).round(2).to_string()
)


# Demand patterns for backtested products
evaluated_products = backtest_results["Product"].unique()

evaluated_demand_pattern = (
    df[df["StockCode"].isin(evaluated_products)]
    .groupby("StockCode")["Demand"]
    .agg(
        Mean_Daily_Demand="mean",
        Std_Daily_Demand="std",
        Zero_Demand_Days=lambda x: (x == 0).mean() * 100,
        Positive_Demand_Days=lambda x: (x > 0).sum()
    )
)

evaluated_demand_pattern["Demand_CV"] = (
    evaluated_demand_pattern["Std_Daily_Demand"]
    / evaluated_demand_pattern["Mean_Daily_Demand"].replace(0, float("nan"))
)

print("\nDemand patterns for backtested products:")
print(
    evaluated_demand_pattern.sort_values(
        "Zero_Demand_Days", ascending=False
    ).round(2).to_string()
)

def prepare_product_history(
    daily_demand,
    stock_code,
    as_of_date
):
    # Only use data strictly before the evaluation date
    history = daily_demand.loc[
        (daily_demand["StockCode"] == stock_code)
        & (daily_demand["Date"] < pd.Timestamp(as_of_date)),
        ["Date", "Demand"]
    ].copy()

    history["Date"] = pd.to_datetime(history["Date"])

    if history.empty:
        return pd.Series(dtype=float)

    # Build a complete daily calendar; missing sales days mean zero recorded sales
    history = history.set_index("Date")["Demand"].sort_index()

    if history.index.has_duplicates:
        raise ValueError(
            f"Duplicate dates found for product {stock_code}"
        )

    if history.isna().any() or (history < 0).any():
        raise ValueError(
            f"Invalid demand values for product {stock_code}"
        )

    full_dates = pd.date_range(
        start=history.index.min(),
        end=history.index.max(),
        freq="D"
    )

    history = history.reindex(full_dates, fill_value=0)
    history = history.clip(lower=0)

    return history

def calculate_inventory_policy(
    history,
    lead_time=3,
    review_period=7,
    service_level=0.95,
    min_history_days=60
):
    if len(history) < min_history_days:
        return None

    if history.isna().any() or (history < 0).any():
        return None

    protection_period = lead_time + review_period

    # Historical demand totals over the lead-time horizon
    lead_time_demand = (
        history.rolling(lead_time)
        .sum()
        .dropna()
    )

    # Historical demand totals over the full protection period
    protection_demand = (
        history.rolling(protection_period)
        .sum()
        .dropna()
    )

    if lead_time_demand.empty or protection_demand.empty:
        return None

    # Empirical demand quantiles
    expected_lead_demand = lead_time_demand.mean()
    reorder_point = lead_time_demand.quantile(service_level)

    expected_protection_demand = protection_demand.mean()
    target_stock = protection_demand.quantile(service_level)

    # Safety stock for the full protection period
    safety_stock = max(
        0,
        target_stock - expected_protection_demand
    )

    return {
        "Lead_Time_Days": lead_time,
        "Review_Period_Days": review_period,
        "Protection_Period_Days": protection_period,
        "Expected_Lead_Demand": expected_lead_demand,
        "Safety_Stock": safety_stock,
        "Reorder_Point": math.ceil(reorder_point),
        "Expected_Protection_Demand": expected_protection_demand,
        "Target_Stock": math.ceil(target_stock),
        "Service_Level_Target": service_level,
        "Historical_Days": len(history),
    }



# ============================================================
# PHASE 5 — INVENTORY POLICY
# ============================================================

def build_inventory_table(
    daily_demand,
    stock_codes,
    as_of_date,
    lead_time=3,
    review_period=7,
    service_level=0.95,
    min_history_days=60
):
    results = []

    for stock_code in stock_codes:
        history = prepare_product_history(
            daily_demand=daily_demand,
            stock_code=stock_code,
            as_of_date=as_of_date
        )

        policy = calculate_inventory_policy(
            history=history,
            lead_time=lead_time,
            review_period=review_period,
            service_level=service_level,
            min_history_days=min_history_days
        )

        if policy is None:
            continue

        policy = policy.copy()
        policy["StockCode"] = stock_code
        results.append(policy)

    columns = [
        "StockCode",
        "Lead_Time_Days",
        "Review_Period_Days",
        "Protection_Period_Days",
        "Expected_Lead_Demand",
        "Safety_Stock",
        "Reorder_Point",
        "Expected_Protection_Demand",
        "Target_Stock",
        "Service_Level_Target",
        "Historical_Days"
    ]

    if not results:
        return pd.DataFrame(columns=columns)

    inventory_table = pd.DataFrame(results)

    return inventory_table[columns]


# ============================================================
# PHASE 6 — INVENTORY SIMULATION
# ============================================================

# 1. Check source data and define simulation dates.
df = df.copy()
df["Date"] = pd.to_datetime(df["Date"])

if df[["Date", "StockCode", "Demand"]].isna().any().any():
    raise ValueError(
        "Source data contains missing Date, StockCode, or Demand values."
    )

if (df["Demand"] < 0).any():
    raise ValueError("Source data contains negative demand.")

if df.duplicated(["StockCode", "Date"]).any():
    raise ValueError(
        "Source data has duplicate product-date rows. "
        "Aggregate or correct these before simulation."
    )

simulation_start = pd.Timestamp("2011-07-21")
simulation_end = df["Date"].max()

if simulation_start > simulation_end:
    raise ValueError("Simulation start is after simulation end.")

simulation_dates = pd.date_range(
    start=simulation_start,
    end=simulation_end,
    freq="D"
)

# Keep the same reproducible sample strategy: first 20 unique product codes.
sample_products = (
    df["StockCode"]
    .drop_duplicates()
    .head(20)
    .tolist()
)
simulation_products = list(dict.fromkeys(sample_products))

print("\nSimulation period:")
print("Start:", simulation_start)
print("End:", simulation_end)
print("Days:", len(simulation_dates))
print("Selected products:", len(simulation_products))


# 2. Build policies using only history before simulation starts.
simulation_policy_table = build_inventory_table(
    daily_demand=df,
    stock_codes=simulation_products,
    as_of_date=simulation_start,
    lead_time=3,
    review_period=7,
    service_level=0.95
)

if simulation_policy_table.empty:
    raise ValueError(
        "No valid inventory policies could be built "
        "for the simulation products."
    )

if simulation_policy_table["StockCode"].duplicated().any():
    raise ValueError(
        "Duplicate products found in simulation policy table."
    )

policy_products = set(
    simulation_policy_table["StockCode"]
)

simulation_products_with_policy = [
    product for product in simulation_products
    if product in policy_products
]

excluded_products = [
    product for product in simulation_products
    if product not in policy_products
]

if not simulation_products_with_policy:
    raise ValueError(
        "No selected products have valid inventory policies."
    )

print("\nProducts with valid policies:",
      len(simulation_products_with_policy))
print("Excluded products:", excluded_products)

print(
    simulation_policy_table[
        ["StockCode", "Reorder_Point", "Target_Stock"]
    ].head()
)


# 3. Build the complete daily product panel.
simulation_data = df.loc[
    df["Date"].between(simulation_start, simulation_end)
    & df["StockCode"].isin(simulation_products_with_policy),
    ["Date", "StockCode", "Demand"]
].copy()

if simulation_data.duplicated(["StockCode", "Date"]).any():
    raise ValueError(
        "Duplicate product-date rows found in simulation data."
    )

if (
    simulation_data["Demand"].isna().any()
    or (simulation_data["Demand"] < 0).any()
):
    raise ValueError("Invalid demand in simulation data.")

full_index = pd.MultiIndex.from_product(
    [simulation_products_with_policy, simulation_dates],
    names=["StockCode", "Date"]
)

simulation_panel = (
    simulation_data
    .set_index(["StockCode", "Date"])[["Demand"]]
    .reindex(full_index, fill_value=0)
    .reset_index()
)

expected_rows = (
    len(simulation_products_with_policy)
    * len(simulation_dates)
)

assert len(simulation_panel) == expected_rows
assert not simulation_panel.duplicated(
    ["StockCode", "Date"]
).any()
assert simulation_panel["Demand"].notna().all()
assert (simulation_panel["Demand"] >= 0).all()

print("\nSimulation panel checks:")
print("Products:", simulation_panel["StockCode"].nunique())
print("Days:", simulation_panel["Date"].nunique())
print("Rows:", len(simulation_panel))
print("Expected rows:", expected_rows)
print("Missing demand:", simulation_panel["Demand"].isna().sum())
print("Negative demand:", (simulation_panel["Demand"] < 0).sum())


# ============================================================
# INVENTORY POLICY SIMULATOR
# ============================================================


def simulate_inventory_policy(
    simulation_panel,
    inventory_policy_table,
    lead_time=3,
    review_period=7,
    initial_stock_by_product=None
):

    if lead_time < 1 or review_period < 1:
        raise ValueError(
            "Lead time and review period must be positive."
        )

    required_panel_columns = {
        "Date", "StockCode", "Demand"
    }

    required_policy_columns = {
        "StockCode", "Target_Stock"
    }

    if not required_panel_columns.issubset(
        simulation_panel.columns
    ):
        raise ValueError("Simulation panel is missing columns.")

    if not required_policy_columns.issubset(
        inventory_policy_table.columns
    ):
        raise ValueError("Inventory policy table is missing columns.")

    panel = simulation_panel.copy()
    panel["Date"] = pd.to_datetime(panel["Date"])

    if panel[["Date", "StockCode", "Demand"]].isna().any().any():
        raise ValueError("Simulation panel contains missing values.")

    if panel.duplicated(["StockCode", "Date"]).any():
        raise ValueError("Duplicate product-date rows found.")

    if (panel["Demand"] < 0).any():
        raise ValueError("Negative demand found.")

    if not np.allclose(
        panel["Demand"],
        np.round(panel["Demand"])
    ):
        raise ValueError(
            "This simulator expects whole-unit demand."
        )

    if inventory_policy_table["StockCode"].duplicated().any():
        raise ValueError("Duplicate product policies found.")

    policies = inventory_policy_table.set_index("StockCode")

    if (
        policies["Target_Stock"].isna().any()
        or (policies["Target_Stock"] < 0).any()
    ):
        raise ValueError("Invalid target stock values found.")

    products = panel["StockCode"].unique().tolist()

    missing_products = [
        product for product in products
        if product not in policies.index
    ]

    if missing_products:
        raise ValueError(
            f"Missing inventory policies for: {missing_products}"
        )

    dates = pd.DatetimeIndex(
        sorted(panel["Date"].unique())
    )

    expected_dates = pd.date_range(
        start=dates.min(),
        end=dates.max(),
        freq="D"
    )

    if not dates.equals(expected_dates):
        raise ValueError(
            "Simulation dates must form a complete daily calendar."
        )

    results = []

    for stock_code in products:
        product_data = (
            panel.loc[panel["StockCode"] == stock_code]
            .sort_values("Date")
            .set_index("Date")
        )

        if not product_data.index.equals(expected_dates):
            raise ValueError(
                f"Incomplete daily data for {stock_code}."
            )

        target_stock = int(
            math.ceil(
                float(policies.loc[stock_code, "Target_Stock"])
            )
        )

        # Simulation assumption: opening stock equals target stock.
        # Use a shared initial stock when supplied.
        if initial_stock_by_product is None:
            on_hand = target_stock
        else:
            on_hand = int(initial_stock_by_product[stock_code])

        if on_hand < 0:
            raise ValueError("Initial stock cannot be negative.")
        on_order = 0
        backorders = 0
        scheduled_receipts = {}

        for day_number, date in enumerate(expected_dates):
            date = pd.Timestamp(date)

            # Receive orders arriving on this date.
            receipts = scheduled_receipts.pop(date, 0)
            on_hand += receipts
            on_order -= receipts

            # Fulfil older backorders first.
            backlog_fulfilled = min(on_hand, backorders)
            on_hand -= backlog_fulfilled
            backorders -= backlog_fulfilled

            demand = int(product_data.loc[date, "Demand"])

            # Fulfil today's demand from remaining stock.
            fulfilled_today = min(on_hand, demand)
            on_hand -= fulfilled_today

            unmet_demand = demand - fulfilled_today
            backorders += unmet_demand

            order_quantity = 0

            # Review on the first day, then every 7 days.
            if day_number % review_period == 0:
                inventory_position = (
                    on_hand + on_order - backorders
                )

                order_quantity = max(
                    0,
                    math.ceil(target_stock - inventory_position)
                )

                if order_quantity > 0:
                    arrival_date = (
                        date + pd.Timedelta(days=lead_time)
                    )

                    scheduled_receipts[arrival_date] = (
                        scheduled_receipts.get(arrival_date, 0)
                        + order_quantity
                    )

                    on_order += order_quantity

            inventory_position = (
                on_hand + on_order - backorders
            )

            results.append({
                "Date": date,
                "StockCode": stock_code,
                "Demand": demand,
                "Receipts": receipts,
                "Backorders_Fulfilled": backlog_fulfilled,
                "Fulfilled_Demand": fulfilled_today,
                "Unmet_Demand": unmet_demand,
                "Backorders": backorders,
                "On_Hand": on_hand,
                "On_Order": on_order,
                "Inventory_Position": inventory_position,
                "Order_Quantity": order_quantity,
                "Target_Stock": target_stock
            })

    return (
        pd.DataFrame(results)
        .sort_values(["StockCode", "Date"])
        .reset_index(drop=True)
    )


# ============================================================
# RUN SIMULATION
# ============================================================

print("\nSimulation policy:")
print(
    simulation_policy_table.loc[
        simulation_policy_table["StockCode"] == "10080",
        ["StockCode", "Target_Stock"]
    ].to_string(index=False)
)

simulation_results = simulate_inventory_policy(
    simulation_panel=simulation_panel,
    inventory_policy_table=simulation_policy_table,
    lead_time=3,
    review_period=7
)


# ============================================================
# SIMULATION METRICS AND VALIDATION
# ============================================================

total_demand = simulation_results["Demand"].sum()
fulfilled_same_day = simulation_results["Fulfilled_Demand"].sum()
total_unmet = simulation_results["Unmet_Demand"].sum()

same_day_fill_rate = (
    fulfilled_same_day / total_demand
    if total_demand > 0
    else float("nan")
)

print("\nSimulation Results:")
print(simulation_results.head(10).to_string(index=False))

print("\nRows:", len(simulation_results))
print("Products:", simulation_results["StockCode"].nunique())
print("Days:", simulation_results["Date"].nunique())

print("\nTotal demand:", total_demand)
print("Fulfilled on demand day:", fulfilled_same_day)
print("Unmet demand on demand day:", total_unmet)
print("Same-day fill rate:",
      round(same_day_fill_rate * 100, 2), "%")
print("Backorders fulfilled later:",
      simulation_results["Backorders_Fulfilled"].sum())
print("Total units ordered:",
      simulation_results["Order_Quantity"].sum())

final_backorders = (
    simulation_results
    .groupby("StockCode")["Backorders"]
    .last()
    .sum()
)

print("Final outstanding backorders:", final_backorders)

assert (simulation_results["On_Hand"] >= 0).all()
assert (simulation_results["On_Order"] >= 0).all()
assert (simulation_results["Backorders"] >= 0).all()
assert (simulation_results["Order_Quantity"] >= 0).all()

print("\nSimulation validation passed.")


# ============================================================
# PHASE 6.1 — PER-PRODUCT SIMULATION EVALUATION
# ============================================================

product_evaluation = (
    simulation_results
    .groupby("StockCode")
    .agg(
        Total_Demand=("Demand", "sum"),
        Same_Day_Fulfilled=("Fulfilled_Demand", "sum"),
        Unmet_Demand=("Unmet_Demand", "sum"),
        Units_Ordered=("Order_Quantity", "sum"),
        Final_Backorders=("Backorders", "last"),
        Average_On_Hand=("On_Hand", "mean"),
        Average_On_Order=("On_Order", "mean"),
        Target_Stock=("Target_Stock", "first")
    )
    .reset_index()
)

product_evaluation["Same_Day_Fill_Rate"] = np.where(
    product_evaluation["Total_Demand"] > 0,
    (
        product_evaluation["Same_Day_Fulfilled"]
        / product_evaluation["Total_Demand"]
    ),
    np.nan
)

product_evaluation["Unmet_Demand_Rate"] = np.where(
    product_evaluation["Total_Demand"] > 0,
    (
        product_evaluation["Unmet_Demand"]
        / product_evaluation["Total_Demand"]
    ),
    np.nan
)

product_evaluation = product_evaluation.sort_values(
    "Unmet_Demand",
    ascending=False
)

print("\nWorst products by unmet demand:")
print(
    product_evaluation[
        [
            "StockCode",
            "Total_Demand",
            "Unmet_Demand",
            "Same_Day_Fill_Rate",
            "Units_Ordered",
            "Target_Stock",
            "Average_On_Hand"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nOverall weighted same-day fill rate:")
total_demand = product_evaluation["Total_Demand"].sum()

if total_demand > 0:
    overall_fill_rate = (
        product_evaluation["Same_Day_Fulfilled"].sum()
        / total_demand
    )
    print(f"{overall_fill_rate:.2%}")
else:
    print("No demand recorded.")


# ============================================================
# Compare inventory policies: Q95 vs Q99 vs Croston Mean
# ============================================================

# 1. Build the Q99 policy using only pre-simulation history
policy_q99 = build_inventory_table(
    daily_demand=df,
    stock_codes=simulation_products,
    as_of_date=simulation_start,
    lead_time=3,
    review_period=7,
    service_level=0.99
)


# 2. Build an experimental Croston-mean policy
def croston_mean_forecast(history, alpha=0.1):
    """Estimate average daily demand for intermittent demand."""
    values = history.to_numpy(dtype=float)
    nonzero_positions = [
        i for i, value in enumerate(values) if value > 0
    ]

    if len(nonzero_positions) == 0:
        return 0.0

    first_position = nonzero_positions[0]
    demand_estimate = values[first_position]
    interval_estimate = first_position + 1

    previous_position = first_position

    for position in nonzero_positions[1:]:
        interval = position - previous_position

        demand_estimate = (
            alpha * values[position]
            + (1 - alpha) * demand_estimate
        )

        interval_estimate = (
            alpha * interval
            + (1 - alpha) * interval_estimate
        )

        previous_position = position

    return demand_estimate / interval_estimate


croston_records = []

protection_period = 3 + 7

for stock_code in simulation_products:
    history = prepare_product_history(
        daily_demand=df,
        stock_code=stock_code,
        as_of_date=simulation_start
    )

    if len(history) < 60:
        continue

    forecast_per_day = croston_mean_forecast(history)

    croston_records.append({
        "StockCode": stock_code,
        "Target_Stock": math.ceil(
            forecast_per_day * protection_period
        )
    })

policy_croston = pd.DataFrame(croston_records)


# ============================================================
# Fair comparison: identical initial inventory
# ============================================================

# Use Q95 target stock as the shared initial inventory.
initial_stock = (
    simulation_policy_table
    .set_index("StockCode")["Target_Stock"]
    .astype(int)
    .to_dict()
)

policies = {
    "Q95": simulation_policy_table,
    "Q99": policy_q99,
    "Croston_Mean": policy_croston
}

comparison_results = []
simulation_outputs = {}

for policy_name, policy_table in policies.items():

    result = simulate_inventory_policy(
        simulation_panel=simulation_panel,
        inventory_policy_table=policy_table,
        lead_time=3,
        review_period=7,
        initial_stock_by_product=initial_stock
    )

    simulation_outputs[policy_name] = result

    total_demand = result["Demand"].sum()
    fulfilled = result["Fulfilled_Demand"].sum()
    unmet = result["Unmet_Demand"].sum()
    order_count = (result["Order_Quantity"] > 0).sum()

    ending = (
        result.sort_values("Date")
        .groupby("StockCode")
        .tail(1)
    )

    comparison_results.append({
        "Policy": policy_name,
        "Total_Demand": total_demand,
        "Same_Day_Fulfilled": fulfilled,
        "Unmet_Demand": unmet,
        "Same_Day_Fill_Rate_%": (
            100 * fulfilled / total_demand
            if total_demand else 0
        ),
        "Average_On_Hand": result["On_Hand"].mean(),
        "Inventory_Unit_Days": result["On_Hand"].sum(),
        "Total_Units_Ordered": result["Order_Quantity"].sum(),
        "Number_of_Orders": order_count,
        "Ending_On_Hand": ending["On_Hand"].sum(),
        "Ending_On_Order": ending["On_Order"].sum(),
        "Ending_Backorders": ending["Backorders"].sum()
    })

fair_comparison = pd.DataFrame(comparison_results)

print("\n=== Fair Inventory Policy Comparison ===")
print(fair_comparison.round(2).to_string(index=False))


# ============================================================
# Improved Inventory Cost Model
# ============================================================

# Illustrative assumptions, not real business costs.
# Same initial stock (Q95 target) is used for all policies.
# Same-day unmet demand is not automatically treated as lost sales; backlog is tracked separately.
holding_cost_per_unit_day = 0.01
fixed_order_cost = 2.0
terminal_backorder_penalty = 1.0

# Sensitivity scenarios for delayed demand
delay_cost_scenarios = [0.01, 0.05, 0.10, 0.25, 0.50]

policy_metrics = {}

for policy_name, result in simulation_outputs.items():

    # Last row per product
    ending = (
        result.sort_values("Date")
        .groupby("StockCode")
        .tail(1)
    )

    on_hand_unit_days = result["On_Hand"].sum()
    on_order_unit_days = result["On_Order"].sum()
    backorder_unit_days = result["Backorders"].sum()

    ending_backorders = ending["Backorders"].sum()
    number_of_orders = result["Order_Quantity"].gt(0).sum()

    policy_metrics[policy_name] = {
        "On_Hand_Unit_Days": on_hand_unit_days,
        "On_Order_Unit_Days": on_order_unit_days,
        "Backorder_Unit_Days": backorder_unit_days,
        "Ending_Backorders": ending_backorders,
        "Number_of_Orders": number_of_orders,
        "Same_Day_Fulfilled": result["Fulfilled_Demand"].sum(),
        "Total_Demand": result["Demand"].sum(),
        "Same_Day_Unmet": result["Unmet_Demand"].sum()
    }


cost_rows = []

for delay_cost in delay_cost_scenarios:

    for policy_name, values in policy_metrics.items():

        # Carrying cost includes on-hand and on-order inventory.
        # This assumes both are subject to the same holding rate.
        holding_cost = (
            values["On_Hand_Unit_Days"]
            + values["On_Order_Unit_Days"]
        ) * holding_cost_per_unit_day

        # Cost of keeping demand waiting over time
        delay_cost_total = (
            values["Backorder_Unit_Days"] * delay_cost
        )

        # Additional penalty for unresolved backlog at the end
        terminal_cost = (
            values["Ending_Backorders"]
            * terminal_backorder_penalty
        )

        ordering_cost = (
            values["Number_of_Orders"] * fixed_order_cost
        )

        total_cost = (
            holding_cost
            + delay_cost_total
            + terminal_cost
            + ordering_cost
        )

        cost_rows.append({
            "Delay_Cost_Per_Unit_Day": delay_cost,
            "Policy": policy_name,
            "Holding_Cost": holding_cost,
            "Delay_Cost": delay_cost_total,
            "Terminal_Backorder_Cost": terminal_cost,
            "Ordering_Cost": ordering_cost,
            "Total_Cost": total_cost
        })


improved_cost_df = pd.DataFrame(cost_rows)

print("\n=== Improved Cost Sensitivity Analysis ===")
print(
    improved_cost_df
    .round(2)
    .sort_values(
        ["Delay_Cost_Per_Unit_Day", "Total_Cost"]
    )
    .to_string(index=False)
)


print("\n=== Lowest-Cost Policy by Scenario ===")

for delay_cost in delay_cost_scenarios:

    scenario = improved_cost_df[
        improved_cost_df["Delay_Cost_Per_Unit_Day"]
        == delay_cost
    ]

    best = scenario.loc[scenario["Total_Cost"].idxmin()]

    print(
        f"Delay cost={delay_cost:.2f}: "
        f"{best['Policy']} "
        f"(total cost={best['Total_Cost']:.2f})"
    )


# ------------------------------------------------------------
# Operational metrics for context
# ------------------------------------------------------------

operational_rows = []

for policy_name, values in policy_metrics.items():

    demand = values["Total_Demand"]
    fulfilled = values["Same_Day_Fulfilled"]

    operational_rows.append({
        "Policy": policy_name,
        "Same_Day_Fill_Rate_%": (
            100 * fulfilled / demand if demand else 0
        ),
        "Same_Day_Unmet_Units": values["Same_Day_Unmet"],
        "Backorder_Unit_Days": values["Backorder_Unit_Days"],
        "Ending_Backorders": values["Ending_Backorders"],
        "On_Hand_Unit_Days": values["On_Hand_Unit_Days"],
        "On_Order_Unit_Days": values["On_Order_Unit_Days"]
    })

operational_df = pd.DataFrame(operational_rows)

print("\n=== Operational Metrics ===")
print(operational_df.round(2).to_string(index=False))


# ============================================================
# Inventory Flow Accounting Audit
# ============================================================

import numpy as np
import pandas as pd

audit_summary = []
audit_missing_columns = []

for policy_name, result in simulation_outputs.items():

    df_audit = result.copy()
    df_audit["Date"] = pd.to_datetime(df_audit["Date"])
    df_audit = df_audit.sort_values(
        ["StockCode", "Date"]
    ).reset_index(drop=True)

    # Required fields for accounting checks
    required_columns = [
        "StockCode", "Date", "Demand", "Receipts",
        "Fulfilled_Demand", "Unmet_Demand",
        "Backorders_Fulfilled", "Backorders",
        "On_Hand", "On_Order", "Order_Quantity"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df_audit.columns
    ]

    if missing_columns:
        print(f"\n{policy_name}: missing columns {missing_columns}")
        audit_missing_columns.append((policy_name, missing_columns))
        continue

    # Check one row per product per date
    duplicate_rows = df_audit.duplicated(
        ["StockCode", "Date"]
    ).sum()

    # Check nonnegative quantities
    quantity_columns = [
        "Demand", "Receipts", "Fulfilled_Demand",
        "Unmet_Demand", "Backorders_Fulfilled",
        "Backorders", "On_Hand", "On_Order",
        "Order_Quantity"
    ]

    negative_values = (
        df_audit[quantity_columns] < 0
    ).any(axis=1).sum()

    # Check daily inventory flow and cumulative balances
    daily_stock_errors = 0
    daily_order_errors = 0
    daily_backorder_errors = 0
    demand_balance_errors_total = 0

    for stock_code, group in df_audit.groupby("StockCode"):
        group = group.sort_values("Date")

        opening_stock = initial_stock[stock_code]

        # End-of-day on-hand balance:
        # previous stock + receipts - fulfilled demand
        expected_on_hand = (
            opening_stock
            + group["Receipts"].cumsum()
            - group["Fulfilled_Demand"].cumsum()
            - group["Backorders_Fulfilled"].cumsum()
        )

        daily_stock_errors += int(
            (~np.isclose(
                group["On_Hand"].to_numpy(),
                expected_on_hand.to_numpy()
            )).sum()
        )

        # Outstanding orders:
        # cumulative orders - cumulative receipts
        expected_on_order = (
            group["Order_Quantity"].cumsum()
            - group["Receipts"].cumsum()
        )

        daily_order_errors += int(
            (~np.isclose(
                group["On_Order"].to_numpy(),
                expected_on_order.to_numpy()
            )).sum()
        )

        # Backlog balance:
        # unmet demand - previously fulfilled backorders
        expected_backorders = (
            group["Unmet_Demand"].cumsum()
            - group["Backorders_Fulfilled"].cumsum()
        )

        daily_backorder_errors += int(
            (~np.isclose(
                group["Backorders"].to_numpy(),
                expected_backorders.to_numpy()
            )).sum()
        )

        # Demand accounting:
        # fulfilled demand + unmet demand = total demand
        demand_balance_errors = int(
            (~np.isclose(
                (
                    group["Fulfilled_Demand"]
                    + group["Unmet_Demand"]
                ).to_numpy(),
                group["Demand"].to_numpy()
            )).sum()
        )

        demand_balance_errors_total += demand_balance_errors

    # Final balances by product
    ending = (
        df_audit.groupby("StockCode")
        .tail(1)
    )

    audit_summary.append({
        "Policy": policy_name,
        "Rows": len(df_audit),
        "Products": df_audit["StockCode"].nunique(),
        "Duplicate_Product_Days": duplicate_rows,
        "Negative_Quantity_Rows": negative_values,
        "On_Hand_Balance_Errors": daily_stock_errors,
        "On_Order_Balance_Errors": daily_order_errors,
        "Backorder_Balance_Errors": daily_backorder_errors,
        "Demand_Balance_Errors": demand_balance_errors_total,
        "Ending_On_Hand": ending["On_Hand"].sum(),
        "Ending_On_Order": ending["On_Order"].sum(),
        "Ending_Backorders": ending["Backorders"].sum()
    })

audit_df = pd.DataFrame(audit_summary)

print("\n=== Inventory Flow Audit ===")
print(audit_df.to_string(index=False))

error_columns = [
    "Duplicate_Product_Days",
    "Negative_Quantity_Rows",
    "On_Hand_Balance_Errors",
    "On_Order_Balance_Errors",
    "Backorder_Balance_Errors",
    "Demand_Balance_Errors"
]

if not audit_df.empty:
    all_passed = (
        not audit_missing_columns
        and audit_df[error_columns].to_numpy().sum() == 0
        and set(audit_df["Policy"]) == set(simulation_outputs)
    )
    print("\nAUDIT STATUS:", "PASS" if all_passed else "CHECK FAILURES")
else:
    all_passed = False
    print("\nAUDIT STATUS: CHECK FAILURES (no policies were audited)")

if audit_missing_columns:
    print("Policies skipped due to missing audit columns:", audit_missing_columns)

if not all_passed:
    raise AssertionError("Inventory accounting audit failed; inspect the audit table above.")


# ============================================================
# Product-Level Policy Analysis
# ============================================================

product_policy_metrics = []

for policy_name, output in simulation_outputs.items():
    for stock_code, group in output.groupby("StockCode"):

        total_demand = group["Demand"].sum()
        same_day_unmet = group["Unmet_Demand"].sum()

        fill_rate = (
            100 * (total_demand - same_day_unmet) / total_demand
            if total_demand > 0
            else np.nan
        )

        product_policy_metrics.append({
            "Policy": policy_name,
            "StockCode": stock_code,
            "Total_Demand": total_demand,
            "Same_Day_Unmet_Units": same_day_unmet,
            "Same_Day_Fill_Rate_%": round(fill_rate, 2),
            "Backorder_Unit_Days": group["Backorders"].sum(),
            "On_Hand_Unit_Days": group["On_Hand"].sum(),
            "On_Order_Unit_Days": group["On_Order"].sum(),
            "Number_of_Orders": (group["Order_Quantity"] > 0).sum()
        })

product_policy_metrics = pd.DataFrame(product_policy_metrics)

print("\n=== Product-Level Policy Analysis ===")
print(
    product_policy_metrics.sort_values(
        ["StockCode", "Policy"]
    ).to_string(index=False)
)


# ============================================================
# Product-Level Cost Analysis
# ============================================================

holding_cost_per_unit_day = 0.01
fixed_order_cost = 2.0
terminal_backorder_penalty = 1.0
delay_cost_per_unit_day = 0.10

product_cost_rows = []

for policy_name, output in simulation_outputs.items():
    for stock_code, group in output.groupby("StockCode"):

        holding_cost = (
            group["On_Hand"].sum()
            + group["On_Order"].sum()
        ) * holding_cost_per_unit_day

        delay_cost = (
            group["Backorders"].sum()
            * delay_cost_per_unit_day
        )

        ending_backorders = group["Backorders"].iloc[-1]

        terminal_cost = (
            ending_backorders
            * terminal_backorder_penalty
        )

        number_of_orders = (
            group["Order_Quantity"] > 0
        ).sum()

        ordering_cost = number_of_orders * fixed_order_cost

        total_cost = (
            holding_cost
            + delay_cost
            + terminal_cost
            + ordering_cost
        )

        total_demand = group["Demand"].sum()
        unmet_units = group["Unmet_Demand"].sum()

        fill_rate = (
            100 * (total_demand - unmet_units) / total_demand
            if total_demand > 0
            else np.nan
        )

        product_cost_rows.append({
            "Policy": policy_name,
            "StockCode": stock_code,
            "Total_Demand": total_demand,
            "Same_Day_Fill_Rate_%": round(fill_rate, 2),
            "Holding_Cost": round(holding_cost, 2),
            "Delay_Cost": round(delay_cost, 2),
            "Terminal_Cost": round(terminal_cost, 2),
            "Ordering_Cost": round(ordering_cost, 2),
            "Total_Cost": round(total_cost, 2)
        })

product_cost_analysis = pd.DataFrame(product_cost_rows)

print("\n=== Product-Level Cost Analysis ===")
print(
    product_cost_analysis.sort_values(
        ["StockCode", "Policy"]
    ).to_string(index=False)
)


# ============================================================
# Lowest-Cost Policy by Product
# ============================================================

eligible_products = product_cost_analysis[
    product_cost_analysis["Total_Demand"] > 0
].copy()

best_policy_by_product = (
    eligible_products
    .sort_values(["StockCode", "Total_Cost"])
    .drop_duplicates("StockCode")
)

print("\n=== Lowest-Cost Policy by Product ===")
print(
    best_policy_by_product[
        [
            "StockCode",
            "Policy",
            "Total_Cost",
            "Same_Day_Fill_Rate_%",
            "Holding_Cost",
            "Delay_Cost",
            "Ordering_Cost"
        ]
    ].to_string(index=False)
)


# ============================================================
# Service-Constrained Policy Selection
# ============================================================

service_thresholds = [80, 90, 95]
selection_rows = []

eligible_products = product_cost_analysis[
    product_cost_analysis["Total_Demand"] > 0
].copy()

for stock_code, group in eligible_products.groupby("StockCode"):

    for threshold in service_thresholds:

        feasible = group[
            group["Same_Day_Fill_Rate_%"] >= threshold
        ]

        if feasible.empty:
            selection_rows.append({
                "StockCode": stock_code,
                "Required_Same_Day_Fill_Rate_%": threshold,
                "Selected_Policy": "No Feasible Policy",
                "Total_Cost": np.nan,
                "Actual_Same_Day_Fill_Rate_%": np.nan
            })
            continue

        best = feasible.sort_values(
            "Total_Cost"
        ).iloc[0]

        selection_rows.append({
            "StockCode": stock_code,
            "Required_Same_Day_Fill_Rate_%": threshold,
            "Selected_Policy": best["Policy"],
            "Total_Cost": best["Total_Cost"],
            "Actual_Same_Day_Fill_Rate_%": best["Same_Day_Fill_Rate_%"]
        })

service_constrained_results = pd.DataFrame(selection_rows)

print("\n=== Service-Constrained Policy Selection ===")
print(
    service_constrained_results.to_string(index=False)
)

# ============================================================
# FINAL PHASE 6 SUMMARY
# ============================================================

reference_delay_cost = 0.10
reference_costs = improved_cost_df.loc[
    improved_cost_df["Delay_Cost_Per_Unit_Day"] == reference_delay_cost,
    ["Policy", "Total_Cost"]
].rename(columns={"Total_Cost": "Illustrative_Total_Cost"})

final_summary = fair_comparison[
    [
        "Policy",
        "Same_Day_Fill_Rate_%",
        "Unmet_Demand",
        "Inventory_Unit_Days",
        "Number_of_Orders",
        "Ending_Backorders"
    ]
].merge(reference_costs, on="Policy", how="left")

final_summary = final_summary.rename(columns={
    "Same_Day_Fill_Rate_%": "Same_Day_Fill_Rate_%",
    "Unmet_Demand": "Same_Day_Unmet_Units",
    "Inventory_Unit_Days": "On_Hand_Unit_Days"
})

print("\n=== FINAL PHASE 6 SUMMARY ===")
print(final_summary.round(2).to_string(index=False))
print("\nCost assumptions (illustrative, not observed business costs):")
print(f"- Holding cost: {holding_cost_per_unit_day:.2f} per unit-day, including on-hand and on-order inventory")
print(f"- Fixed ordering cost: {fixed_order_cost:.2f} per order event")
print(f"- Terminal backorder penalty: {terminal_backorder_penalty:.2f} per ending backorder unit")
print(f"- Reference delay cost in final table: {reference_delay_cost:.2f} per backorder unit-day")
print("- All policies use the same initial stock, based on Q95 target stock.")
print("- Same-day unmet demand is tracked as backorders; it is not automatically treated as lost sales.")
print("- Policy choice depends on the service level required and the cost assumptions.")

# ============================================================
# PHASE 7 — ROLLING-ORIGIN INVENTORY POLICY BACKTEST
# ============================================================

print("\n" + "=" * 70)
print("ROLLING-ORIGIN INVENTORY POLICY BACKTEST")
print("=" * 70)

BACKTEST_WINDOW = 28
BACKTEST_LEAD_TIME = 3
BACKTEST_REVIEW_PERIOD = 7
BACKTEST_MIN_HISTORY = 60

# Prepare simulation data
backtest_panel = simulation_panel.copy()
backtest_panel["Date"] = pd.to_datetime(backtest_panel["Date"])
backtest_panel["StockCode"] = backtest_panel["StockCode"].astype(str)

backtest_dates = pd.DatetimeIndex(
    sorted(backtest_panel["Date"].unique())
)

backtest_products = [
    str(product) for product in simulation_products_with_policy
]

if len(backtest_dates) == 0:
    raise ValueError("No dates available for inventory backtesting.")

if not backtest_dates.equals(
    pd.date_range(
        backtest_dates.min(),
        backtest_dates.max(),
        freq="D"
    )
):
    raise ValueError("Backtest requires a complete daily calendar.")

backtest_fold_results = []

for fold_number, start_index in enumerate(
    range(0, len(backtest_dates), BACKTEST_WINDOW),
    start=1
):
    fold_dates = backtest_dates[
        start_index:start_index + BACKTEST_WINDOW
    ]

    if len(fold_dates) < BACKTEST_WINDOW:
        print(
            f"Skipping incomplete fold {fold_number}: "
            f"{len(fold_dates)} days."
        )
        continue

    fold_start = pd.Timestamp(fold_dates[0])
    fold_end = pd.Timestamp(fold_dates[-1])

    print(
        f"\nFold {fold_number}: "
        f"{fold_start.date()} to {fold_end.date()}"
    )

    # Build Q95 and Q99 policies using historical data only.
    fold_q95 = build_inventory_table(
        daily_demand=df,
        stock_codes=backtest_products,
        as_of_date=fold_start,
        lead_time=BACKTEST_LEAD_TIME,
        review_period=BACKTEST_REVIEW_PERIOD,
        service_level=0.95,
        min_history_days=BACKTEST_MIN_HISTORY
    )

    fold_q99 = build_inventory_table(
        daily_demand=df,
        stock_codes=backtest_products,
        as_of_date=fold_start,
        lead_time=BACKTEST_LEAD_TIME,
        review_period=BACKTEST_REVIEW_PERIOD,
        service_level=0.99,
        min_history_days=BACKTEST_MIN_HISTORY
    )

    # Build Croston policy from the same historical cutoff.
    croston_records = []

    protection_period = (
        BACKTEST_LEAD_TIME + BACKTEST_REVIEW_PERIOD
    )

    for product in backtest_products:
        history = prepare_product_history(
            daily_demand=df,
            stock_code=product,
            as_of_date=fold_start
        )

        if (
            len(history) < BACKTEST_MIN_HISTORY
            or history.isna().any()
            or (history < 0).any()
        ):
            continue

        daily_forecast = croston_mean_forecast(history)

        croston_records.append({
            "StockCode": product,
            "Target_Stock": math.ceil(
                daily_forecast * protection_period
            )
        })

    fold_croston = pd.DataFrame(
        croston_records,
        columns=["StockCode", "Target_Stock"]
    )

    fold_policies = {
        "Q95": fold_q95,
        "Q99": fold_q99,
        "Croston_Mean": fold_croston
    }

    # Normalize identifiers.
    for policy_table in fold_policies.values():
        if not policy_table.empty:
            policy_table["StockCode"] = (
                policy_table["StockCode"].astype(str)
            )

    # Compare the policies on exactly the same products.
    common_products = set(backtest_products)

    for policy_table in fold_policies.values():
        common_products &= set(policy_table["StockCode"])

    common_products = sorted(common_products)

    if not common_products:
        print("Skipping fold: no products shared by all policies.")
        continue

    fold_panel = backtest_panel.loc[
        backtest_panel["Date"].isin(fold_dates)
        & backtest_panel["StockCode"].isin(common_products)
    ].copy()

    # Use the Q95 target as common starting inventory for fairness.
    q95_targets = (
        fold_q95.assign(
            StockCode=fold_q95["StockCode"].astype(str)
        )
        .set_index("StockCode")["Target_Stock"]
    )

    shared_initial_stock = {
        product: int(math.ceil(float(q95_targets.loc[product])))
        for product in common_products
    }

    for policy_name, policy_table in fold_policies.items():
        policy_table = policy_table.loc[
            policy_table["StockCode"].isin(common_products)
        ].copy()

        result = simulate_inventory_policy(
            simulation_panel=fold_panel,
            inventory_policy_table=policy_table,
            lead_time=BACKTEST_LEAD_TIME,
            review_period=BACKTEST_REVIEW_PERIOD,
            initial_stock_by_product=shared_initial_stock
        )

        total_demand = result["Demand"].sum()
        fulfilled = result["Fulfilled_Demand"].sum()

        ending = (
            result.sort_values(["StockCode", "Date"])
            .groupby("StockCode")
            .tail(1)
        )

        backtest_fold_results.append({
            "Fold": fold_number,
            "Start_Date": fold_start,
            "End_Date": fold_end,
            "Policy": policy_name,
            "Products": result["StockCode"].nunique(),
            "Total_Demand": total_demand,
            "Same_Day_Fulfilled": fulfilled,
            "Unmet_Demand": result["Unmet_Demand"].sum(),
            "Same_Day_Fill_Rate_%": (
                100 * fulfilled / total_demand
                if total_demand > 0 else np.nan
            ),
            "On_Hand_Unit_Days": result["On_Hand"].sum(),
            "On_Order_Unit_Days": result["On_Order"].sum(),
            "Backorder_Unit_Days": result["Backorders"].sum(),
            "Units_Ordered": result["Order_Quantity"].sum(),
            "Number_of_Orders": int(
                result["Order_Quantity"].gt(0).sum()
            ),
            "Ending_Backorders": ending["Backorders"].sum()
        })

        fill_rate = (
            100 * fulfilled / total_demand
            if total_demand > 0 else float("nan")
        )

        print(
            f"  {policy_name}: "
            f"fill rate = {fill_rate:.2f}%"
        )

# Validate and summarize results.
if not backtest_fold_results:
    raise ValueError(
        "No backtest results were generated. "
        "Check product history and policy eligibility."
    )

rolling_backtest_df = pd.DataFrame(backtest_fold_results)

print("\n=== Results by Fold ===")
print(
    rolling_backtest_df[
        [
            "Fold",
            "Start_Date",
            "End_Date",
            "Policy",
            "Products",
            "Same_Day_Fill_Rate_%",
            "Unmet_Demand",
            "On_Hand_Unit_Days",
            "Units_Ordered"
        ]
        ].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

# Aggregate demand-weighted metrics across folds.
rolling_backtest_summary = (
    rolling_backtest_df
    .groupby("Policy")
    .agg(
        Folds=("Fold", "nunique"),
        Total_Demand=("Total_Demand", "sum"),
        Same_Day_Fulfilled=("Same_Day_Fulfilled", "sum"),
        Unmet_Demand=("Unmet_Demand", "sum"),
        On_Hand_Unit_Days=("On_Hand_Unit_Days", "sum"),
        On_Order_Unit_Days=("On_Order_Unit_Days", "sum"),
        Backorder_Unit_Days=("Backorder_Unit_Days", "sum"),
        Units_Ordered=("Units_Ordered", "sum"),
        Number_of_Orders=("Number_of_Orders", "sum"),
        Sum_of_Fold_Ending_Backorders=("Ending_Backorders", "sum")
    )
    .reset_index()
)

rolling_backtest_summary["Same_Day_Fill_Rate_%"] = np.where(
    rolling_backtest_summary["Total_Demand"] > 0,
    (
        100
        * rolling_backtest_summary["Same_Day_Fulfilled"]
        / rolling_backtest_summary["Total_Demand"]
    ),
    np.nan
)

# Average on-hand stock per product-day observation.
observation_counts = (
    rolling_backtest_df.assign(
        Observations=lambda x: (
            x["Products"]
            * ((x["End_Date"] - x["Start_Date"]).dt.days + 1)
        )
    )
    .groupby("Policy")["Observations"]
    .sum()
)

rolling_backtest_summary["Average_On_Hand"] = (
    rolling_backtest_summary["On_Hand_Unit_Days"]
    / rolling_backtest_summary["Policy"].map(observation_counts)
)

print("\n=== Rolling-Origin Backtest Summary ===")
print(
    rolling_backtest_summary[
        [
            "Policy",
            "Folds",
            "Total_Demand",
            "Same_Day_Fill_Rate_%",
            "Unmet_Demand",
            "Average_On_Hand",
            "Units_Ordered",
            "Number_of_Orders",
            "Sum_of_Fold_Ending_Backorders"
        ]
    ].round(2).to_string(index=False)
)

# Export results.
rolling_backtest_df.to_csv(
    "rolling_backtest_by_fold.csv",
    index=False
)

rolling_backtest_summary.to_csv(
    "rolling_backtest_summary.csv",
    index=False
)

print("\nBacktest files saved:")
print("- rolling_backtest_by_fold.csv")
print("- rolling_backtest_summary.csv")
print("\nRolling-origin inventory backtest completed.")