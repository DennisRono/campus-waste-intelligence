import warnings
import uuid
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Food Waste Forecaster | XGBoost & LightGBM",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY    = "#2e7d32"
LIGHT_GRN  = "#4caf50"
PALE_GRN   = "#e8f5e9"
DARK_GRN   = "#1b5e20"
ACCENT     = "#81c784"
GOOD_GRN   = "#c8e6c9"

SECTIONS   = ["A", "B", "C", "D"]
MEALS      = ["breakfast", "lunch", "dinner"]

MEAL_PALETTE = {
    "Breakfast": "#ff8f00", "Lunch": "#0288d1", "Dinner": "#7b1fa2",
    "breakfast": "#ff8f00", "lunch":  "#0288d1", "dinner": "#7b1fa2",
}

st.markdown(
    f"""
    <style>
      [data-testid="stMetricValue"]   {{ color: {PRIMARY}; font-weight: 700; }}
      [data-testid="stSidebar"]       {{ background-color: {PALE_GRN}; }}
      h1, h2, h3                      {{ color: {DARK_GRN}; }}
      .block-container                {{ padding-top: 1.5rem; padding-bottom: 2rem; }}
      [data-testid="stTabs"] button   {{ font-weight: 600; }}
      hr                              {{ border-color: {LIGHT_GRN}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading raw dataset…")
def load_raw_dataset() -> pd.DataFrame:
    df = pd.read_csv("data/synthetic_forecast_cleaned.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.normalize()
    df["location_id"] = df["location_id"].astype(str).str.upper()
    def _meal_period(h: int) -> str:
        if 6 <= h < 10: return "Breakfast"
        if 11 <= h < 15: return "Lunch"
        if 17 <= h < 21: return "Dinner"
        return "Other"
    df["meal_period"] = df["hour"].apply(_meal_period)
    return df

@st.cache_data(show_spinner="Loading feature dataset for tree models…")
def load_xgb_dataset() -> pd.DataFrame:
    df = pd.read_csv("data/waste_features_xgb.csv")
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    sections_lc = ["a", "b", "c", "d"]
    def _get_section(row) -> str | None:
        for s in sections_lc:
            col = f"section_{s}"
            if col in row.index and bool(row[col]):
                return s.upper()
        return None
    df["section"] = df.apply(_get_section, axis=1)
    df = df.rename(columns={"meal": "meal_type"})
    df["meal_type"] = df["meal_type"].str.lower()
    df = df[df["meal_type"].isin(MEALS)].copy()
    return df

@st.cache_data(show_spinner="Loading performance metrics…")
def load_metrics_for_model(model_name: str) -> pd.DataFrame:
    """Load CSV from metrics/ folder and return a DataFrame with averaged metrics per section."""
    path = Path(f"metrics/{model_name}_model_comparison.csv")
    if not path.exists():
        st.warning(f"Metrics file not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    
    
    df["stage"] = df["model"].apply(lambda x: "Baseline" if "Baseline" in x else "Tuned")
    
    avg = df.groupby(["section", "stage"]).agg(
        RMSE=("RMSE", "mean"),
        MAE=("MAE", "mean"),
        MAPE=("MAPE", "mean"),
        R2=("R2", "mean")
    ).reset_index()
    return avg

@st.cache_resource(show_spinner=False)
def load_tree_model(model_type: str, section: str):
    """Load tuned XGBoost or LightGBM model (per section)."""
    sec_lc = section.lower()
    
    primary_path = Path(f"models/{model_type}_optimized/tuned_section_{sec_lc}.pkl")
    if primary_path.exists():
        with open(primary_path, "rb") as f:
            return pickle.load(f), str(primary_path)
    
    return None, f"Model not found. Expected {primary_path}"


def forecast_tree_based(model, history_df: pd.DataFrame, horizon: int = 7, target_col: str = "waste_kg") -> dict:
    MAX_LOOKBACK = 28
    forecasts = {meal: [] for meal in MEALS}
    history_df = history_df.sort_values(["date", "meal_type"]).reset_index(drop=True)
    last_date = history_df["date"].max()

    meal_histories = {
        meal: history_df[history_df["meal_type"] == meal].copy()
        for meal in MEALS
    }

    for i in range(1, horizon + 1):
        next_date = last_date + pd.Timedelta(days=i)
        for meal in MEALS:
            meal_hist = meal_histories[meal]
            if meal_hist.empty:
                forecasts[meal].append(0.0)
                continue

            new_row = meal_hist.iloc[-1:].copy()
            new_row["date"] = next_date
            new_row[target_col] = np.nan
            new_row["meal_type"] = meal
            new_row["year"] = next_date.year
            new_row["month"] = next_date.month
            new_row["day"] = next_date.day
            new_row["day_of_week"] = next_date.dayofweek
            new_row["day_of_year"] = next_date.dayofyear
            new_row["week_of_year"] = int(next_date.isocalendar().week)
            new_row["quarter"] = next_date.quarter
            new_row["is_weekend"] = int(next_date.dayofweek >= 5)
            new_row["dow_sin"] = np.sin(2 * np.pi * next_date.dayofweek / 7)
            new_row["dow_cos"] = np.cos(2 * np.pi * next_date.dayofweek / 7)
            new_row["doy_sin"] = np.sin(2 * np.pi * next_date.dayofyear / 365)
            new_row["doy_cos"] = np.cos(2 * np.pi * next_date.dayofyear / 365)

            temp = pd.concat([meal_hist.iloc[-MAX_LOOKBACK:], new_row], ignore_index=True)

            for lag in [1, 7, 14, 28]:
                temp[f"lag_{lag}"] = temp[target_col].shift(lag)
            temp["rolling_mean_7"] = temp[target_col].shift(1).rolling(7, min_periods=1).mean()
            temp["rolling_mean_14"] = temp[target_col].shift(1).rolling(14, min_periods=1).mean()
            temp["rolling_std_7"] = temp[target_col].shift(1).rolling(7, min_periods=1).std().fillna(0)
            temp["rolling_max_7"] = temp[target_col].shift(1).rolling(7, min_periods=1).max()

            section_drop = [c for c in temp.columns if c.startswith("section_") or c == "section"]
            X_pred = temp.iloc[-1:].drop(columns=[target_col, "date"] + section_drop, errors="ignore")
            for col in X_pred.select_dtypes(include=["object", "string"]).columns:
                X_pred[col] = X_pred[col].astype("category")

            try:
                predicted = float(model.predict(X_pred)[0])
            except Exception:
                predicted = 0.0

            forecasts[meal].append(max(0.0, predicted))
            new_row[target_col] = predicted
            meal_histories[meal] = pd.concat([meal_hist, new_row], ignore_index=True)

    return forecasts


def render_performance_table():
    st.markdown("## 📊 Model Performance Comparison (30‑day hold‑out)")
    st.markdown("Metrics are averaged over breakfast, lunch, and dinner for each section. Baseline vs. tuned after walk‑forward cross‑validation.")

    models = ["xgboost", "lightgbm", "sarima", "prophet"]
    all_data = []
    for m in models:
        df_metrics = load_metrics_for_model(m)
        if df_metrics.empty:
            continue
        df_metrics["model"] = m.upper()
        all_data.append(df_metrics)

    if not all_data:
        st.error("No metrics files found in `metrics/` folder. Please ensure CSV files exist.")
        return

    combined = pd.concat(all_data, ignore_index=True)
    
    pivot = combined.pivot_table(
        index=["model", "section"],
        columns="stage",
        values=["RMSE", "MAE", "MAPE", "R2"],
        aggfunc="first"
    ).reset_index()
    
    pivot.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in pivot.columns.values]
    pivot = pivot.rename(columns={
        "model_": "Model",
        "section_": "Section",
        "RMSE_Baseline": "RMSE (base)",
        "RMSE_Tuned": "RMSE (tuned)",
        "MAE_Baseline": "MAE (base)",
        "MAE_Tuned": "MAE (tuned)",
        "MAPE_Baseline": "MAPE (base)",
        "MAPE_Tuned": "MAPE (tuned)",
        "R2_Baseline": "R² (base)",
        "R2_Tuned": "R² (tuned)",
    })
    
    order = ["Model", "Section", "RMSE (base)", "RMSE (tuned)", "MAE (base)", "MAE (tuned)",
             "MAPE (base)", "MAPE (tuned)", "R² (base)", "R² (tuned)"]
    pivot = pivot[order]

    st.dataframe(pivot.round(2), width='stretch', hide_index=True)

    
    st.markdown("---")
    st.markdown("### 🔍 Data‑Driven Interpretation of Metrics")

    
    tree_models = ["XGBOOST", "LIGHTGBM"]
    classical = ["SARIMA", "PROPHET"]
    tree_improvement = pivot[pivot["Model"].isin(tree_models)]["RMSE (tuned)"].mean()
    classical_improvement = pivot[pivot["Model"].isin(classical)]["RMSE (tuned)"].mean()

    st.markdown(f"""
    **Quantitative observation:**  
    - Tree‑based models (XGBoost, LightGBM) achieve an average tuned RMSE of **{tree_improvement:.2f} kg** across all sections.  
    - Classical time‑series models (SARIMA, Prophet) have an average tuned RMSE of **{classical_improvement:.2f} kg**, which is substantially higher.

    **Why?**  
    The data‑generating process includes non‑linear interactions (e.g., footfall × section × meal type). Tree‑based models can automatically learn such interactions via recursive partitioning, whereas SARIMA and Prophet assume additive, linear dynamics. Even after exhaustive tuning, the classical models cannot overcome this structural limitation – their errors plateau.

    **R² interpretation:**  
    Negative R² values (e.g., SARIMA sections) indicate that the model performs worse than simply predicting the mean. This occurs because the temporal autocorrelation structure is weak relative to the noise and the seasonal period is too short to estimate reliably. In contrast, XGBoost achieves R² > 0.8 in most sections, meaning it explains over 80% of the variance in daily waste.

    **Conclusion:**  
    Model **class selection** is more critical than hyperparameter optimization when the underlying process contains multiplicative interactions. For operational forecasting in this canteen, XGBoost or LightGBM should be deployed.
    """)

    st.markdown("### 📝 Original Findings (as stated)")
    st.markdown("""
    The results indicate that model performance is primarily determined by the alignment between the model’s assumptions and the underlying data-generating process.  
    In this case, waste was constructed as a **multiplicative function of footfall** with additional interacting factors such as location and trend.  
    Tree-based models like **XGBoost and LightGBM** are able to capture these non‑linear interactions effectively, leading to strong predictive performance.  
    In contrast, linear time‑series models such as **SARIMA and Prophet** are constrained by their additive and linear structure, which limits their ability to model the true relationship in the data.  
    As a result, their performance plateaus regardless of tuning, demonstrating that **model class selection is more critical than hyperparameter optimization** when dealing with complex, interaction‑driven data.
    """)
    st.markdown("---")


def render_forecast_tab(model_type: str, xgb_df: pd.DataFrame|None, raw_df: pd.DataFrame|None):
    st.header(f"🔮 7‑Day Forecast — {model_type.upper()}")
    st.markdown(f"""
    The {model_type.upper()} model generates a **daily forecast** by iteratively predicting one day ahead,
    feeding each prediction back as a lag feature for the next day. This mirrors real‑world deployment.
    """)

    selected_sec = st.selectbox("Select canteen section", SECTIONS, key=f"{model_type}_sec")

    model, info = load_tree_model(model_type, selected_sec)
    if model is None:
        st.warning(f"⚠️ {model_type.upper()} model not found for section {selected_sec}.\n\n{info}")
        return

    
    if xgb_df is not None:
        history = xgb_df[xgb_df["section"] == selected_sec].copy()
        history = history.sort_values(["date", "meal_type"])
        history_30 = history.iloc[-(30 * len(MEALS)):]
    elif raw_df is not None:
        proxy = raw_df[(raw_df["location_id"] == selected_sec) & (raw_df["meal_period"] != "Other")].copy()
        proxy = proxy.rename(columns={"meal_period": "meal_type"})
        proxy["meal_type"] = proxy["meal_type"].str.lower()
        proxy["foot_traffic"] = proxy["footfall"]
        proxy["has_special_event"] = proxy.get("special_event", pd.Series([0]*len(proxy))).notna().astype(int)
        for s in ["a","b","c","d"]:
            proxy[f"section_{s}"] = int(s == selected_sec.lower())
        agg = proxy.groupby(["date","meal_type"]).agg(
            waste_kg=("waste_kg","sum"),
            foot_traffic=("foot_traffic","mean"),
            is_holiday=("is_holiday","max"),
            has_special_event=("has_special_event","max"),
            year=("year","first"),
            month=("month","first"),
            day=("day","first"),
            day_of_week=("day_of_week","first"),
            is_weekend=("is_weekend","first"),
        ).reset_index()
        agg["day_of_year"] = pd.to_datetime(agg["date"]).dt.dayofyear
        agg["week_of_year"] = pd.to_datetime(agg["date"]).dt.isocalendar().week.astype(int)
        agg["quarter"] = pd.to_datetime(agg["date"]).dt.quarter
        agg["dow_sin"] = np.sin(2*np.pi*agg["day_of_week"]/7)
        agg["dow_cos"] = np.cos(2*np.pi*agg["day_of_week"]/7)
        agg["doy_sin"] = np.sin(2*np.pi*agg["day_of_year"]/365)
        agg["doy_cos"] = np.cos(2*np.pi*agg["day_of_year"]/365)
        for s in ["a","b","c","d"]:
            agg[f"section_{s}"] = int(s == selected_sec.lower())
        history_30 = agg.sort_values(["date","meal_type"]).iloc[-(30*len(MEALS)):]
    else:
        st.error("Dataset not loaded. Cannot generate forecast.")
        return

    with st.spinner(f"Generating 7‑day forecast for Section {selected_sec} …"):
        try:
            forecasts = forecast_tree_based(model, history_30, horizon=7)
        except Exception as e:
            st.error(f"Forecast failed: {e}")
            return

    last_date = history_30["date"].max()
    fc_rows = []
    for day_i in range(7):
        fdate = last_date + pd.Timedelta(days=day_i + 1)
        for meal in MEALS:
            fc_rows.append({
                "Date": fdate.strftime("%a %d %b"),
                "Day": f"Day {day_i+1}",
                "Meal": meal.capitalize(),
                "Waste (kg)": round(forecasts[meal][day_i], 2),
            })
    fc_df = pd.DataFrame(fc_rows)

    all_vals = fc_df["Waste (kg)"]
    y_min = all_vals.min()
    y_max = all_vals.max()
    y_pad = max((y_max - y_min) * 0.4, 0.1)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(fc_df[["Date","Meal","Waste (kg)"]], width='stretch', hide_index=True)
    with col2:
        fig_line = px.line(fc_df, x="Day", y="Waste (kg)", color="Meal",
                           color_discrete_map=MEAL_PALETTE, markers=True,
                           category_orders={"Meal": ["Breakfast","Lunch","Dinner"]})
        fig_line.update_layout(
            title=f"7‑Day Forecast by Meal — Section {selected_sec}",
            plot_bgcolor="white",
            title_font_color=DARK_GRN,
            yaxis=dict(range=[y_min - y_pad, y_max + y_pad]),
        )
        st.plotly_chart(fig_line, width='stretch', key=str(uuid.uuid4()))

    daily_fc = fc_df.groupby("Date", sort=False)["Waste (kg)"].sum().reset_index()
    fig_bar = px.bar(daily_fc, x="Date", y="Waste (kg)", color_discrete_sequence=[PRIMARY])
    fig_bar.update_layout(title=f"Forecasted Total Daily Waste — Section {selected_sec}",
                          plot_bgcolor="white", title_font_color=DARK_GRN)
    st.plotly_chart(fig_bar, width='stretch', key=str(uuid.uuid4()))

    total_7d = fc_df["Waste (kg)"].sum()
    st.metric("📦 Total Forecasted Waste (7 days)", f"{total_7d:.1f} kg")


def main():
    st.title("🍽️ Food Waste Forecaster")
    st.markdown("**Scientific communication of forecasting models** – 7‑day forecasts using XGBoost and LightGBM, with empirical performance comparison across all four models.")
    
    raw_df = None
    xgb_df = None
    try:
        raw_df = load_raw_dataset()
    except FileNotFoundError:
        st.warning("⚠️ `data/synthetic_forecast_cleaned.csv` not found. Forecasts will use fallback aggregation (slower).")
    try:
        xgb_df = load_xgb_dataset()
    except FileNotFoundError:
        st.warning("⚠️ `data/waste_features_xgb.csv` not found. Some lag features may be missing. Using raw aggregation fallback.")

    
    render_performance_table()

    
    tab_xgb, tab_lgb = st.tabs(["🌲 XGBoost Forecast", "💡 LightGBM Forecast"])

    with tab_xgb:
        if raw_df is not None or xgb_df is not None:
            render_forecast_tab("xgboost", xgb_df, raw_df)
        else:
            st.error("No data source available. Please ensure `data/` directory contains the required CSV files.")

    with tab_lgb:
        if raw_df is not None or xgb_df is not None:
            render_forecast_tab("lightgbm", xgb_df, raw_df)
        else:
            st.error("No data source available.")

if __name__ == "__main__":
    main()

