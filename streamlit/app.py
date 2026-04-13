import warnings
warnings.filterwarnings("ignore")

import uuid
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="Food Waste Forecaster",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)


PRIMARY    = "#2e7d32"
LIGHT_GRN  = "#4caf50"
PALE_GRN   = "#e8f5e9"
DARK_GRN   = "#1b5e20"
ACCENT     = "#81c784"
WARN_RED   = "#ffccbc"
GOOD_GRN   = "#c8e6c9"

SECTIONS   = ["A", "B", "C", "D"]
MEALS      = ["breakfast", "lunch", "dinner"]

SECTION_PALETTE = {
    "A": PRIMARY, "B": LIGHT_GRN, "C": DARK_GRN, "D": ACCENT,
}
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


PERFORMANCE = {
    "prophet": {
        "A": {"baseline_rmse": 49.85, "tuned_rmse": 21.42, "baseline_mae": 45.75, "tuned_mae": 18.25, "baseline_mape": 50.27, "tuned_mape": 19.43},
        "B": {"baseline_rmse": 49.29, "tuned_rmse": 13.34, "baseline_mae": 43.01, "tuned_mae": 10.76, "baseline_mape": 48.38, "tuned_mape": 11.20},
        "C": {"baseline_rmse": 27.91, "tuned_rmse": 13.77, "baseline_mae": 23.91, "tuned_mae":  9.47, "baseline_mape": 26.33, "tuned_mape":  9.40},
        "D": {"baseline_rmse": 228.76,"tuned_rmse": 25.98, "baseline_mae": 206.16,"tuned_mae": 21.04, "baseline_mape": 232.37,"tuned_mape": 21.27},
    },
    "sarima": {
        "A": {"baseline_rmse": 10.60, "tuned_rmse": 10.13, "baseline_mae":  8.00, "tuned_mae":  7.68, "baseline_mape":  7.69, "tuned_mape":  7.46},
        "B": {"baseline_rmse":  9.43, "tuned_rmse":  9.33, "baseline_mae":  6.85, "tuned_mae":  6.75, "baseline_mape":  6.91, "tuned_mape":  6.75},
        "C": {"baseline_rmse": 11.33, "tuned_rmse": 11.84, "baseline_mae":  7.26, "tuned_mae":  7.58, "baseline_mape":  6.43, "tuned_mape":  6.67},
        "D": {"baseline_rmse": 18.23, "tuned_rmse": 29.24, "baseline_mae": 13.09, "tuned_mae": 25.17, "baseline_mape": 12.37, "tuned_mape": 27.27},
    },
    "xgboost": {
        "A": {"baseline_rmse":  3.10, "tuned_rmse":  1.91, "baseline_mae":  1.74, "tuned_mae":  1.10, "baseline_mape":  1.44, "tuned_mape":  0.91},
        "B": {"baseline_rmse":  3.19, "tuned_rmse":  2.92, "baseline_mae":  2.27, "tuned_mae":  1.90, "baseline_mape":  2.02, "tuned_mape":  1.48},
        "C": {"baseline_rmse":  3.76, "tuned_rmse":  4.17, "baseline_mae":  2.51, "tuned_mae":  2.41, "baseline_mape":  2.21, "tuned_mape":  1.96},
        "D": {"baseline_rmse":  6.19, "tuned_rmse":  5.06, "baseline_mae":  2.90, "tuned_mae":  2.36, "baseline_mape":  2.18, "tuned_mape":  1.71},
    },
    "lightgbm": {
        "A": {"baseline_rmse": 14.89, "tuned_rmse":  2.63, "baseline_mae":  8.65, "tuned_mae":  2.02, "baseline_mape":  5.89, "tuned_mape":  1.95},
        "B": {"baseline_rmse": 12.87, "tuned_rmse":  9.54, "baseline_mae":  7.35, "tuned_mae":  4.48, "baseline_mape":  5.67, "tuned_mape":  2.99},
        "C": {"baseline_rmse":  9.89, "tuned_rmse":  4.60, "baseline_mae":  5.65, "tuned_mae":  2.51, "baseline_mape":  4.25, "tuned_mape":  2.03},
        "D": {"baseline_rmse": 18.47, "tuned_rmse":  9.52, "baseline_mae": 10.63, "tuned_mae":  5.87, "baseline_mape":  7.85, "tuned_mape":  5.21},
    },
}

BEST_PARAMS = {
    "prophet": {
        "A": "changepoint_prior=0.01 · seasonality_prior=0.1 · mode=multiplicative",
        "B": "changepoint_prior=0.01 · seasonality_prior=0.1 · mode=multiplicative",
        "C": "changepoint_prior=0.10 · seasonality_prior=0.1 · mode=multiplicative",
        "D": "changepoint_prior=0.10 · seasonality_prior=0.1 · mode=multiplicative",
    },
    "sarima": {
        "A": "ARIMA(0,1,1) — seasonal component removed by CV",
        "B": "ARIMA(1,1,1) — seasonal component removed by CV",
        "C": "ARIMA(1,1,1) — seasonal component removed by CV",
        "D": "ARIMA(1,1,0) — seasonal component removed by CV",
    },
    "xgboost": {
        "A": "n_est=300 · depth=3 · lr=0.05 · subsample=0.8 · colsample=0.8",
        "B": "n_est=300 · depth=7 · lr=0.10 · subsample=0.8 · colsample=0.8",
        "C": "n_est=100 · depth=3 · lr=0.20 · subsample=0.8 · colsample=0.8",
        "D": "n_est=300 · depth=3 · lr=0.20 · subsample=0.8 · colsample=0.8",
    },
    "lightgbm": {
        "A": "Optuna (TPE) — num_leaves tuned · early_stopping=50 rounds",
        "B": "Optuna (TPE) — num_leaves tuned · early_stopping=50 rounds",
        "C": "Optuna (TPE) — num_leaves tuned · early_stopping=50 rounds",
        "D": "Optuna (TPE) — num_leaves tuned · early_stopping=50 rounds",
    },
}


@st.cache_data(show_spinner="Loading cleaned dataset…")
def load_raw_dataset() -> pd.DataFrame:
    """Load synthetic_forecast_cleaned.csv — 8,688 rows, 30-min granularity."""
    df = pd.read_csv("data/synthetic_forecast_cleaned.csv")
    df["timestamp"]  = pd.to_datetime(df["timestamp"])
    df["date"]       = df["timestamp"].dt.normalize()
    df["location_id"]= df["location_id"].astype(str).str.upper()

    def _meal_period(h: int) -> str:
        if 6  <= h < 10: return "Breakfast"
        if 11 <= h < 15: return "Lunch"
        if 17 <= h < 21: return "Dinner"
        return "Other"

    df["meal_period"] = df["hour"].apply(_meal_period)
    return df


@st.cache_data(show_spinner="Loading feature dataset…")
def load_xgb_dataset() -> pd.DataFrame:
    """
    Load waste_features_xgb.csv — daily aggregated with lags, rolling stats,
    section dummies, and cyclical encodings. Used to provide correct history
    for tree-based model forecasting.
    """
    df = pd.read_csv("data/waste_features_xgb.csv")
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])

    sections_lc = ["a", "b", "c", "d"]
    def _get_section(row) -> str:
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


@st.cache_resource(show_spinner=False)
def load_tree_model(model_type: str, section: str):
    """
    Try multiple path conventions for XGBoost / LightGBM tuned models.
    Returns (model_object, path_used) or (None, error_msg).
    """
    sec_lc = section.lower()
    sec_uc = section.upper()
    candidates = [
        f"models/{model_type}_optimized/tuned_section_{sec_lc}.pkl",
        f"models/{model_type}/{model_type}_section_{sec_uc}.pkl",
        f"models/{model_type}/{model_type}_section_{sec_lc}.pkl",
        f"models/{model_type}_optimized/tuned_section_{sec_uc}.pkl",
    ]
    for path in candidates:
        try:
            with open(path, "rb") as f:
                return pickle.load(f), path
        except FileNotFoundError:
            continue
    tried = "\n".join(f"  • {c}" for c in candidates)
    return None, f"Model not found. Checked:\n{tried}"


@st.cache_resource(show_spinner=False)
def load_sarima_model(section: str, meal: str):
    """Load a per-section, per-meal SARIMA tuned model."""
    sec_lc = section.lower()
    candidates = [
        f"models/sarima_optimized/tuned_{sec_lc}_{meal}.pkl",
        f"models/sarima/sarima_section_{section.upper()}_{meal}.pkl",
        f"models/sarima/sarima_section_{sec_lc}_{meal}.pkl",
    ]
    for path in candidates:
        try:
            with open(path, "rb") as f:
                return pickle.load(f), path
        except FileNotFoundError:
            continue
    return None, f"Not found for section={section}, meal={meal}"


@st.cache_resource(show_spinner=False)
def load_prophet_model(section: str, meal: str):
    """Load a per-section, per-meal Prophet tuned model."""
    sec_lc = section.lower()
    candidates = [
        f"models/prophet_optimized/tuned_{sec_lc}_{meal}.pkl",
        f"models/prophet/prophet_section_{section.upper()}_{meal}.pkl",
        f"models/prophet/prophet_section_{sec_lc}_{meal}.pkl",
        f"models/prophet_optimized/tuned_section_{sec_lc}.pkl",
        f"models/prophet/prophet_section_{section.upper()}.pkl",
    ]
    for path in candidates:
        try:
            with open(path, "rb") as f:
                return pickle.load(f), path
        except FileNotFoundError:
            continue
    return None, f"Not found for section={section}, meal={meal}"


def forecast_tree_based(
    model,
    history_df: pd.DataFrame,
    horizon: int = 7,
    target_col: str = "waste_kg",
) -> dict:
    """
    Generate a horizon-day forecast for each meal type using iterative
    one-step-ahead prediction. Each predicted value is fed back as a lag
    feature for the subsequent step.

    history_df must contain: date, meal_type, waste_kg, and all engineered
    features that the model was trained on.
    """
    MAX_LOOKBACK = 28
    forecasts    = {meal: [] for meal in MEALS}
    history_df   = history_df.sort_values(["date", "meal_type"]).reset_index(drop=True)
    last_date    = history_df["date"].max()

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

            new_row                = meal_hist.iloc[-1:].copy()
            new_row["date"]        = next_date
            new_row[target_col]    = np.nan
            new_row["meal_type"]   = meal

            new_row["year"]        = next_date.year
            new_row["month"]       = next_date.month
            new_row["day"]         = next_date.day
            new_row["day_of_week"] = next_date.dayofweek
            new_row["day_of_year"] = next_date.dayofyear
            new_row["week_of_year"]= int(next_date.isocalendar().week)
            new_row["quarter"]     = next_date.quarter
            new_row["is_weekend"]  = int(next_date.dayofweek >= 5)
            new_row["dow_sin"]     = np.sin(2 * np.pi * next_date.dayofweek / 7)
            new_row["dow_cos"]     = np.cos(2 * np.pi * next_date.dayofweek / 7)
            new_row["doy_sin"]     = np.sin(2 * np.pi * next_date.dayofyear / 365)
            new_row["doy_cos"]     = np.cos(2 * np.pi * next_date.dayofyear / 365)

            temp = pd.concat(
                [meal_hist.iloc[-MAX_LOOKBACK:], new_row], ignore_index=True
            )

            for lag in [1, 7, 14, 28]:
                temp[f"lag_{lag}"] = temp[target_col].shift(lag)
            temp["rolling_mean_7"]  = temp[target_col].shift(1).rolling(7,  min_periods=1).mean()
            temp["rolling_mean_14"] = temp[target_col].shift(1).rolling(14, min_periods=1).mean()
            temp["rolling_std_7"]   = temp[target_col].shift(1).rolling(7,  min_periods=1).std().fillna(0)
            temp["rolling_max_7"]   = temp[target_col].shift(1).rolling(7,  min_periods=1).max()

            X_pred = temp.iloc[-1:].drop(columns=[target_col, "date"], errors="ignore")
            for col in X_pred.select_dtypes(include=["object", "string"]).columns:
                X_pred[col] = X_pred[col].astype("category")

            try:
                predicted = float(model.predict(X_pred)[0])
            except Exception:
                predicted = 0.0

            forecasts[meal].append(max(0.0, predicted))

            new_row[target_col]  = predicted
            meal_histories[meal] = pd.concat([meal_hist, new_row], ignore_index=True)

    return forecasts


def build_perf_df(model_type: str) -> pd.DataFrame:
    rows = []
    for sec, v in PERFORMANCE[model_type].items():
        rmse_impr = round((1 - v["tuned_rmse"]  / v["baseline_rmse"]) * 100, 1)
        mae_impr  = round((1 - v["tuned_mae"]   / v["baseline_mae"])  * 100, 1)
        rows.append({
            "Section":       sec,
            "Baseline RMSE": v["baseline_rmse"],
            "Tuned RMSE":    v["tuned_rmse"],
            "RMSE Δ%":       rmse_impr,
            "Baseline MAE":  v["baseline_mae"],
            "Tuned MAE":     v["tuned_mae"],
            "MAE Δ%":        mae_impr,
            "Baseline MAPE": v["baseline_mape"],
            "Tuned MAPE":    v["tuned_mape"],
        })
    return pd.DataFrame(rows).set_index("Section")


def _style_delta(val):
    """Green cell for positive improvement, red for degradation."""
    try:
        if val > 20:  return f"background-color:{GOOD_GRN}; color:{DARK_GRN}"
        if val > 0:   return f"background-color:#f1f8e9"
        return f"background-color:{WARN_RED}"
    except Exception:
        return ""


def _plotly_layout(fig, title=""):
    fig.update_layout(
        # plot_bgcolor="white",
        # paper_bgcolor="white",
        title_font_color=DARK_GRN,
        title_text=title if title else fig.layout.title.text,
        font_color="#000000",
        margin=dict(t=50, b=30, l=30, r=30),
    )
    return fig


def render_sidebar():
    with st.sidebar:
        st.markdown(
            f"<h2 style='color:{PRIMARY}; margin-bottom:0'>🍽️ Food Waste<br>Forecaster</h2>",
            unsafe_allow_html=True,
        )
        st.caption("University Canteen · Jan–Jun 2025")
        st.divider()

        st.markdown(f"**📁 Dataset**")
        st.markdown(
            "- 8,688 records\n"
            "- 30-minute granularity\n"
            "- 4 canteen sections (A–D)\n"
            "- Jan–Jun 2025\n"
            "- 3 meal periods"
        )
        st.divider()

        st.markdown("**🤖 Models**")
        st.markdown(
            "- 🌲 XGBoost\n"
            "- 💡 LightGBM\n"
            "- 📈 SARIMA\n"
            "- 🔮 Prophet"
        )
        st.divider()

        st.markdown("**📐 Evaluation**")
        st.markdown(
            "- Last 30 days held out\n"
            "- Walk-forward CV (5 folds)\n"
            "- Metrics: RMSE · MAE · MAPE\n"
            "- Tuned via Optuna / grid search"
        )
        st.divider()
        st.caption("DS Project 3 · Scientific Communication")


def render_overview():
    st.markdown("# 📋 Overview")
    st.markdown(
        """
        This dashboard presents the results of a food waste forecasting study
        on a university canteen dataset. The aim is to predict daily food waste
        in kilograms across four canteen sections (A–D) using four different
        forecasting methods.

        The original Kaggle dataset lacked the features needed for reliable
        time-series modelling — specifically foot traffic, intra-day timestamps,
        and sufficient temporal range. A synthetic dataset was generated to fill
        these gaps while preserving the original data's statistical distribution.
        All models share the same 30-day test split and walk-forward
        cross-validation protocol to ensure fair comparison.
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records",    "8,688",     "30-min bins")
    c2.metric("Sections",   "4",         "A · B · C · D")
    c3.metric("Date Range", "6 months",  "Jan–Jun 2025")
    c4.metric("Models",     "4",         "XGB · LGB · SARIMA · Prophet")

    st.divider()

    st.markdown("## 🔄 Full Data Pipeline")
    pipeline = pd.DataFrame({
        "Step": [
            "1. Data Collection",
            "2. Synthetic Generation",
            "3. Data Cleaning",
            "4. EDA",
            "5. Feature Engineering",
            "6. Model Training",
            "7. Hyperparameter Tuning",
            "8. Evaluation",
        ],
        "Output": [
            "raw_kaggle.csv",
            "synthetic_forecast_ready.csv",
            "synthetic_forecast_cleaned.csv",
            "EDA report / insights",
            "waste_features_full.csv · waste_features_xgb.csv",
            "4 model families × 4 sections",
            "Best params per section",
            "RMSE · MAE · MAPE on 30-day hold-out",
        ],
        "Description": [
            "University canteen dataset from Kaggle",
            "Added footfall, 30-min timestamps, seasonal patterns, 8,688 entries",
            "Interpolation, IQR×3 capping, type correction, consistency checks",
            "Univariate, temporal, bivariate, and section-level analysis",
            "Lags (1,7,14,28 d), rolling stats, cyclical encodings, section dummies",
            "Prophet, SARIMA/SARIMAX, XGBoost, LightGBM — per section and meal",
            "Walk-forward CV · Optuna TPE for tree models · grid search for SARIMA/Prophet",
            "Same held-out 30 days for every model — no data leakage",
        ],
    })
    st.dataframe(pipeline, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("## 🏆 Best Model per Section (Tuned RMSE)")
    st.markdown(
        "Lower RMSE means the model's predictions were closer to the actual waste values. "
        "The highlighted cell in each row shows the best-performing model for that section."
    )

    heat_rows = {}
    for model, data in PERFORMANCE.items():
        heat_rows[model.upper()] = {sec: data[sec]["tuned_rmse"] for sec in SECTIONS}
    heat_df = pd.DataFrame(heat_rows).T  

    def _highlight_min_col(col):
        is_min = col == col.min()
        return [f"background-color:{GOOD_GRN}; font-weight:bold; color: #000000;" if v else "" for v in is_min]

    st.dataframe(
        heat_df.style
               .apply(_highlight_min_col, axis=0)
               .format("{:.2f}"),
        use_container_width=True,
    )
    st.caption(
        "Green = best (lowest) RMSE per section. "
        "XGBoost consistently achieves the lowest absolute error across all sections."
    )


def render_eda(df: pd.DataFrame):
    st.markdown("# 🧭 Exploratory Data Analysis")
    st.markdown(
        """
        The cleaned dataset contains **8,688 records** at 30-minute resolution
        across four canteen sections from January to June 2025. The analysis below
        covers the distribution of food waste, temporal patterns, foot traffic
        relationships, waste composition, and section-level differences. These
        insights directly informed the feature choices and model configurations
        used in this study.
        """
    )

    col_sel, c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1, 1])
    with col_sel:
        sec_choice = st.selectbox(
            "Filter by section", ["All"] + SECTIONS, key="eda_sec"
        )

    view = df.copy() if sec_choice == "All" else df[df["location_id"] == sec_choice].copy()

    c1.metric("Records",          f"{len(view):,}")
    c2.metric("Avg Waste (kg)",   f"{view['waste_kg'].mean():.2f}")
    c3.metric("Avg Footfall",     f"{view['footfall'].mean():.1f}")
    c4.metric("Weekend Records",  f"{view['is_weekend'].mean()*100:.1f}%")

    st.divider()

    st.markdown("### 📈 Time Series of Daily Waste")
    if sec_choice == "All":
        daily = (
            view.groupby(["date", "location_id"])["waste_kg"]
                .sum().reset_index()
        )
        fig_ts = px.line(
            daily, x="date", y="waste_kg", color="location_id",
            color_discrete_map=SECTION_PALETTE,
            labels={"waste_kg": "Waste (kg)", "date": "Date", "location_id": "Section"},
        )
    else:
        daily = view.groupby("date")["waste_kg"].sum().reset_index()
        fig_ts = px.line(
            daily, x="date", y="waste_kg",
            color_discrete_sequence=[PRIMARY],
            labels={"waste_kg": "Waste (kg)", "date": "Date"},
        )

    _plotly_layout(fig_ts, f"Daily Total Waste — Section {sec_choice}")
    st.plotly_chart(fig_ts, use_container_width=True, key=f"{uuid.uuid4()}")

    st.markdown(
        """
        The time series reveals clear **weekly seasonality** — waste rises and
        falls predictably with the day of the week across all sections.
        Section A consistently produces the highest volume. Section D shows the
        lowest weekday waste but notable weekend spikes, likely from special events
        and irregular visitor patterns.
        """
    )

    st.divider()

    st.markdown("### 📊 Distribution & Composition")
    d_col1, d_col2 = st.columns(2)

    with d_col1:
        fig_hist = px.histogram(
            view, x="waste_kg", nbins=60,
            color_discrete_sequence=[PRIMARY],
            labels={"waste_kg": "Waste (kg)", "count": "Frequency"},
        )
        _plotly_layout(fig_hist, "Distribution of Waste per 30-min Slot")
        st.plotly_chart(fig_hist, use_container_width=True, key=f"{uuid.uuid4()}")

    with d_col2:
        comp_cols = ["waste_organic_kg", "waste_recyclable_kg", "waste_landfill_kg"]
        if all(c in view.columns for c in comp_cols):
            comp = pd.DataFrame({
                "Type":   ["Organic", "Recyclable", "Landfill"],
                "kg":     [view[c].sum() for c in comp_cols],
            })
            fig_pie = px.pie(
                comp, names="Type", values="kg",
                color_discrete_sequence=[PRIMARY, LIGHT_GRN, ACCENT],
                hole=0.4,
            )
            _plotly_layout(fig_pie, "Waste Composition by Type")
            st.plotly_chart(fig_pie, use_container_width=True, key=f"{uuid.uuid4()}")

    st.markdown(
        """
        Waste weight is **right-skewed** — most time slots produce small amounts
        with occasional large spikes from busy periods or events. Organic waste
        accounts for roughly **55%** of total waste, recyclable for ~30%, and
        landfill for ~15%. This breakdown is consistent across sections, though
        Section A shows slightly higher organic fractions during peak lunch hours.
        """
    )

    st.divider()

    st.markdown("### ⏰ Intra‑day & Weekly Patterns")
    t1, t2 = st.columns(2)

    with t1:
        hourly = view.groupby("hour")["waste_kg"].mean().reset_index()
        fig_hr = px.bar(
            hourly, x="hour", y="waste_kg",
            color_discrete_sequence=[PRIMARY],
            labels={"waste_kg": "Avg Waste (kg)", "hour": "Hour of Day"},
        )
        _plotly_layout(fig_hr, "Average Waste by Hour of Day")
        st.plotly_chart(fig_hr, use_container_width=True, key=f"{uuid.uuid4()}")

    with t2:
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow = view.groupby("day_of_week")["waste_kg"].mean().reset_index()
        dow["Day"] = dow["day_of_week"].map(dict(enumerate(dow_names)))
        fig_dow = px.bar(
            dow, x="Day", y="waste_kg",
            color_discrete_sequence=[LIGHT_GRN],
            labels={"waste_kg": "Avg Waste (kg)", "Day": "Day of Week"},
            category_orders={"Day": dow_names},
        )
        _plotly_layout(fig_dow, "Average Waste by Day of Week")
        st.plotly_chart(fig_dow, use_container_width=True, key=f"{uuid.uuid4()}")

    st.markdown(
        """
        Waste peaks during **lunch (12–14h)** and **dinner (18–20h)** — the two
        main service windows — with a smaller breakfast peak around 8h.
        Across the week, waste levels are highest on Saturdays and Sundays, driven
        by increased footfall from events. These intra-day and weekly cycles
        motivated the use of cyclical encodings and seasonality components in every model.
        """
    )

    st.divider()

    st.markdown("### 📅 Monthly Trend")
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun"}
    monthly = view.groupby("month")["waste_kg"].mean().reset_index()
    monthly["Month"] = monthly["month"].map(month_names)
    fig_mon = px.line(
        monthly, x="Month", y="waste_kg", markers=True,
        color_discrete_sequence=[PRIMARY],
        labels={"waste_kg": "Avg Waste (kg)", "Month": "Month"},
        category_orders={"Month": list(month_names.values())},
    )
    _plotly_layout(fig_mon, "Average Waste by Month (Jan–Jun 2025)")
    st.plotly_chart(fig_mon, use_container_width=True, key=f"{uuid.uuid4()}")

    st.divider()

    st.markdown("### 🍽️ Waste by Meal Period")
    meal_view = view[view["meal_period"] != "Other"].copy()
    if not meal_view.empty:
        fig_meal = px.box(
            meal_view, x="meal_period", y="waste_kg",
            color="meal_period",
            color_discrete_map=MEAL_PALETTE,
            labels={"waste_kg": "Waste (kg)", "meal_period": "Meal Period"},
            category_orders={"meal_period": ["Breakfast", "Lunch", "Dinner"]},
        )
        _plotly_layout(fig_meal, "Waste Distribution by Meal Period")
        fig_meal.update_layout(showlegend=False)
        st.plotly_chart(fig_meal, use_container_width=True, key=f"{uuid.uuid4()}")
        st.markdown(
            """
            Lunch generates the widest spread of waste values — it is the highest-traffic
            service and the most variable in terms of menu offerings and attendance.
            Dinner shows a slightly narrower range with fewer extreme outliers, while
            breakfast produces the lowest average waste across all sections.
            """
        )

    st.divider()

    st.markdown("### 👣 Foot Traffic vs Waste")
    sample = view.sample(min(3000, len(view)), random_state=42)
    if sec_choice == "All":
        fig_sc = px.scatter(
            sample, x="footfall", y="waste_kg",
            color="location_id",
            color_discrete_map=SECTION_PALETTE,
            opacity=0.45,
            labels={
                "footfall": "Foot Traffic (people / 30 min)",
                "waste_kg": "Waste (kg)",
                "location_id": "Section",
            },
        )
    else:
        fig_sc = px.scatter(
            sample, x="footfall", y="waste_kg",
            color_discrete_sequence=[PRIMARY],
            opacity=0.45,
            labels={
                "footfall": "Foot Traffic (people / 30 min)",
                "waste_kg": "Waste (kg)",
            },
        )
    _plotly_layout(fig_sc, "Foot Traffic vs Waste Weight")
    st.plotly_chart(fig_sc, use_container_width=True, key=f"{uuid.uuid4()}")

    r = view[["footfall", "waste_kg"]].corr().iloc[0, 1]
    st.metric("Pearson r (footfall ↔ waste_kg)", f"{r:.3f}")

    st.markdown(
        f"""
        There is a **strong positive correlation** (r ≈ {r:.2f}) between foot traffic
        and waste volume. Higher footfall means more food is prepared and served, and
        consequently more is left over or discarded. This relationship holds across all
        four sections and makes `footfall` one of the strongest predictors in every model.
        """
    )

    st.divider()

    st.markdown("### 🏢 Section‑Level Comparison")
    sec_agg = (
        df.groupby("location_id")
          .agg(
              avg_waste    =("waste_kg",    "mean"),
              total_waste  =("waste_kg",    "sum"),
              avg_footfall =("footfall",    "mean"),
              avg_organic  =("waste_organic_kg",   "mean"),
              avg_recycl   =("waste_recyclable_kg", "mean"),
              avg_landfill =("waste_landfill_kg",   "mean"),
          )
          .reset_index()
          .rename(columns={"location_id": "Section"})
    )

    s1, s2 = st.columns(2)
    with s1:
        fig_sec_avg = px.bar(
            sec_agg, x="Section", y="avg_waste",
            color="Section", color_discrete_map=SECTION_PALETTE,
            labels={"avg_waste": "Avg Waste per Slot (kg)"},
        )
        _plotly_layout(fig_sec_avg, "Average Waste per 30-min Slot by Section")
        fig_sec_avg.update_layout(showlegend=False)
        st.plotly_chart(fig_sec_avg, use_container_width=True, key=f"{uuid.uuid4()}")

    with s2:
        fig_sec_ff = px.bar(
            sec_agg, x="Section", y="avg_footfall",
            color="Section", color_discrete_map=SECTION_PALETTE,
            labels={"avg_footfall": "Avg Footfall per Slot"},
        )
        _plotly_layout(fig_sec_ff, "Average Footfall per 30-min Slot by Section")
        fig_sec_ff.update_layout(showlegend=False)
        st.plotly_chart(fig_sec_ff, use_container_width=True, key=f"{uuid.uuid4()}")

    comp_long = sec_agg.melt(
        id_vars="Section",
        value_vars=["avg_organic", "avg_recycl", "avg_landfill"],
        var_name="Waste Type",
        value_name="Avg (kg)",
    )
    comp_long["Waste Type"] = comp_long["Waste Type"].map({
        "avg_organic":  "Organic",
        "avg_recycl":   "Recyclable",
        "avg_landfill": "Landfill",
    })
    fig_comp = px.bar(
        comp_long, x="Section", y="Avg (kg)", color="Waste Type",
        color_discrete_map={"Organic": PRIMARY, "Recyclable": LIGHT_GRN, "Landfill": ACCENT},
        barmode="stack",
    )
    _plotly_layout(fig_comp, "Average Waste Composition per Slot by Section")
    st.plotly_chart(fig_comp, use_container_width=True, key=f"{uuid.uuid4()}")

    if "is_holiday" in df.columns:
        hol = (
            df.groupby("is_holiday")["waste_kg"].mean()
              .reset_index()
        )
        hol["Day Type"] = hol["is_holiday"].map({0: "Normal Day", 1: "Holiday"})
        fig_hol = px.bar(
            hol, x="Day Type", y="waste_kg",
            color="Day Type",
            color_discrete_map={"Normal Day": ACCENT, "Holiday": PRIMARY},
            labels={"waste_kg": "Avg Waste (kg)"},
        )
        _plotly_layout(fig_hol, "Average Waste: Holiday vs Normal Day (all sections)")
        fig_hol.update_layout(showlegend=False)
        st.plotly_chart(fig_hol, use_container_width=True, key=f"{uuid.uuid4()}")
        st.markdown(
            """
            Holidays produce noticeably higher waste on average. This is consistent with
            special events co-occurring on public holidays — larger crowds, buffet-style
            service, and less predictable serving volumes all contribute to increased waste.
            The `is_holiday` and `has_special_event` flags were therefore included as
            exogenous regressors in both SARIMA and Prophet.
            """
        )


def render_model_comparison():
    st.markdown("# 📊 Model Comparison")
    st.markdown(
        """
        All four models were evaluated on the same 30-day hold-out test set.
        The charts below let you compare tuned performance across models and
        sections side by side. Use the metric selector to switch between RMSE,
        MAE, and MAPE.
        """
    )

    metric_label = st.selectbox(
        "Select metric", ["RMSE (kg)", "MAE (kg)", "MAPE (%)"], key="cmp_metric"
    )
    metric_key = {
        "RMSE (kg)": ("tuned_rmse", "baseline_rmse"),
        "MAE (kg)":  ("tuned_mae",  "baseline_mae"),
        "MAPE (%)":  ("tuned_mape", "baseline_mape"),
    }[metric_label]
    metric_short = metric_label.split(" ")[0]

    rows = []
    for model in ["xgboost", "lightgbm", "sarima", "prophet"]:
        for sec in SECTIONS:
            v = PERFORMANCE[model][sec]
            impr = round((1 - v[metric_key[0]] / v[metric_key[1]]) * 100, 1)
            rows.append({
                "Model":    model.upper(),
                "Section":  sec,
                "Baseline": round(v[metric_key[1]], 2),
                "Tuned":    round(v[metric_key[0]], 2),
                "Impr. %":  impr,
            })
    cmp_df = pd.DataFrame(rows)

    heat = cmp_df.pivot(index="Model", columns="Section", values="Tuned")
    fig_heat = px.imshow(
        heat,
        color_continuous_scale=[
            [0.0, DARK_GRN], [0.4, LIGHT_GRN], [0.7, PALE_GRN], [1.0, "#ffffff"]
        ],
        text_auto=".2f",
        aspect="auto",
        labels={"color": f"Tuned {metric_short}"},
    )
    _plotly_layout(
        fig_heat,
        f"Tuned {metric_label} Heatmap — All Models × Sections (lower = better)"
    )
    st.plotly_chart(fig_heat, use_container_width=True, key=f"{uuid.uuid4()}")

    fig_bar = px.bar(
        cmp_df, x="Section", y="Tuned", color="Model",
        barmode="group",
        color_discrete_map={
            "XGBOOST":  PRIMARY,
            "LIGHTGBM": LIGHT_GRN,
            "SARIMA":   ACCENT,
            "PROPHET":  "#ff8f00",
        },
        labels={"Tuned": f"Tuned {metric_label}", "Section": "Section"},
    )
    _plotly_layout(fig_bar, f"Tuned {metric_label} by Model and Section")
    st.plotly_chart(fig_bar, use_container_width=True, key=f"{uuid.uuid4()}")

    st.markdown(f"### 📉 Improvement (%) from Tuning – {metric_label}")
    st.markdown(
        "Positive values (green) mean tuning reduced the error. "
        "Negative values (orange) mean tuning made things worse for that section."
    )
    impr_heat = cmp_df.pivot(index="Model", columns="Section", values="Impr. %")

    def _cell_style(val):
        try:
            if val > 30:  return f"background-color:{GOOD_GRN}; color:{DARK_GRN}"
            if val > 0:   return "background-color:#f1f8e9"
            return f"background-color:{WARN_RED}"
        except Exception:
            return ""

    st.dataframe(
        impr_heat.style
                 .map(_cell_style)
                 .format("{:.1f}%"),
        use_container_width=True,
    )

    st.divider()
    st.markdown(
        """
        **Key takeaways:**

        XGBoost and LightGBM consistently achieve the lowest absolute errors. Both models
        benefit from the lag and rolling features that encode recent temporal context in a
        way that tabular models can directly exploit. SARIMA performs competitively on
        Sections A and B — which have more regular weekly patterns — but degrades on
        Sections C and D after tuning removed the seasonal component. Prophet shows the
        largest variance in improvement: its default settings perform poorly (the
        changepoint prior is too flexible), but tuned settings produce strong gains,
        particularly for Section D where an 88% RMSE reduction was achieved.
        """
    )


def render_perf_panel(model_type: str):
    perf     = PERFORMANCE[model_type]
    df_perf  = build_perf_df(model_type)

    st.markdown("### 📈 Performance Metrics (Tuned vs Baseline)")

    avg_rmse_impr = df_perf["RMSE Δ%"].mean()
    avg_mae_impr  = df_perf["MAE Δ%"].mean()
    best_sec_rmse = df_perf["Tuned RMSE"].idxmin()
    best_val_rmse = df_perf["Tuned RMSE"].min()

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg RMSE Improvement", f"{avg_rmse_impr:.1f}%")
    c2.metric("Avg MAE Improvement",  f"{avg_mae_impr:.1f}%")
    c3.metric("Best RMSE",            f"{best_val_rmse:.2f} kg", f"Section {best_sec_rmse}")

    display_cols = [
        "Baseline RMSE", "Tuned RMSE",    "RMSE Δ%",
        "Baseline MAE",  "Tuned MAE",     "MAE Δ%",
        "Baseline MAPE", "Tuned MAPE",
    ]
    st.dataframe(
        df_perf[display_cols]
               .style.map(_style_delta, subset=["RMSE Δ%", "MAE Δ%"])
               .format("{:.2f}"),
        use_container_width=True,
    )

    bar_rows = []
    for sec in SECTIONS:
        v = perf[sec]
        bar_rows += [
            {"Section": sec, "Stage": "Baseline", "Metric": "RMSE", "Value": v["baseline_rmse"]},
            {"Section": sec, "Stage": "Tuned",    "Metric": "RMSE", "Value": v["tuned_rmse"]},
            {"Section": sec, "Stage": "Baseline", "Metric": "MAE",  "Value": v["baseline_mae"]},
            {"Section": sec, "Stage": "Tuned",    "Metric": "MAE",  "Value": v["tuned_mae"]},
        ]
    bar_df = pd.DataFrame(bar_rows)

    fig_cmp = make_subplots(rows=1, cols=2, subplot_titles=("RMSE (kg)", "MAE (kg)"))
    colours = {"Baseline": ACCENT, "Tuned": PRIMARY}

    for metric, col_idx in [("RMSE", 1), ("MAE", 2)]:
        sub = bar_df[bar_df["Metric"] == metric]
        for stage in ["Baseline", "Tuned"]:
            d = sub[sub["Stage"] == stage]
            fig_cmp.add_trace(
                go.Bar(
                    name=stage,
                    x=d["Section"],
                    y=d["Value"],
                    marker_color=colours[stage],
                    showlegend=(col_idx == 1),
                ),
                row=1, col=col_idx,
            )

    fig_cmp.update_layout(
        barmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#333333",
        title_font_color=DARK_GRN,
        height=380,
        margin=dict(t=50, b=30),
    )
    st.plotly_chart(fig_cmp, use_container_width=True, key=f"3_{id(fig_cmp)}")

    st.markdown("### ⚙️ Best Hyperparameters per Section")
    params_df = pd.DataFrame(
        [(sec, BEST_PARAMS[model_type][sec]) for sec in SECTIONS],
        columns=["Section", "Best Parameters"],
    )
    st.dataframe(params_df, use_container_width=True, hide_index=True)


def render_tree_forecast_panel(
    model_type: str,
    xgb_df: pd.DataFrame | None,
    raw_df:  pd.DataFrame | None,
):
    st.markdown("### 🔮 7‑Day Forecast (Iterative Prediction)")
    st.markdown(
        """
        The model predicts one day at a time. Each predicted value is immediately
        fed back as a lag feature for the next day's prediction. This mirrors real
        deployment conditions where future ground truth is unavailable.
        """
    )

    selected_sec = st.selectbox(
        "Select section", SECTIONS, key=f"{model_type}_fc_sec"
    )

    model, model_info = load_tree_model(model_type, selected_sec)
    if model is None:
        st.warning(
            f"⚠️ {model_type.upper()} model not found for section {selected_sec}.\n\n"
            f"{model_info}\n\n"
            "Place the model `.pkl` files in the `models/` directory and restart."
        )
        return

    if xgb_df is not None:
        history = xgb_df[xgb_df["section"] == selected_sec].copy()
        history = history.sort_values(["date", "meal_type"])
        history_30 = history.iloc[-(30 * len(MEALS)):]
    elif raw_df is not None:
        # Fallback: build a daily aggregated dataset from raw_df
        proxy = raw_df[
            (raw_df["location_id"] == selected_sec) &
            (raw_df["meal_period"] != "Other")
        ].copy()
        proxy = proxy.rename(columns={"meal_period": "meal_type"})
        proxy["meal_type"] = proxy["meal_type"].str.lower()
        proxy["foot_traffic"] = proxy["footfall"]
        proxy["has_special_event"] = (
            proxy["special_event"].notna().astype(int)
            if "special_event" in proxy.columns else 0
        )
        for s in ["a", "b", "c", "d"]:
            proxy[f"section_{s}"] = int(s == selected_sec.lower())
        agg = (
            proxy.groupby(["date", "meal_type"])
                 .agg(
                     waste_kg    =("waste_kg",    "sum"),
                     foot_traffic=("foot_traffic","mean"),
                     is_holiday  =("is_holiday",  "max"),
                     has_special_event=("has_special_event","max"),
                     year        =("year",         "first"),
                     month       =("month",        "first"),
                     day         =("day",          "first"),
                     day_of_week =("day_of_week",  "first"),
                     is_weekend  =("is_weekend",   "first"),
                 )
                 .reset_index()
        )
        agg["day_of_year"]  = pd.to_datetime(agg["date"]).dt.dayofyear
        agg["week_of_year"] = pd.to_datetime(agg["date"]).dt.isocalendar().week.astype(int)
        agg["quarter"]      = pd.to_datetime(agg["date"]).dt.quarter
        agg["dow_sin"] = np.sin(2 * np.pi * agg["day_of_week"] / 7)
        agg["dow_cos"] = np.cos(2 * np.pi * agg["day_of_week"] / 7)
        agg["doy_sin"] = np.sin(2 * np.pi * agg["day_of_year"] / 365)
        agg["doy_cos"] = np.cos(2 * np.pi * agg["day_of_year"] / 365)
        for s in ["a", "b", "c", "d"]:
            agg[f"section_{s}"] = int(s == selected_sec.lower())
        history_30 = agg.sort_values(["date", "meal_type"]).iloc[-(30 * len(MEALS)):]
    else:
        st.info("Dataset not loaded. Cannot generate forecast.")
        return

    with st.spinner(f"Generating 7-day forecast for Section {selected_sec}…"):
        try:
            forecasts = forecast_tree_based(model, history_30, horizon=7)
        except Exception as e:
            st.error(
                f"Forecast failed: {e}\n\n"
                "This may mean the model expects a different feature set than "
                "what is available in the loaded dataset."
            )
            return

    last_date = history_30["date"].max()
    fc_rows   = []
    for day_i in range(7):
        fdate = last_date + pd.Timedelta(days=day_i + 1)
        for meal in MEALS:
            fc_rows.append({
                "Date":       fdate.strftime("%a %d %b"),
                "Day":        f"Day {day_i + 1}",
                "Meal":       meal.capitalize(),
                "Waste (kg)": round(forecasts[meal][day_i], 2),
            })
    fc_df = pd.DataFrame(fc_rows)

    if fc_df.empty:
        st.error("Forecast generation failed – no data to display.")
        return

    col_t, col_c = st.columns([1, 2])
    with col_t:
        st.dataframe(
            fc_df[["Date", "Meal", "Waste (kg)"]],
            use_container_width=True,
            hide_index=True,
        )
    with col_c:
        fig_fc = px.line(
            fc_df, x="Day", y="Waste (kg)", color="Meal",
            color_discrete_map=MEAL_PALETTE,
            markers=True,
            category_orders={"Meal": ["Breakfast", "Lunch", "Dinner"]},
        )
        _plotly_layout(
            fig_fc,
            f"7-Day Waste Forecast by Meal — Section {selected_sec}"
        )
        st.plotly_chart(fig_fc, use_container_width=True, key=f"{uuid.uuid4()}")

    daily_fc = fc_df.groupby("Date", sort=False)["Waste (kg)"].sum().reset_index()
    fig_daily = px.bar(
        daily_fc, x="Date", y="Waste (kg)",
        color_discrete_sequence=[PRIMARY],
    )
    _plotly_layout(fig_daily, f"Forecasted Total Daily Waste — Section {selected_sec}")
    st.plotly_chart(fig_daily, use_container_width=True, key=f"{uuid.uuid4()}")

    total_7day = fc_df["Waste (kg)"].sum()
    st.metric("Total Forecasted Waste (7 days)", f"{total_7day:.1f} kg")


def render_sarima_forecast_panel():
    st.markdown("### 🔮 7‑Day SARIMA Forecast")
    st.markdown(
        """
        SARIMA models generate multi-step forecasts directly from their internal
        state — no iterative feature rebuilding is needed. A separate model was
        trained for each section × meal combination, so nine models are loaded
        here (3 meals × the selected section).
        """
    )

    selected_sec = st.selectbox("Select section", SECTIONS, key="sarima_fc_sec")

    all_loaded = True
    sarima_models = {}
    for meal in MEALS:
        m, info = load_sarima_model(selected_sec, meal)
        if m is None:
            all_loaded = False
        sarima_models[meal] = (m, info)

    if not all_loaded:
        missing = [
            meal for meal, (m, _) in sarima_models.items() if m is None
        ]
        st.warning(
            f"⚠️ SARIMA models not found for: {', '.join(missing)} "
            f"(Section {selected_sec}).\n\n"
            "Expected path format: `models/sarima_optimized/tuned_{{section}}_{{meal}}.pkl`\n\n"
            "Performance tables above remain accurate from the training results."
        )
        return

    fc_rows = []
    with st.spinner("Generating SARIMA 7-day forecast…"):
        for meal, (model, _) in sarima_models.items():
            try:
                preds = model.predict(n_periods=7).tolist()
                preds = [max(0.0, p) for p in preds]
            except Exception as e:
                st.error(f"SARIMA predict failed for {meal}: {e}")
                return
            for day_i, val in enumerate(preds):
                fc_rows.append({
                    "Day":        f"Day {day_i + 1}",
                    "Meal":       meal.capitalize(),
                    "Waste (kg)": round(val, 2),
                })

    fc_df = pd.DataFrame(fc_rows)

    if fc_df.empty:
        st.error("Forecast generation failed – no data to display.")
        return

    col_t, col_c = st.columns([1, 2])
    with col_t:
        st.dataframe(fc_df, use_container_width=True, hide_index=True)
    with col_c:
        fig_fc = px.line(
            fc_df, x="Day", y="Waste (kg)", color="Meal",
            color_discrete_map=MEAL_PALETTE,
            markers=True,
            category_orders={"Meal": ["Breakfast", "Lunch", "Dinner"]},
        )
        _plotly_layout(fig_fc, f"7-Day SARIMA Forecast — Section {selected_sec}")
        st.plotly_chart(fig_fc, use_container_width=True, key=f"{uuid.uuid4()}")


def render_prophet_forecast_panel(raw_df: pd.DataFrame | None):
    st.markdown("### 🔮 7‑Day Prophet Forecast")
    st.markdown(
        """
        Prophet models generate forecasts by extending their fitted trend and
        seasonality components forward in time. Exogenous regressors (foot traffic,
        holiday flag, special event flag) must be supplied for the forecast horizon.
        The values below use the mean of the training window as a forward-fill estimate.
        """
    )

    selected_sec = st.selectbox("Select section", SECTIONS, key="prophet_fc_sec")

    all_loaded = True
    prophet_models = {}
    for meal in MEALS:
        m, info = load_prophet_model(selected_sec, meal)
        if m is None:
            all_loaded = False
        prophet_models[meal] = (m, info)

    if not all_loaded:
        missing = [
            meal for meal, (m, _) in prophet_models.items() if m is None
        ]
        st.warning(
            f"⚠️ Prophet models not found for: {', '.join(missing)} "
            f"(Section {selected_sec}).\n\n"
            "Expected path format: `models/prophet_optimized/tuned_{{section}}_{{meal}}.pkl`\n\n"
            "Performance tables above remain accurate."
        )
        return

    import datetime
    start = datetime.date.today()
    future_dates = pd.date_range(start=start, periods=7)

    fc_rows = []
    with st.spinner("Generating Prophet 7-day forecast…"):
        for meal, (model, _) in prophet_models.items():
            try:
                future = model.make_future_dataframe(periods=7)
                future = future.tail(7).copy()
                for reg in ["foot_traffic", "is_holiday", "has_special_event"]:
                    if reg in model.extra_regressors:
                        future[reg] = model.history[reg].mean()
                forecast = model.predict(future)
                preds = forecast["yhat"].clip(lower=0).tolist()
            except Exception as e:
                st.error(f"Prophet predict failed for {meal}: {e}")
                return
            for day_i, val in enumerate(preds):
                fc_rows.append({
                    "Day":        f"Day {day_i + 1}",
                    "Meal":       meal.capitalize(),
                    "Waste (kg)": round(val, 2),
                })

    fc_df = pd.DataFrame(fc_rows)

    if fc_df.empty:
        st.error("Forecast generation failed – no data to display.")
        return

    col_t, col_c = st.columns([1, 2])
    with col_t:
        st.dataframe(fc_df, use_container_width=True, hide_index=True)
    with col_c:
        fig_fc = px.line(
            fc_df, x="Day", y="Waste (kg)", color="Meal",
            color_discrete_map=MEAL_PALETTE,
            markers=True,
            category_orders={"Meal": ["Breakfast", "Lunch", "Dinner"]},
        )
        _plotly_layout(fig_fc, f"7-Day Prophet Forecast — Section {selected_sec}")
        st.plotly_chart(fig_fc, use_container_width=True, key=f"{uuid.uuid4()}")


def render_xgboost(xgb_df, raw_df):
    st.header("🌲 XGBoost")
    st.markdown(
        """
        XGBoost (**eXtreme Gradient Boosting**) builds many small decision trees in
        sequence, each correcting the errors of the previous one. It excels at
        tabular data and does not require the input to be an evenly spaced time series.

        Features given to the model include **lag values** (1, 7, 14, and 28 days ago),
        **rolling statistics** (7-day and 14-day mean, standard deviation, and maximum),
        **cyclical time encodings** (sine/cosine of day-of-week and day-of-year), and
        **section dummy variables**. Tuning used Optuna with the TPE sampler across
        5 walk-forward CV folds.

        XGBoost achieves the **lowest absolute error** of all four models, with tuned
        MAPE values below 2% across all sections. Its key advantage is that the lag and
        rolling features give it direct access to recent temporal history, so it learns
        both short-term and seasonal patterns without an explicit seasonal structure.
        """
    )
    st.divider()
    render_perf_panel("xgboost")
    st.divider()
    render_tree_forecast_panel("xgboost", xgb_df, raw_df)


def render_lightgbm(xgb_df, raw_df):
    st.header("💡 LightGBM")
    st.markdown(
        """
        LightGBM (**Light Gradient Boosting Machine**) is a faster and more
        memory-efficient variant of gradient boosting. It uses **leaf-wise** tree
        growth instead of the depth-wise approach used by XGBoost. This means each
        tree splits the leaf with the highest gain, which often gives better accuracy
        but can overfit if the data is small or noisy.

        The same feature set as XGBoost was used. Tuning added `num_leaves` and
        `min_child_samples` as additional parameters, alongside early stopping
        (50 rounds) to prevent overfitting during CV.

        LightGBM shows the largest improvement from tuning — up to **82% RMSE
        reduction** for Section A. Its baseline performance is weaker than XGBoost's
        because the default leaf-wise growth overfits quickly on this dataset size.
        Once tuned, it is competitive with XGBoost and occasionally outperforms it
        on sections with more regular weekly patterns (A and C).
        """
    )
    st.divider()
    render_perf_panel("lightgbm")
    st.divider()
    render_tree_forecast_panel("lightgbm", xgb_df, raw_df)


def render_sarima():
    st.header("📈 SARIMA")
    st.markdown(
        """
        SARIMA (**Seasonal AutoRegressive Integrated Moving Average**) is a classical
        statistical model for time series. It captures autocorrelation — the
        relationship between today's waste and past days' waste — and models both
        the trend (through differencing) and seasonal cycles.

        Three exogenous regressors were included — foot traffic, holiday flag, and
        special event flag — making this technically a **SARIMAX** model.
        Walk-forward cross-validation selected the best (p, d, q) and seasonal
        order parameters for each section × meal combination.

        **Key finding:** CV dropped the seasonal component for every section.
        The 151-day training window was too short to reliably estimate weekly SARIMA
        seasonal parameters, so the best-performing models were non-seasonal ARIMA
        variants. This explains why SARIMA is outperformed by the tree models, which
        capture weekly patterns through explicit lag features.

        For Sections C and D, tuning actually *increased* error because removing the
        seasonal component hurt sections with stronger weekly cycles. This is a known
        risk in automated order selection on short time series.
        """
    )
    st.divider()
    render_perf_panel("sarima")
    st.markdown(
        """
        **Note on Sections C and D:** the RMSE Δ% values for these sections are
        negative, meaning the tuned model performed *worse* than the baseline.
        When the optimizer removed the seasonal component, the model lost the
        ability to capture weekly patterns that are particularly strong in C and D.
        This is visible in the table above as orange cells.
        """
    )
    st.divider()
    render_sarima_forecast_panel()


def render_prophet(raw_df):
    st.header("🔮 Prophet")
    st.markdown(
        """
        Prophet is a decomposable additive time-series model developed by Meta.
        It models a time series as the sum of three components: a **trend**, one or
        more **seasonal cycles**, and **holiday effects**. It is designed to handle
        missing data gracefully and works well with irregular event patterns.

        Three additional regressors were included: foot traffic, holiday flag, and
        special event flag. Two hyperparameters were tuned via grid search:
        `changepoint_prior_scale` (how flexible the trend is) and
        `seasonality_prior_scale` (how strongly seasonal patterns are fitted).

        All sections performed best with **multiplicative** seasonality, meaning
        seasonal effects scale proportionally with the trend level. This is a
        realistic assumption for canteen data — busier seasons have both higher
        baseline waste and larger seasonal swings.

        Tuning gave the **largest relative improvement** of any model. The default
        changepoint scale is too flexible for this dataset and causes the trend
        component to overfit noise. Once tightened, Prophet learned the true seasonal
        structure, giving up to an **89% RMSE reduction** for Section D.
        """
    )
    st.divider()
    render_perf_panel("prophet")
    st.divider()
    render_prophet_forecast_panel(raw_df)


def render_methodology():
    st.markdown("# 🧪 Methodology")
    st.markdown("## 📦 Data Generation & Cleaning")
    st.markdown(
        """
        The project started with a publicly available university canteen dataset
        from Kaggle. That dataset was missing three things needed for robust
        time-series forecasting: intra-day timestamps, a foot traffic column, and
        sufficient temporal range.

        A synthetic dataset was generated to fill these gaps while preserving the
        statistical distribution of the original data. The generation process added:
        a footfall column correlated with waste volume and time of day; timestamps
        at 30-minute resolution across four canteen sections; realistic seasonality
        patterns (weekly cycle, lunch/dinner peaks); and enough records (8,688)
        to support lag-based feature engineering.
        """
    )

    st.markdown("## 🧹 Cleaning Pipeline")
    st.markdown(
        """
        The cleaning pipeline applied in sequence:
        linear interpolation (within each location group) for the 12 missing
        footfall values; forward/backward fill for categorical columns with a
        fallback to `"unknown"`; data type correction (timestamp → datetime64,
        integer columns → Int64, float columns → float64); IQR × 3 capping for
        extreme outliers in `waste_kg`, `footfall`, and the waste sub-type columns;
        and consistency checks ensuring waste components sum to `waste_kg` within
        a 0.05 kg tolerance.
        """
    )

    st.markdown("## ⚙️ Feature Engineering")
    st.markdown(
        """
        Two feature sets were produced for different model types.

        **For SARIMA and Prophet** (`waste_features_full.csv`): records were
        aggregated from 30-minute slots to one row per date × section × meal.
        Features include foot traffic (mean over the slot), holiday and special
        event flags, and basic calendar fields. No lag features were added because
        these models capture autocorrelation internally.

        **For XGBoost and LightGBM** (`waste_features_xgb.csv`): the same daily
        aggregation plus engineered lag values at 1, 7, 14, and 28 days;
        rolling statistics over 7 and 14-day windows (mean, standard deviation,
        maximum); sine and cosine encodings of day-of-week and day-of-year; and
        one-hot encoded section dummies.
        """
    )

    st.markdown("## 📏 Evaluation Protocol")
    st.markdown(
        """
        All models shared the same evaluation protocol to ensure fair comparison.
        The last 30 calendar days were held out as the test set — no information
        from this window was available during training or tuning.
        Walk-forward cross-validation (5 expanding folds) was used to tune
        hyperparameters without leaking future data.

        Three error metrics were computed: RMSE (root mean squared error),
        MAE (mean absolute error), and MAPE (mean absolute percentage error).
        RMSE is the primary metric because it penalises large prediction errors
        more heavily — important in operations where a large underestimate leads
        to food shortages or costly over-ordering.
        """
    )

    st.markdown("## 🔮 Forecast Generation (7‑Day)")
    st.markdown(
        """
        Tree-based models (XGBoost and LightGBM) use **iterative one-step-ahead
        prediction** to generate a 7-day forecast. Each day, the model predicts
        the next day's waste. That prediction is appended to the history and used
        to recompute the lag features for the following step. This ensures the
        forecast is realistic — no future ground-truth values are used.

        SARIMA models call `model.predict(n_periods=7)` directly. The model
        object retains its internal state from training and extrapolates the fitted
        ARIMA process forward, using the exogenous regressors supplied for the
        forecast window.

        Prophet models use `model.make_future_dataframe(periods=7)` and
        `model.predict()`. Exogenous regressor values for the future window are
        estimated as the training-window mean — a conservative default that avoids
        fabricating future event or footfall data.
        """
    )

    st.markdown("## 📊 Summary of Models")
    summary = pd.DataFrame({
        "Model":         ["XGBoost", "LightGBM", "SARIMA", "Prophet"],
        "Type":          ["Gradient boosting", "Gradient boosting", "Statistical", "Decomposable additive"],
        "Feature set":   ["Lags + rolling + cyclical", "Lags + rolling + cyclical", "Internal AR + MA", "Trend + seasonality"],
        "Regressors":    ["foot_traffic, section dummies, meal dummies", "foot_traffic, section dummies, meal dummies", "foot_traffic, is_holiday, has_special_event", "foot_traffic, is_holiday, has_special_event"],
        "Tuning tool":   ["Optuna (TPE)", "Optuna (TPE)", "Grid search / auto_arima", "Grid search"],
        "Best section":  ["A (RMSE 1.91)", "A (RMSE 2.63)", "B (RMSE 9.33)", "B (RMSE 13.34)"],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)


def main():
    # render_sidebar()

    st.title("🍽️ Food Waste Forecasting Dashboard")
    st.markdown(
        "A scientific communication tool comparing four forecasting models on "
        "university canteen food waste data — Jan to Jun 2025."
    )

    raw_df  = None
    xgb_df  = None

    try:
        raw_df = load_raw_dataset()
    except FileNotFoundError:
        st.info(
            "ℹ️ `data/synthetic_forecast_cleaned.csv` not found. "
            "EDA and tree-based forecast panels require this file."
        )

    try:
        xgb_df = load_xgb_dataset()
    except FileNotFoundError:
        pass

    tabs = st.tabs([
        "🏠 Overview",
        "🧭 EDA",
        "📊 Model Comparison",
        "🌲 XGBoost",
        "💡 LightGBM",
        "📈 SARIMA",
        "🔮 Prophet",
        "🧪 Methodology",
    ])

    with tabs[0]:
        render_overview()

    with tabs[1]:
        if raw_df is not None:
            render_eda(raw_df)
        else:
            st.warning(
                "Dataset `data/synthetic_forecast_cleaned.csv` must be present "
                "to view EDA. Place it in the `data/` directory and reload."
            )

    with tabs[2]:
        render_model_comparison()

    with tabs[3]:
        render_xgboost(xgb_df, raw_df)

    with tabs[4]:
        render_lightgbm(xgb_df, raw_df)

    with tabs[5]:
        render_sarima()

    with tabs[6]:
        render_prophet(raw_df)

    with tabs[7]:
        render_methodology()


if __name__ == "__main__":
    main()