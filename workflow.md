## 1. Data Preprocessing & Aggregation

**Goal**: Convert the raw transaction‑level data into a time series with a 30‑minute granularity for each canteen section.

- **Parse timestamps**  
  Convert the `Date` column to datetime.  
  ```python
  df['datetime'] = pd.to_datetime(df['Date'])
  ```

- **Create 30‑minute bins**  
  Floor each timestamp to the nearest 30‑minute mark (e.g., 12:00:00 → 12:00:00, 12:15:00 → 12:00:00, 12:45:00 → 12:30:00).  
  ```python
  df['time_bin'] = df['datetime'].dt.floor('30min')
  ```

- **Aggregate per (canteen section, time bin)**  
  Group by `Canteen_Section` and `time_bin`, summing numeric values of interest.  
  ```python
  agg_dict = {
      'Waste_Weight_kg': 'sum',           # total waste
      'Cost_Loss': 'sum',                 # total cost loss
      'Foot_Traffic': 'mean'              # average foot traffic (could also sum)
  }
  # Keep categorical columns for later feature engineering
  # The original Meal, Food_Category are lost – that's fine for total waste per section.
  df_agg = df.groupby(['Canteen_Section', 'time_bin']).agg(agg_dict).reset_index()
  ```

- **Fill missing time bins**  
  Some 30‑minute intervals may have no records (e.g., when the canteen is closed). For a complete time series, we need to ensure every interval exists.  
  - Create a full date range for the period covered, with 30‑minute frequency.  
  - For each canteen section, reindex the aggregated series to this full range, filling missing waste with 0 (no waste) and foot traffic with NaN (or 0).  
  ```python
  full_range = pd.date_range(start=df_agg['time_bin'].min(), 
                             end=df_agg['time_bin'].max(), 
                             freq='30min')
  df_agg.set_index('time_bin', inplace=True)
  df_agg = df_agg.groupby('Canteen_Section').apply(
      lambda x: x.reindex(full_range, fill_value=0)
  ).drop('Canteen_Section', axis=1).reset_index()
  ```

- **Add time‑based features**  
  From the `time_bin` column, create features that will help models capture patterns:  
  - `hour` (0‑23), `minute` (0 or 30) – can be used as cyclical features.  
  - `weekday` (0‑6, Monday=0), `month`, `day_of_year`, `quarter`, `is_weekend`.  
  - Cyclical encoding: `sin_hour = sin(2π * hour/24)`, `cos_hour`, similarly for weekday, minute.  



## 2. Feature Engineering for Machine Learning Models (XGBoost, SVM)

These models require a tabular format where each row corresponds to a time step and contains features that predict the target.

- **Lagged target values**  
  For each canteen section, create lagged waste values for the last few intervals. Because we have strong daily seasonality (48 intervals per day), include lags at 1, 2, 48, 96, 336 (7 days). Use a sliding window; be careful not to use future data.  
  ```python
  for lag in [1, 2, 48, 96, 336]:
      df[f'waste_lag_{lag}'] = df.groupby('Canteen_Section')['Waste_Weight_kg'].shift(lag)
  ```

- **Rolling statistics**  
  Compute rolling means and standard deviations over windows of e.g. 48 (1 day) and 336 (7 days). These help capture trends. Again, use `shift` to avoid leakage.  
  ```python
  df['waste_rolling_mean_48'] = df.groupby('Canteen_Section')['Waste_Weight_kg'].transform(
      lambda x: x.rolling(48, min_periods=1).mean().shift(1)
  )
  ```

- **Foot traffic**  
  The original `Foot_Traffic` column may be used as an exogenous regressor. We can also create lags of foot traffic if it’s known in advance (e.g., expected foot traffic). For forecasting, the user may provide future foot traffic values. Include foot traffic as a feature.

- **Event flags**  
  Create binary features for known events (e.g., public holidays, special days). If the dataset doesn’t contain this, you can manually add them from external holiday calendars. For testing, you might simulate simple flags like “is_holiday”.

- **Drop rows with NaN lags**  
  Lags introduce missing values at the beginning of each series. Drop those rows before training.



## 3. Train/Validation/Test Split (Time‑Series Safe)

- **Chronological split**  
  Because we are predicting the future, we must respect time order.  
  - Define a cutoff date for training, another for validation.  
  - Example: Use first 70% of the data for training, next 15% for validation, last 15% for testing.  
  - Alternatively, use a rolling‑window cross‑validation (e.g., `TimeSeriesSplit` from scikit‑learn).  
  - Ensure no data from the future is used in training (e.g., when creating lags, we used `shift`, which is safe).

- **Separate per canteen section**  
  Since we group by section, splits must be applied after the aggregation and feature engineering, but before feeding into models. For each section, the split dates should be the same to keep temporal consistency.



## 4. Modeling Approaches

### 4.1 XGBoost (Gradient Boosting)
- **Input**: Feature matrix (time features, lags, rolling stats, foot traffic, event flags)  
- **Target**: `Waste_Weight_kg` at current time step (or multi‑step direct/recursive).  
- **Forecast**: Recursive strategy – use the model to predict one step ahead, then feed that prediction back as lag for the next step, along with known future exogenous variables (foot traffic, events). This yields a full 336‑step forecast.  
- **Hyperparameter tuning**: Use `TimeSeriesSplit` on the training set to avoid leakage. Search over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, etc.

### 4.2 SARIMA (Seasonal ARIMA)
- **Univariate per canteen section** (optionally with exogenous variables via SARIMAX).  
- **Seasonality**: Daily (48) and weekly (336) are likely.  
- **Process**:  
  1. Check stationarity (ADF test) – apply differencing if needed.  
  2. Use auto‑ARIMA (`pmdarima.auto_arima`) to find (p,d,q)(P,D,Q,s) based on training data, with `seasonal=True` and `m=48` (or 336 if weekly seasonality is stronger).  
  3. Fit `SARIMAX` with exogenous foot traffic/events if available.  
- **Forecast**: Call `forecast(steps=336, exog=...)`.  
- **Evaluation**: Compare fitted vs validation/test.

### 4.3 Prophet
- **Input**: A dataframe with columns `ds` (timestamp) and `y` (waste), plus optional `add_regressor` for foot traffic and events.  
- **Seasonalities**: Prophet automatically detects daily, weekly, yearly seasonalities. For 30‑minute data, you can add custom seasonalities (e.g., `add_seasonality(name='half_hourly', period=0.5, fourier_order=5)`).  
- **Forecast**: Use `make_future_dataframe(periods=336, freq='30min')` and pass future foot traffic values via the regressors.  
- **Tuning**: Adjust `changepoint_prior_scale`, `seasonality_prior_scale`, `holidays_prior_scale`.

### 4.4 Support Vector Machine (SVR)
- Similar to XGBoost: treat as regression problem with the same feature set.  
- **Kernel**: Typically RBF.  
- **Scaling**: Features must be standardized (e.g., using `StandardScaler` fitted on training data).  
- **Hyperparameters**: `C`, `gamma` – tune with `TimeSeriesSplit`.  
- **Forecast**: Recursive, same as XGBoost.



## 5. Multi‑Step Forecasting Strategy

For models that predict one step at a time (XGBoost, SVM), we need to generate a 336‑step forecast:

- Start with the last known data point.  
- For each future time step \( t \):
  1. Construct the feature vector using:
     - Known future features (time features, foot traffic, events).
     - Lag values: these may come from actual past values (if still within the historical window) or from previous predictions.
  2. Predict \( \hat{y}_t \).
  3. Append \( \hat{y}_t \) to the series for use in later lags.  

This recursive method can accumulate error, but it’s simple and widely used.



## 6. Evaluation Metrics & Cross‑Validation

- **Metrics**:  
  - MAE (Mean Absolute Error) – easy to interpret.  
  - RMSE (Root Mean Squared Error) – penalizes large errors more.  
  - MAPE (Mean Absolute Percentage Error) – scale‑independent, but sensitive to zero values.  
  - sMAPE (symmetric MAPE) – handles zeros better.  

  Compute these per forecast horizon (e.g., 1‑step, 48‑step, 336‑step) to understand model behavior.

- **Cross‑Validation**:  
  Use `TimeSeriesSplit` from scikit‑learn. For each fold, train on past data, validate on the next contiguous block. Tune hyperparameters using the validation set, then evaluate on the final test set.

- **Avoid leakage**:  
  - All feature engineering (lags, rolling stats) must be done on the training data only, then applied to validation/test (using `shift` or rolling windows that only look at the past).  
  - Do not use future information (like global statistics) when creating features.



## 7. Handling Optional Inputs (Events & Foot Traffic)

- **During training**: Include event flags and foot traffic as features (for XGBoost/SVM) or as exogenous regressors (for SARIMAX/Prophet).  
- **During prediction**:  
  - The user can supply a dataframe with expected foot traffic and event flags for each 30‑minute interval of the forecast period.  
  - For SARIMAX and Prophet, these are passed as `exog` or `add_regressor` values.  
  - For XGBoost/SVM, they become part of the feature vector constructed for each future step.



## 8. Additional Targets (Cost Loss)

- **Direct approach**: Train a separate model for `Cost_Loss` using the same feature set (or a different one). This allows predicting both waste and cost.  
- **Indirect approach**: Use the waste predictions and multiply by `Unit_Price_per_kg` (if the unit price is known). However, unit price may vary over time, so direct modeling is safer.



## 9. Model Selection & Production

- Compare models using test set metrics, also considering computational cost, ease of deployment, and interpretability.  
- For production, the selected model should be retrained periodically (e.g., weekly) on the latest data to capture new patterns.  
- Implement an API that accepts:
  - Current date (to know where to start forecasting).
  - Optional foot traffic and event flags for the forecast window.
  - Returns predictions for waste (and cost) per canteen section, per 30‑minute interval.



## 10. Implementation Tools

- **Libraries**: pandas, numpy, matplotlib (visualization), scikit‑learn (preprocessing, metrics, TimeSeriesSplit), xgboost, statsmodels (SARIMAX), prophet (or fbprophet), pmdarima (auto‑ARIMA), scikit‑learn’s SVR.  
- **Version control**: Use git to track code changes.  


## Summary of Key Steps

1. **Aggregate** raw data to 30‑minute intervals per canteen section.  
2. **Engineer features** (lags, rolling stats, time features, foot traffic, events).  
3. **Split data** chronologically (train/validation/test).  
4. **Build models** (XGBoost, SARIMA, Prophet, SVM, LSTM) with appropriate forecasting strategies.  
5. **Tune hyperparameters** using time‑series cross‑validation.  
6. **Evaluate** on test set with multiple metrics, comparing across horizons.  
7. **Select best model** and prepare it for production, allowing optional future inputs.  

By following this workflow, you’ll have a robust forecasting system that respects time‑series principles, avoids leakage, and can be extended with external factors like events and foot traffic.