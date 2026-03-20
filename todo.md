## Todo

1. Dataset Selection
  [x] Choose a dataset to use

2. Data Cleaning
  [x] Handle missing values
  [x] Standardize formats
  [x] Remove duplicates
  [x] Ensure temporal consistency

3. Dataset Expansion (Synthetic Data)
  [ ] Generate additional synthetic observations
  [ ] Add location types (e.g., dining hall, dorm, academic building)
  [ ] Add event indicators
  [ ] Simulate foot traffic patterns
  [ ] Preserve temporal structure of original data

4. Exploratory Data Analysis (EDA)
  [x] Analyze trends
  [x] Identify seasonality
  [x] Examine distribution of waste volumes
  [x] Analyze correlations with features (day of week, location type, etc.)

5. Define Forecasting Granularity and Target
  [ ] Determine temporal resolution (e.g., daily, hourly)
  [ ] Define spatial granularity (e.g., by location type)
  [ ] Confirm target variable (e.g., waste weight or volume)
  [ ] Ensure alignment with cleaned and augmented dataset

6. Feature Engineering for Time-Series Modeling
  [ ] Create time-based features
  [ ] Day of week
  [ ] Month
  [ ] Holiday flags
  [ ] Academic calendar events
  [ ] Weather variables (if available)
  [ ] Generate lag features
  [ ] Previous day
  [ ] Previous week
  [ ] Same day last week
  [ ] Create rolling statistics
  [ ] 7-day moving average
  [ ] Incorporate external regressors
  [ ] Event schedules
  [ ] Foot traffic estimates
  [ ] Synthetic contamination indicators

7. Temporal Data Splitting
  [ ] Split dataset chronologically
  [ ] 70% training
  [ ] 15% validation
  [ ] 15% testing
  [ ] Ensure realistic future validation and test periods

8. Baseline Forecasting Models
  [ ] Implement moving average model (e.g., last 7 days)
  [ ] Implement seasonal naive model (e.g., same day previous week)
  [ ] Compute evaluation metrics
  [ ] MAE
  [ ] RMSE

9. Advanced Forecasting Models
  [ ] SARIMA
  [ ] Perform grid search for (p,d,q)(P,D,Q)s
  [ ] Prophet
  [ ] Add seasonalities
  [ ] Include holidays and regressors
  [ ] XGBoost
  [ ] Train with lag and engineered features
  [ ] Tune hyperparameters using time-series cross-validation
  [ ] Document configurations and preprocessing steps

10. Model Evaluation and Comparison
  [ ] Compute metrics on validation set
  [ ] MAE
  [ ] RMSE
  [ ] MAPE (optional)
  [ ] Compare models (accuracy and efficiency)
  [ ] Evaluate best model on test set

11. Forecast Output Preparation
  [ ] Generate point forecasts
  [ ] Ensure correct temporal and spatial resolution
  [ ] Export outputs
  [ ] CSV
  [ ] DataFrame
  [ ] Provide uncertainty intervals (if possible)

12. Documentation and Code Review
  [ ] Document workflow and assumptions
  [ ] Record final model parameters
