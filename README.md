# Campus Waste Intelligence System
Joint Prediction and Optimization of Waste Generation and Contamination
Forecasting Food Waste (kgs) per Campus Location (A, B, C, D)

```bash
python3 -m venv venv
```

```bash
pip install -r requirements.txt
```

## our workflow

 start with synthetic.ipynb - then data_cleaning.ipynb - then eda.py then - preprocessing and feature engineering

 then any of the models notebooks - xgboost.ipynb, or lightgbm.ipynb, or sarima.ipynb, or prophet.iynb

the models will be saved in the models folder


run streamlit application

```bash
streamlit run streamlit/app.py
```
