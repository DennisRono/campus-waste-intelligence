import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

LOC_COMMODITY_MAP = {
    "dining_hall": ["Cabbages", "Lettuce and chicory", "Tomatoes", "Potatoes", "Cauliflowers and broccoli", "Spinach"],
    "dormitory": ["Apples", "Bananas", "Oranges", "Potatoes", "Pears", "Grapes"],
    "academic_bldg": ["Apples", "Lemons and limes", "Other vegetables, fresh n.e.c."],
    "library": ["Apples", "Other vegetables, fresh n.e.c."],
    "gym": ["Bananas", "Pineapples", "Watermelons"],
    "student_union": ["Tomatoes", "Lettuce and chicory", "Cucumbers and gherkins", "Potatoes", "Mushrooms and truffles"],
}


def load_baselines(fao_path, unep_path):
    fao = pd.read_csv(fao_path)
    unep = pd.read_csv(unep_path)

    fao_campus = fao[fao['food_supply_stage'].isin(['Retail', 'Households', 'Whole supply chain'])]
    stage_loss = fao_campus.groupby('food_supply_stage')['loss_percentage'].median()
    max_loss = fao['loss_percentage'].max()

    commodity_prior = (
        fao_campus.groupby('commodity')['loss_percentage']
        .median()
        .div(max_loss)
        .clip(0, 1)
    )

    def get_prior(loc):
        comms = LOC_COMMODITY_MAP.get(loc, [])
        vals = commodity_prior[commodity_prior.index.isin(comms)]
        return vals.mean() if len(vals) else commodity_prior.mean()

    usa = unep[unep['Country'] == 'United States of America'].iloc[0]
    fs_g = usa['Food service estimate (kg/capita/year)'] * 1000 / 365
    hh_g = usa['Household estimate (kg/capita/year)'] * 1000 / 365
    ret_g = usa['Retail estimate (kg/capita/year)'] * 1000 / 365

    locations = {
        "dining_hall": dict(bins=12, base_contam=stage_loss['Retail'] / max_loss, volume_g=fs_g, occupancy=500),
        "dormitory": dict(bins=20, base_contam=stage_loss['Households'] / max_loss, volume_g=hh_g, occupancy=300),
        "academic_bldg": dict(bins=18, base_contam=stage_loss['Whole supply chain'] / max_loss, volume_g=fs_g, occupancy=400),
        "library": dict(bins=8, base_contam=stage_loss['Whole supply chain'] / max_loss * 0.8, volume_g=ret_g, occupancy=150),
        "gym": dict(bins=6, base_contam=stage_loss['Retail'] / max_loss, volume_g=fs_g, occupancy=100),
        "student_union": dict(bins=10, base_contam=stage_loss['Retail'] / max_loss, volume_g=fs_g, occupancy=350),
    }
    for loc, cfg in locations.items():
        cfg['commodity_prior'] = get_prior(loc)

    return locations


def build_dataset(locations, n_days=365, start=datetime(2018, 1, 1)):
    dates = [start + timedelta(days=i) for i in range(n_days)]
    records = []

    for date in dates:
        dow = date.weekday()
        month = date.month
        is_weekend = int(dow >= 5)
        is_summer = int(month in [6, 7, 8])
        is_holiday = int(month == 12 and date.day >= 20)
        has_event = int(np.random.rand() < (0.35 if not is_weekend else 0.15))
        temp_f = 60 + 20 * np.sin((month - 3) * np.pi / 6) + np.random.normal(0, 5)
        is_rainy = int(np.random.rand() < 0.15)

        for loc, cfg in locations.items():
            for bin_id in range(cfg['bins']):
                vol = cfg['volume_g'] * cfg['occupancy'] / cfg['bins']
                vol *= (0.55 if is_weekend else 1.0)
                vol *= (0.45 if is_summer else 1.0)
                vol *= (1.30 if has_event else 1.0)
                vol = max(0.0, vol + np.random.normal(0, vol * 0.12))

                capacity_fill = float(np.clip((vol / 1000) / 20 + np.random.normal(0, 0.05), 0, 1))

                c_rate = 0.5 * cfg['base_contam'] + 0.5 * cfg['commodity_prior']
                c_rate += 0.10 * capacity_fill
                c_rate += 0.05 * has_event
                c_rate -= 0.06 * is_summer
                c_rate += 0.03 * is_rainy
                c_rate += 0.04 * is_weekend
                c_rate += 0.001 * bin_id
                c_rate = float(np.clip(c_rate + np.random.normal(0, 0.04), 0.01, 0.99))

                records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "location_type": loc,
                    "bin_id": f"{loc}_bin_{bin_id:02d}",
                    "real_base_contam": round(cfg['base_contam'], 4),
                    "commodity_prior": round(cfg['commodity_prior'], 4),
                    "bin_volume_g": round(vol, 2),
                    "month": month,
                    "day_of_week": dow,
                    "is_weekend": is_weekend,
                    "is_summer": is_summer,
                    "is_holiday": is_holiday,
                    "has_event": has_event,
                    "temp_f": round(temp_f, 1),
                    "is_rainy": is_rainy,
                    "bin_position": bin_id,
                    "capacity_fill": round(capacity_fill, 4),
                    "contamination_rate": round(c_rate, 4),
                    "is_contaminated": int(c_rate >= 0.20),
                })

    return pd.DataFrame(records)


def engineer_features(df):
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["event_x_fill"] = df["has_event"] * df["capacity_fill"]
    df["prior_x_fill"] = df["commodity_prior"] * df["capacity_fill"]
    df["prior_x_event"] = df["commodity_prior"] * df["has_event"]
    df = pd.get_dummies(df, columns=["location_type"], prefix="loc")
    return df


if __name__ == "__main__":
    locations = load_baselines(
        fao_path="./data/Data.csv",
        unep_path="./data/Food_Waste_data_and_research_-_by_country.csv"
    )
    df = build_dataset(locations)
    df = engineer_features(df)

    print(f"Dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Contamination prevalence: {df['is_contaminated'].mean():.1%}")

    df.to_csv("./data/bins.csv", index=False)
    print("Saved to ./data/bins.csv")