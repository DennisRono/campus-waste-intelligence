import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


PLOT_DIR = './plots'
DATA_DIR = './data'


def get_parts() -> tuple[list, list, list]:
    try:
        preds  = pd.read_csv(DATA_DIR + "/contamination_predictions.csv", parse_dates=["date"])
        bins   = pd.read_csv(DATA_DIR + "/bins.csv", parse_dates=["date"])
        merged = preds.merge(bins[["date", "bin_id", "contamination_rate"]], on=["date", "bin_id"])

        return preds, merged, preds.groupby("location_type")["predicted_contam_prob"].mean().sort_values(ascending=False).index.tolist()
    except Exception as e:
        print(f"Error getting location order: {e}")
        return [], [], []


def generate_contam_loc_graph(loc_order: list, preds: list) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = preds.groupby("location_type")["predicted_contam_prob"].mean().reindex(loc_order)
    bars = ax.bar(vals.index, vals.values)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Avg Predicted Contamination Probability by Location", fontsize=12)
    ax.set_ylabel("Avg Contamination Probability")

    for bar, v in zip(bars, vals.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.0%}", ha="center", va="bottom", fontsize=9)
    
    plt.xticks(rotation=20, ha="right")
    plt.savefig("./plots/contam_by_location.png", dpi=150)
    plt.close()
    print("Saved: contam_by_location.png")


def generate_pred_graph(mreged: list) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    sample = merged.sample(min(3000, len(merged)), random_state=42)
    
    ax.scatter(sample["contamination_rate"], sample["predicted_contam_rate"],alpha=0.2, s=8)
    
    ax.plot([0, 1], [0, 1], "r--", label="Perfect prediction")
    ax.set_xlabel("Actual Contamination Rate")
    ax.set_ylabel("Predicted Contamination Rate")
    ax.set_title("Predicted vs Actual Contamination Rate", fontsize=12)
    ax.legend()
    plt.savefig("./plots/predicted_vs_actual.png", dpi=150)
    plt.close()
    print("Saved: predicted_vs_actual.png")


if __name__ == "__main__":
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
    
    preds, merged, loc_order = get_parts()
    generate_contam_loc_graph(loc_order, preds)
    generate_pred_graph(merged)
