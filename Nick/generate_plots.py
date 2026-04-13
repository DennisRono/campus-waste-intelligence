"""
generate_plots.py — save contamination model charts to ./plots/
Run from the Nick/ directory: python generate_plots.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

os.makedirs("./plots", exist_ok=True)

preds  = pd.read_csv("./data/contamination_predictions.csv", parse_dates=["date"])
bins   = pd.read_csv("./data/bins.csv", parse_dates=["date"])
merged = preds.merge(bins[["date", "bin_id", "contamination_rate"]], on=["date", "bin_id"])

LOC_ORDER = preds.groupby("location_type")["predicted_contam_prob"].mean().sort_values(ascending=False).index.tolist()


fig, ax = plt.subplots(figsize=(7, 4))
vals = preds.groupby("location_type")["predicted_contam_prob"].mean().reindex(LOC_ORDER)
bars = ax.bar(vals.index, vals.values, color="#2E86AB", edgecolor="white", width=0.6)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_title("Avg Predicted Contamination Probability by Location", fontsize=12)
ax.set_ylabel("Avg Contamination Probability")
for bar, v in zip(bars, vals.values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.0%}", ha="center", va="bottom", fontsize=9)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("./plots/contam_by_location.png", dpi=150)
plt.close()
print("Saved: contam_by_location.png")


fig, ax = plt.subplots(figsize=(5, 5))
sample = merged.sample(min(3000, len(merged)), random_state=42)
ax.scatter(sample["contamination_rate"], sample["predicted_contam_rate"],
           alpha=0.2, s=8, color="#2E86AB")
ax.plot([0, 1], [0, 1], "r--", lw=1, label="Perfect prediction")
ax.set_xlabel("Actual Contamination Rate")
ax.set_ylabel("Predicted Contamination Rate")
ax.set_title("Predicted vs Actual Contamination Rate", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig("./plots/predicted_vs_actual.png", dpi=150)
plt.close()
print("Saved: predicted_vs_actual.png")

print("\nDone.")
