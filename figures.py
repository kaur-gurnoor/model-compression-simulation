import json
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
with open("./results.json") as f:
    R = json.load(f)
available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
plt.rcParams["font.family"] = "Times New Roman" if "Times New Roman" in available_fonts else "DejaVu Serif"
plt.rcParams["font.size"] = 11
compression = [max(0.0, v) for v in R["compression_ratio_pct"]]
overall = [v * 100 for v in R["overall_accuracy"]]
minority = [v * 100 for v in R["minority_class_accuracy"]]
majority = [v * 100 for v in R["majority_class_avg_accuracy"]]
fig, ax = plt.subplots(figsize=(6.5, 4.3), dpi=600)

ax.plot(compression, overall, marker="o", linewidth=2, color="#1a1a1a",
        label="Overall test accuracy", zorder=3)
ax.plot(compression, majority, marker="s", linewidth=1.8, color="#4472C4",
        linestyle="--", label="Majority-class average accuracy", zorder=2)
ax.plot(compression, minority, marker="^", linewidth=2.2, color="#C00000",
        label="Minority-class (digit 8) accuracy", zorder=3)

ax.set_xlabel("Parameters removed from fc1 layer via truncated SVD (%)")
ax.set_ylabel("Classification accuracy (%)")
ax.set_ylim(-5, 105)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.legend(loc="lower left", frameon=False, fontsize=9.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig("./fig1_aggregate_vs_subgroup.png", dpi=600, bbox_inches="tight")
plt.savefig("./fig1_aggregate_vs_subgroup.pdf", bbox_inches="tight")
plt.close()
print("Saved fig1_aggregate_vs_subgroup.png / .pdf")
per_class = np.array(R["per_class_accuracy"]) * 100   # shape (n_ranks, 10)

fig, ax = plt.subplots(figsize=(6.8, 4.3), dpi=600)
im = ax.imshow(per_class.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)

ax.set_xticks(range(len(R["ranks"])))
ax.set_xticklabels([f"{c:.0f}%" for c in compression], fontsize=9)
ax.set_yticks(range(10))
ax.set_yticklabels([str(d) for d in range(10)], fontsize=9)
ax.set_xlabel("Parameters removed via truncated SVD (%)")
ax.set_ylabel("Digit class")

# Outline the minority class row so it stands out
ax.add_patch(plt.Rectangle((-0.5, 8 - 0.5), len(R["ranks"]), 1, fill=False,
                           edgecolor="black", linewidth=2))

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Class accuracy (%)", fontsize=9.5)

plt.tight_layout()
plt.savefig("./fig2_per_class_heatmap.png", dpi=600, bbox_inches="tight")
plt.savefig("./fig2_per_class_heatmap.pdf", bbox_inches="tight")
plt.close()