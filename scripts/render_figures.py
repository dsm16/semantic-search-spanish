"""Plot frozen public measurements only; never load encoders or raw data."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ["TF-IDF", "BETO\nfirst-token", "BETO\nmasked mean",
          "Robertuito\nfirst-token", "Robertuito\nmasked mean", "MiniLM"]

def plot_metric(results, key, title, ylabel, *, upper=None):
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    values = [s[key] for s in results["systems"]]
    bars = ax.bar(LABELS, values, color="#286985", width=0.65)
    ax.set_title(title, loc="left", fontweight="bold", pad=16)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, upper if upper is not None else max(values) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f"{v:.4f}" if key != "seconds" else f"{v:.2f}" for v in values], padding=5)
    return fig

def render(results, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for key, title, ylabel, upper, name in [
        ("precision_at_5", "Precision@5 by retrieval system", "Macro Precision@5", 1, "precision_at_5"),
        ("mrr_at_10", "MRR@10 by retrieval system", "MRR@10", 1, "mrr_at_10"),
        ("seconds", "Runtime by retrieval system", "Seconds (CPU)", None, "runtime"),
    ]:
        fig = plot_metric(results, key, title, ylabel, upper=upper)
        fig.savefig(directory / f"{name}.png", dpi=160)
        plt.close(fig)

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    render(json.loads((root / "evaluation/benchmark_results.json").read_text(encoding="utf-8")), root / "docs/figures")
