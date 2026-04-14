"""
Plot remote benchmark summary from 260406_remote_testing_results_summary.txt.

Shows index-read time and data-read time per method as box-and-whisker
plots, grouped by consolidated_metadata (CR) status.

Usage:
    python examples/plot_benchmark_summary.py [path/to/summary.txt]
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Parse ─────────────────────────────────────────────────────────────────────

LINE_RE = re.compile(
    r"^(?P<method>\w+)\s+read:.*?time=(?P<index>[\d.]+)s\s+(?P<data>[\d.]+)s?\s+\(CR=(?P<cr>[^)]+)\)"
)

def load(path: Path):
    rows = []
    for raw in path.read_text().splitlines():
        m = LINE_RE.match(raw.strip())
        if m:
            rows.append({
                "method": m["method"],
                "index_s": float(m["index"]),
                "data_s":  float(m["data"]),
                "cr":      m["cr"].strip(),
            })
    return rows


# ── Organise ───────────────────────────────────────────────────────────────────

def organise(rows):
    """Return nested dict: cr_label -> method -> {index_s: [...], data_s: [...]}"""
    out = defaultdict(lambda: defaultdict(lambda: {"index_s": [], "data_s": []}))
    for r in rows:
        cr = r["cr"]
        out[cr][r["method"]]["index_s"].append(r["index_s"])
        out[cr][r["method"]]["data_s"].append(r["data_s"])
    return out


# ── Plot ───────────────────────────────────────────────────────────────────────

METHOD_ORDER = ["POSIX", "SSH", "HTTP", "S3"]
CR_ORDER     = ["False", "True", "UNAVAILBLE"]   # note typo preserved from data
CR_COLORS    = {"False": "#2196F3", "True": "#FF7043", "UNAVAILBLE": "#9C27B0"}
CR_LABELS    = {"False": "CR=False", "True": "CR=True", "UNAVAILBLE": "CR=N/A (SSH)"}


def _box_group(ax, data_by_cr, cr_groups, methods, time_key, *, width=0.20):
    """One cluster of boxes per method; one box per CR group within each cluster."""
    n = len(cr_groups)
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width
    legend_handles = []

    for offset, cr in zip(offsets, cr_groups):
        color = CR_COLORS.get(cr, "#888")
        positions = [i + 1 + offset for i in range(len(methods))]
        data = [data_by_cr.get(cr, {}).get(meth, {}).get(time_key, [np.nan])
                for meth in methods]

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.85,
            patch_artist=True,
            manage_ticks=False,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
            flierprops=dict(marker="o", color=color, markersize=4, alpha=0.5),
            boxprops=dict(facecolor=color, alpha=0.70, linewidth=0.8),
        )
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.70,
                           label=CR_LABELS.get(cr, cr))
        )

    return legend_handles


def plot(rows, out_path: Path | None = None):
    data = organise(rows)
    cr_groups = [cr for cr in CR_ORDER if cr in data]

    fig, (ax_idx, ax_dat) = plt.subplots(1, 2, figsize=(11, 5), sharey=False)

    for ax, time_key, title in [
        (ax_idx, "index_s", "Index read time"),
        (ax_dat, "data_s",  "Data read time"),
    ]:
        handles = _box_group(ax, data, cr_groups, METHOD_ORDER, time_key)
        ax.set_xticks(range(1, len(METHOD_ORDER) + 1))
        ax.set_xticklabels(METHOD_ORDER, fontsize=11)
        ax.set_xlim(0.4, len(METHOD_ORDER) + 0.6)
        ax.set_ylabel("Time (s)")
        ax.set_title(title, fontsize=11)
        ax.legend(handles=handles, fontsize=9, title="CR status")
        ax.grid(axis="y", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle("Remote read benchmark by method and consolidated-metadata status",
                 fontsize=12)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else \
          Path(__file__).parent.parent / "260406_remote_testing_results_summary.txt"

    rows = load(src)
    if not rows:
        sys.exit(f"No data parsed from {src}")
    print(f"Parsed {len(rows)} rows from {src}")

    out = src.with_suffix(".png")
    plot(rows, out_path=out)
