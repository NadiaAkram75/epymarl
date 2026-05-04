#!/usr/bin/env python3
"""
plot_results.py
───────────────
Parse epymarl Sacred / CSV logs from all runs and produce:
  - Learning curves (mean ± std across seeds)  per env × alg
  - Final-performance bar chart
  - Gradient-norm stability plots
  - Per-algorithm seed-variance heatmap
  - One consolidated HTML report

Usage:
    python plot_all_results.py --results_root ~/marl_cv
    python plot_all_results.py --results_root ~/marl_cv --metric test_return_mean
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ALG_COLORS = {
    "qmix":   "#2196F3",
    "mappo":  "#F44336",
    "maddpg": "#4CAF50",
}
ALG_LABELS = {"qmix": "QMIX", "mappo": "MAPPO", "maddpg": "MADDPG"}
ENV_LABELS = {
    "lbf":       "LBF",
    "rware":     "RWARE",
    "mpe":       "MPE Spread",
    "overcooked":"Overcooked",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#cccccc",
    "axes.labelcolor":  "black",
    "text.color":       "black",
    "xtick.color":      "black",
    "ytick.color":      "black",
    "grid.color":       "#eeeeee",
    "legend.facecolor": "white",
    "legend.edgecolor": "#cccccc",
    "font.size":        11,
})


# ─────────────────────────────────────────────────────────────────────────────
# Log parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_sacred_metrics(log_dir: Path) -> pd.DataFrame:
    """
    epymarl writes Sacred JSON metrics to  <log_dir>/metrics.json
    or  <log_dir>/1/metrics.json  (Sacred run directory).
    Returns a tidy DataFrame with columns: step, metric, value.
    """
    candidates = list(log_dir.rglob("metrics.json"))
    if not candidates:
        return pd.DataFrame()

    frames = []
    for f in candidates:
        try:
            with open(f) as fh:
                raw = json.load(fh)
            for metric_name, payload in raw.items():
                steps  = payload.get("steps", [])
                values = payload.get("values", [])
                df = pd.DataFrame({"step": steps, "value": values})
                df["metric"] = metric_name
                frames.append(df)
        except Exception:
            pass

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def parse_plain_log(log_file: Path) -> pd.DataFrame:
    """
    Parse epymarl plain-text .log files.
    Stats are logged in multi-line blocks like:
      [INFO ...] my_main Recent Stats | t_env: 10000 | Episode: 200
      agent_grad_norm: 0.05   critic_grad_norm: 0.12   test_return_mean: 0.45 ...
    All key=value metrics are extracted and returned in long format.
    """
    STAT_BLOCK = re.compile(
        r"Recent Stats \| t_env:\s*(\d+).*?Episode.*?\n((?:.*\n)*?)(?=\[INFO|\Z)",
        re.MULTILINE
    )
    KV = re.compile(r"([\w_]+):\s*([-\d.]+)")

    rows = []
    try:
        text = log_file.read_text(errors="ignore")
        for m in STAT_BLOCK.finditer(text):
            step = int(m.group(1))
            stats = dict(KV.findall(m.group(2)))
            for metric_name, val in stats.items():
                rows.append({
                    "step":   step,
                    "metric": metric_name,
                    "value":  float(val),
                })
    except Exception:
        pass
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_run(log_file: Path) -> pd.DataFrame:
    """
    Try Sacred JSON first (legacy), then fall back to plain-text parsing.
    log_file is the direct .log path, e.g. logs/maddpg_lbf_s1.log
    """
    # Try Sacred metrics.json in the same directory
    df = parse_sacred_metrics(log_file.parent)
    if not df.empty:
        return df
    # Fall back to plain log parsing
    return parse_plain_log(log_file)


def collect_all_runs(results_root: Path):
    """
    Walk results_root/logs/ and collect DataFrames indexed by (alg, env, seed).
    Supports both naming conventions:
      - Plain log files:  <alg>_<env>_s<N>.log   (e.g. maddpg_lbf_s1.log)
      - Sacred run dirs:  <alg>__<env>__seed<N>   (e.g. maddpg__lbf__seed1)
    """
    data = defaultdict(list)   # (alg, env) -> list of DataFrames (one per seed)
    logs_dir = results_root / "logs"
    if not logs_dir.exists():
        sys.exit(f"[ERROR] logs directory not found: {logs_dir}")

    for entry in sorted(logs_dir.iterdir()):
        # ── Pattern 1: plain .log files like maddpg_lbf_s1.log ──────────────
        if entry.is_file() and entry.suffix == ".log":
            m = re.match(r"^(maddpg|mappo|qmix)_(lbf|rware|mpe|overcooked)_s(\d+)\.log$",
                         entry.name)
            if not m:
                continue
            alg, env, seed = m.group(1), m.group(2), int(m.group(3))
            df = load_run(entry)

        # ── Pattern 2: Sacred run directories like maddpg__lbf__seed1 ───────
        elif entry.is_dir():
            m = re.match(r"^([a-z]+)__([a-z]+)__seed(\d+)$", entry.name)
            if not m:
                continue
            alg, env, seed = m.group(1), m.group(2), int(m.group(3))
            df = load_run(entry / "train.log")  # fallback inside dir
            if df.empty:
                df = parse_sacred_metrics(entry)
        else:
            continue

        if not df.empty:
            df["alg"]  = alg
            df["env"]  = env
            df["seed"] = seed
            data[(alg, env)].append(df)
            print(f"  loaded  {entry.name}")
        else:
            print(f"  EMPTY   {entry.name}  (skipped)")

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def smooth(values, weight=0.6):
    """Exponential moving average."""
    smoothed, last = [], values[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return np.array(smoothed)


def plot_learning_curves(data, results_root, metric="test_return_mean"):
    """One figure per environment showing all algorithms (mean ± std across seeds)."""
    envs = sorted({env for _, env in data.keys()})
    plots_dir = results_root / "plots" / "learning_curves"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for env in envs:
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.suptitle(f"Learning Curves — {ENV_LABELS.get(env, env)}",
                     fontsize=14, fontweight="bold", y=1.01)

        has_data = False
        for alg in ["qmix", "mappo", "maddpg"]:
            seed_dfs = data.get((alg, env), [])
            if not seed_dfs:
                continue

            # Interpolate each seed onto a common step grid
            valid_steps = [
                df[df.metric == metric].step.max()
                for df in seed_dfs
                if metric in df.metric.values
            ]
            if not valid_steps:
                continue
            max_step = max(valid_steps)
            grid = np.linspace(0, max_step, 200)
            curves = []
            for df in seed_dfs:
                sub = df[df.metric == metric].sort_values("step")
                if sub.empty:
                    continue
                interp = np.interp(grid, sub.step.values, sub.value.values)
                curves.append(smooth(interp))

            if not curves:
                continue
            has_data = True
            arr  = np.array(curves)
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)

            color = ALG_COLORS[alg]
            label = ALG_LABELS[alg]
            ax.plot(grid, mean, color=color, lw=2, label=label)
            ax.fill_between(grid, mean - std, mean + std,
                            color=color, alpha=0.18)

        ax.set_xlabel("Environment Steps")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, linewidth=0.4)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{int(x/1e3)}k"
        ))
        if has_data:
            ax.legend(loc="lower right", framealpha=0.7)

        out = plots_dir / f"{env}__{metric}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [saved] {out.relative_to(results_root)}")


def plot_final_performance(data, results_root, metric="test_return_mean", last_pct=0.1):
    """Bar chart: mean of last {last_pct*100}% of training across seeds."""
    plots_dir = results_root / "plots"
    plots_dir.mkdir(exist_ok=True)

    envs = sorted({env for _, env in data.keys()})
    algs = ["qmix", "mappo", "maddpg"]

    n_env  = len(envs)
    n_alg  = len(algs)
    x      = np.arange(n_env)
    width  = 0.22
    offsets= np.linspace(-(n_alg-1)/2*width, (n_alg-1)/2*width, n_alg)

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Final Performance (mean of last 10 % training)",
                 fontsize=14, fontweight="bold")

    for i, alg in enumerate(algs):
        means, errs = [], []
        for env in envs:
            seed_dfs = data.get((alg, env), [])
            vals = []
            for df in seed_dfs:
                sub = df[df.metric == metric].sort_values("step")
                if sub.empty:
                    continue
                n = max(1, int(len(sub) * last_pct))
                vals.append(sub.value.iloc[-n:].mean())
            if vals:
                means.append(np.mean(vals))
                errs.append(np.std(vals))
            else:
                means.append(0); errs.append(0)

        bars = ax.bar(x + offsets[i], means, width,
                      label=ALG_LABELS[alg],
                      color=ALG_COLORS[alg],
                      yerr=errs, capsize=4,
                      error_kw={"ecolor": "#FFFFFF88", "lw": 1.5})

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.01 * abs(mean) + 0.001),
                    f"{mean:.3f}", ha="center", va="bottom",
                    fontsize=8, color="#D0D6F0")

    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABELS.get(e, e) for e in envs])
    ax.set_ylabel("Mean Test Return")
    ax.axhline(0, color="#666", lw=0.8, ls="--")
    ax.grid(True, axis="y", linewidth=0.4)
    ax.legend(framealpha=0.7)

    out = plots_dir / "final_performance.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out.relative_to(results_root)}")


def plot_gradient_norms(data, results_root):
    plot_learning_curves(data, results_root, metric="critic_grad_norm")


def plot_seed_variance_heatmap(data, results_root, metric="test_return_mean", last_pct=0.1):
    """Heatmap of std-dev across seeds — shows which combos are unstable."""
    plots_dir = results_root / "plots"
    plots_dir.mkdir(exist_ok=True)
    envs = sorted({env for _, env in data.keys()})
    algs = ["qmix", "mappo", "maddpg"]

    matrix = np.zeros((len(algs), len(envs)))
    for i, alg in enumerate(algs):
        for j, env in enumerate(envs):
            seed_dfs = data.get((alg, env), [])
            vals = []
            for df in seed_dfs:
                sub = df[df.metric == metric].sort_values("step")
                if sub.empty:
                    continue
                n = max(1, int(len(sub) * last_pct))
                vals.append(sub.value.iloc[-n:].mean())
            matrix[i, j] = np.std(vals) if len(vals) > 1 else float("nan")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Std-dev across seeds")
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels([ENV_LABELS.get(e, e) for e in envs])
    ax.set_yticks(range(len(algs)))
    ax.set_yticklabels([ALG_LABELS[a] for a in algs])
    ax.set_title("Seed Variance Heatmap (final performance std-dev)", fontsize=12)

    for i in range(len(algs)):
        for j in range(len(envs)):
            v = matrix[i, j]
            txt = f"{v:.3f}" if not np.isnan(v) else "N/A"
            ax.text(j, i, txt, ha="center", va="center",
                    color="black" if v < matrix.max()*0.6 else "white",
                    fontsize=9, fontweight="bold")

    out = plots_dir / "seed_variance_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out.relative_to(results_root)}")


def export_metrics_csv(data, results_root, metric="test_return_mean", last_pct=0.1):
    """Write a tidy CSV of final performance per (alg, env, seed)."""
    metrics_dir = results_root / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    rows = []
    for (alg, env), seed_dfs in data.items():
        for df in seed_dfs:
            seed = df.seed.iloc[0]
            sub  = df[df.metric == metric].sort_values("step")
            if sub.empty:
                continue
            n    = max(1, int(len(sub) * last_pct))
            rows.append({
                "alg":   alg,
                "env":   env,
                "seed":  seed,
                "final_mean": sub.value.iloc[-n:].mean(),
                "final_std":  sub.value.iloc[-n:].std(),
                "max_return": sub.value.max(),
                "steps":      sub.step.max(),
            })
    if rows:
        out = results_root / "metrics" / f"{metric}_summary.csv"
        pd.DataFrame(rows).sort_values(["env","alg","seed"]).to_csv(out, index=False)
        print(f"  [saved] {out.relative_to(results_root)}")


def build_html_report(results_root):
    """Collect all saved PNGs and write a simple HTML dashboard."""
    plots = sorted((results_root / "plots").rglob("*.png"))
    if not plots:
        return

    imgs_html = "\n".join(
        f'<figure><img src="{p.relative_to(results_root)}" '
        f'style="max-width:100%;border-radius:8px;"></figure>'
        for p in plots
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>MARL Pixel Sweep — Results</title>
  <style>
    body {{background:#12151F;color:#D0D6F0;font-family:'Segoe UI',sans-serif;margin:0;padding:24px}}
    h1   {{color:#7BBFEA;letter-spacing:2px}}
    h2   {{color:#52C96E;border-bottom:1px solid #2E3450;padding-bottom:6px}}
    figure{{margin:0 0 32px 0}}
    img  {{box-shadow:0 4px 24px #000a}}
  </style>
</head>
<body>
  <h1>Shared vs Independent Visual Encoders — Experiment Results</h1>
  <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
  <h2>Plots</h2>
  {imgs_html}
</body>
</html>"""

    out = results_root / "report.html"
    out.write_text(html)
    print(f"  [saved] report.html")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", required=True,
                        help="Path to results directory containing logs/ folder")
    parser.add_argument("--metric", default="test_return_mean",
                        help="Primary metric to plot")
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    if not results_root.exists():
        sys.exit(f"[ERROR] results_root not found: {results_root}")

    print(f"\n[INFO] Collecting runs from: {results_root}/logs/")
    data = collect_all_runs(results_root)
    if not data:
        sys.exit("[ERROR] No parsed run data found. "
                 "Check that logs exist and contain metrics.json or .log files.")

    combos = len(data)
    total_seeds = sum(len(v) for v in data.values())
    print(f"[INFO] Found {combos} (alg, env) combos — {total_seeds} seed runs total\n")

    print("[STEP 1/6] Learning curves (test return)...")
    plot_learning_curves(data, results_root, metric=args.metric)

    print("[STEP 2/6] Final performance bar chart...")
    plot_final_performance(data, results_root, metric=args.metric)

    print("[STEP 3/6] Gradient norm stability...")
    plot_gradient_norms(data, results_root)

    print("[STEP 4/6] Seed variance heatmap...")
    plot_seed_variance_heatmap(data, results_root, metric=args.metric)

    print("[STEP 5/5] Exporting metrics CSV...")
    export_metrics_csv(data, results_root, metric=args.metric)

    print(f"\n✔  All done. Plots saved to {results_root}/plots/")


if __name__ == "__main__":
    main()