# Python script to read the filled CSV and generate bar charts, trade-off scatter plots,
# energy-centric plots, and radar charts. Each chart is its own figure, using matplotlib only.
# It saves PNGs to /mnt/data/kernel_charts and prints the saved file paths.
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import zipfile
# from adjustText import adjust_text



def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Try to coerce numeric fields
    for col in ["Latency_cycles", "Throughput_gops_per_s", "Area_units", "Power_mW"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Drop rows with missing Kernel or Method
    df = df.dropna(subset=["Kernel", "Method"])
    return df

def derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Derived metrics
    df["Perf_per_Area"] = df["Throughput_gops_per_s"] / df["Area_units"]
    df["Perf_per_Power"] = df["Throughput_gops_per_s"] / df["Power_mW"]
    df["Energy_per_Op_J"] = df["Power_mW"]*(10**(-3)) / (df["Throughput_gops_per_s"]*(10**(-9)))
    # Delay (s) from Latency_cycles for EDP; if latency is missing, use NaN
    df["Delay_s"] = df["Latency_cycles"] * 1e-9
    df["EDP"] = df["Energy_per_Op_J"] * df["Delay_s"]
    return df

def _save_show(fig, filename):
    filepath = output_dir / filename
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(filepath)


def _grouped_bar(ax, categories, series_a, series_b, label_a, label_b, title, ylabel, log_scale=False):
    x = np.arange(len(categories))
    width = 0.38
    ax.bar(x - width/2, series_a, width, label=label_a)
    ax.bar(x + width/2, series_b, width, label=label_b)
    ax.set_xticks(x)
    ax.tick_params(axis='x')
    ax.set_xticklabels(categories, rotation=0, fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=12)

    # Apply log scale if requested
    if log_scale:
        ax.set_yscale('log')


def _bar_two_methods(df_metric: pd.DataFrame, metric_col: str, metric_name: str, filename_prefix: str, log_scale=False):
    # Expect df_metric to have exactly two methods per kernel
    kernels = sorted(df_metric["Kernel"].unique().tolist())
    methods = sorted(df_metric["Method"].unique().tolist())
    if len(methods) != 2:
        # If more or fewer methods exist, build one bar per method without grouping
        for m in methods:
            dfm = df_metric[df_metric["Method"] == m]
            vals = [dfm[dfm["Kernel"] == k][metric_col].values[0] if not dfm[dfm["Kernel"] == k].empty else np.nan for k in kernels]
            fig, ax = plt.subplots()
            ax.bar(np.arange(len(kernels)), vals)
            ax.set_xticks(np.arange(len(kernels)))
            ax.set_xticklabels(kernels)
            ax.set_title(f"{metric_name} - {m}")
            ax.set_ylabel(metric_name)
            path = _save_show(fig, f"{filename_prefix}_{m}.png")
            saved_files.append(path)
        return

    # Build aligned vectors for two methods
    m0, m1 = methods
    vals0, vals1 = [], []
    for k in kernels:
        v0 = df_metric[(df_metric["Kernel"] == k) & (df_metric["Method"] == m0)][metric_col]
        v1 = df_metric[(df_metric["Kernel"] == k) & (df_metric["Method"] == m1)][metric_col]
        vals0.append(v0.values[0] if len(v0) else np.nan)
        vals1.append(v1.values[0] if len(v1) else np.nan)

    fig, ax = plt.subplots()
    _grouped_bar(ax, kernels, vals0, vals1, m0, m1, f"{metric_name} by Kernel", metric_name, log_scale=log_scale)
    path = _save_show(fig, f"{filename_prefix}.png")
    saved_files.append(path)

def _jitter_duplicates(df, xcol, ycol, frac=0.005):
    """Spread identical (x,y) points in a small ring so labels/markers aren’t stacked."""
    out = df.copy()
    # axis spans used to scale jitter
    xspan = (out[xcol].max() - out[xcol].min()) or 1.0
    yspan = (out[ycol].max() - out[ycol].min()) or 1.0
    r_x, r_y = xspan * frac, yspan * frac

    g = out.groupby([xcol, ycol], dropna=False)
    for (_, _), idx in g.groups.items():
        idx = list(idx)
        n = len(idx)
        if n <= 1:
            continue
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        out.loc[idx, xcol] = out.loc[idx, xcol].values + r_x * np.cos(angles)
        out.loc[idx, ycol] = out.loc[idx, ycol].values + r_y * np.sin(angles)
    return out

def area_vs_throughput_scatter(df: pd.DataFrame, filename="scatter_area_vs_throughput.png"):
    fig, ax = plt.subplots()
    
    # — optional: jitter perfectly-identical coordinates so you can see duplicates —
    # df = _jitter_duplicates(df, xcol="Area_units", ycol="Throughput_gops_per_s", frac=0.004)

    # Point per (kernel, method): x=Area, y=Throughput, size=Power, marker by method
    methods = sorted(df["Method"].unique().tolist())
    markers = ["o", "s", "^", "D", "P", "X", "*"]
    
    texts = []
    for i, m in enumerate(methods):
        d = df[df["Method"] == m]
        sizes = (d["Power_mW"].fillna(0) + 1e-12)  # avoid zeros
        # scale bubble size
        size_scale = 800.0 / (np.nanmax(sizes) if np.nanmax(sizes) > 0 else 1.0)
        # marker = markers[i % len(markers)]
        marker = markers[0]
        sc = ax.scatter(d["Area_units"], d["Throughput_gops_per_s"], s=sizes * size_scale, marker=marker, label=m)
        
        # add labels
        for _, row in d.iterrows():
            if not (pd.isna(row["Area_units"]) or pd.isna(row["Throughput_gops_per_s"])):
                t = ax.annotate(
                    str(row["Kernel"]),
                    (row["Area_units"], row["Throughput_gops_per_s"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9,
                    # bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
                    # zorder=3
                )
                texts.append(t)

    ax.set_xlabel("Area (mm²)", fontsize=14)
    ax.set_ylabel("Throughput (Gops/s)", fontsize=14)
    ax.set_title("Trade-off: Area vs Throughput (bubble size = Power)", fontsize=15, fontweight="bold")
    ax.legend(fontsize=13, labelspacing=1.0, handletextpad=0.8, borderaxespad=0.6)
    ax.tick_params(axis='both', labelsize=12)
    ax.margins(0.06)
    
    # # nudge labels to avoid overlaps; add subtle leader lines
    # adjust_text(
    #     texts, ax=ax,
    #     expand_text=(1.05, 1.2), expand_points=(1.05, 1.2),
    #     arrowprops=dict(arrowstyle="-", lw=0.5, color="0.3", alpha=0.6)
    # )

    path = _save_show(fig, filename)
    saved_files.append(path)

def latency_vs_energy_scatter(df: pd.DataFrame, filename="scatter_latency_vs_energy.png"):
    fig, ax = plt.subplots()

    methods = sorted(df["Method"].unique().tolist())
    markers = ["o", "s", "^", "D", "P", "X", "*"]

    for i, m in enumerate(methods):
        d = df[df["Method"] == m]
        # marker = markers[i % len(markers)]
        marker = markers[0]
        ax.scatter(d["Latency_cycles"] / 1e3, d["Energy_per_Op_J"], marker=marker, label=m)
        for _, row in d.iterrows():
            if not (pd.isna(row["Latency_cycles"]) or pd.isna(row["Energy_per_Op_J"])):
                ax.annotate(str(row["Kernel"]), (row["Latency_cycles"] / 1e3, row["Energy_per_Op_J"]), textcoords="offset points", xytext=(5,5), fontsize=8)

    ax.set_xlabel("Latency (µs)", fontsize=14, labelpad=10)
    ax.set_ylabel("Energy per Op (J)", fontsize=14)
    ax.set_title("Energy-Centric Lens: Latency vs Energy per Op", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    ax.margins(0.06)

    path = _save_show(fig, filename)
    saved_files.append(path)

def energy_and_efficiency_bars(df: pd.DataFrame):
    # Energy/op
    _bar_two_methods(df, "Energy_per_Op_J", "Energy per Op (J)", "bar_energy_per_op")
    # Perf/Area and Perf/Power
    _bar_two_methods(df, "Perf_per_Area", "Performance per Area (Gops/s per mm²)", "bar_perf_per_area")
    _bar_two_methods(df, "Perf_per_Power", "Performance per Power (Gops/s/mW)", "bar_perf_per_power")
    # EDP
    _bar_two_methods(df, "EDP", "Energy-Delay Product (J·s)", "bar_edp", log_scale=True)

def radar_chart_per_kernel(df: pd.DataFrame):
    # Radar chart comparing methods for each kernel
    # Metrics: Latency_cycles (lower better), Throughput_gops_per_s (higher better), Area_units (lower better), Power_mW (lower better)
    metrics = ["Latency_cycles", "Throughput_gops_per_s", "Area_units", "Power_mW"]
    label_map = {
        "Latency_cycles": "Latency",
        "Throughput_gops_per_s": "1/Throughput",
        "Area_units": "Area",
        "Power_mW": "Power"
    }

    kernels = sorted(df["Kernel"].unique().tolist())
    methods = sorted(df["Method"].unique().tolist())

    # Normalize each metric within a kernel across methods to [0,1] where 1 is best
    # For metrics where lower is better: norm = 1 - (x - min)/(max - min)
    # For higher-is-better (Throughput): norm = (x - min)/(max - min)
    def normalize(vals, higher_is_better):
        arr = np.array(vals, dtype=float)
        vmin = np.nanmin(arr)
        vmax = np.nanmax(arr)
        if np.allclose(vmin, vmax) or np.isnan(vmin) or np.isnan(vmax):
            # If no variation or NaNs, default to 0.5
            return np.ones_like(arr) * 0.5

        # base = (arr - vmin) / (vmax - vmin)
        # return base if higher_is_better else 1.0 - base
        if higher_is_better:
            base = arr/vmax
        else:
            base = vmin/arr
        return base

    for k in kernels:
        dk = df[df["Kernel"] == k]
        if dk.empty:
            continue
        # Collect metric arrays aligned to methods
        values_per_metric = []
        for met in metrics:
            aligned = [dk[dk["Method"] == m][met].values[0] if not dk[dk["Method"] == m].empty else np.nan for m in methods]
            if met == "Throughput_gops_per_s":
                norm = normalize(aligned, higher_is_better=False)
            else:
                norm = normalize(aligned, higher_is_better=True)
            values_per_metric.append(norm)

        # angles
        num_vars = len(metrics)
        angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # close the loop

        fig = plt.figure()
        ax = plt.subplot(111, polar=True)

        for mi, m in enumerate(methods):
            vals = [values_per_metric[pi][mi] for pi in range(num_vars)]
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=2, label=m)
            ax.fill(angles, vals, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([label_map[x] for x in metrics], fontsize=11)
        ax.tick_params(axis='x', pad=10)
        # ax.set_ylim(0, 1.15)
        ax.set_yticklabels([])
        ax.set_title(f"Radar — {k} (normalized)", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=10)
        path = _save_show(fig, f"radar_{k}.png")
        saved_files.append(path)

def make_zip(zip_name="kernel_charts.zip"):
    zip_path = output_dir / zip_name
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in saved_files:
            if os.path.exists(f):
                zf.write(f, arcname=os.path.basename(f))
    return str(zip_path)


if __name__ == "__main__":
    # -------- Configuration --------
    # Change this to your uploaded/edited CSV path if different
    default_csv_path = "./data/kernel_compare.csv"
    output_dir = Path("./data/")
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------- Run pipeline --------
    csv_path = default_csv_path if os.path.exists(default_csv_path) else default_csv_path
    df_in = load_data(csv_path)
    df = derived_metrics(df_in)

    # Store saved file paths
    saved_files = []

    # Bar charts: Latency & Throughput
    _bar_two_methods(df, "Latency_cycles", "Latency (Cycles)", "bar_latency", log_scale=True)
    _bar_two_methods(df, "Throughput_gops_per_s", "Throughput (Gops/s)", "bar_throughput")
    _bar_two_methods(df, "Area_units", "Area (mm²)", "bar_area")
    _bar_two_methods(df, "Power_mW", "Power (mW)", "bar_power")

    # Efficiency and energy-centric bar charts
    energy_and_efficiency_bars(df)

    # Trade-off scatters
    area_vs_throughput_scatter(df)
    latency_vs_energy_scatter(df)

    # Radar charts (one per kernel)
    radar_chart_per_kernel(df)

    # Package into a ZIP for easy download
    # zip_path = make_zip()

    # Show a quick summary table with derived metrics
    display_cols = ["Kernel","Method","Latency_cycles","Throughput_gops_per_s","Area_units","Power_mW","Perf_per_Area","Perf_per_Power","Energy_per_Op_J","EDP"]
    summary_df = df[display_cols].copy()

    # import caas_jupyter_tools
    # caas_jupyter_tools.display_dataframe_to_user("Derived Metrics Summary", summary_df)

    # print("Saved files:")
    # for f in saved_files:
    #     print(f)
    # print("ZIP bundle:", zip_path)

    # zip_path
