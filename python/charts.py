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


spatial_array_total_area_mm2 = 0.84573
spatial_array_avg_Power = 193.0

spatial_array_area_margin_coeff = 1.2  # 20% area margin for interconnect, etc.
spatial_array_power_margin_coeff = 1.2  # 20% power margin for overheads

spatial_array_static_power_fraction = 0.2  # fraction of power that is static/leakage
spatial_array_size = (8,8)  # rows, cols

MatVec_input_size = "1024x16"
MatVec_weights_size = "16x1"
MatVec_Type = "complex_float"

MatMat_input_size = "1024x16"
MatMat_weights_size = "16x16"
MatMat_Type = "float"

FIR_input_size = "1024x1"
FIR_weights_size = "32x1"
FIR_Type = "complex_float"

VectorMagSq_input_size = "1024x1"
VectorMagSq_weights_size = "1024x1"
VectorMagSq_Type = "complex_float"

OuterProduct_input_size = "1024x32"
OuterProduct_weights_size = "1024x32"
OuterProduct_Type = "complex_float"

spatial_array_results_path = "./data/spatial_array_results.csv"
hls_matmat_results_path = "./data/MatMat_results.csv"
hls_matvec_results_path = "./data/MatVec_results.csv"
hls_fir_results_path = "./data/FIR_results.csv"
hls_outerproduct_results_path = "./data/OutProd_results.csv"
hls_vecmagsq_results_path = "./data/VecMagSq_results.csv"
hls_matmat_sweep_results_path = "./data/MatMat_1024x16_16x16_sweep_results.csv"
hls_vecmagsq_sweep_results_path = "./data/VecMagSq_1024x1_sweep_results.csv"


# def load_data(csv_path: str) -> pd.DataFrame:
#     df = pd.read_csv(csv_path)
#     # Try to coerce numeric fields
#     for col in ["Latency", "Throughput", "Area", "Power", "Utilization"]:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors="coerce")
#     # Drop rows with missing Kernel or Method
#     df = df.dropna(subset=["Kernel", "Method"])
#     return df

def load_data() -> pd.DataFrame:

    def _coerce_metrics(df: pd.DataFrame, metric_cols):
        for col in metric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=lambda c: (
            str(c)
            .replace("\ufeff", "")     # strip BOM
            .replace("\u00a0", " ")    # NBSP -> space
            .strip()                   # trim spaces
        ))
        # --- Normalize all string cells ---
        df = df.map(
            lambda x: (
                str(x)
                .replace("\ufeff", "")   # strip BOM
                .replace("\u00a0", " ")  # NBSP → space
                .strip()
            ) if isinstance(x, str) else x
        )
        return df

    metric_cols = ["Latency", "Throughput", "Area", "Power", "Utilization"]
    keep_cols = ["Kernel", "Method", "Inputs", "Weights", "Type", "Multipliers"] + metric_cols
    df = pd.read_csv(spatial_array_results_path)
    # Try to coerce numeric fields
    df = _normalize(df)
    df = _coerce_metrics(df, metric_cols)
    # keep only columns listed in keep_cols (drop all others)
    df["Method"] = "SpatialArray"
    df["Multipliers"] = np.prod(spatial_array_size)
    df["Area"] = spatial_array_area_margin_coeff * spatial_array_total_area_mm2
    cols_to_keep = [c for c in keep_cols if c in df.columns]
    df = df[cols_to_keep]
    df = df.reindex(columns=keep_cols)
    # Drop rows with missing Kernel or Method
    df = df.dropna(subset=["Kernel", "Method"])

    for kernel, path in [
            # ("MatMat", hls_matmat_results_path),
            ("MatMat_sweep", hls_matmat_sweep_results_path),
            # ("MatVec", hls_matvec_results_path),
            # ("FIR", hls_fir_results_path),
            # ("OuterProduct", hls_outerproduct_results_path),
            # ("VectorMagSq", hls_vecmagsq_results_path),
            ("VectorMagSq_sweep", hls_vecmagsq_sweep_results_path),
        ]:
        if os.path.exists(path):
            df_hls = pd.read_csv(path)
            df_hls = _normalize(df_hls)
            df_hls = _coerce_metrics(df_hls, metric_cols)
            if kernel.endswith("_sweep"):
                # keep only rows with 2 <= Multipliers <= 64 (inclusive)
                # keep only rows whose Multipliers value is in an explicit allowed list
                allowed_multipliers = [4, 16, 64]  # modify this list as needed
                df_hls["Multipliers"] = pd.to_numeric(df_hls["Multipliers"], errors="coerce")
                df_hls = df_hls[df_hls["Multipliers"].isin(allowed_multipliers)].copy()
                df_hls = df_hls.reset_index(drop=True)
                df_hls["Method"] = "HLS_" + df_hls["Multipliers"].astype(str)
                kernel = kernel.strip("_sweep")
            else:
                df_hls["Method"] = "HLS"
            df_hls["Kernel"] = kernel
            df_hls["Utilization"] = 100.0  # HLS designs assumed fully utilized
            cols_to_keep = [c for c in keep_cols if c in df.columns]
            df = df[cols_to_keep]
            df_hls = df_hls.reindex(columns=keep_cols)
            df = pd.concat([df, df_hls], ignore_index=True, sort=False, join="inner")

    # Keep only rows matching the kernel-specific (input_size, weight_size, type) specs.
    specs = {
        "MatVec": (MatVec_input_size, MatVec_weights_size, MatVec_Type),
        "MatMat": (MatMat_input_size, MatMat_weights_size, MatMat_Type),
        "FIR": (FIR_input_size, FIR_weights_size, FIR_Type),
        "OuterProduct": (OuterProduct_input_size, OuterProduct_weights_size, OuterProduct_Type),
        "VectorMagSq": (VectorMagSq_input_size, VectorMagSq_weights_size, VectorMagSq_Type),
    }
    # helper to find likely column name among variants
    def _find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    input_col = _find_col(df, ["Inputs"])
    weight_col = _find_col(df, ["Weights"])
    type_col = _find_col(df, ["Type"])

    # If none of the identifying columns exist, skip filtering (nothing to match)
    if any([input_col, weight_col, type_col]):
        mask_keep = pd.Series(False, index=df.index)
        for kernel, (insz, wgtsz, typ) in specs.items():
            kmask = df["Kernel"] == kernel
            if not kmask.any():
                continue
            cond = kmask
            if input_col:
                cond = cond & (df[input_col].astype(str) == str(insz))
            if weight_col:
                cond = cond & (df[weight_col].astype(str) == str(wgtsz))
            if type_col:
                cond = cond & df[type_col].fillna('').astype(str).str.startswith(str(typ))
            mask_keep = mask_keep | cond
        df = df[mask_keep].reset_index(drop=True)
    print(df)
    return df

def derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Derived metrics
    static_power = spatial_array_avg_Power * spatial_array_static_power_fraction
    dynamic_power = spatial_array_avg_Power - static_power
    mask = df["Method"].eq("SpatialArray")
    df.loc[mask, "Power"] = (
        (dynamic_power * df.loc[mask, "Utilization"] / 100.0 + static_power)
        * spatial_array_power_margin_coeff
    )
    
    mask = df["Method"].eq("SpatialArray")
    df.loc[mask, "Area"] = spatial_array_area_margin_coeff * spatial_array_total_area_mm2
    df["Perf_per_Area"] = df["Throughput"] / df["Area"]
    df["Perf_per_Power"] = df["Throughput"] / df["Power"]
    df["Energy_per_Op_J"] = df["Power"]*(10**(-3)) / (df["Throughput"]*(10**(-9)))
    # Delay (s) from Latency for EDP; if latency is missing, use NaN
    df["Delay_s"] = df["Latency"] * 1e-9
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
    ax.bar(x - width/2, series_a, width, label=label_a, color='#FF6F00')
    ax.bar(x + width/2, series_b, width, label=label_b, color='#57068C')
    ax.set_xticks(x)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xticklabels(categories, rotation=15, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=17, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=16)
    ax.legend(fontsize=14)

    # Apply log scale if requested
    if log_scale:
        ax.set_yscale('log')


def _bar_two_methods(df_metric: pd.DataFrame, metric_col: str, metric_name: str, filename_prefix: str, log_scale=False):
    if metric_col in ["Area", "Power", "Multipliers"]:
        df_metric = df_metric.copy()
        # Create an "All" summary row per method:
        # - For SpatialArray: Area and Power = max across that method's rows
        # - For HLS (and other methods): Area and Power = sum across that method's rows
        if "Kernel" in df_metric.columns and "Method" in df_metric.columns:
            # drop any pre-existing "All" rows to avoid duplicates
            df_metric = df_metric[df_metric["Kernel"] != "All"].copy()
            methods = df_metric["Method"].unique().tolist()
            summary_rows = []
            for m in methods:
                subset = df_metric[df_metric["Method"] == m]
                if subset.empty:
                    continue
                # compute area and power with NaN-safe ops
                area_val = np.nan
                power_val = np.nan
                mult_val = np.nan
                if "Area" in subset.columns:
                    if str(m).lower().startswith("spatial"):
                        area_val = float(np.nanmax(subset["Area"].values))
                    else:
                        area_val = float(np.nansum(subset["Area"].values))
                if "Power" in subset.columns:
                    if str(m).lower().startswith("spatial"):
                        power_val = float(np.nanmax(subset["Power"].values))
                    else:
                        power_val = float(np.nansum(subset["Power"].values))
                if "Multipliers" in subset.columns:
                    if str(m).lower().startswith("spatial"):
                        mult_val = int(np.nanmax(subset["Multipliers"].values))
                    else:
                        mult_val = int(np.nansum(subset["Multipliers"].values))
                # build a full row matching df_metric's columns (fill missing with NaN)
                row = {c: np.nan for c in df_metric.columns}
                row.update({"Kernel": "All", "Method": m})
                if "Area" in df_metric.columns:
                    row["Area"] = area_val
                if "Power" in df_metric.columns:
                    row["Power"] = power_val
                if "Multipliers" in df_metric.columns:
                    row["Multipliers"] = mult_val
                summary_rows.append(row)
            if summary_rows:
                df_metric = pd.concat([df_metric, pd.DataFrame(summary_rows)], ignore_index=True, sort=False)
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
    # df = _jitter_duplicates(df, xcol="Area", ycol="Throughput", frac=0.004)

    # Point per (kernel, method): x=Area, y=Throughput, size=Power, marker by method
    methods = sorted(df["Method"].unique().tolist())
    markers = ["o", "s", "^", "D", "P", "X", "*"]
    colors = ['#FF6F00', '#57068C']
    
    texts = []
    for i, m in enumerate(methods):
        d = df[df["Method"] == m]
        sizes = (d["Power"].fillna(0) + 1e-12)  # avoid zeros
        # scale bubble size
        size_scale = 800.0 / (np.nanmax(sizes) if np.nanmax(sizes) > 0 else 1.0)
        marker = markers[0]
        color = colors[i % len(colors)]
        sc = ax.scatter(d["Area"], d["Throughput"], s=sizes * size_scale, marker=marker, label=m, color=color)

        # add labels
        for _, row in d.iterrows():
            if not (pd.isna(row["Area"]) or pd.isna(row["Throughput"])):
                t = ax.annotate(
                    str(row["Kernel"]),
                    (row["Area"], row["Throughput"]),
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
    colors = ['#FF6F00', '#57068C']

    for i, m in enumerate(methods):
        d = df[df["Method"] == m]
        marker = markers[0]
        color = colors[i % len(colors)]
        ax.scatter(d["Latency"] / 1e3, d["Energy_per_Op_J"], marker=marker, label=m, color=color)
        for _, row in d.iterrows():
            if not (pd.isna(row["Latency"]) or pd.isna(row["Energy_per_Op_J"])):
                ax.annotate(str(row["Kernel"]), (row["Latency"] / 1e3, row["Energy_per_Op_J"]), textcoords="offset points", xytext=(5,5), fontsize=8)

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
    _bar_two_methods(df, "Perf_per_Area", "Performance per Area (Gops/s/mm²)", "bar_perf_per_area")
    _bar_two_methods(df, "Perf_per_Power", "Performance per Power (Gops/s/mW)", "bar_perf_per_power")
    # EDP
    _bar_two_methods(df, "EDP", "Energy-Delay Product (J·s)", "bar_edp", log_scale=True)

def radar_chart_per_kernel(df: pd.DataFrame):
    # Radar chart comparing methods for each kernel
    # Metrics: Latency (lower better), Throughput (higher better), Area (lower better), Power (lower better)
    metrics = ["Area", "Power", "Latency", "Throughput"]
    label_map = {
        "Latency": "Latency",
        "Throughput": "1/Throughput",
        "Area": "Area",
        "Power": "Power"
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
            if met == "Throughput":
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

        colors = ['#FF6F00', '#00A3E0', '#00B050', '#C00000']
        for mi, m in enumerate(methods):
            vals = [values_per_metric[pi][mi] for pi in range(num_vars)]
            vals += vals[:1]
            if m == "SpatialArray":
                color = '#57068C'
            else:
                color = colors[mi % len(colors)]
            ax.plot(angles, vals, linewidth=2, label=m, color=color)
            ax.fill(angles, vals, alpha=0.15, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([label_map[x] for x in metrics], fontsize=14, fontweight="bold")
        ax.tick_params(axis='x', pad=10, rotation=0)
        # ax.set_ylim(0, 1.15)
        ax.set_yticklabels([])
        ax.set_title(f"{k} (normalized)", fontsize=16, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=14)
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
    output_dir = Path("./figs/")
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------- Run pipeline --------
    csv_path = default_csv_path if os.path.exists(default_csv_path) else default_csv_path
    df_in = load_data()
    df = derived_metrics(df_in)

    # Store saved file paths
    saved_files = []

    # Bar charts: Latency & Throughput
    _bar_two_methods(df, "Latency", "Latency (Cycles)", "bar_latency", log_scale=True)
    _bar_two_methods(df, "Throughput", "Throughput (Gops/s)", "bar_throughput")
    _bar_two_methods(df, "Area", "Area (mm²)", "bar_area")
    _bar_two_methods(df, "Power", "Power (mW)", "bar_power")
    _bar_two_methods(df, "Multipliers", "Number of Multipliers", "bar_multipliers")

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
    display_cols = ["Kernel","Method","Latency","Throughput","Area","Power","Perf_per_Area","Perf_per_Power","Energy_per_Op_J","EDP"]
    summary_df = df[display_cols].copy()

    # import caas_jupyter_tools
    # caas_jupyter_tools.display_dataframe_to_user("Derived Metrics Summary", summary_df)

    # print("Saved files:")
    # for f in saved_files:
    #     print(f)
    # print("ZIP bundle:", zip_path)

    # zip_path
