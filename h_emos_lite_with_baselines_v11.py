#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# H-EMOS-Lite Comparison Framework: 
# Implements 4 approaches: 1) H-EMOS-Lite, 2) Independent DSM, 
# 3) Sequential optimization, 4) PSO metaheuristic

import argparse
import json
import time
import hashlib
import os
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import numpy as np

# Optional AC power flow validation 
try:
    import pandapower as pp
    import pandapower.networks as pn
except Exception:
    pp = None
    pn = None

from pathlib import Path

# === AUTO-FIX FOR IDE RUNS (VS / VS CODE) ===
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATADIR = SCRIPT_DIR / "datadir"

# ------------------------- Evaluation correctness helpers -------------------------

def energy_cost(sched: np.ndarray, prices: np.ndarray) -> float:
    """True energy cost (no peak penalty)."""
    return float((sched * prices[None, :]).sum())


def enforce_energy_conservation(
    sched: np.ndarray,
    loads_ref: np.ndarray,
    prices: np.ndarray,
    eps: float = 1e-9,
    spread_frac: float = 0.15
) -> np.ndarray:
    """
    Ensure per-household energy is conserved after shifting and clipping.

    Key fix vs earlier versions: distribute any deficit/surplus across multiple low/high price slots
    to avoid creating artificial spikes.
    """
    H, T = sched.shape
    target = loads_ref.sum(axis=1)
    current = sched.sum(axis=1)
    delta = target - current  # positive => need to add, negative => need to remove

    k = max(1, int(T * spread_frac))  # number of slots to spread over
    low_idx = np.argsort(prices)[:k]          # cheapest k
    high_idx = np.argsort(prices)[-k:][::-1]  # most expensive k

    out = np.clip(sched.copy(), 0.0, None)

    for h in range(H):
        d = float(delta[h])
        if abs(d) <= eps:
            continue

        if d > 0:
            add_each = d / len(low_idx)
            out[h, low_idx] += add_each
        else:
            remaining = -d
            for t in high_idx:
                if remaining <= eps:
                    break
                take = min(out[h, t], remaining / max(1, (len(high_idx))))
                out[h, t] -= take
                remaining -= take
            # If still remaining (rare), take from anywhere nonzero
            if remaining > eps:
                for t in np.argsort(out[h])[::-1]:
                    if remaining <= eps:
                        break
                    take = min(out[h, t], remaining)
                    out[h, t] -= take
                    remaining -= take

    return np.clip(out, 0.0, None)


def safe_reduction_pct(baseline: float, value: float) -> float:
    """Compute (baseline-value)/baseline*100 with safe bounds."""
    if baseline <= 1e-12:
        return 0.0
    pct = 100.0 * (baseline - value) / baseline
    # bound to avoid nonsense due to bugs/numerics
    return float(np.clip(pct, -100.0, 100.0))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

# ------------------------------ Utilities ------------------------------

def set_seed(seed: int = 42):
    np.random.seed(seed)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def create_comparison_plots(results_df: pd.DataFrame, output_dir: Path, make_convergence: bool = False) -> List[str]:
    """Create figures (PNG + PDF).

    Produces:
      - performance_metrics.(png/pdf): 2x2 bar chart of key metrics
      - load_profiles.(png/pdf): 2x2 before/after aggregate load profiles
      - (optional) convergence_history.(png/pdf): only if make_convergence=True and data available
    """
    ensure_dir(output_dir)
    generated: List[str] = []

    import matplotlib.pyplot as plt

    #  defaults
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
    })

    # --- Figure A: Performance metrics (bar charts) ---
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)


    metrics = [
       ("peak_reduction_pct", "Peak load reduction (%)", "Peak reduction (%)"),
    ("runtime_sec", "Computational time (s)", "Runtime (s)"),
    ]

    approaches = results_df["approach"].tolist()

    for ax, (col, title, ylabel) in zip(axes, metrics):
        vals = results_df[col].astype(float).values
        bars = ax.bar(approaches, vals)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        # value labels
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h, f"{h:.2f}" if abs(h) < 1000 else f"{h:.1f}",
                    ha="center", va="bottom", fontsize=8)

        # Make cost/peak reductions easier to read
        if col in ("peak_reduction_pct", "cost_reduction_pct"):
            ax.axhline(0, linewidth=1)

    fig.savefig(output_dir / "performance_metrics.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "performance_metrics.pdf", bbox_inches="tight")
    generated += ["performance_metrics.png", "performance_metrics.pdf"]
    plt.close(fig)

    # --- Figure B: Load profiles before/after (aggregate) ---
    # Expect columns: before_profile, after_profile as list-like; if not present, skip.
    if "load_profile_before" in results_df.columns and "load_profile_after" in results_df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
        for ax, (_, row) in zip(axes.ravel(), results_df.iterrows()):
            before = np.array(row["load_profile_before"], dtype=float)
            after = np.array(row["load_profile_after"], dtype=float)
            ax.plot(before, label="Before", linewidth=1.5)
            ax.plot(after, label="After", linewidth=1.5)
            ax.set_title(str(row["approach"]).upper())
            ax.set_xlabel("Time step (15-min)")
            ax.set_ylabel("Aggregate load (kW)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

        fig.savefig(output_dir / "load_profiles.png", dpi=600, bbox_inches="tight")
        fig.savefig(output_dir / "load_profiles.pdf", bbox_inches="tight")
        generated += ["load_profiles.png", "load_profiles.pdf"]
        plt.close(fig)

    return generated

    # --- Optional Figure C: Convergence (only if present) ---
    if make_convergence and "convergence_history" in results_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)

        for _, row in results_df.iterrows():
            name = str(row["approach"])
            conv = row["convergence_history"]
            peak = row["peak_history"] if "peak_history" in results_df.columns else None
            if isinstance(conv, (list, tuple)) and len(conv) > 1:
                axes[0].plot(conv, label=name, linewidth=1.5)
            if peak is not None and isinstance(peak, (list, tuple)) and len(peak) > 1:
                axes[1].plot(peak, label=name, linewidth=1.5)

        axes[0].set_title("Objective (best-so-far)")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Objective")
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title("Peak load (best-so-far)")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Peak (kW)")
        axes[1].grid(True, alpha=0.3)

        for ax in axes:
            ax.legend(loc="best", ncol=2)

        fig.savefig(output_dir / "convergence_history.png", dpi=600, bbox_inches="tight")
        fig.savefig(output_dir / "convergence_history.pdf", bbox_inches="tight")
        generated += ["convergence_history.png", "convergence_history.pdf"]
        plt.close(fig)

# ------------------------------ Real-data loaders ------------------------------

def _read_csv_any(path: Path) -> pd.DataFrame:
    """Read CSV with a friendly error message."""
    if path is None:
        raise ValueError("CSV path is None")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

def load_real_loads(loads_csv: Path) -> np.ndarray:
    """
    Loads shape expected: H x T numeric matrix in CSV (no index).
    If 'house' and 't' columns exist, it will pivot automatically.
    """
    df = _read_csv_any(loads_csv)
    # long format?
    cols = {c.lower() for c in df.columns}
    if {"house", "t", "load"}.issubset(cols) or {"house", "time", "load"}.issubset(cols):
        hcol = "house" if "house" in df.columns else [c for c in df.columns if c.lower()=="house"][0]
        tcol = "t" if "t" in df.columns else ("time" if "time" in df.columns else [c for c in df.columns if c.lower() in ("t","time")][0])
        lcol = "load" if "load" in df.columns else [c for c in df.columns if c.lower()=="load"][0]
        piv = df.pivot(index=hcol, columns=tcol, values=lcol).sort_index(axis=0).sort_index(axis=1)
        return piv.values.astype(float)
    # wide format: all numeric columns
    return df.select_dtypes(include=[np.number]).values.astype(float)

def load_real_prices(prices_csv: Path) -> np.ndarray:
    """Prices as a vector length T. Accepts column 'price' or first numeric column."""
    df = _read_csv_any(prices_csv)
    if "price" in df.columns:
        return df["price"].values.astype(float)
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        raise ValueError("prices.csv must contain a numeric column (e.g., 'price').")
    return num.iloc[:, 0].values.astype(float)

def load_real_ami(ami_csv: Path) -> np.ndarray:
    """AMI coordinates: expects columns (x,y) or (lat,lon)."""
    df = _read_csv_any(ami_csv)
    for pair in (("x","y"), ("lat","lon"), ("latitude","longitude")):
        if all(c in df.columns for c in pair):
            return df[list(pair)].values.astype(float)
    # otherwise take first two numeric columns
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        raise ValueError("ami.csv must contain x,y (or lat,lon) numeric columns.")
    return num.iloc[:, :2].values.astype(float)



# ------------------------------ Dataset-specific loaders  ------------------------------

def _infer_timestamp_column(df: pd.DataFrame) -> str:
    """Try to find a timestamp/datetime column in a dataframe."""
    cand = [c for c in df.columns if str(c).lower() in ("timestamp", "time", "datetime", "date", "utc", "localtime")]
    if cand:
        return cand[0]
    # also accept columns that contain 'time'
    cand = [c for c in df.columns if "time" in str(c).lower() or "date" in str(c).lower()]
    return cand[0] if cand else None

def _infer_value_column(df: pd.DataFrame) -> str:
    """Try to find a power/load/value column in a dataframe."""
    for key in ("power", "load", "demand", "kw", "kW", "value", "p"):
        cand = [c for c in df.columns if str(c).lower() == key.lower() or key.lower() in str(c).lower()]
        if cand:
            # prefer numeric
            for c in cand:
                if pd.api.types.is_numeric_dtype(df[c]):
                    return c
            return cand[0]
    # fallback: first numeric column
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        return None
    return num.columns[0]

def load_smartstar_directory(
    smartstar_dir: Path,
    H: int,
    T: int,
    freq: str = "15min",
    timezone: str = None,
) -> np.ndarray:
    """
     residential demand from a directory of per-house CSV files.

    Expected per-file: a time column + a numeric power/load column.

    Output: H x T numpy array (households x time steps), clipped to be non-negative.
    """
    smartstar_dir = Path(smartstar_dir)
    if not smartstar_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {smartstar_dir}")

    files = sorted([p for p in smartstar_dir.rglob("*.csv") if p.is_file()])
    if len(files) == 0:
        raise FileNotFoundError(f"No CSV files found under: {smartstar_dir}")

    series_list = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue

        tcol = _infer_timestamp_column(df)
        vcol = _infer_value_column(df)
        if tcol is None or vcol is None:
            continue

        # parse time
        try:
            ts = pd.to_datetime(df[tcol], errors="coerce")
        except Exception:
            continue
        ser = pd.Series(df[vcol].astype(float).values, index=ts).dropna()
        ser = ser[~ser.index.duplicated(keep="first")].sort_index()

        if timezone is not None:
            # make timezone-aware if possible
            try:
                if ser.index.tz is None:
                    ser.index = ser.index.tz_localize(timezone, ambiguous="infer", nonexistent="shift_forward")
                else:
                    ser.index = ser.index.tz_convert(timezone)
            except Exception:
                pass

        # resample to target frequency
        try:
            ser = ser.resample(freq).mean()
        except Exception:
            # if not a DatetimeIndex, skip
            continue

        # keep a contiguous window of length T (typically one day if freq=15min and T=96)
        ser = ser.dropna()
        if len(ser) < T:
            continue

        # take the first T samples
        ser = ser.iloc[:T]
        series_list.append(ser.values)

        if len(series_list) >= H:
            break

    if len(series_list) < max(1, min(H, 5)):
        raise ValueError(
            f"Dataset loader could not build enough series from {smartstar_dir}. "
            f"Parsed {len(series_list)} usable household files. "
            "Tip: pre-process dataset into a HxT matrix and pass it via --loads_csv."
        )

    loads = np.vstack(series_list)
    # if fewer than H, pad by repeating (for robustness in demos)
    if loads.shape[0] < H:
        reps = int(np.ceil(H / loads.shape[0]))
        loads = np.tile(loads, (reps, 1))[:H, :]

    loads = np.clip(loads, 0.0, None)
    return loads[:, :T]

def load_entsoe_prices_csv(prices_csv: Path, T: int) -> np.ndarray:
    """
    Load  day-ahead prices from a CSV export (best-effort).
    Output: vector length T.
    """
    df = _read_csv_any(prices_csv)

    # Prefer explicit 'price'
    if "price" in df.columns:
        prices = df["price"].astype(float).values
        return prices[:T]

    # ENTSO-E exports often contain a column like 'Price [EUR/MWh]'
    price_cols = [c for c in df.columns if "price" in str(c).lower()]
    if price_cols:
        # choose first numeric-like price column
        for c in price_cols:
            try:
                prices = pd.to_numeric(df[c], errors="coerce").dropna().values
                if len(prices) >= T:
                    return prices[:T].astype(float)
            except Exception:
                pass

    # fallback: first numeric column
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        # try coercing all columns
        for c in df.columns:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().sum() >= T:
                return coerced.dropna().values[:T].astype(float)
        raise ValueError("Could not infer a numeric price column from CSV.")
    return num.iloc[:, 0].values.astype(float)[:T]



def build_entsoe_prices_via_api(
    out_csv: Path,
    zone: str,
    start: str,
    end: str,
    api_key: str,
    tz: str = "Europe/Brussels",
    freq: str = "15min",
    T: int = 96,
) -> Path:
    """
    Download ENTSO-E day-ahead prices via the Transparency Platform Web API and write a canonical prices.csv.

    Notes:
      - ENTSO-E prices are often hourly; we upsample to `freq` by forward-fill.
      - `start`/`end` must be YYYY-MM-DD strings; timestamps are made timezone-aware.
    """
    try:
        from entsoe import EntsoePandasClient
    except Exception as e:
        raise ImportError("ENTSO-E API mode requires entsoe-py: pip install entsoe-py") from e

    start_ts = pd.Timestamp(start, tz=tz)
    end_ts = pd.Timestamp(end, tz=tz)
    if end_ts <= start_ts:
        raise ValueError(f"Invalid ENTSO-E time range: end ({end}) must be after start ({start}).")

    client = EntsoePandasClient(api_key=api_key)

    # Returns a pandas Series indexed by timestamps
    s = client.query_day_ahead_prices(zone, start=start_ts, end=end_ts)
    if isinstance(s, pd.DataFrame):
        # safety: pick first numeric column if a DF is returned
        s = s.select_dtypes(include=[np.number]).iloc[:, 0]

    s = s.sort_index().dropna()

    # Normalize to desired resolution (15-min by default)
    s = s.resample(freq).ffill().dropna()

    if len(s) < T:
        raise RuntimeError(f"Not enough price points after resampling: got {len(s)}, need {T}.")

    prices = s.iloc[:T].astype(float).values

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"price": prices}).to_csv(out_csv, index=False)
    return out_csv


def load_bus_map(bus_map_json: Path) -> Dict[str, Any]:
    """Load simplified bus map JSON used by the proxy model; required for current code."""
    p = Path(bus_map_json)
    if not p.exists():
        raise FileNotFoundError(f"bus_map.json not found: {p}")
    return json.loads(p.read_text())

class ACPowerFlowValidator:
    """
    Post-hoc AC power flow validation for IEEE 33-bus using pandapower.
   
    """
    def __init__(self, vmin: float = 0.95, vmax: float = 1.05):
        self.vmin = float(vmin)
        self.vmax = float(vmax)

    def validate_33bus(self, agg_load_kw: np.ndarray) -> Dict[str, Any]:
        if pp is None or pn is None:
            raise ImportError("pandapower is not installed. Install via: pip install pandapower")
        net = pn.case33bw()  # IEEE 33-bus radial feeder
        # Map total load to all PQ loads proportionally (simple, reproducible)
        base_p_mw = net.load["p_mw"].values.copy()
        base_total = float(base_p_mw.sum()) if base_p_mw.sum() > 0 else 1.0
        results = {"timesteps": [], "violations": 0, "min_v": [], "max_v": []}
        for t, PkW in enumerate(agg_load_kw):
            Pmw = float(PkW) / 1000.0
            scale = Pmw / base_total
            net.load.loc[:, "p_mw"] = base_p_mw * scale
            try:
                pp.runpp(net, algorithm="nr", init="auto", max_iteration=20, tolerance_mva=1e-6)
                v = net.res_bus["vm_pu"].values
                vmin_t = float(np.min(v))
                vmax_t = float(np.max(v))
                bad = (vmin_t < self.vmin) or (vmax_t > self.vmax)
                if bad:
                    results["violations"] += 1
                    results["timesteps"].append(int(t))
                results["min_v"].append(vmin_t)
                results["max_v"].append(vmax_t)
            except Exception:
                # treat solver failures as violations
                results["violations"] += 1
                results["timesteps"].append(int(t))
                results["min_v"].append(float("nan"))
                results["max_v"].append(float("nan"))
        return results

def generate_synthetic_data(base: Path, H: int = 100, T: int = 96, grid_buses: int = 33):
    """Generate synthetic data for all approaches"""
    ensure_dir(base)
    set_seed(7)
    
    # Loads with realistic patterns
    t = np.arange(T)
    diurnal = 1.2 + 0.8 * np.sin(2 * np.pi * (t - 24) / 96) ** 2
    loads = np.clip(diurnal + 0.3 * np.random.randn(H, T), 0.05, None)
    
    # Add appliance-specific patterns
    ev_idx = np.random.choice(H, size=H // 2, replace=False)
    for i in ev_idx:
        start = np.random.randint(70, 88)
        width = np.random.randint(4, 10)
        loads[i, start:start + width] += np.linspace(0.5, 1.5, width)
    
    # Add HVAC patterns
    hvac_idx = np.random.choice(H, size=H // 3, replace=False)
    for i in hvac_idx:
        day_start = 32  # 8:00
        day_end = 68    # 17:00
        loads[i, day_start:day_end] += 0.5 + 0.3 * np.random.randn(day_end - day_start)
    
    pd.DataFrame(loads).to_csv(base / "loads.csv", index=False)
    
    # Prices with peak/off-peak structure
    prices = 0.4 + 0.15 * np.sin(2 * np.pi * (t - 10) / 96) ** 2
    prices[60:84] += 0.35  # Evening peak
    prices += 0.05 * np.random.rand(T)
    pd.DataFrame({"price": prices}).to_csv(base / "prices.csv", index=False)
    
    # AMI coordinates
    N = 5000
    x = 10000 * np.random.rand(N)
    y = 10000 * np.random.rand(N)
    pd.DataFrame({"x": x, "y": y}).to_csv(base / "ami.csv", index=False)
    
    # Bus map with voltage sensitivity
    bus_map = {
        "num_buses": int(grid_buses),
        "volt_sens_scalar": 0.000015,
        "line_impedances": (np.random.randn(grid_buses - 1) * 0.01 + 0.05).tolist()
    }
    (base / "bus_map.json").write_text(json.dumps(bus_map, indent=2))
    
    print(f"[data] Generated synthetic data in {base}")

# ------------------------------ Voltage Model --------------------------

class VoltageModel:
    """Simplified voltage model for feasibility checking"""
    def __init__(self, num_buses: int = 33, vmin: float = 0.95, vmax: float = 1.05):
        self.num_buses = num_buses
        self.vmin = vmin
        self.vmax = vmax
        
    def check_voltage(self, agg_load: np.ndarray) -> Tuple[bool, List[int]]:
        """Check voltage constraints and return violations"""
        # Simplified voltage calculation: V = 1.0 - sensitivity * load
        sensitivity = 0.00002  # Voltage drop per kW
        base_voltage = 1.0
        voltages = base_voltage - sensitivity * agg_load
        
        violations = []
        for t, v in enumerate(voltages):
            if v < self.vmin or v > self.vmax:
                violations.append(t)
        
        return len(violations) == 0, violations

# ------------------------------ K-Means & Covering ---------------------

def kmeans_numpy(points: np.ndarray, K: int, iters: int = 10, seed: int = 123) -> Tuple[np.ndarray, np.ndarray]:
    """Simple K-means implementation"""
    set_seed(seed)
    N = points.shape[0]
    assert K <= N, "K must be <= number of points"
    idx = np.random.choice(N, K, replace=False)
    centers = points[idx].copy()
    
    for _ in range(iters):
        # Assign points to nearest center
        d2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        
        # Update centers
        for k in range(K):
            sel = (labels == k)
            if np.any(sel):
                centers[k] = points[sel].mean(axis=0)
    
    return centers, labels

def greedy_cover(points: np.ndarray, centers: np.ndarray, R: float, redundancy: int = 1):
    """Greedy covering algorithm for DAP placement"""
    R2 = R * R
    N = points.shape[0]
    K = centers.shape[0]
    
    chosen = []
    covered_count = np.zeros(N, dtype=int)
    remaining = set(range(K))
    
    # Precompute coverage
    cover_sets = []
    for k in range(K):
        d2 = ((points - centers[k]) ** 2).sum(axis=1)
        cover_sets.append(np.where(d2 <= R2)[0])
    
    while True:
        if np.all(covered_count >= redundancy):
            break
        if not remaining:
            break
            
        best_k, best_gain = None, -1
        for k in list(remaining):
            gain = np.sum(covered_count[cover_sets[k]] < redundancy)
            if gain > best_gain:
                best_k, best_gain = k, gain
        
        if best_k is None or best_gain <= 0:
            break
            
        chosen.append(best_k)
        remaining.remove(best_k)
        covered_count[cover_sets[best_k]] += 1
    
    return np.array(chosen, dtype=int)

# ------------------------- Base Optimization Classes -------------------

@dataclass
class OptimizationResult:
    """Container for optimization results"""
    approach: str
    peak_before: float
    peak_after: float
    peak_reduction_pct: float
    cost_before: float
    cost_after: float
    cost_reduction_pct: float
    runtime_sec: float
    voltage_violations: int
    voltage_violation_timesteps: List[int]
    load_profile_before: np.ndarray
    load_profile_after: np.ndarray
    convergence_history: Optional[List[float]] = None
    peak_history: Optional[List[float]] = None
    coverage_ratio: Optional[float] = None

class BaseOptimizer:
    """Base class for all optimization approaches"""
    def __init__(self, data, params):
        self.data = data
        self.params = params
        self.voltage_model = VoltageModel(vmin=params.vmin, vmax=params.vmax)
        
    def objective(self, sched: np.ndarray, prices: np.ndarray) -> float:
        """Objective function: total cost + peak penalty"""
        agg = sched.sum(axis=0)
        cost = (sched * prices[None, :]).sum()
        peak_penalty = 0.20 * agg.max() ** 2  # Quadratic penalty for peaks
        return cost + peak_penalty
    
    def check_feasibility(self, sched: np.ndarray) -> Tuple[bool, List[int]]:
        """Check voltage feasibility"""
        agg_load = sched.sum(axis=0)
        return self.voltage_model.check_voltage(agg_load)
    
    def calculate_dap_coverage(self, ami_points: np.ndarray) -> float:
        """Calculate DAP coverage ratio"""
        if ami_points.shape[0] == 0:
            return 0.0
        
        centers, _ = kmeans_numpy(ami_points, K=min(100, ami_points.shape[0] // 10))
        chosen = greedy_cover(ami_points, centers, R=1200.0)
        return len(chosen) / max(1, len(centers))

# ------------------------- 1. H-EMOS-Lite Approach ---------------------

class HEMOSLiteOptimizer(BaseOptimizer):
    """Original H-EMOS-Lite implementation"""
    
    def __init__(self, data, params):
        super().__init__(data, params)
        self.macro = params.macro
        
    def temporal_coarsening(self, data: np.ndarray) -> np.ndarray:
        """Aggregate 15-min data to hourly"""
        H, T = data.shape
        assert T % self.macro == 0
        M = T // self.macro
        return data.reshape(H, M, self.macro).mean(axis=2)
    
    def greedy_dsm(self, loads: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """Greedy DSM scheduling"""
        H, T = loads.shape
        sched = loads.copy()
        
        # Coarsen to macro level
        loads_macro = self.temporal_coarsening(loads)
        prices_macro = prices.reshape(-1, self.macro).mean(axis=1)
        
        # Shift loads from high-price to low-price periods
        for h in range(H):
            load_h = loads_macro[h]
            
            # Find highest and lowest price periods
            src_idx = int(np.argmax(load_h * prices_macro))  # expensive-load block
            # send to 3 cheapest blocks to avoid new peaks
            dst_blocks = np.argsort(prices_macro)[:6]

            if src_idx not in dst_blocks:
                move_amount = 0.45 * load_h[src_idx]
                load_h[src_idx] -= move_amount
                for d in dst_blocks:
                    load_h[int(d)] += move_amount / len(dst_blocks)
                
                # Expand back to original resolution
                sched[h] = np.repeat(np.clip(load_h, 0.0, None), self.macro)
        
        return sched
    
    def optimize(self) -> OptimizationResult:
        """Main optimization routine"""
        start_time = time.time()
        
        # Initial state
        loads = self.data.loads
        prices = self.data.prices
        agg_before = loads.sum(axis=0)
        cost_before = energy_cost(loads, prices)
        best_obj = self.objective(loads, prices)
        best_energy = energy_cost(loads, prices)
        
        # Main optimization loop
        best_sched = loads.copy()
        best_peak = float(agg_before.max())
        peak_history = [best_peak]

        target_peak = min(float(self.params.target_peak_factor) * best_peak,
                          (1.0 - float(self.params.min_peak_improve)) * best_peak)

        def hemos_obj(sched: np.ndarray, tp: float) -> float:
            agg = sched.sum(axis=0)
            energy = energy_cost(sched, prices)
            peak_excess = max(0.0, float(agg.max()) - float(tp))
            return energy + float(self.params.h_emos_peak_weight) * (peak_excess ** 2)

        best_cost = hemos_obj(best_sched, target_peak)
        convergence = [best_cost]
        for iteration in range(self.params.iters):
            # Exploration vs exploitation
            if np.random.rand() < self.params.epsilon:
                # Exploration: perturb prices
                price_perturbation = self.data.prices * (1 + 0.1 * np.random.randn())
                price_perturbation = np.clip(price_perturbation, 0.5, 2.0)
                current_prices = price_perturbation
            else:
                current_prices = prices
            
            # Apply DSM
            candidate_sched = self.greedy_dsm(best_sched, current_prices)
            candidate_sched = enforce_energy_conservation(candidate_sched, loads, prices)
            
            # Check feasibility
            feasible, violations = self.check_feasibility(candidate_sched)
            
            if feasible:
                candidate_cost = hemos_obj(candidate_sched, target_peak)
                candidate_peak = float(candidate_sched.sum(axis=0).max())
                candidate_energy = energy_cost(candidate_sched, prices)
                
                # Acceptance criterion:
                #  (1) strict improvement in objective AND not worse than target peak; OR
                #  (2) peak improves meaningfully while keeping energy cost within +1% of current best.
                if (((candidate_peak <= target_peak) and (candidate_cost < best_cost))
                    or ((candidate_peak < best_peak) and (candidate_energy <= 1.01 * best_energy))):
                    best_sched = candidate_sched
                    best_cost = candidate_cost
                    best_peak = candidate_sched.sum(axis=0).max()
                    best_energy = candidate_energy
            
            convergence.append(best_cost)
            peak_history.append(best_peak)
            
            # Adaptive target adjustment
            if iteration % 10 == 0:
                target_peak = self.params.target_peak_factor * best_peak

        def hemos_obj(sched: np.ndarray, tp: float) -> float:
            agg = sched.sum(axis=0)
            energy = energy_cost(sched, prices)
            peak_excess = max(0.0, float(agg.max()) - float(tp))
            return energy + self.params.h_emos_peak_weight * (peak_excess ** 2)

        
        runtime = time.time() - start_time
        agg_after = best_sched.sum(axis=0)
        cost_after = energy_cost(best_sched, prices)
        
        # Check final feasibility
        _, violations = self.check_feasibility(best_sched)
        
        # Calculate DAP coverage
        coverage = self.calculate_dap_coverage(self.data.ami)
        
        return OptimizationResult(
            approach="h_emos",
            peak_before=agg_before.max(),
            peak_after=agg_after.max(),
            peak_reduction_pct=safe_reduction_pct(float(agg_before.max()), float(agg_after.max())),
            cost_before=cost_before,
            cost_after=cost_after,
            cost_reduction_pct=safe_reduction_pct(float(cost_before), float(cost_after)),
            runtime_sec=runtime,
            voltage_violations=len(violations),
            voltage_violation_timesteps=violations,
            load_profile_before=agg_before,
            load_profile_after=agg_after,
            convergence_history=convergence,
            peak_history=peak_history,
            coverage_ratio=coverage
        )

# ------------------------- 2. Independent DSM Approach -----------------

class IndependentDSMOptimizer(BaseOptimizer):
    """Independent DSM optimization without coordination"""
    
    def optimize(self) -> OptimizationResult:
        """Independent DSM optimization"""
        start_time = time.time()
        
        loads = self.data.loads
        prices = self.data.prices
        agg_before = loads.sum(axis=0)
        cost_before = energy_cost(loads, prices)
        
        # Independent DSM baseline (energy-conserving peak-aware shifting)
        # Goal: reduce peak without coordination by shifting a fraction of each household's load
        # from system-peak timesteps to low-load/low-price timesteps.
        sched = loads.copy()
        H, T = sched.shape

        agg_base = loads.sum(axis=0)
        k_peak = max(1, int(0.10 * T))
        k_off = max(1, int(0.10 * T))

        # Peak timesteps: highest aggregate load (system peaks)
        peak_times = np.argsort(agg_base)[-k_peak:]
        # Off-peak candidates: low aggregate load AND low price
        score_off = 0.7 * (agg_base / (agg_base.max() + 1e-9)) + 0.3 * (prices / (prices.max() + 1e-9))
        off_times = np.argsort(score_off)[:k_off]

        alpha = 0.20  # shift 20% from peak times
        for h in range(H):
            removed = 0.0
            for t in peak_times:
                amt = alpha * sched[h, t]
                sched[h, t] -= amt
                removed += amt
            # distribute removed energy across off_times to avoid new peaks
            sched[h, off_times] += removed / len(off_times)

        sched = np.clip(sched, 0.0, None)
        sched = enforce_energy_conservation(sched, loads, prices)

        runtime = time.time() - start_time
        agg_after = sched.sum(axis=0)
        cost_after = energy_cost(sched, prices)
        
        # Check feasibility
        _, violations = self.check_feasibility(sched)
        
        return OptimizationResult(
            approach="independent_dsm",
            peak_before=agg_before.max(),
            peak_after=agg_after.max(),
            peak_reduction_pct=safe_reduction_pct(float(agg_before.max()), float(agg_after.max())),
            cost_before=cost_before,
            cost_after=cost_after,
            cost_reduction_pct=safe_reduction_pct(float(cost_before), float(cost_after)),
            runtime_sec=runtime,
            voltage_violations=len(violations),
            voltage_violation_timesteps=violations,
            load_profile_before=agg_before,
            load_profile_after=agg_after
        )

# ------------------------- 3. Sequential Optimization ------------------

class SequentialOptimizer(BaseOptimizer):
    """Sequential optimization: DSM first, then feasibility check"""
    
    def optimize(self) -> OptimizationResult:
        """Sequential optimization approach"""
        start_time = time.time()
        
        loads = self.data.loads
        prices = self.data.prices
        agg_before = loads.sum(axis=0)
        cost_before = energy_cost(loads, prices)
        
        # Step 1: DSM optimization (peak-aware shifting)
        sched = loads.copy()
        H, T = sched.shape

        agg_base = loads.sum(axis=0)
        k_peak = max(1, int(0.12 * T))
        k_off = max(1, int(0.12 * T))
        peak_times = np.argsort(agg_base)[-k_peak:]
        score_off = 0.6 * (agg_base / (agg_base.max() + 1e-9)) + 0.4 * (prices / (prices.max() + 1e-9))
        off_times = np.argsort(score_off)[:k_off]

        alpha = 0.30  # more aggressive than independent
        for h in range(H):
            removed = 0.0
            for t in peak_times:
                amt = alpha * sched[h, t]
                sched[h, t] -= amt
                removed += amt
            sched[h, off_times] += removed / len(off_times)

        sched = np.clip(sched, 0.0, None)
        sched = enforce_energy_conservation(sched, loads, prices)

        # Step 2: Feasibility check and repair
        feasible, violations = self.check_feasibility(sched)
        
        if not feasible and len(violations) > 0:
            # Simple repair: reduce load in violation periods
            for violation_time in violations:
                total_violation_load = sched[:, violation_time].sum()
                if total_violation_load > 0:
                    # Reduce by 10%
                    reduction = 0.1 * sched[:, violation_time]
                    sched[:, violation_time] -= reduction
                    
                    # Redistribute to non-violation periods
                    non_violation_times = [t for t in range(T) if t not in violations]
                    if non_violation_times:
                        redistribute_to = np.random.choice(non_violation_times, 
                                                         min(5, len(non_violation_times)))
                        for dst in redistribute_to:
                            sched[:, dst] += reduction / len(redistribute_to)
        
        sched = np.clip(sched, 0.0, None)
        sched = enforce_energy_conservation(sched, loads, prices)
        
        runtime = time.time() - start_time
        agg_after = sched.sum(axis=0)
        cost_after = energy_cost(sched, prices)
        
        # Final feasibility check
        _, final_violations = self.check_feasibility(sched)
        
        return OptimizationResult(
            approach="sequential",
            peak_before=agg_before.max(),
            peak_after=agg_after.max(),
            peak_reduction_pct=safe_reduction_pct(float(agg_before.max()), float(agg_after.max())),
            cost_before=cost_before,
            cost_after=cost_after,
            cost_reduction_pct=safe_reduction_pct(float(cost_before), float(cost_after)),
            runtime_sec=runtime,
            voltage_violations=len(final_violations),
            voltage_violation_timesteps=final_violations,
            load_profile_before=agg_before,
            load_profile_after=agg_after
        )

# ------------------------- 4. PSO Optimization -------------------------

class PSOOptimizer(BaseOptimizer):
    """Particle Swarm Optimization implementation"""
    
    def __init__(self, data, params,c1=1.5, c2=1.5, w=0.7):
        super().__init__(data, params)
        self.n_particles = params.n_particles if hasattr(params, 'n_particles') else 30
        self.w = 0.7  # inertia
        #self.c1 = 1.5 if c1 is None else float(c1)
        #self.c2 = 1.5 if c2 is None else float(c2)

        self.c1 = 1.5  # cognitive parameter
        self.c2 = 1.5  # social parameter
        
    def encode_solution(self, sched: np.ndarray) -> np.ndarray:
        """Encode schedule as 1D array"""
        return sched.flatten()
    
    def decode_solution(self, encoded: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """Decode 1D array back to schedule"""
        return encoded.reshape(shape)
    
    def optimize(self) -> OptimizationResult:
        """PSO optimization"""
        start_time = time.time()
        
        loads = self.data.loads
        prices = self.data.prices
        H, T = loads.shape
        agg_before = loads.sum(axis=0)
        cost_before = energy_cost(loads, prices)
        
        # Initialize particles
        n_vars = H * T
        particles = np.random.uniform(0.5, 1.5, (self.n_particles, n_vars)) * loads.flatten()
        velocities = np.random.uniform(-0.1, 0.1, (self.n_particles, n_vars))
        
        pbest_positions = particles.copy()
        pbest_values = np.full(self.n_particles, np.inf)
        gbest_position = None
        gbest_value = np.inf
        
        convergence = []
        peak_history = []
        
        for iteration in range(self.params.iters):
            for i in range(self.n_particles):
                # Decode particle
                particle_sched = self.decode_solution(particles[i], (H, T))
                
                # Check feasibility
                feasible, _ = self.check_feasibility(particle_sched)
                
                if feasible:
                    # Calculate objective
                    value = self.objective(particle_sched, prices)
                    
                    # Update personal best
                    if value < pbest_values[i]:
                        pbest_values[i] = value
                        pbest_positions[i] = particles[i].copy()
                    
                    # Update global best
                    if value < gbest_value:
                        gbest_value = value
                        gbest_position = particles[i].copy()
            
                
                      # Update velocities and positions
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(2)
                # --- SAFETY DEFAULTS (manual patch) ---
                
                if getattr(self, "c2", None) is None:
                  self.c2 = 1.5
                if getattr(self, "c1", None) is None:
                    self.c1 = 1.5
                if getattr(self, "w", None) is None:
                     self.w = 0.7

                 # ---- SAFE GLOBAL BEST ----
                if gbest_position is None:
                   gbest_ref = pbest_positions[i]   # fallback
                else:
                   gbest_ref = gbest_position   

                # --- SAFETY DEFAULTS (must be right before use) ---
                if self.c1 is None: self.c1 = 1.5
                if self.c2 is None: self.c2 = 1.5
                if self.w  is None: self.w  = 0.7

                # --- SAFE global-best reference ---
                gbest_ref = pbest_positions[i] if (gbest_position is None) else gbest_position

                cognitive = float(self.c1) * r1 * (pbest_positions[i] - particles[i])
                social    = float(self.c2) * r2 * (gbest_ref - particles[i])

                
                velocities[i] = self.w * velocities[i] + cognitive + social
                particles[i] += velocities[i]
                
                # Boundary check
                particles[i] = np.clip(particles[i], 0.1, 2.0 * loads.flatten())
            
            convergence.append(gbest_value if gbest_value < np.inf else cost_before)
            
            # Track peak
            if gbest_position is not None:
                best_sched = self.decode_solution(gbest_position, (H, T))
                peak_history.append(best_sched.sum(axis=0).max())
            else:
                peak_history.append(agg_before.max())
        
        runtime = time.time() - start_time
        
        # Get best solution
        if gbest_position is not None:
            best_sched = self.decode_solution(gbest_position, (H, T))
        else:
            best_sched = loads
        
        agg_after = best_sched.sum(axis=0)
        cost_after = gbest_value if gbest_value < np.inf else cost_before
        
        # Check final feasibility
        _, violations = self.check_feasibility(best_sched)
        
        return OptimizationResult(
            approach="pso",
            peak_before=agg_before.max(),
            peak_after=agg_after.max(),
            peak_reduction_pct=safe_reduction_pct(float(agg_before.max()), float(agg_after.max())),
            cost_before=cost_before,
            cost_after=cost_after,
            cost_reduction_pct=safe_reduction_pct(float(cost_before), float(cost_after)),
            runtime_sec=runtime,
            voltage_violations=len(violations),
            voltage_violation_timesteps=violations,
            load_profile_before=agg_before,
            load_profile_after=agg_after,
            convergence_history=convergence,
            peak_history=peak_history
        )

# ------------------------- Comparison Runner ---------------------------

class ComparisonRunner:
    """Runs and compares all optimization approaches"""
    
    def __init__(
        self,
        datadir: str,
        params,
        data_mode: str = "synthetic",
        loads_csv: Optional[str] = None,
        prices_csv: Optional[str] = None,
        ami_csv: Optional[str] = None,
        bus_map_json: Optional[str] = None,
    ):
        self.datadir = Path(datadir)
        self.params = params
        self.data_mode = str(data_mode).lower().strip()

        # Load data (real or synthetic). We always read from datadir/* unless explicit CSV paths are provided.
        self.data = type('Data', (), {})()

        if self.data_mode != "real":
            raise ValueError("Synthetic mode is disabled. Use real data only.")

        # Decide actual file paths
        loads_path = Path(loads_csv) if loads_csv else (self.datadir / "loads.csv")
        prices_path = Path(prices_csv) if prices_csv else (self.datadir / "prices.csv")
        ami_path = Path(ami_csv) if ami_csv else (self.datadir / "ami.csv")
        bus_map_path = Path(bus_map_json) if bus_map_json else (self.datadir / "bus_map.json")

        # Read
        self.data.loads = load_real_loads(loads_path)
        self.data.prices = load_real_prices(prices_path)
        self.data.ami = load_real_ami(ami_path)
        self.data.bus_map = load_bus_map(bus_map_path)
        self.data.T = int(self.data.prices.shape[0])

    def run_all_approaches(self) -> List[OptimizationResult]:
        """Run all optimization approaches"""
        results = []
        
        # 1. H-EMOS-Lite
        print("Running H-EMOS-Lite...")
        hemo_optimizer = HEMOSLiteOptimizer(self.data, self.params)
        results.append(hemo_optimizer.optimize())
        
        # 2. Independent DSM
        print("Running Independent DSM...")
        indep_optimizer = IndependentDSMOptimizer(self.data, self.params)
        results.append(indep_optimizer.optimize())
        
        # 3. Sequential Optimization
        print("Running Sequential Optimization...")
        seq_optimizer = SequentialOptimizer(self.data, self.params)
        results.append(seq_optimizer.optimize())
        
        # 4. PSO
        print("Running PSO...")
        pso_params = type('Params', (), {})()
        for attr in vars(self.params):
            setattr(pso_params, attr, getattr(self.params, attr))
        pso_params.n_particles = 30
        pso_optimizer = PSOOptimizer(self.data, pso_params)
        results.append(pso_optimizer.optimize())
        
        return results
    
    def create_summary_dataframe(self, results: List[OptimizationResult]) -> pd.DataFrame:
        """Create summary DataFrame from results"""
        summary_data = []
        
        for result in results:
            summary_data.append({
                'approach': result.approach,
                'peak_before': result.peak_before,
                'peak_after': result.peak_after,
                'peak_reduction_pct': result.peak_reduction_pct,
                'cost_before': result.cost_before,
                'cost_after': result.cost_after,
                'cost_reduction_pct': result.cost_reduction_pct,
                'runtime_sec': result.runtime_sec,
                'voltage_violations': result.voltage_violations,
                'voltage_violation_timesteps': result.voltage_violation_timesteps,
                'load_profile_before': result.load_profile_before.tolist(),
                'load_profile_after': result.load_profile_after.tolist(),
                'convergence_history': result.convergence_history,
                'peak_history': result.peak_history,
                'coverage_ratio': result.coverage_ratio
            })
        
        return pd.DataFrame(summary_data)

# ------------------------------- Main ----------------------------------

def main():
    parser = argparse.ArgumentParser(description="H-EMOS-Lite Comparison Framework")
    parser.add_argument(
    "--datadir",
    type=str,
    default=str(DEFAULT_DATADIR),
    help="Directory containing loads.csv, prices.csv, ami.csv, bus_map.json"
)

    parser.add_argument("--H", type=int, default=100, help="Number of households (H).")
    parser.add_argument("--T", type=int, default=96, help="Number of time steps (T). For 15-min slots in a day, T=96.")
    parser.add_argument("--data_mode", type=str, default="real", choices=["real"],
                        help="Real data only (synthetic mode disabled).")
    parser.add_argument("--loads_csv", type=str, default=None, help="Path to real loads.csv (H x T) or long format.")
    parser.add_argument("--prices_csv", type=str, default=None, help="Path to real prices.csv (vector length T).")
    parser.add_argument("--ami_csv", type=str, default=None, help="Path to real ami.csv (x,y or lat,lon).")
    parser.add_argument("--bus_map_json", type=str, default=None, help="Path to bus_map.json (required for proxy model).")

    # Dataset-specific helpers (optional). If provided, the script can generate the required CSV inputs.
    parser.add_argument("--smartstar_dir", type=str, default=None,
                        help="Directory containing Smart* per-house CSV files (optional). Used if --loads_csv is not provided.")
    parser.add_argument("--entsoe_csv", type=str, default=None,
                        help="ENTSO-E prices CSV export (optional). Used if --prices_csv is not provided.")

    parser.add_argument("--entsoe_api", action="store_true",
                        help="Download ENTSO-E day-ahead prices via Web API and create prices.csv (requires token).")
    parser.add_argument("--entsoe_zone", type=str, default="FR",
                        help="ENTSO-E bidding zone/country code (e.g., FR, DE_LU, BE, NL).")
    parser.add_argument("--entsoe_start", type=str, default="2024-01-01",
                        help="Start date (YYYY-MM-DD) in Europe/Brussels timezone.")
    parser.add_argument("--entsoe_end", type=str, default="2024-01-02",
                        help="End date (YYYY-MM-DD), typically next day for a 24h window.")
    parser.add_argument("--entsoe_tz", type=str, default="Europe/Brussels",
                        help="Timezone for ENTSO-E timestamps.")
    parser.add_argument("--entsoe_key", type=str, default=None,
                        help="ENTSO-E API token. If omitted, read env var ENTSOE_API_KEY.")
    parser.add_argument("--ieee33_bus_csv", type=str, default=None,
                        help="IEEE 33-bus bus data CSV (optional; used for documentation/traceability).")
    parser.add_argument("--ieee33_line_csv", type=str, default=None,
                        help="IEEE 33-bus line/branch data CSV (optional; used for documentation/traceability).")
    parser.add_argument("--ac_validate", action="store_true",
                        help="Run post-hoc AC power-flow validation on IEEE 33-bus using pandapower (slower).")
    parser.add_argument("--iters", type=int, default=100, help="Number of iterations")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Exploration rate")
    parser.add_argument("--vmin", type=float, default=0.95, help="Minimum voltage")
    parser.add_argument("--vmax", type=float, default=1.05, help="Maximum voltage")
    parser.add_argument("--target_peak_factor", type=float, default=0.9, help="Target peak reduction factor")
    parser.add_argument("--h_emos_peak_weight", type=float, default=6.0, help="H-EMOS peak-excess penalty weight (higher => more peak shaving).")
    parser.add_argument("--min_peak_improve", type=float, default=0.01, help="Minimum relative peak shaving pressure for H-EMOS (e.g., 0.01 = 1%).")
    parser.add_argument("--cost_tolerance", type=float, default=0.005, help="Allowed relative cost increase when accepting peak-improving moves (e.g., 0.005 = 0.5%).")
    parser.add_argument("--make_convergence", action="store_true", help="Also generate convergence plots (optional).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Prepare data directory / inputs
    base = Path(args.datadir)
    ensure_dir(base)

   
    args.data_mode = "real"

    #  We only:
    # (a) use explicit CSV paths, OR
    # (b) build canonical loads.csv/prices.csv from Smart* / ENTSO-E exports, OR
    # (c) use canonical files already present in --datadir .

    # --- Smart* -> loads.csv (only if loads source not provided)
    if args.loads_csv is None and args.smartstar_dir is not None:
        print(f"[data] Building loads.csv from Smart* directory: {args.smartstar_dir}")
        loads = load_smartstar_directory(Path(args.smartstar_dir), H=args.H, T=args.T)
        pd.DataFrame(loads).to_csv(base / "loads.csv", index=False)
        args.loads_csv = str(base / "loads.csv")
        print(f"[data] Wrote {base / 'loads.csv'}")

    # --- ENTSO-E -> prices.csv (only if prices source not provided)
    # --- ENTSO-E Web API -> prices.csv (preferred)
    if args.prices_csv is None and getattr(args, "entsoe_api", False):
        api_key = args.entsoe_key or os.getenv("ENTSOE_API_KEY")
        if not api_key:
            raise RuntimeError("ENTSO-E API key missing. Use --entsoe_key or set ENTSOE_API_KEY env var.")
        print(f"[data] Downloading ENTSO-E prices via API: zone={args.entsoe_zone} {args.entsoe_start}->{args.entsoe_end}")
        outp = build_entsoe_prices_via_api(
            out_csv=base / "prices.csv",
            zone=args.entsoe_zone,
            start=args.entsoe_start,
            end=args.entsoe_end,
            api_key=api_key,
            tz=args.entsoe_tz,
            freq="15min",
            T=args.T,
        )
        args.prices_csv = str(outp)
        print(f"[data] Wrote {outp}")

    if args.prices_csv is None and args.entsoe_csv is not None:
        print(f"[data] Building prices.csv from ENTSO-E export: {args.entsoe_csv}")
        prices = load_entsoe_prices_csv(Path(args.entsoe_csv), T=args.T)
        pd.DataFrame({"price": prices}).to_csv(base / "prices.csv", index=False)
        args.prices_csv = str(base / "prices.csv")
        print(f"[data] Wrote {base / 'prices.csv'}")

    # Require canonical files/paths (NO fallback).
    loads_path = Path(args.loads_csv) if args.loads_csv else (base / "loads.csv")
    prices_path = Path(args.prices_csv) if args.prices_csv else (base / "prices.csv")
    ami_path = Path(args.ami_csv) if args.ami_csv else (base / "ami.csv")
    bus_map_path = Path(args.bus_map_json) if args.bus_map_json else (base / "bus_map.json")

    missing = []
    if not loads_path.exists():
        missing.append("loads (provide --loads_csv or --smartstar_dir, or place loads.csv in --datadir)")
    if not prices_path.exists():
        missing.append("prices (provide --prices_csv or --entsoe_csv, or place prices.csv in --datadir)")
    if not ami_path.exists():
        missing.append("AMI coords (provide --ami_csv or place ami.csv in --datadir)")
    if not bus_map_path.exists():
        missing.append("bus_map (provide --bus_map_json or place bus_map.json in --datadir)")
    if missing:
        raise FileNotFoundError("STRICT REAL MODE: missing required inputs: " + "; ".join(missing))

    # Proof logs (paths + SHA256 fingerprints) to show exactly what data were used.
    def _sha16(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    print(f"[DATA] mode=real")
    print(f"[DATA] loads_path={loads_path} sha16={_sha16(loads_path)}")
    print(f"[DATA] prices_path={prices_path} sha16={_sha16(prices_path)}")
    print(f"[DATA] ami_path={ami_path} sha16={_sha16(ami_path)}")
    print(f"[DATA] bus_map_path={bus_map_path} sha16={_sha16(bus_map_path)}")
# Create parameters object
    params = type('Params', (), {})()
    params.iters = args.iters
    params.epsilon = args.epsilon
    params.vmin = args.vmin
    params.vmax = args.vmax
    params.target_peak_factor = args.target_peak_factor
    params.h_emos_peak_weight = args.h_emos_peak_weight
    params.min_peak_improve = args.min_peak_improve
    params.cost_tolerance = args.cost_tolerance
    params.macro = 4
    params.seed = args.seed
    
    # Run comparison
    print("\n" + "="*60)
    print("H-EMOS-Lite Comparison Framework")
    print("="*60 + "\n")
    
    runner = ComparisonRunner(args.datadir, params, data_mode=args.data_mode, loads_csv=args.loads_csv, prices_csv=args.prices_csv, ami_csv=args.ami_csv, bus_map_json=args.bus_map_json)
    results = runner.run_all_approaches()
    
    # Create summary
    df = runner.create_summary_dataframe(results)

    # Optional: post-hoc AC power-flow validation (IEEE 33-bus) for the final aggregate profile
    if args.ac_validate:

        print("\nRunning AC power-flow validation (IEEE 33-bus) using pandapower...")
        validator = ACPowerFlowValidator(vmin=args.vmin, vmax=args.vmax)
        ac_reports = {}
        for _, row in df.iterrows():
            agg_after = np.array(row["load_profile_after"], dtype=float)
            ac_reports[str(row["approach"])] = validator.validate_33bus(agg_after)
        (results_dir := base / "results")  # ensure exists later too
        ensure_dir(results_dir)
        (results_dir / "ac_validation.json").write_text(json.dumps(ac_reports, indent=2))
        # Add summary column
        df["ac_violations"] = df["approach"].map(lambda a: ac_reports.get(str(a), {}).get("violations", None))
    
    # Display results
    print("\n" + "="*60)
    print("COMPARISON RESULTS SUMMARY")
    print("="*60)
    
    display_cols = ['approach', 'peak_reduction_pct', 'cost_reduction_pct', 
                   'runtime_sec', 'voltage_violations']
    print(df[display_cols].to_string(index=False))
    
    # Save detailed results
    results_dir = base / "results"
    ensure_dir(results_dir)
    
    # Save JSON results
    df.to_json(results_dir / "detailed_results.json", indent=2, orient='records')
    
    
    
    # Create comparison plots
    print("\nGenerating comparison plots...")
    generated_files = create_comparison_plots(df, results_dir, make_convergence=args.make_convergence)
    
    print(f"\nResults saved to: {results_dir}")
    print("\n" + "="*60)
    print("KEY FINDINGS:")
    print("="*60)
    
    # Analyze results
    best_peak_reduction = df.loc[df['peak_reduction_pct'].idxmax()]
    fastest = df.loc[df['runtime_sec'].idxmin()]
   
    
   
    print(f"Best Cost Reduction: {best_cost_reduction['approach']} ({best_cost_reduction['cost_reduction_pct']:.2f}%)")
    print(f"Fastest: {fastest['approach']} ({fastest['runtime_sec']:.2f} seconds)")
   
    
    print("\n" + "="*60)
    print("Plot files generated:")
    print("="*60)
    print("1. performance_metrics.png / performance_metrics.pdf")
print("2. load_profiles.png / load_profiles.pdf")
print("3. convergence_history.png / convergence_history.pdf (only if --make_convergence)")
if __name__ == "__main__":
    main()