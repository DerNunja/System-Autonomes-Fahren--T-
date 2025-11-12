"""
# Einzelne Fahrt
python3.11 analyse.py --mode single --csv path/to/run.csv --out run.html --start 0 --end 60 --downsample 2

# Vergleich zweier Fahrten
python3.11 analyse.py --mode compare --csv good.csv --csv2 bad.csv \
  --out compare.html --metrics_out compare_metrics.csv --start 0 --end 60 --downsample 2
"""
  
import argparse
import math
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------- Konfiguration -------------------------------- #

TIME_COL = "timestamp"

VECTOR_COLUMNS = {
    "rrp_pos": ("rrp_pos_x", "rrp_pos_y", "rrp_pos_z"),
    "rrp_lin_vel": ("rrp_lv_x", "rrp_lv_y", "rrp_lv_z"),
    "rrp_rot_vel": ("rrp_rv_x", "rrp_rv_y", "rrp_rv_z"),
    # Achtung: Reihenfolge hier als (w,x,y,z) interpretiert – bei Bedarf anpassen!
    "rrp_quat": ("rrp_qw", "rrp_qx", "rrp_qy", "rrp_qz"),
}

SCALAR_MAYBE_INTERESTING = [
    "wheel_position",
    "wheel_adas_position_K_p",
    "wheel_adas_position_K_D",
    "wheel_adas_velocity_K_v_FF",
    "wheel_adas_velocity_K_p",
    "wheel_adas_velocity_K_i",
    "wheel_adas_max_velocity_n_max_LW",
    "wheel_adas_max_torque_M_max_LW",
    "wheel_adas_max_acceleration_a_max_LW",
    "wheel_raw_position_adas",
    "wheel_raw_torque_adas",
]

# ------------------------------- CLI -------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CSV-Plot & Analyse (Plotly) – single & compare")
    p.add_argument("--mode", choices=["single", "compare"], default="single", help="Analysemodus")
    p.add_argument("--csv", type=str, required=True, help="Pfad zur CSV-Datei (single/compare: Run A)")
    p.add_argument("--csv2", type=str, help="Pfad zur zweiten CSV (nur compare: Run B)")
    p.add_argument("--out", type=str, default="plots.html", help="Ziel-HTML mit interaktiven Plots")
    p.add_argument("--metrics_out", type=str, default=None, help="(compare) CSV für Kennzahlen-Tabelle")
    p.add_argument("--start", type=float, default=None, help="Startzeit in s (inklusive)")
    p.add_argument("--end", type=float, default=None, help="Endzeit in s (exklusiv)")
    p.add_argument("--downsample", type=int, default=1, help="Jeden n-ten Punkt behalten (>=1)")
    p.add_argument("--labelA", type=str, default="good", help="Label für CSV A (compare)")
    p.add_argument("--labelB", type=str, default="bad", help="Label für CSV B (compare)")
    return p.parse_args()

# ------------------------------- Utils -------------------------------- #

def is_units_row(row: pd.Series) -> bool:
    v = str(row.get(TIME_COL, "")).strip().lower()
    return v in {"s", "sec", "seconds"}

def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def expand_vector_column(df: pd.DataFrame, col: str, axis_names: Tuple[str, ...]) -> pd.DataFrame:
    if col not in df.columns:
        return df
    raw = df[col].astype(str).str.strip().str.replace('"', "", regex=False)
    parts = raw.apply(lambda x: [p.strip() for p in x.split(",")] if x != "" else [np.nan]*len(axis_names))
    fixed = parts.apply(lambda lst: (lst + [np.nan]*len(axis_names))[: len(axis_names)])
    for i, axis in enumerate(axis_names):
        df[axis] = pd.to_numeric(fixed.apply(lambda lst: lst[i]), errors="coerce")
    return df

def quat_to_euler_zyx(qw, qx, qy, qz):
    norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if norm == 0:
        return (np.nan, np.nan, np.nan)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    siny_cosp = 2.0 * (qw*qz + qx*qy);  cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
    sinp = 2.0 * (qw*qy - qz*qx)
    pitch = math.degrees(math.copysign(math.pi/2, sinp)) if abs(sinp) >= 1 else math.degrees(math.asin(sinp))
    sinr_cosp = 2.0 * (qw*qx + qy*qz);  cosr_cosp = 1.0 - 2.0 * (qx*qx + qy*qy)
    roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))
    return yaw, pitch, roll

def compute_sampling_rate(t: pd.Series) -> Optional[float]:
    t = pd.to_numeric(t, errors="coerce").dropna()
    if len(t) < 2: return None
    dt = np.diff(t.values); dt = dt[dt > 0]
    if len(dt) == 0: return None
    return 1.0 / np.median(dt)

def downsample_df(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n is None or n <= 1: return df
    return df.iloc[::n, :].reset_index(drop=True)

# ------------------------------- Pipeline: Laden, Ableiten -------------------------------- #

def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    mask_units = df.apply(is_units_row, axis=1)
    if mask_units.any():
        df = df.loc[~mask_units].reset_index(drop=True)
    df.replace("-", np.nan, inplace=True)
    if TIME_COL in df.columns:
        df[TIME_COL] = coerce_numeric(df[TIME_COL])
    for col, axes in VECTOR_COLUMNS.items():
        if col in df.columns:
            df = expand_vector_column(df, col, axes)
    if all(c in df.columns for c in VECTOR_COLUMNS.get("rrp_quat", ())):
        qw, qx, qy, qz = (df["rrp_qw"], df["rrp_qx"], df["rrp_qy"], df["rrp_qz"])
        eulers = [quat_to_euler_zyx(w, x, y, z) for w, x, y, z in zip(qw, qx, qy, qz)]
        df["yaw_deg"], df["pitch_deg"], df["roll_deg"] = zip(*eulers)
    for c in SCALAR_MAYBE_INTERESTING:
        if c in df.columns:
            df[c] = coerce_numeric(df[c])
    return df

def filter_time_window(df: pd.DataFrame, start: Optional[float], end: Optional[float]) -> pd.DataFrame:
    if TIME_COL not in df.columns: return df
    mask = pd.Series(True, index=df.index)
    if start is not None: mask &= df[TIME_COL] >= start
    if end is not None:   mask &= df[TIME_COL] < end
    return df.loc[mask].reset_index(drop=True)

def derive_signals(df: pd.DataFrame, time_col=TIME_COL) -> pd.DataFrame:
    out = df.copy()
    if time_col in out.columns:
        t = out[time_col].astype(float).values
    else:
        t = np.arange(len(out), dtype=float)
    # Speed
    if set(["rrp_lv_x","rrp_lv_y","rrp_lv_z"]).issubset(out.columns):
        out["speed"] = np.sqrt(out["rrp_lv_x"]**2 + out["rrp_lv_y"]**2 + out["rrp_lv_z"]**2)
    # Yaw-Rate
    if "rrp_rv_z" in out.columns:
        out["yaw_rate"] = out["rrp_rv_z"]
    elif "yaw_deg" in out.columns:
        yaw_rad = np.deg2rad(pd.to_numeric(out["yaw_deg"], errors="coerce").fillna(method="ffill").fillna(0.0))
        out["yaw_rate"] = np.gradient(yaw_rad, t)
    # a_lat & jerk
    if "speed" in out.columns and "yaw_rate" in out.columns:
        out["a_lat"] = out["speed"] * out["yaw_rate"]
        out["jerk_lat"] = np.gradient(out["a_lat"].astype(float).values, t)
    # Lenkgeschwindigkeit + reversals
    if "wheel_position" in out.columns:
        wp = pd.to_numeric(out["wheel_position"], errors="coerce").fillna(method="ffill").fillna(0.0).values
        out["steer_rate"] = np.gradient(wp, t)
        sr = np.sign(pd.Series(out["steer_rate"]).fillna(0).values)
        reversals = (np.roll(sr, 1) * sr < 0).astype(int); reversals[0] = 0
        out["steer_reversal"] = reversals
    # Torque-Ausnutzung
    if "wheel_raw_torque_adas" in out.columns and "wheel_adas_max_torque_M_max_LW" in out.columns:
        denom = pd.to_numeric(out["wheel_adas_max_torque_M_max_LW"], errors="coerce").replace(0, np.nan)
        out["torque_util"] = pd.to_numeric(out["wheel_raw_torque_adas"], errors="coerce").abs() / denom
    return out

# ------------------------------- Kennzahlen & Vergleich -------------------------------- #

def compute_metrics(df: pd.DataFrame, label:str, time_col=TIME_COL) -> pd.Series:
    s = pd.Series(name=label, dtype=float)

    def qstats(col):
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(x)==0: return np.nan, np.nan, np.nan, np.nan
        return x.mean(), x.std(), x.quantile(0.95), x.max()

    for col, key in [("speed","speed"), ("a_lat","a_lat"), ("jerk_lat","jerk_lat"),
                     ("steer_rate","steer_rate"), ("wheel_raw_torque_adas","torque")]:
        if col in df.columns:
            m, sd, p95, mx = qstats(col)
            s[f"{key}_mean"] = m; s[f"{key}_std"] = sd; s[f"{key}_p95"] = p95; s[f"{key}_max"] = mx

    if "steer_rate" in df.columns:
        sr = pd.to_numeric(df["steer_rate"], errors="coerce").dropna()
        s["steer_rate_rms"] = float(np.sqrt(np.mean(sr**2))) if len(sr) else np.nan

    if "steer_reversal" in df.columns and time_col in df.columns and df[time_col].notna().any():
        dt_total = df[time_col].max() - df[time_col].min()
        minutes = dt_total/60 if dt_total and dt_total>0 else np.nan
        s["steer_reversals_per_min"] = float(df["steer_reversal"].sum() / minutes) if minutes else np.nan

    if "torque_util" in df.columns:
        s["torque_sat_frac"] = float((df["torque_util"] > 0.9).mean())

    if "wheel_position" in df.columns and "yaw_rate" in df.columns:
        a = pd.to_numeric(df["wheel_position"], errors="coerce").fillna(0).values
        b = pd.to_numeric(df["yaw_rate"], errors="coerce").fillna(0).values
        if a.std() > 1e-9 and b.std() > 1e-9:
            a = (a - a.mean())/a.std(); b = (b - b.mean())/b.std()
            xcorr = np.correlate(a, b, mode="full")
            lag_idx = xcorr.argmax() - (len(a)-1)
            s["steer_yaw_corr"] = float(xcorr.max()/len(a))
            if time_col in df.columns and df[time_col].notna().sum()>1:
                dt = np.median(np.diff(df[time_col].dropna()))
                s["steer_yaw_lag_s"] = float(lag_idx * (dt if dt>0 else np.nan))
        else:
            s["steer_yaw_corr"] = np.nan
            s["steer_yaw_lag_s"] = np.nan
    return s

def compare_runs(dfA: pd.DataFrame, dfB: pd.DataFrame, labelA="good", labelB="bad") -> pd.DataFrame:
    mA = compute_metrics(dfA, labelA)
    mB = compute_metrics(dfB, labelB)
    comp = pd.concat([mA, mB], axis=1)
    comp["delta(B-A)"] = comp.get(labelB, pd.Series()) - comp.get(labelA, pd.Series())
    return comp

# ------------------------------- Plot-Builders -------------------------------- #

def build_single_figure(df: pd.DataFrame, title="Fahrdaten – Interaktive Analyse") -> go.Figure:
    rows = 4
    specs = [[{"secondary_y": True}], [{}], [{}], [{}]]
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, specs=specs,
        subplot_titles=("Lenkung & Torque", "Position rrp_pos (x,y,z)", "Euler (Yaw/Pitch/Roll)", "Lin. Geschwindigkeit rrp_lin_vel (x,y,z)")
    )
    x = df[TIME_COL] if TIME_COL in df.columns else np.arange(len(df))
    if "wheel_position" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["wheel_position"], mode="lines", name="wheel_position"), row=1, col=1, secondary_y=False)
    if "wheel_raw_torque_adas" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["wheel_raw_torque_adas"], mode="lines", name="wheel_raw_torque_adas"), row=1, col=1, secondary_y=True)
    for axis in ["rrp_pos_x", "rrp_pos_y", "rrp_pos_z"]:
        if axis in df.columns:
            fig.add_trace(go.Scatter(x=x, y=df[axis], mode="lines", name=axis), row=2, col=1)
    for axis in ["yaw_deg", "pitch_deg", "roll_deg"]:
        if axis in df.columns:
            fig.add_trace(go.Scatter(x=x, y=df[axis], mode="lines", name=axis), row=3, col=1)
    any_lv = False
    for axis in ["rrp_lv_x", "rrp_lv_y", "rrp_lv_z"]:
        if axis in df.columns:
            fig.add_trace(go.Scatter(x=x, y=df[axis], mode="lines", name=axis), row=4, col=1)
            any_lv = True
    if not any_lv:
        fig.update_yaxes(title_text="(keine rrp_lin_vel-Spalten gefunden)", row=4, col=1)
    fig.update_xaxes(title_text="Zeit [s]" if TIME_COL in df.columns else "Index", row=rows, col=1)
    fig.update_yaxes(title_text="Lenkung", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Torque", row=1, col=1, secondary_y=True)
    fig.update_layout(height=900, title=title, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=60, r=20, t=60, b=40))
    return fig

def build_compare_figure(dfA: pd.DataFrame, dfB: pd.DataFrame, labelA="good", labelB="bad", title="Vergleich zweier Fahrten") -> go.Figure:
    cols = [
        ("wheel_position", "Lenkwinkel"),
        ("yaw_rate", "Yaw-Rate"),
        ("a_lat", "Laterale Beschl."),
        ("speed", "Geschwindigkeit"),
    ]
    fig = make_subplots(rows=len(cols), cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=[c[1] for c in cols])
    for i, (col, _) in enumerate(cols, start=1):
        if col in dfA.columns:
            fig.add_trace(go.Scatter(x=dfA[TIME_COL], y=dfA[col], name=f"{labelA} – {col}", mode="lines"), row=i, col=1)
        if col in dfB.columns:
            fig.add_trace(go.Scatter(x=dfB[TIME_COL], y=dfB[col], name=f"{labelB} – {col}", mode="lines"), row=i, col=1)
    fig.update_xaxes(title_text="Zeit [s]", row=len(cols), col=1)
    fig.update_layout(height=300*len(cols), title=title, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# ------------------------------- Helpers -------------------------------- #

def print_summary(df: pd.DataFrame) -> None:
    print("\n=== Zusammenfassung ===")
    if TIME_COL in df.columns and df[TIME_COL].notna().any():
        tmin, tmax = float(df[TIME_COL].min()), float(df[TIME_COL].max())
        sr = compute_sampling_rate(df[TIME_COL])
        print(f"Zeitbereich: {tmin:.3f} s .. {tmax:.3f} s")
        if sr: print(f"≈ Samplingrate (Median): {sr:.2f} Hz")
    for c in ["wheel_position", "wheel_raw_torque_adas"]:
        if c in df.columns and df[c].notna().any():
            s = df[c].dropna()
            print(f"{c}: min={s.min():.3f}, mean={s.mean():.3f}, max={s.max():.3f}")
    for c in ["yaw_deg", "pitch_deg", "roll_deg"]:
        if c in df.columns and df[c].notna().any():
            s = df[c].dropna()
            print(f"{c}: min={s.min():.2f}°, mean={s.mean():.2f}°, max={s.max():.2f}°")

def align_overlap(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Schneidet beide DataFrames auf den gemeinsamen Zeitbereich."""
    if TIME_COL not in a.columns or TIME_COL not in b.columns:
        return a.reset_index(drop=True), b.reset_index(drop=True)
    t0 = max(a[TIME_COL].min(), b[TIME_COL].min())
    t1 = min(a[TIME_COL].max(), b[TIME_COL].max())
    a2 = a[(a[TIME_COL] >= t0) & (a[TIME_COL] <= t1)].reset_index(drop=True)
    b2 = b[(b[TIME_COL] >= t0) & (b[TIME_COL] <= t1)].reset_index(drop=True)
    return a2, b2

# ------------------------------- Main -------------------------------- #

def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {csv_path}")

    if args.mode == "single":
        df = load_and_prepare(csv_path)
        df = filter_time_window(df, args.start, args.end)
        df = downsample_df(df, args.downsample)
        df = derive_signals(df)

        print_summary(df)

        fig = build_single_figure(df)
        out_path = Path(args.out)
        fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
        print(f"\nInteraktive Plots gespeichert: {out_path.resolve()}")

    else:  # compare
        if not args.csv2:
            raise SystemExit("Im compare-Modus ist --csv2 erforderlich.")
        csv2_path = Path(args.csv2)
        if not csv2_path.exists():
            raise FileNotFoundError(f"CSV2 nicht gefunden: {csv2_path}")

        dfA = load_and_prepare(csv_path)
        dfB = load_and_prepare(csv2_path)

        # Optionales Zeitfenster auf beide anwenden
        dfA = filter_time_window(dfA, args.start, args.end)
        dfB = filter_time_window(dfB, args.start, args.end)

        # Downsampling
        dfA = downsample_df(dfA, args.downsample)
        dfB = downsample_df(dfB, args.downsample)

        # Ableitungen
        dfA = derive_signals(dfA)
        dfB = derive_signals(dfB)

        # Gemeinsamen Zeitbereich schneiden
        dfA, dfB = align_overlap(dfA, dfB)

        print(f"A: {len(dfA)} Punkte | B: {len(dfB)} Punkte (nach Overlap)")
        comp = compare_runs(dfA, dfB, args.labelA, args.labelB)
        print("\n=== Kennzahlenvergleich ===")
        print(comp)

        # Kennzahlen speichern (optional)
        if args.metrics_out:
            Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
            comp.to_csv(args.metrics_out)
            print(f"Kennzahlen gespeichert: {Path(args.metrics_out).resolve()}")

        # Overlay-Plot
        fig = build_compare_figure(dfA, dfB, args.labelA, args.labelB)
        out_path = Path(args.out)
        fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
        print(f"Vergleichs-Plots gespeichert: {out_path.resolve()}")

if __name__ == "__main__":
    main()