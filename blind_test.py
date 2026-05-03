# ==============================================================================
# BLIND TEST — Drilling Trajectory Advisory Model (Self-Contained)
# ==============================================================================
# SKENARIO A : Pakai joblib yang sudah ada → test pada sumur BARU (belum di-train)
# SKENARIO B : Leave-One-Well-Out CV      → retrain per fold, semua sumur jadi
#              blind test sekali. Cocok jika semua sumur sudah masuk training.
#
# Cara pakai di Google Colab:
#   1. Upload file ini + file data sumur ke Colab
#   2. Jalankan cell instalasi di bawah (Cell 1)
#   3. Jalankan sisa cells secara berurutan
# ==============================================================================

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 — INSTALL (jalankan sekali)
# ═══════════════════════════════════════════════════════════════════════════════
# !pip install -q xgboost lightgbm scikit-learn joblib pandas numpy openpyxl xlrd matplotlib

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 — IMPORT
# ═══════════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import warnings
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

print("✅ Library berhasil diimport.")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 — KONFIGURASI
# ═══════════════════════════════════════════════════════════════════════════════

# ── Pilih skenario ─────────────────────────────────────────────────────────────
# 'A'  = pakai joblib yang ada, test sumur baru (belum pernah di-train)
# 'B'  = Leave-One-Well-Out CV (retrain per fold, semua sumur B-J)
# 'AB' = jalankan keduanya
SKENARIO = 'B'

# ── Daftar sumur yang dipakai saat training (untuk Skenario B) ────────────────
PASANGAN_SUMUR = [
    {'nama': 'Sumur B', 'def': 'Sumur B definitive.xls',  'plan': 'Sumur B plan.xls'},
    {'nama': 'Sumur C', 'def': 'Sumur C definitive.xlsx', 'plan': 'Sumur C plan.xlsx'},
    {'nama': 'Sumur D', 'def': 'Sumur D definitive.csv',  'plan': 'Sumur D plan.csv'},
    {'nama': 'Sumur E', 'def': 'Sumur E definitive.csv',  'plan': 'Sumur E plan.csv'},
    {'nama': 'Sumur F', 'def': 'Sumur F definitive.csv',  'plan': 'Sumur F plan.csv'},
    {'nama': 'Sumur G', 'def': 'Sumur G definitive.xlsx', 'plan': 'Sumur G plan.xlsx'},
    {'nama': 'Sumur H', 'def': 'Sumur H definitive.xlsx', 'plan': 'Sumur H plan.xlsx'},
    {'nama': 'Sumur I', 'def': 'Sumur I definitive.xlsx', 'plan': 'Sumur I plan.xlsx'},
    {'nama': 'Sumur J', 'def': 'Sumur J definitive.xlsx', 'plan': 'Sumur J plan.xlsx'},
]

# ── Untuk Skenario A: sumur baru yang BELUM pernah di-train ───────────────────
SUMUR_BARU = [
    {'nama': 'Sumur K', 'def': 'Sumur K definitive.xlsx', 'plan': 'Sumur K plan.xlsx'},
]

# ── Path joblib (untuk Skenario A) ────────────────────────────────────────────
ARTIFACT_PATH = 'artifacts/drilling_advisory_full.joblib'

# ── Output ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = 'blind_test_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Model yang diikutkan di LOWO (Skenario B) ─────────────────────────────────
LOWO_MODELS = ['RandomForest', 'XGBoost']
if LIGHTGBM_AVAILABLE:
    LOWO_MODELS.append('LightGBM')

# ── Konstanta (harus sama dengan saat training) ────────────────────────────────
BASE_FEATURES = [
    'measured_depth', 'inclination', 'azimuth', 'dogleg_severity',
    'delta_inc', 'delta_azi', 'inc_error', 'azi_error',
    'rolling_dls_mean', 'rolling_inc_std',
    'depth_norm', 'cumulative_inc', 'section_code', 'dls_vs_plan',
]
TARGET           = 'distance_to_plan'
MERGE_TOLERANCE  = 30
MIN_ROWS_PER_WELL = 10
OUTLIER_QUANTILE = 0.99
RANDOM_STATE     = 42

ADVISORY = {
    'green':  (0.0,  5.0,  '🟢 ON TRACK',      'Lintasan tepat sesuai rencana.'),
    'yellow': (5.0,  15.0, '🟡 MONITOR',        'Deviasi mulai terjadi.'),
    'orange': (15.0, 30.0, '🟠 CORRECTIVE',     'Koreksi segera diperlukan.'),
    'red':    (30.0, 50.0, '🔴 CRITICAL',       'Deviasi signifikan!'),
    'abort':  (50.0, 9999, '⛔ ABORT SECTION',  'Deviasi ekstrem.'),
}

# ── Mapping keyword kolom ke nama standar ─────────────────────────────────────
HEADER_KEYWORDS: Dict[str, set] = {
    'MD':       {'MD', 'MEASURED DEPTH', 'DEPTH', 'MDEPTH', 'M DEPTH',
                 'MD (M)', 'MD(M)', 'DEPTH (M)', 'DEPTH(M)'},
    'INC':      {'INC', 'INCLINATION', 'INCL', 'ANGLE', 'INCL (DEG)',
                 'INC (DEG)', 'INC(DEG)', 'INCLINATION (DEG)'},
    'AZI':      {'AZI', 'AZIMUTH', 'AZIM', 'BEARING',
                 'AZI (DEG)', 'AZI(DEG)', 'AZIMUTH (DEG)'},
    'TVD':      {'TVD', 'TRUE VERTICAL DEPTH', 'TVDSS', 'TVD (M)', 'TVD(M)',
                 'VERTICAL DEPTH', 'TRUE VERTICAL'},
    'DLS':      {'DLS', 'DOGLEG', 'DOGLEG SEVERITY', 'DLS (DEG/30M)',
                 'DLS(DEG/30M)', 'DOGLEG (DEG/30M)', 'DOG LEG'},
    'NORTHING': {'NORTHING', 'NORTH', 'NS', 'N/S', 'N-S', 'Y', 'NORTH (M)'},
    'EASTING':  {'EASTING', 'EAST', 'EW', 'E/W', 'E-W', 'X', 'EAST (M)'},
}

print(f"✅ Konfigurasi berhasil. Skenario: {SKENARIO}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 — FUNGSI BACA FILE & FEATURE ENGINEERING (SELF-CONTAINED)
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_col(raw: str) -> Optional[str]:
    c = str(raw).strip().upper()
    c_clean = c.replace('(','').replace(')','').replace('/','').replace('-',' ').strip()
    for std, kws in HEADER_KEYWORDS.items():
        kws_clean = {k.replace('(','').replace(')','').replace('/','').replace('-',' ').strip()
                     for k in kws}
        if c in kws or c_clean in kws_clean:
            return std
    return None


def smart_read_file(filepath: str) -> pd.DataFrame:
    fp = str(filepath)
    df_raw = None

    if fp.lower().endswith('.csv'):
        for enc in ['utf-8', 'latin1', 'cp1252']:
            for sep in [',', ';', '\t', '|']:
                try:
                    tmp = pd.read_csv(fp, header=None, names=range(100),
                                      encoding=enc, sep=sep, low_memory=False)
                    if tmp.shape[1] >= 3 and tmp.shape[0] >= 5:
                        df_raw = tmp
                        break
                except Exception:
                    continue
            if df_raw is not None:
                break
    else:
        try:
            xl = pd.ExcelFile(fp)
            best_rows = 0
            for sheet in xl.sheet_names:
                try:
                    tmp = pd.read_excel(fp, sheet_name=sheet, header=None)
                    if tmp.shape[0] > best_rows:
                        best_rows = tmp.shape[0]
                        df_raw = tmp
                except Exception:
                    continue
        except Exception:
            df_raw = pd.read_excel(fp, header=None)

    if df_raw is None or df_raw.empty:
        raise ValueError(f"Gagal membaca file: {fp}")

    # Scan header row
    header_idx = -1
    for i, row in df_raw.head(60).iterrows():
        vals  = row.fillna('').astype(str).str.strip().str.upper().tolist()
        found = {_normalize_col(v) for v in vals if _normalize_col(v)}
        if 'MD' in found and 'INC' in found:
            header_idx = i
            break

    if header_idx >= 0:
        raw_cols = df_raw.iloc[header_idx].fillna('').astype(str).str.strip().str.upper().tolist()
        df = df_raw.iloc[header_idx + 1:].copy()
    else:
        raw_cols = df_raw.iloc[0].fillna('').astype(str).str.strip().str.upper().tolist()
        df = df_raw.iloc[1:].copy()

    df.columns = raw_cols
    df = df.reset_index(drop=True)

    rename_map: Dict[str, str] = {}
    already_mapped: set = set()
    for col in df.columns:
        std = _normalize_col(col)
        if std and std not in already_mapped:
            rename_map[col] = std
            already_mapped.add(std)

    df = df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.duplicated()]

    for col in list(HEADER_KEYWORDS.keys()):
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '', regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def _estimate_dls(df: pd.DataFrame) -> pd.Series:
    delta_md  = df['MD'].diff().replace(0, np.nan).abs()
    delta_inc = df['INC'].diff().abs()
    delta_azi = df['AZI'].diff().abs()
    dls_est   = (np.sqrt(delta_inc**2 +
                 (delta_azi * np.sin(np.radians(df['INC'].clip(lower=0.1))))**2)
                 / delta_md * 30)
    return dls_est.fillna(0).clip(lower=0)


def _classify_section(inc: pd.Series) -> pd.Series:
    delta = inc.diff().fillna(0)
    sec   = pd.Series(2, index=inc.index)
    sec[inc < 5]                            = 0
    sec[(inc >= 5) & (delta >  0.5)]        = 1
    sec[(inc >= 5) & (delta < -0.5)]        = 3
    return sec


def engineer_features(df_def: pd.DataFrame,
                      df_plan: pd.DataFrame,
                      well_name: str) -> Optional[pd.DataFrame]:
    required = ['MD', 'INC', 'AZI', 'TVD']
    for r in required:
        if r not in df_def.columns:
            raise ValueError(f"[{well_name}] Kolom '{r}' tidak ada di data AKTUAL.")
        if r not in df_plan.columns:
            raise ValueError(f"[{well_name}] Kolom '{r}' tidak ada di data PLAN.")

    if 'DLS' not in df_def.columns:
        df_def  = df_def.copy()
        df_def['DLS'] = _estimate_dls(df_def)
    if 'DLS' not in df_plan.columns:
        df_plan = df_plan.copy()
        df_plan['DLS'] = _estimate_dls(df_plan)

    df_def  = df_def.dropna(subset=['MD','TVD']).sort_values('MD').reset_index(drop=True)
    df_plan = df_plan.dropna(subset=['MD','TVD']).sort_values('MD').reset_index(drop=True)

    df_m = pd.merge_asof(
        df_def, df_plan, on='MD',
        suffixes=('_act','_plan'),
        direction='nearest',
        tolerance=MERGE_TOLERANCE,
    ).dropna(subset=['TVD_act','TVD_plan'])

    if len(df_m) < MIN_ROWS_PER_WELL:
        return None

    if all(c in df_m.columns for c in ['NORTHING_act','NORTHING_plan',
                                        'EASTING_act', 'EASTING_plan']):
        dn   = df_m['NORTHING_act'] - df_m['NORTHING_plan']
        de   = df_m['EASTING_act']  - df_m['EASTING_plan']
        dv   = df_m['TVD_act']      - df_m['TVD_plan']
        dist = np.sqrt(dn**2 + de**2 + dv**2)
    else:
        dist = (df_m['TVD_act'] - df_m['TVD_plan']).abs()

    feat = pd.DataFrame(index=df_m.index)
    feat['measured_depth']  = df_m['MD'].values
    feat['inclination']     = df_m['INC_act'].values
    feat['azimuth']         = df_m['AZI_act'].values
    feat['dogleg_severity'] = df_m['DLS_act'].clip(lower=0).values
    feat['inc_error']       = (df_m['INC_act'] - df_m['INC_plan']).values
    feat['azi_error']       = (df_m['AZI_act'] - df_m['AZI_plan']).values
    feat['dls_vs_plan']     = (df_m['DLS_act'] - df_m['DLS_plan']).values
    feat['delta_inc']       = feat['inclination'].diff().fillna(0)
    feat['delta_azi']       = feat['azimuth'].diff().fillna(0)
    feat['rolling_dls_mean']= feat['dogleg_severity'].rolling(5, min_periods=1).mean()
    feat['rolling_inc_std'] = feat['inclination'].rolling(5, min_periods=1).std().fillna(0)
    max_md                  = feat['measured_depth'].max()
    feat['depth_norm']      = feat['measured_depth'] / max_md if max_md > 0 else 0.0
    feat['cumulative_inc']  = feat['inclination'].cumsum()
    feat['section_code']    = _classify_section(feat['inclination']).values
    feat[TARGET]            = dist.values
    feat['well_name']       = well_name

    return feat.dropna(subset=BASE_FEATURES + [TARGET]).reset_index(drop=True)


print("✅ Fungsi helper berhasil didefinisikan.")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — FUNGSI EVALUASI & HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred) -> dict:
    return {
        'MAE':  round(float(mean_absolute_error(y_true, y_pred)), 4),
        'RMSE': round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        'R2':   round(float(r2_score(y_true, y_pred)), 4),
        'MaxE': round(float(np.max(np.abs(np.array(y_true) - np.array(y_pred)))), 4),
        'n':    int(len(y_true)),
    }


def print_metrics(metrics: dict, label: str):
    print(f"  ┌─ {label}")
    print(f"  │  MAE   : {metrics['MAE']:.4f} m")
    print(f"  │  RMSE  : {metrics['RMSE']:.4f} m")
    print(f"  │  R²    : {metrics['R2']:.4f}")
    print(f"  │  MaxErr: {metrics['MaxE']:.4f} m")
    print(f"  └─ n     : {metrics['n']} titik survei")


def severity_label(dtp: float) -> str:
    for key, (lo, hi, lbl, _) in ADVISORY.items():
        if lo <= dtp < hi:
            return lbl
    return '⛔ ABORT'


def _build_model(mname: str):
    if mname == 'RandomForest':
        return RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_split=4,
            min_samples_leaf=2, max_features='sqrt',
            n_jobs=-1, random_state=RANDOM_STATE,
        )
    elif mname == 'XGBoost':
        return xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.04, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            n_jobs=-1, random_state=RANDOM_STATE, verbosity=0,
        )
    elif mname == 'LightGBM' and LIGHTGBM_AVAILABLE:
        return lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.04, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
            n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
        )
    elif mname == 'ANN':
        return MLPRegressor(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation='relu', solver='adam', alpha=0.001,
            batch_size=64, max_iter=600, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=25,
            random_state=RANDOM_STATE, verbose=False,
        )
    return None


print("✅ Fungsi evaluasi berhasil didefinisikan.")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 — SKENARIO A: Pakai Joblib yang Sudah Ada, Test Sumur Baru
# ═══════════════════════════════════════════════════════════════════════════════

def run_scenario_a():
    print(f"\n{'═'*62}")
    print("  SKENARIO A — Blind Test dengan Joblib yang Sudah Ada")
    print(f"{'═'*62}")

    if not os.path.exists(ARTIFACT_PATH):
        print(f"  ❌ '{ARTIFACT_PATH}' tidak ditemukan.")
        print("     Jalankan train_model_v2.py dulu untuk generate artifact.")
        return None

    artifact      = joblib.load(ARTIFACT_PATH)
    models_loaded = artifact['models']
    scaler        = artifact['scaler']
    best_mdl      = artifact['best_model']
    print(f"  ✅ Artifact dimuat. Model terbaik: {best_mdl}")
    print(f"     Model tersedia: {list(models_loaded.keys())}\n")

    results_a = {}

    for sumur in SUMUR_BARU:
        name, f_def, f_plan = sumur['nama'], sumur['def'], sumur['plan']

        if not os.path.exists(f_def) or not os.path.exists(f_plan):
            print(f"  ⚠️  {name}: File tidak ditemukan, dilewati.")
            continue

        try:
            df_feat = engineer_features(smart_read_file(f_def),
                                        smart_read_file(f_plan), name)
            if df_feat is None:
                print(f"  ⚠️  {name}: Data tidak cukup setelah merge.")
                continue

            X_blind = df_feat[BASE_FEATURES].values
            y_blind = df_feat[TARGET].values
            print(f"  ✅ {name} — {len(y_blind)} titik survei")
            results_a[name] = {}

            for mname, mobj in models_loaded.items():
                y_pred = (mobj.predict(scaler.transform(X_blind))
                          if mname == 'ANN' else mobj.predict(X_blind))
                m = compute_metrics(y_blind, y_pred)
                results_a[name][mname] = m
                print_metrics(m, f"{name} — {mname}")

            # Advisory detail per titik
            df_adv              = df_feat[['measured_depth', TARGET]].copy()
            best_m              = models_loaded[best_mdl]
            df_adv['predicted'] = (best_m.predict(scaler.transform(X_blind))
                                   if best_mdl == 'ANN' else best_m.predict(X_blind))
            df_adv['severity']  = df_adv['predicted'].apply(severity_label)
            df_adv['error_m']   = (df_adv['predicted'] - df_adv[TARGET]).abs()

            csv_path = f"{OUTPUT_DIR}/skenario_a_{name.replace(' ','_')}.csv"
            df_adv.to_csv(csv_path, index=False)
            print(f"  💾 Detail: {csv_path}\n")

        except Exception as e:
            print(f"  ❌ {name}: {type(e).__name__}: {e}\n")

    if not results_a:
        print("  ⚠️  Tidak ada sumur baru yang berhasil diuji.")
        print("     Isi SUMUR_BARU dengan file yang tersedia, atau gunakan Skenario B.\n")

    return results_a


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 — SKENARIO B: Leave-One-Well-Out CV (Retrain Per Fold)
# ═══════════════════════════════════════════════════════════════════════════════

def run_scenario_b():
    print(f"\n{'═'*62}")
    print("  SKENARIO B — Leave-One-Well-Out Cross Validation (LOWO-CV)")
    print(f"{'═'*62}")
    print("  Setiap sumur dijadikan blind test sekali (retrain per fold).\n")

    # Load semua sumur
    all_frames = []
    for sumur in PASANGAN_SUMUR:
        name, f_def, f_plan = sumur['nama'], sumur['def'], sumur['plan']
        if not os.path.exists(f_def) or not os.path.exists(f_plan):
            print(f"  ⚠️  {name}: File tidak ditemukan, dilewati.")
            continue
        try:
            df_feat = engineer_features(smart_read_file(f_def),
                                        smart_read_file(f_plan), name)
            if df_feat is not None and len(df_feat) >= MIN_ROWS_PER_WELL:
                all_frames.append(df_feat)
                print(f"  ✅ Loaded: {name} ({len(df_feat)} baris)")
        except Exception as e:
            print(f"  ❌ {name}: {e}")

    if len(all_frames) < 2:
        print("\n  ❌ Minimal 2 sumur diperlukan untuk LOWO-CV.")
        return None, None

    y_all      = pd.concat([f[TARGET] for f in all_frames])
    y_cap      = y_all.quantile(OUTLIER_QUANTILE)
    well_names = [f['well_name'].iloc[0] for f in all_frames]

    lowo_results = {mname: {} for mname in LOWO_MODELS}

    print(f"\n  {'─'*60}")
    print(f"  {'WELL':<14} {'MODEL':<14} {'MAE':>7} {'RMSE':>7} {'R²':>7} {'n':>5}")
    print(f"  {'─'*60}")

    for i, blind_name in enumerate(well_names):
        train_frames = [f for f in all_frames if f['well_name'].iloc[0] != blind_name]
        df_train = pd.concat(train_frames, ignore_index=True)

        df_train = df_train[df_train[TARGET] <= y_cap]
        df_blind = all_frames[i][all_frames[i][TARGET] <= y_cap]

        X_tr = df_train[BASE_FEATURES].values
        y_tr = df_train[TARGET].values
        X_bl = df_blind[BASE_FEATURES].values
        y_bl = df_blind[TARGET].values

        if len(y_bl) < 3:
            continue

        scaler_fold = RobustScaler().fit(X_tr)

        for mname in LOWO_MODELS:
            m = _build_model(mname)
            if m is None:
                continue

            if mname == 'ANN':
                m.fit(scaler_fold.transform(X_tr), y_tr)
                y_pred = m.predict(scaler_fold.transform(X_bl))
            else:
                m.fit(X_tr, y_tr)
                y_pred = m.predict(X_bl)

            met = compute_metrics(y_bl, y_pred)
            lowo_results[mname][blind_name] = met
            print(f"  {blind_name:<14} {mname:<14} "
                  f"{met['MAE']:>7.4f} {met['RMSE']:>7.4f} "
                  f"{met['R2']:>7.4f} {met['n']:>5}")

    print(f"  {'─'*60}")

    # Agregasi
    print(f"\n  RATA-RATA LOWO-CV PER MODEL")
    print(f"  {'─'*50}")
    print(f"  {'MODEL':<14} {'MAE (mean±std)':>18} {'RMSE':>8} {'R² (mean±std)':>16}")
    print(f"  {'─'*50}")

    summary = {}
    for mname, well_dict in lowo_results.items():
        if not well_dict:
            continue
        maes  = [v['MAE']  for v in well_dict.values()]
        rmses = [v['RMSE'] for v in well_dict.values()]
        r2s   = [v['R2']   for v in well_dict.values()]
        summary[mname] = {
            'MAE_mean':  round(np.mean(maes),  4),
            'MAE_std':   round(np.std(maes),   4),
            'RMSE_mean': round(np.mean(rmses), 4),
            'R2_mean':   round(np.mean(r2s),   4),
            'R2_std':    round(np.std(r2s),    4),
        }
        s = summary[mname]
        print(f"  {mname:<14} "
              f"{s['MAE_mean']:>8.4f} ± {s['MAE_std']:.4f}  "
              f"{s['RMSE_mean']:>8.4f}  "
              f"{s['R2_mean']:>7.4f} ± {s['R2_std']:.4f}")

    print(f"  {'─'*50}")

    if summary:
        best_lowo = min(summary, key=lambda k: summary[k]['MAE_mean'])
        print(f"\n  🏆 Model terbaik (LOWO): {best_lowo}  "
              f"(MAE = {summary[best_lowo]['MAE_mean']:.4f} ± "
              f"{summary[best_lowo]['MAE_std']:.4f} m)")

    # Simpan CSV
    rows = [{'Model': mn, 'Well': wn, **met}
            for mn, wd in lowo_results.items()
            for wn, met in wd.items()]
    if rows:
        csv_path = f'{OUTPUT_DIR}/lowo_cv_detail.csv'
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\n  💾 Detail per fold: {csv_path}")

    _plot_lowo(lowo_results, summary, well_names)
    return lowo_results, summary


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8 — VISUALISASI
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_lowo(lowo_results, summary, well_names):
    active = {k: v for k, v in lowo_results.items() if v}
    if not active:
        return

    n_models = len(active)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Leave-One-Well-Out CV — Blind Test Results',
                 fontsize=13, fontweight='bold')

    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    x      = np.arange(len(well_names))
    width  = 0.8 / n_models

    # MAE per sumur
    ax1 = axes[0]
    for j, (mname, well_dict) in enumerate(active.items()):
        maes = [well_dict.get(w, {}).get('MAE', 0) for w in well_names]
        offset = j * width - (n_models - 1) * width / 2
        bars = ax1.bar(x + offset, maes, width, label=mname,
                       color=colors[j], edgecolor='white', alpha=0.85)
        for bar, val in zip(bars, maes):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                         f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    for val, ls, color, lbl in [(5,'--','green','🟢 5m'),
                                 (15,'-.','orange','🟡 15m'),
                                 (30,':','red','🟠 30m')]:
        ax1.axhline(val, color=color, linestyle=ls, lw=1.2, alpha=0.6, label=lbl)

    ax1.set_xticks(x)
    ax1.set_xticklabels(well_names, rotation=25, ha='right', fontsize=9)
    ax1.set_ylabel('MAE (meter)')
    ax1.set_title('MAE per Sumur — Blind Test', fontweight='bold')
    ax1.legend(fontsize=8)

    # R² per sumur
    ax2 = axes[1]
    for j, (mname, well_dict) in enumerate(active.items()):
        r2s    = [well_dict.get(w, {}).get('R2', 0) for w in well_names]
        offset = j * width - (n_models - 1) * width / 2
        bars   = ax2.bar(x + offset, r2s, width, label=mname,
                         color=colors[j], edgecolor='white', alpha=0.85)
        for bar, val in zip(bars, r2s):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     max(bar.get_height(), 0) + 0.01,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    ax2.axhline(0.8, color='green',  linestyle='--', lw=1.2, alpha=0.6, label='R²=0.8 target')
    ax2.axhline(0.6, color='orange', linestyle='--', lw=1.2, alpha=0.6, label='R²=0.6 baseline')
    ax2.set_xticks(x)
    ax2.set_xticklabels(well_names, rotation=25, ha='right', fontsize=9)
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² per Sumur — Blind Test', fontweight='bold')
    ax2.set_ylim(bottom=min(0, ax2.get_ylim()[0]))
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plot_path = f'{OUTPUT_DIR}/lowo_cv_results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  📊 Plot disimpan: {plot_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9 — JALANKAN
# ═══════════════════════════════════════════════════════════════════════════════

print("╔══════════════════════════════════════════════════════════╗")
print("║        BLIND TEST — Drilling Trajectory Model           ║")
print("╚══════════════════════════════════════════════════════════╝")
print(f"  Skenario aktif: {SKENARIO}\n")

if SKENARIO in ('A', 'AB'):
    results_a = run_scenario_a()

if SKENARIO in ('B', 'AB'):
    results_b, summary_b = run_scenario_b()

print(f"\n{'═'*62}")
print(f"  ✅ Selesai. Output tersimpan di folder: {OUTPUT_DIR}/")
print(f"{'═'*62}")
print("""
  INTERPRETASI HASIL:
  ──────────────────────────────────────────────────────────
  MAE < 5m    → Model sangat andal (threshold ON TRACK)
  MAE 5–15m   → Cukup baik untuk advisory level MONITOR
  MAE > 15m   → Perlu lebih banyak data atau tuning model

  R² > 0.8    → Model menjelaskan >80% pola deviasi
  R² 0.6–0.8  → Acceptable untuk advisory pendukung keputusan
  R² < 0.6    → Kurang andal, pertimbangkan fitur tambahan

  Jika MAE blind test >> MAE training:
  → Model overfit ke sumur tertentu
  → Solusi: tambah data sumur atau kurangi kompleksitas model
  ──────────────────────────────────────────────────────────
""")
