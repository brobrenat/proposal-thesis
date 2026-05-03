# ==============================================================================
# DRILLING TRAJECTORY ADVISORY SYSTEM  v2.0
# Multi-Model ML Pipeline untuk DSS Integrasi
# Models : RandomForest | XGBoost | LightGBM | ANN (MLP) | LSTM
# Target  : Prediksi Distance-to-Plan & Advisory Koreksi Lintasan
# Author  : Generated for DSS Integration
# ==============================================================================

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 — INSTALL DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════
# Jalankan cell ini sekali di Colab

# !pip install -q xgboost lightgbm scikit-learn joblib pandas numpy openpyxl xlrd matplotlib seaborn
# !pip install -q tensorflow   # untuk LSTM (sudah tersedia di Colab)

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 — IMPORT LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import os
import warnings
import json
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

import xgboost as xgb

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM tidak tersedia. Install: pip install lightgbm")

# TensorFlow untuk LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    tf.random.set_seed(42)
    TENSORFLOW_AVAILABLE = True
    print(f"✅ TensorFlow {tf.__version__} tersedia — LSTM aktif")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️  TensorFlow tidak tersedia — LSTM dilewati")

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)

print("✅ Semua library berhasil diimport.")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 — KONFIGURASI UTAMA
# ═══════════════════════════════════════════════════════════════════════════════

# ── Daftar Sumur B s/d J ─────────────────────────────────────────────────────
# Sesuaikan ekstensi file (.xls / .xlsx / .csv) dengan file yang ada di folder Colab
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

# ── Hyperparameter ─────────────────────────────────────────────────────────────
LSTM_SEQ_LEN      = 10    # Panjang window sequence untuk LSTM (jumlah survey points)
LSTM_EPOCHS       = 150
LSTM_BATCH_SIZE   = 32
MERGE_TOLERANCE   = 30    # Toleransi merge_asof dalam meter
MIN_ROWS_PER_WELL = 10    # Minimum baris valid agar sumur disertakan
RANDOM_STATE      = 42
OUTLIER_QUANTILE  = 0.99  # Potong top 1% outlier target

OUTPUT_DIR = 'artifacts'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Fitur Engineering ──────────────────────────────────────────────────────────
BASE_FEATURES = [
    'measured_depth',    # MD titik survei
    'inclination',       # INC aktual
    'azimuth',           # AZI aktual
    'dogleg_severity',   # DLS aktual (°/30m)
    'delta_inc',         # Perubahan INC antar survei
    'delta_azi',         # Perubahan AZI antar survei
    'inc_error',         # INC_aktual - INC_plan
    'azi_error',         # AZI_aktual - AZI_plan
    'rolling_dls_mean',  # Rata-rata DLS 5 titik terakhir
    'rolling_inc_std',   # Std INC 5 titik terakhir (tren volatilitas)
    'depth_norm',        # MD ternormalisasi (0–1)
    'cumulative_inc',    # Total inklinasi kumulatif
    'section_code',      # 0=Vertikal, 1=Build, 2=Hold, 3=Drop
    'dls_vs_plan',       # DLS_aktual - DLS_plan
]
TARGET = 'distance_to_plan'

# ── Thresholds Advisory ────────────────────────────────────────────────────────
ADVISORY = {
    'green':  (0.0,  5.0,  '🟢 ON TRACK',       'Lintasan tepat sesuai rencana. Pertahankan parameter drilling saat ini.'),
    'yellow': (5.0,  15.0, '🟡 MONITOR',         'Deviasi mulai terjadi. Pantau tren dan siapkan penyesuaian ringan.'),
    'orange': (15.0, 30.0, '🟠 CORRECTIVE',      'Koreksi segera diperlukan. Sesuaikan WOB/RPM atau toolface orientation.'),
    'red':    (30.0, 50.0, '🔴 CRITICAL',        'Deviasi signifikan! Lakukan survey ulang dan pertimbangkan sidetrack.'),
    'abort':  (50.0, 9999, '⛔ ABORT SECTION',   'Deviasi ekstrem. Hentikan operasi, evaluasi ulang seluruh lintasan.'),
}

print(f"✅ Konfigurasi berhasil.")
print(f"   Sumur dikonfigurasi : {len(PASANGAN_SUMUR)} sumur (B–J)")
print(f"   Fitur per model     : {len(BASE_FEATURES)}")
print(f"   Output artifacts    : ./{OUTPUT_DIR}/")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 — SMART FILE READER v2 (Auto-Scan Header, Multi-Format, Multi-Encoding)
# ═══════════════════════════════════════════════════════════════════════════════

# Mapping keyword kolom ke nama standar
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


def _normalize_col(raw: str) -> Optional[str]:
    """Kembalikan nama standar untuk raw column string, atau None jika tidak dikenal."""
    c = str(raw).strip().upper()
    # strip karakter dekorasi umum
    c_clean = c.replace('(', '').replace(')', '').replace('/', '').replace('-', ' ').strip()
    for std, kws in HEADER_KEYWORDS.items():
        kws_clean = {k.replace('(', '').replace(')', '').replace('/', '').replace('-', ' ').strip() for k in kws}
        if c in kws or c_clean in kws_clean:
            return std
    return None


def smart_read_file(filepath: str) -> pd.DataFrame:
    """
    Robust reader: auto-detects header row dalam 60 baris pertama,
    menormalisasi nama kolom, mendukung Excel & CSV multi-encoding/separator.
    """
    fp = str(filepath)

    # ── Baca raw tanpa header ──────────────────────────────────────────────────
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
        # Excel: coba setiap sheet, ambil yang paling banyak baris numerik
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

    # ── Scan baris header (MD + INC wajib ada) ─────────────────────────────────
    header_idx = -1
    for i, row in df_raw.head(60).iterrows():
        vals = row.fillna('').astype(str).str.strip().str.upper().tolist()
        found = {_normalize_col(v) for v in vals if _normalize_col(v)}
        if 'MD' in found and 'INC' in found:
            header_idx = i
            break

    # ── Terapkan header ────────────────────────────────────────────────────────
    if header_idx >= 0:
        raw_cols = df_raw.iloc[header_idx].fillna('').astype(str).str.strip().str.upper().tolist()
        df = df_raw.iloc[header_idx + 1:].copy()
    else:
        raw_cols = df_raw.iloc[0].fillna('').astype(str).str.strip().str.upper().tolist()
        df = df_raw.iloc[1:].copy()

    df.columns = raw_cols
    df = df.reset_index(drop=True)

    # ── Rename ke nama standar (hindari override kolom sudah benar) ────────────
    rename_map: Dict[str, str] = {}
    already_mapped: set = set()
    for col in df.columns:
        std = _normalize_col(col)
        if std and std not in already_mapped:
            rename_map[col] = std
            already_mapped.add(std)

    df = df.rename(columns=rename_map)

    # Hapus duplikasi kolom
    df = df.loc[:, ~df.columns.duplicated()]

    # ── Konversi numerik & bersihkan koma ribuan ───────────────────────────────
    for col in list(HEADER_KEYWORDS.keys()):
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '', regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — FEATURE ENGINEERING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def _estimate_dls(df: pd.DataFrame) -> pd.Series:
    """Estimasi DLS dari perubahan INC/AZI jika kolom DLS tidak tersedia."""
    delta_md  = df['MD'].diff().replace(0, np.nan).abs()
    delta_inc = df['INC'].diff().abs()
    delta_azi = df['AZI'].diff().abs()
    # Simplified minimum curvature approximation
    dls_est = (np.sqrt(delta_inc**2 +
               (delta_azi * np.sin(np.radians(df['INC'].clip(lower=0.1))))**2)
               / delta_md * 30)
    return dls_est.fillna(0).clip(lower=0)


def _classify_section(inc: pd.Series) -> pd.Series:
    """
    Klasifikasikan seksi lintasan:
      0 = Vertical  (INC < 5°)
      1 = Build     (INC naik >0.5°/station)
      2 = Hold      (INC stabil ±0.5°/station)
      3 = Drop      (INC turun >0.5°/station)
    """
    delta = inc.diff().fillna(0)
    sec = pd.Series(2, index=inc.index)  # default Hold
    sec[inc < 5] = 0
    sec[(inc >= 5) & (delta >  0.5)] = 1
    sec[(inc >= 5) & (delta < -0.5)] = 3
    return sec


def engineer_features(df_def: pd.DataFrame,
                      df_plan: pd.DataFrame,
                      well_name: str) -> Optional[pd.DataFrame]:
    """
    Gabungkan data aktual & plan, lalu hitung semua fitur engineering.
    Kembalikan None jika data tidak mencukupi.
    """

    # ── Validasi kolom wajib ───────────────────────────────────────────────────
    required = ['MD', 'INC', 'AZI', 'TVD']
    for r in required:
        if r not in df_def.columns:
            raise ValueError(f"[{well_name}] Kolom wajib '{r}' tidak ada di data AKTUAL.")
        if r not in df_plan.columns:
            raise ValueError(f"[{well_name}] Kolom wajib '{r}' tidak ada di data PLAN.")

    # ── Estimasi DLS jika tidak ada ────────────────────────────────────────────
    if 'DLS' not in df_def.columns:
        df_def = df_def.copy()
        df_def['DLS'] = _estimate_dls(df_def)
    if 'DLS' not in df_plan.columns:
        df_plan = df_plan.copy()
        df_plan['DLS'] = _estimate_dls(df_plan)

    # ── Bersihkan & urutkan ────────────────────────────────────────────────────
    df_def  = df_def.dropna(subset=['MD', 'TVD']).sort_values('MD').reset_index(drop=True)
    df_plan = df_plan.dropna(subset=['MD', 'TVD']).sort_values('MD').reset_index(drop=True)

    # ── Merge dengan toleransi ─────────────────────────────────────────────────
    df_m = pd.merge_asof(
        df_def, df_plan,
        on='MD',
        suffixes=('_act', '_plan'),
        direction='nearest',
        tolerance=MERGE_TOLERANCE
    ).dropna(subset=['TVD_act', 'TVD_plan'])

    if len(df_m) < MIN_ROWS_PER_WELL:
        return None

    # ── Hitung jarak ke plan ───────────────────────────────────────────────────
    # 3D distance jika Northing/Easting tersedia, fallback ke TVD diff
    if all(c in df_m.columns for c in ['NORTHING_act', 'NORTHING_plan',
                                         'EASTING_act',  'EASTING_plan']):
        dn = df_m['NORTHING_act'] - df_m['NORTHING_plan']
        de = df_m['EASTING_act']  - df_m['EASTING_plan']
        dv = df_m['TVD_act']      - df_m['TVD_plan']
        dist = np.sqrt(dn**2 + de**2 + dv**2)
    else:
        dist = (df_m['TVD_act'] - df_m['TVD_plan']).abs()

    # ── Build feature DataFrame ────────────────────────────────────────────────
    feat = pd.DataFrame(index=df_m.index)

    feat['measured_depth']  = df_m['MD'].values
    feat['inclination']     = df_m['INC_act'].values
    feat['azimuth']         = df_m['AZI_act'].values
    feat['dogleg_severity'] = df_m['DLS_act'].clip(lower=0).values

    feat['inc_error']   = (df_m['INC_act'] - df_m['INC_plan']).values
    feat['azi_error']   = (df_m['AZI_act'] - df_m['AZI_plan']).values
    feat['dls_vs_plan'] = (df_m['DLS_act'] - df_m['DLS_plan']).values

    feat['delta_inc'] = feat['inclination'].diff().fillna(0)
    feat['delta_azi'] = feat['azimuth'].diff().fillna(0)

    feat['rolling_dls_mean'] = feat['dogleg_severity'].rolling(5, min_periods=1).mean()
    feat['rolling_inc_std']  = feat['inclination'].rolling(5, min_periods=1).std().fillna(0)

    max_md = feat['measured_depth'].max()
    feat['depth_norm']      = feat['measured_depth'] / max_md if max_md > 0 else 0.0
    feat['cumulative_inc']  = feat['inclination'].cumsum()
    feat['section_code']    = _classify_section(feat['inclination']).values

    feat[TARGET]     = dist.values
    feat['well_name'] = well_name

    # Simpan referensi plan untuk advisory
    feat['tvd_plan'] = df_m['TVD_plan'].values
    feat['inc_plan'] = df_m['INC_plan'].values
    feat['azi_plan'] = df_m['AZI_plan'].values

    return feat.dropna(subset=BASE_FEATURES + [TARGET]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 — LOAD & PROSES SEMUA SUMUR
# ═══════════════════════════════════════════════════════════════════════════════

all_frames: List[pd.DataFrame] = []
well_stats: Dict[str, dict]    = {}

print(f"{'═'*62}")
print(f"  MEMUAT DATA: {len(PASANGAN_SUMUR)} SUMUR DIKONFIGURASI")
print(f"{'═'*62}")

for sumur in PASANGAN_SUMUR:
    name   = sumur['nama']
    f_def  = sumur['def']
    f_plan = sumur['plan']

    if not os.path.exists(f_def):
        print(f"  ⚠️  {name:<12}: File aktual '{f_def}' tidak ditemukan.")
        continue
    if not os.path.exists(f_plan):
        print(f"  ⚠️  {name:<12}: File plan '{f_plan}' tidak ditemukan.")
        continue

    try:
        df_def  = smart_read_file(f_def)
        df_plan = smart_read_file(f_plan)
        df_feat = engineer_features(df_def, df_plan, name)

        if df_feat is None:
            print(f"  ⚠️  {name:<12}: Kurang dari {MIN_ROWS_PER_WELL} baris setelah merge.")
            continue

        all_frames.append(df_feat)
        well_stats[name] = {
            'rows':     len(df_feat),
            'md_range': f"{df_feat['measured_depth'].min():.0f}–{df_feat['measured_depth'].max():.0f} m",
            'max_dtp':  round(float(df_feat[TARGET].max()), 2),
            'mean_dtp': round(float(df_feat[TARGET].mean()), 2),
        }
        print(f"  ✅ {name:<12}: {len(df_feat):>4} baris | "
              f"MD {well_stats[name]['md_range']:<18} | "
              f"Max DTP: {well_stats[name]['max_dtp']:.1f} m")

    except Exception as e:
        print(f"  ❌ {name:<12}: {type(e).__name__}: {e}")

print(f"{'═'*62}")

if not all_frames:
    raise RuntimeError("❌ Tidak ada data yang berhasil dimuat! Periksa nama file di folder Colab.")

df_all = pd.concat(all_frames, ignore_index=True)

print(f"\n📦 TOTAL DATASET : {len(df_all):,} baris dari {len(all_frames)} sumur")
print(f"   {TARGET} — Mean: {df_all[TARGET].mean():.2f} m | "
      f"Median: {df_all[TARGET].median():.2f} m | "
      f"Max: {df_all[TARGET].max():.2f} m")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 — PERSIAPAN DATA & SCALER
# ═══════════════════════════════════════════════════════════════════════════════

X_all = df_all[BASE_FEATURES].copy()
y_all = df_all[TARGET].copy()

# Potong outlier ekstrem (top 1% mendistorsi gradien model ANN/LSTM)
y_cap = y_all.quantile(OUTLIER_QUANTILE)
mask  = y_all <= y_cap
X_all, y_all = X_all[mask].reset_index(drop=True), y_all[mask].reset_index(drop=True)
print(f"✂️  Outlier cap (>{y_cap:.1f} m, top {(1-OUTLIER_QUANTILE)*100:.0f}%): "
      f"{(~mask).sum()} baris dibuang")

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=RANDOM_STATE
)
print(f"📊 Train: {len(X_train):,} | Test: {len(X_test):,}")

# RobustScaler lebih tahan terhadap sisa outlier dibandingkan StandardScaler
scaler_X = RobustScaler()
X_train_sc = scaler_X.fit_transform(X_train)
X_test_sc  = scaler_X.transform(X_test)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8 — LSTM SEQUENCE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_sequences(X_arr: np.ndarray,
                    y_arr: np.ndarray,
                    seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding-window sequences untuk input LSTM."""
    Xs, ys = [], []
    for i in range(seq_len, len(X_arr)):
        Xs.append(X_arr[i - seq_len: i])
        ys.append(y_arr[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


if TENSORFLOW_AVAILABLE:
    seq_frames_X, seq_frames_y = [], []

    for well in df_all['well_name'].unique():
        wdf = df_all[df_all['well_name'] == well].sort_values('measured_depth')
        wx  = wdf[BASE_FEATURES].values
        wy  = wdf[TARGET].values
        # Terapkan cap yang sama
        valid = wy <= y_cap
        wx, wy = wx[valid], wy[valid]
        if len(wx) > LSTM_SEQ_LEN + 2:
            wx_sc = scaler_X.transform(wx)
            xs, ys = build_sequences(wx_sc, wy, LSTM_SEQ_LEN)
            seq_frames_X.append(xs)
            seq_frames_y.append(ys)

    if seq_frames_X:
        X_seq = np.concatenate(seq_frames_X, axis=0)
        y_seq = np.concatenate(seq_frames_y, axis=0)
        n_split = int(len(X_seq) * 0.8)
        X_seq_tr, X_seq_te = X_seq[:n_split], X_seq[n_split:]
        y_seq_tr, y_seq_te = y_seq[:n_split], y_seq[n_split:]
        print(f"🔁 LSTM Sequences : {len(X_seq_tr):,} train | {len(X_seq_te):,} test "
              f"| Shape: {X_seq_tr.shape}")
    else:
        TENSORFLOW_AVAILABLE = False
        print("⚠️  Data per-sumur tidak cukup untuk sequences LSTM.")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9 — TRAINING SEMUA MODEL
# ═══════════════════════════════════════════════════════════════════════════════

results: Dict[str, dict] = {}
models:  Dict            = {}


def _eval(y_true, y_pred, name: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    print(f"  ┌─ {name}")
    print(f"  │  MAE  : {mae:.4f} m")
    print(f"  │  RMSE : {rmse:.4f} m")
    print(f"  └─ R²   : {r2:.4f}")
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


print(f"\n{'═'*50}")
print("  🚀 TRAINING SEMUA MODEL")
print(f"{'═'*50}\n")

# ── 1. Random Forest ──────────────────────────────────────────────────────────
print("1️⃣  Random Forest...")
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
rf.fit(X_train, y_train)
results['RandomForest'] = _eval(y_test, rf.predict(X_test), 'Random Forest')
models['RandomForest']  = rf

# ── 2. XGBoost ────────────────────────────────────────────────────────────────
print("\n2️⃣  XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=400,
    learning_rate=0.04,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbosity=0,
    objective='reg:squarederror',
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
results['XGBoost'] = _eval(y_test, xgb_model.predict(X_test), 'XGBoost')
models['XGBoost']  = xgb_model

# ── 3. LightGBM ───────────────────────────────────────────────────────────────
if LIGHTGBM_AVAILABLE:
    print("\n3️⃣  LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    lgb_model.fit(X_train, y_train)
    results['LightGBM'] = _eval(y_test, lgb_model.predict(X_test), 'LightGBM')
    models['LightGBM']  = lgb_model
else:
    print("\n3️⃣  LightGBM — dilewati (tidak tersedia)")

# ── 4. ANN / MLP ──────────────────────────────────────────────────────────────
print("\n4️⃣  ANN (Multi-Layer Perceptron)...")
ann = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=64,
    learning_rate='adaptive',
    max_iter=600,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=25,
    random_state=RANDOM_STATE,
    verbose=False,
)
ann.fit(X_train_sc, y_train)
results['ANN'] = _eval(y_test, ann.predict(X_test_sc), 'ANN (MLP)')
models['ANN']  = ann

# ── 5. LSTM ───────────────────────────────────────────────────────────────────
if TENSORFLOW_AVAILABLE:
    print("\n5️⃣  LSTM (Recurrent Neural Network)...")
    n_features = X_seq_tr.shape[2]

    lstm_model = Sequential([
        LSTM(64, input_shape=(LSTM_SEQ_LEN, n_features), return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1,  activation='linear'),
    ], name='DrillLSTM')

    lstm_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='huber',        # Lebih robust terhadap outlier sisa
        metrics=['mae'],
    )

    callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss', verbose=0),
        ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6, verbose=0),
    ]

    history = lstm_model.fit(
        X_seq_tr, y_seq_tr,
        validation_data=(X_seq_te, y_seq_te),
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
    )

    y_pred_lstm = lstm_model.predict(X_seq_te, verbose=0).flatten()
    results['LSTM'] = _eval(y_seq_te, y_pred_lstm, 'LSTM')
    models['LSTM']  = lstm_model
    print(f"     Epochs konvergen : {len(history.history['loss'])}")
else:
    print("\n5️⃣  LSTM — dilewati")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 10 — EVALUASI & VISUALISASI PERBANDINGAN MODEL
# ═══════════════════════════════════════════════════════════════════════════════

df_res = pd.DataFrame(results).T.round(4).sort_values('MAE')
df_res['Rank'] = range(1, len(df_res) + 1)
best_model_name = df_res.index[0]

print(f"\n{'═'*60}")
print("  📊 RANGKUMAN PERFORMA SEMUA MODEL")
print(f"{'═'*60}")
print(df_res[['MAE', 'RMSE', 'R2', 'Rank']].to_string())
print(f"{'═'*60}")
print(f"  🏆 MODEL TERBAIK : {best_model_name}  "
      f"(MAE={df_res.loc[best_model_name,'MAE']:.4f} m | "
      f"R²={df_res.loc[best_model_name,'R2']:.4f})")

# ── Visualisasi ───────────────────────────────────────────────────────────────
n_models = len(results)
palette  = ['#4CAF50' if m == best_model_name else '#2196F3' for m in df_res.index]

fig = plt.figure(figsize=(20, 14))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Plot 1: MAE
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.barh(df_res.index, df_res['MAE'], color=palette, edgecolor='white', height=0.6)
ax1.set_title('Mean Absolute Error  (↓ lebih baik)', fontweight='bold', pad=8)
ax1.set_xlabel('MAE (meter)')
for bar, val in zip(bars, df_res['MAE']):
    ax1.text(val + 0.005, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=9)

# Plot 2: R²
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.barh(df_res.index, df_res['R2'], color=palette, edgecolor='white', height=0.6)
ax2.set_title('R² Score  (↑ lebih baik)', fontweight='bold', pad=8)
ax2.set_xlabel('R²')
ax2.set_xlim(0, 1.08)
for bar, val in zip(bars, df_res['R2']):
    ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=9)

# Plot 3: RMSE
ax3 = fig.add_subplot(gs[0, 2])
bars = ax3.barh(df_res.index, df_res['RMSE'], color=palette, edgecolor='white', height=0.6)
ax3.set_title('RMSE  (↓ lebih baik)', fontweight='bold', pad=8)
ax3.set_xlabel('RMSE (meter)')
for bar, val in zip(bars, df_res['RMSE']):
    ax3.text(val + 0.005, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=9)

# Plot 4: Feature Importance (tree-based model terbaik)
ax4 = fig.add_subplot(gs[1, :2])
tree_candidates = {k: results[k] for k in ['RandomForest', 'XGBoost', 'LightGBM']
                   if k in models}
if tree_candidates:
    best_tree = min(tree_candidates, key=lambda k: tree_candidates[k]['MAE'])
    m = models[best_tree]
    if hasattr(m, 'feature_importances_'):
        imp = pd.Series(m.feature_importances_, index=BASE_FEATURES).sort_values()
        colors_fi = ['#FF9800' if imp[f] == imp.max() else '#78909C' for f in imp.index]
        imp.plot(kind='barh', ax=ax4, color=colors_fi, edgecolor='white')
        ax4.set_title(f'Feature Importance — {best_tree}', fontweight='bold', pad=8)
        ax4.set_xlabel('Importance Score')
        ax4.axvline(imp.mean(), color='red', linestyle='--', alpha=0.6,
                    label=f'Mean={imp.mean():.3f}')
        ax4.legend(fontsize=8)

# Plot 5: Actual vs Predicted (model terbaik)
ax5 = fig.add_subplot(gs[1, 2])
if best_model_name == 'LSTM' and TENSORFLOW_AVAILABLE:
    y_pred_best = models['LSTM'].predict(X_seq_te, verbose=0).flatten()
    y_true_best = y_seq_te
elif best_model_name == 'ANN':
    y_pred_best = models['ANN'].predict(X_test_sc)
    y_true_best = y_test.values
else:
    y_pred_best = models[best_model_name].predict(X_test)
    y_true_best = y_test.values

sample_n = min(300, len(y_true_best))
idx_s    = np.random.choice(len(y_true_best), sample_n, replace=False)
lim      = max(y_true_best[idx_s].max(), y_pred_best[idx_s].max()) * 1.05
ax5.scatter(y_true_best[idx_s], y_pred_best[idx_s],
            alpha=0.45, s=18, color='#9C27B0', zorder=3)
ax5.plot([0, lim], [0, lim], 'r--', lw=1.5, label='Perfect fit')
ax5.fill_between([0, lim], [0*0.8, lim*0.8], [0*1.2, lim*1.2],
                 alpha=0.07, color='green', label='±20% band')
ax5.set_title(f'Actual vs Predicted — {best_model_name}', fontweight='bold', pad=8)
ax5.set_xlabel('Actual DTP (m)')
ax5.set_ylabel('Predicted DTP (m)')
ax5.legend(fontsize=8)
ax5.set_xlim(0, lim)
ax5.set_ylim(0, lim)

plt.suptitle('Drilling Trajectory Advisory System v2.0 — Model Comparison',
             fontsize=14, fontweight='bold', y=1.01)
plt.savefig(f'{OUTPUT_DIR}/model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"📊 Plot disimpan: {OUTPUT_DIR}/model_comparison.png")

# Plot distribusi target per sumur
fig2, ax = plt.subplots(figsize=(14, 5))
for frame in all_frames:
    wn = frame['well_name'].iloc[0]
    ax.plot(frame['measured_depth'], frame[TARGET], lw=1.2, label=wn, alpha=0.8)

# Advisory thresholds sebagai garis horizontal
colors_adv = {'green': '#4CAF50', 'yellow': '#FFC107',
               'orange': '#FF9800', 'red': '#F44336', 'abort': '#9C27B0'}
for key, (lo, hi, lbl, _) in ADVISORY.items():
    if hi < 9999:
        ax.axhline(hi, linestyle='--', alpha=0.5, color=colors_adv[key], lw=1,
                   label=f'{lbl} threshold ({hi}m)')

ax.set_title('Distance to Plan per Sumur vs Advisory Thresholds', fontweight='bold')
ax.set_xlabel('Measured Depth (m)')
ax.set_ylabel('Distance to Plan (m)')
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/dtp_per_well.png', dpi=150, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 11 — ADVISORY SYSTEM CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class DrillingAdvisorySystem:
    """
    Real-time advisory engine untuk DSS.
    Input  : dict survey point (fitur dari BASE_FEATURES)
    Output : severity, prediksi DTP, tren, rekomendasi koreksi spesifik
    """

    SECTION_NAMES = {0: 'Vertical', 1: 'Build', 2: 'Hold', 3: 'Drop'}

    # Panduan koreksi per situasi dan seksi lintasan
    CORRECTIONS = {
        'inc_high': {
            'Vertical': "Cek vertikalitas BHA. Pertimbangkan stabilizer atau pergantian bit.",
            'Build':    "Kurangi WOB, naikkan RPM untuk mengurangi laju build inclination.",
            'Hold':     "Slide dengan toolface ~270° untuk drop inclination ke target.",
            'Drop':     "Laju drop terlalu agresif. Kurangi sliding percentage.",
        },
        'inc_low': {
            'Vertical': "Inclination masih di batas vertikal. Monitor bila mendekati 5°.",
            'Build':    "Naikkan WOB, pertahankan RPM untuk mempercepat build inclination.",
            'Hold':     "Slide dengan toolface ~90° untuk build inclination ke target.",
            'Drop':     "Kurangi sliding percentage untuk memperlambat laju drop.",
        },
        'azi_right': "Toolface terlalu ke kanan. Rotasi toolface CCW (counter-clockwise), kurangi sliding %.",
        'azi_left':  "Toolface terlalu ke kiri. Rotasi toolface CW (clockwise), tambah sliding %.",
        'dls_high':  "DLS melebihi rencana! Kurangi WOB dan sliding %. Waspadai tubular fatigue.",
        'on_track':  "Parameter drilling dalam batas normal. Pertahankan kondisi saat ini.",
    }

    def __init__(self,
                 models_dict:   Dict,
                 scaler:        RobustScaler,
                 feature_names: List[str],
                 best_model:    str,
                 results_dict:  Dict,
                 lstm_seq_len:  int = 10,
                 tf_available:  bool = False):
        self.models       = models_dict
        self.scaler       = scaler
        self.features     = feature_names
        self.best_model   = best_model
        self.results      = results_dict
        self.seq_len      = lstm_seq_len
        self.tf_available = tf_available

        # Hitung bobot ensemble dari MAE (MAE lebih kecil = bobot lebih besar)
        mae_vals = {k: v['MAE'] for k, v in results_dict.items() if k != 'LSTM'}
        inv_mae  = {k: 1.0 / v for k, v in mae_vals.items() if v > 0}
        total    = sum(inv_mae.values())
        self._weights = {k: v / total for k, v in inv_mae.items()} if total > 0 else {}

    def _get_severity(self, dist: float) -> Tuple[str, str, str]:
        for key, (lo, hi, label, msg) in ADVISORY.items():
            if lo <= dist < hi:
                return key, label, msg
        return 'abort', ADVISORY['abort'][2], ADVISORY['abort'][3]

    def _predict_scalar(self, x_row: np.ndarray, model_name: str) -> float:
        x2 = x_row.reshape(1, -1)
        if model_name == 'ANN':
            return float(self.models[model_name].predict(self.scaler.transform(x2))[0])
        return float(self.models[model_name].predict(x2)[0])

    def analyze(self,
                survey_point: Dict,
                history: Optional[pd.DataFrame] = None) -> Dict:
        """
        Analisis satu titik survei.

        Parameters
        ----------
        survey_point : dict  — nilai setiap fitur di BASE_FEATURES
        history      : DataFrame opsional dengan kolom 'distance_to_plan'
                       (urutan MD ascending) untuk analisis tren

        Returns
        -------
        dict berisi severity, prediksi, tren, rekomendasi
        """
        x = np.array([survey_point.get(f, 0.0) for f in self.features],
                     dtype=np.float64)

        # ── Prediksi dari setiap model (kecuali LSTM — butuh sequence) ─────────
        preds = {}
        for mname in self.models:
            if mname == 'LSTM':
                continue
            try:
                preds[mname] = self._predict_scalar(x, mname)
            except Exception:
                pass

        if not preds:
            return {'error': 'Tidak ada model tersedia.'}

        # Ensemble terbobot (1/MAE)
        if self._weights:
            ensemble = sum(preds.get(m, 0) * w
                           for m, w in self._weights.items() if m in preds)
            weight_sum = sum(self._weights.get(m, 0) for m in preds)
            ensemble   = ensemble / weight_sum if weight_sum > 0 else np.mean(list(preds.values()))
        else:
            ensemble = np.mean(list(preds.values()))

        best_pred = preds.get(self.best_model, ensemble)

        # ── Severity ───────────────────────────────────────────────────────────
        sev_key, sev_label, sev_msg = self._get_severity(ensemble)

        # ── Tren dari riwayat ──────────────────────────────────────────────────
        trend = 'N/A'
        trend_detail = ''
        if history is not None and len(history) >= 3:
            recent = history[TARGET].tail(6).values
            if len(recent) >= 2:
                slope = np.polyfit(range(len(recent)), recent, 1)[0]
                if slope > 0.3:
                    trend = '📈 MEMBURUK'
                    trend_detail = f"Deviasi naik ~{slope:.1f}m/survey point."
                elif slope < -0.3:
                    trend = '📉 MEMBAIK'
                    trend_detail = f"Deviasi turun ~{abs(slope):.1f}m/survey point."
                else:
                    trend = '➡️  STABIL'
                    trend_detail = "Deviasi relatif konstan."

        # ── Seksi lintasan ─────────────────────────────────────────────────────
        inc     = survey_point.get('inclination', 0.0)
        sec_raw = int(round(survey_point.get('section_code', 2)))
        section = 'Vertical' if inc < 5 else self.SECTION_NAMES.get(sec_raw, 'Hold')

        # ── Rekomendasi koreksi spesifik ───────────────────────────────────────
        recs    = []
        inc_err = survey_point.get('inc_error', 0.0)
        azi_err = survey_point.get('azi_error', 0.0)
        dls_pl  = survey_point.get('dls_vs_plan', 0.0)
        dls     = survey_point.get('dogleg_severity', 0.0)

        if abs(inc_err) > 1.5:
            key = 'inc_high' if inc_err > 0 else 'inc_low'
            rec = self.CORRECTIONS[key].get(section, self.CORRECTIONS[key]['Hold'])
            recs.append(f"INC Error ({inc_err:+.1f}°): {rec}")

        if abs(azi_err) > 2.5:
            key = 'azi_right' if azi_err > 0 else 'azi_left'
            recs.append(f"AZI Error ({azi_err:+.1f}°): {self.CORRECTIONS[key]}")

        if dls_pl > 0.5:
            recs.append(f"DLS ({dls:.2f} °/30m vs plan): {self.CORRECTIONS['dls_high']}")

        if not recs:
            recs = [self.CORRECTIONS['on_track']]

        return {
            'md':               survey_point.get('measured_depth', 0.0),
            'section':          section,
            'ensemble_dtp':     round(float(ensemble), 3),
            'best_model_dtp':   round(float(best_pred), 3),
            'all_predictions':  {k: round(float(v), 3) for k, v in preds.items()},
            'severity_key':     sev_key,
            'severity_label':   sev_label,
            'severity_msg':     sev_msg,
            'trend':            trend,
            'trend_detail':     trend_detail,
            'inc_error':        round(float(inc_err), 2),
            'azi_error':        round(float(azi_err), 2),
            'dls':              round(float(dls), 3),
            'recommendations':  recs,
        }

    def print_report(self, r: Dict):
        if 'error' in r:
            print(f"❌ {r['error']}")
            return
        print(f"\n{'═'*62}")
        print(f"  DRILLING TRAJECTORY ADVISORY REPORT")
        print(f"{'═'*62}")
        print(f"  Measured Depth    : {r['md']:.1f} m")
        print(f"  Seksi Lintasan    : {r['section']}")
        print(f"{'─'*62}")
        print(f"  Prediksi DTP      : {r['ensemble_dtp']:.3f} m  (ensemble weighted)")
        print(f"  Best Model DTP    : {r['best_model_dtp']:.3f} m  ({self.best_model})")
        print(f"{'─'*62}")
        print(f"  Status Advisory   : {r['severity_label']}")
        print(f"  Keterangan        : {r['severity_msg']}")
        print(f"{'─'*62}")
        print(f"  Tren Deviasi      : {r['trend']}")
        if r['trend_detail']:
            print(f"  Detail Tren       : {r['trend_detail']}")
        print(f"{'─'*62}")
        print(f"  Error INC         : {r['inc_error']:+.2f}°")
        print(f"  Error AZI         : {r['azi_error']:+.2f}°")
        print(f"  DLS Aktual        : {r['dls']:.3f} °/30m")
        print(f"{'─'*62}")
        print(f"  REKOMENDASI KOREKSI:")
        for i, rec in enumerate(r['recommendations'], 1):
            # Word-wrap sederhana
            words = rec.split()
            line = f"  {i}. "
            for w in words:
                if len(line) + len(w) > 75:
                    print(line)
                    line = "     "
                line += w + " "
            if line.strip():
                print(line)
        print(f"{'─'*62}")
        print(f"  Prediksi per Model:")
        for mname, val in r['all_predictions'].items():
            marker = " ← best" if mname == self.best_model else ""
            print(f"    {mname:<14}: {val:.3f} m{marker}")
        print(f"{'═'*62}\n")

    def batch_analyze(self, df_survey: pd.DataFrame) -> pd.DataFrame:
        """
        Analisis batch seluruh dataframe survei sekaligus.
        Berguna untuk post-run analysis atau audit lintasan.
        """
        out = []
        for i in range(len(df_survey)):
            row  = df_survey.iloc[i].to_dict()
            hist = df_survey.iloc[:i] if i > 0 else None
            res  = self.analyze(row, history=hist)
            out.append({
                'MD':             res['md'],
                'section':        res['section'],
                'ensemble_dtp':   res['ensemble_dtp'],
                'severity':       res['severity_label'],
                'trend':          res['trend'],
                'inc_error':      res['inc_error'],
                'azi_error':      res['azi_error'],
                'recommendation': res['recommendations'][0],
            })
        return pd.DataFrame(out)


# Inisialisasi
advisor = DrillingAdvisorySystem(
    models_dict   = models,
    scaler        = scaler_X,
    feature_names = BASE_FEATURES,
    best_model    = best_model_name,
    results_dict  = results,
    lstm_seq_len  = LSTM_SEQ_LEN,
    tf_available  = TENSORFLOW_AVAILABLE,
)
print("✅ Advisory System berhasil diinisialisasi.")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 12 — EXPORT SEMUA ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*55}")
print("  💾 MENYIMPAN ARTIFACTS")
print(f"{'═'*55}\n")

saved: List[str] = []

# ── Per-model artifact ─────────────────────────────────────────────────────────
for mname, mobj in models.items():
    if mname == 'LSTM':
        continue
    art = {
        'model':            mobj,
        'scaler':           scaler_X if mname == 'ANN' else None,
        'feature_names':    BASE_FEATURES,
        'target':           TARGET,
        'metrics':          results.get(mname, {}),
        'advisory_config':  ADVISORY,
        'version':          '2.0',
    }
    fp = f'{OUTPUT_DIR}/{mname.lower()}_model.joblib'
    joblib.dump(art, fp, compress=3)
    saved.append(fp)
    print(f"  ✅ {mname:<14}: {fp}")

# ── LSTM artifact (Keras tidak langsung picklable, simpan via config+weights) ──
if TENSORFLOW_AVAILABLE and 'LSTM' in models:
    lstm_art = {
        'model_config':    models['LSTM'].to_json(),
        'model_weights':   models['LSTM'].get_weights(),
        'scaler':          scaler_X,
        'seq_len':         LSTM_SEQ_LEN,
        'feature_names':   BASE_FEATURES,
        'target':          TARGET,
        'metrics':         results.get('LSTM', {}),
        'advisory_config': ADVISORY,
        'version':         '2.0',
        # Helper snippet untuk reload
        '_reload_instructions': (
            "import joblib, tensorflow as tf\n"
            "art = joblib.load('artifacts/lstm_model.joblib')\n"
            "model = tf.keras.models.model_from_json(art['model_config'])\n"
            "model.set_weights(art['model_weights'])"
        ),
    }
    fp = f'{OUTPUT_DIR}/lstm_model.joblib'
    joblib.dump(lstm_art, fp, compress=3)
    saved.append(fp)
    print(f"  ✅ {'LSTM':<14}: {fp}")

# ── Full system artifact (untuk DSS integration) ───────────────────────────────
full_art = {
    # Semua model non-LSTM
    'models':           {k: v for k, v in models.items() if k != 'LSTM'},
    'scaler':           scaler_X,
    'feature_names':    BASE_FEATURES,
    'target':           TARGET,
    'best_model':       best_model_name,
    'ensemble_weights': advisor._weights,
    'metrics':          results,
    'advisory_config':  ADVISORY,
    'well_stats':       well_stats,
    'version':          '2.0',
}
# Sertakan LSTM dalam full artifact
if TENSORFLOW_AVAILABLE and 'LSTM' in models:
    full_art['lstm_config']   = models['LSTM'].to_json()
    full_art['lstm_weights']  = models['LSTM'].get_weights()
    full_art['lstm_seq_len']  = LSTM_SEQ_LEN

fp = f'{OUTPUT_DIR}/drilling_advisory_full.joblib'
joblib.dump(full_art, fp, compress=3)
saved.append(fp)
print(f"\n  ✅ {'FULL SYSTEM':<14}: {fp}  ← Gunakan ini di DSS")

# ── Backward compatible (untuk app.py yang sudah ada) ─────────────────────────
joblib.dump(models.get(best_model_name, list(models.values())[0]),
            'drilling_model.pkl')
print(f"  ✅ {'LEGACY PKL':<14}: drilling_model.pkl  ← Backward compatible")

# ── Metadata JSON ──────────────────────────────────────────────────────────────
meta = {
    'version':       '2.0',
    'best_model':    best_model_name,
    'total_wells':   len(all_frames),
    'total_rows':    int(len(df_all)),
    'features':      BASE_FEATURES,
    'target':        TARGET,
    'metrics':       {k: {m: float(v) for m, v in vals.items()}
                      for k, vals in results.items()},
    'well_stats':    well_stats,
    'advisory':      {k: {'min': lo, 'max': hi, 'label': lbl}
                      for k, (lo, hi, lbl, _) in ADVISORY.items()},
    'artifacts':     saved,
}
with open(f'{OUTPUT_DIR}/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"  ✅ metadata.json   : {OUTPUT_DIR}/metadata.json")

print(f"\n{'═'*55}")
print(f"  🎉 {len(saved)} artifacts disimpan di '{OUTPUT_DIR}/'")
print(f"{'═'*55}")
print(f"\n  RINGKASAN AKHIR")
print(f"  {'─'*50}")
print(f"  Sumur berhasil   : {len(all_frames)} sumur")
print(f"  Total dataset    : {len(df_all):,} baris")
print(f"  Model terbaik    : {best_model_name}")
print(f"  MAE terbaik      : {results[best_model_name]['MAE']:.4f} m")
print(f"  R² terbaik       : {results[best_model_name]['R2']:.4f}")
print(f"  {'─'*50}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 13 — DEMO ADVISORY + CARA LOAD DI DSS
# ═══════════════════════════════════════════════════════════════════════════════

print("═"*62)
print("  🧪 DEMO ADVISORY — Simulasi Survey Point Real-time")
print("═"*62)

# Contoh survey point (ganti dengan data real-time dari sensor DSS)
demo_survey = {
    'measured_depth':   2500.0,
    'inclination':      35.2,
    'azimuth':          125.0,
    'dogleg_severity':   1.8,
    'delta_inc':         0.5,
    'delta_azi':         1.2,
    'inc_error':         2.1,    # INC 2.1° lebih tinggi dari plan
    'azi_error':        -3.5,    # AZI 3.5° ke kiri dari plan
    'rolling_dls_mean':  1.6,
    'rolling_inc_std':   0.3,
    'depth_norm':        0.72,
    'cumulative_inc':   8400.0,
    'section_code':      2.0,    # Hold section
    'dls_vs_plan':       0.3,
}

result = advisor.analyze(demo_survey)
advisor.print_report(result)

# ── Batch analysis dari data sumur aktual ─────────────────────────────────────
if all_frames:
    sample_well = all_frames[0].copy()
    print(f"📋 Batch analysis: {sample_well['well_name'].iloc[0]} "
          f"({len(sample_well)} titik survei)")
    batch_result = advisor.batch_analyze(sample_well[BASE_FEATURES + [TARGET]])
    print(batch_result[['MD', 'section', 'ensemble_dtp', 'severity', 'trend']].head(10).to_string())
    batch_result.to_csv(f'{OUTPUT_DIR}/batch_advisory_sample.csv', index=False)
    print(f"💾 Batch result: {OUTPUT_DIR}/batch_advisory_sample.csv")

# ── Snippet integrasi DSS ──────────────────────────────────────────────────────
print(f"""
{'─'*62}
  📌 INTEGRASI KE APLIKASI DSS (app.py / dss_core.py):
{'─'*62}

import joblib
import numpy as np

# ── Load sekali saat startup ──
artifact = joblib.load('artifacts/drilling_advisory_full.joblib')
models   = artifact['models']
scaler   = artifact['scaler']
features = artifact['feature_names']
best_mdl = artifact['best_model']
weights  = artifact['ensemble_weights']

# ── Fungsi prediksi real-time ──
def predict_dtp(survey_dict: dict) -> dict:
    x = np.array([survey_dict.get(f, 0.0) for f in features]).reshape(1, -1)
    preds = {{}}
    for mname, mobj in models.items():
        if mname == 'ANN':
            preds[mname] = float(mobj.predict(scaler.transform(x))[0])
        else:
            preds[mname] = float(mobj.predict(x)[0])
    # Ensemble terbobot
    total_w = sum(weights.get(m, 0) for m in preds)
    ensemble = sum(preds.get(m, 0) * weights.get(m, 0) for m in preds)
    return {{
        'ensemble_dtp': round(ensemble / total_w, 3) if total_w > 0 else 0.0,
        'per_model':    {{k: round(v, 3) for k, v in preds.items()}},
    }}

# ── Contoh pemakaian ──
result = predict_dtp(survey_dict)
print(f"Predicted DTP: {{result['ensemble_dtp']}} m")
{'─'*62}
""")
