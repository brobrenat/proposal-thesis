# ==============================================================================
# BLIND TEST — Drilling Trajectory Advisory Model
# ==============================================================================
# SKENARIO A : Pakai joblib yang sudah ada → test pada sumur BARU (belum di-train)
# SKENARIO B : Leave-One-Well-Out CV      → retrain per fold, semua sumur jadi
#              blind test sekali. Cocok jika semua sumur sudah masuk training.
# ==============================================================================

import pandas as pd
import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

warnings.filterwarnings('ignore')

# ── Import helper dari train_model_v2 ─────────────────────────────────────────
# Pastikan train_model_v2.py ada di folder yang sama
from train_model_v2 import (
    smart_read_file,
    engineer_features,
    BASE_FEATURES,
    TARGET,
    ADVISORY,
    MERGE_TOLERANCE,
    MIN_ROWS_PER_WELL,
    PASANGAN_SUMUR,
)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# ==============================================================================
# KONFIGURASI BLIND TEST
# ==============================================================================

# ── Pilih skenario ─────────────────────────────────────────────────────────────
# 'A' = pakai joblib yang ada, test sumur baru
# 'B' = Leave-One-Well-Out CV (retrain per fold)
# 'AB' = jalankan keduanya
SKENARIO = 'B'

# ── Untuk Skenario A: sumur baru yang belum pernah di-train ───────────────────
# Ganti dengan nama file sumur baru Anda
SUMUR_BARU = [
    {'nama': 'Sumur K', 'def': 'Sumur K definitive.xlsx', 'plan': 'Sumur K plan.xlsx'},
    # Tambahkan sumur lain di sini jika ada
]

# ── Path joblib yang sudah ada ─────────────────────────────────────────────────
ARTIFACT_PATH = 'artifacts/drilling_advisory_full.joblib'

# ── Output ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = 'blind_test_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Model yang diikutkan di LOWO (Skenario B) ─────────────────────────────────
LOWO_MODELS = ['RandomForest', 'XGBoost']   # Tambah 'LightGBM' jika tersedia
if LIGHTGBM_AVAILABLE:
    LOWO_MODELS.append('LightGBM')

RANDOM_STATE  = 42
OUTLIER_QUANTILE = 0.99


# ==============================================================================
# FUNGSI EVALUASI
# ==============================================================================

def compute_metrics(y_true, y_pred) -> dict:
    return {
        'MAE':  round(float(mean_absolute_error(y_true, y_pred)), 4),
        'RMSE': round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        'R2':   round(float(r2_score(y_true, y_pred)), 4),
        'MaxE': round(float(np.max(np.abs(y_true - y_pred))), 4),
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


# ==============================================================================
# SKENARIO A — Gunakan joblib existing, test sumur baru
# ==============================================================================

def run_scenario_a():
    print(f"\n{'═'*62}")
    print("  SKENARIO A — Blind Test dengan Joblib yang Sudah Ada")
    print(f"{'═'*62}")
    print(f"  Artifact : {ARTIFACT_PATH}\n")

    if not os.path.exists(ARTIFACT_PATH):
        print(f"  ❌ File '{ARTIFACT_PATH}' tidak ditemukan. Jalankan train_model_v2.py dulu.")
        return

    artifact = joblib.load(ARTIFACT_PATH)
    models_loaded = artifact['models']
    scaler        = artifact['scaler']
    best_mdl      = artifact['best_model']
    print(f"  Model terbaik dari training : {best_mdl}")
    print(f"  Model tersedia              : {list(models_loaded.keys())}\n")

    results_a = {}

    for sumur in SUMUR_BARU:
        name   = sumur['nama']
        f_def  = sumur['def']
        f_plan = sumur['plan']

        if not os.path.exists(f_def) or not os.path.exists(f_plan):
            print(f"  ⚠️  {name}: File tidak ditemukan, dilewati.")
            continue

        try:
            df_def  = smart_read_file(f_def)
            df_plan = smart_read_file(f_plan)
            df_feat = engineer_features(df_def, df_plan, name)

            if df_feat is None:
                print(f"  ⚠️  {name}: Data tidak cukup setelah merge.")
                continue

            X_blind = df_feat[BASE_FEATURES].values
            y_blind = df_feat[TARGET].values

            print(f"  ✅ {name} — {len(y_blind)} titik survei dimuat")
            results_a[name] = {}

            for mname, mobj in models_loaded.items():
                if mname == 'ANN':
                    y_pred = mobj.predict(scaler.transform(X_blind))
                else:
                    y_pred = mobj.predict(X_blind)

                m = compute_metrics(y_blind, y_pred)
                results_a[name][mname] = m
                print_metrics(m, f"{name} — {mname}")

            # Advisory breakdown per titik
            df_adv = df_feat[['measured_depth', TARGET]].copy()
            best_m = models_loaded[best_mdl]
            if best_mdl == 'ANN':
                df_adv['predicted'] = best_m.predict(scaler.transform(X_blind))
            else:
                df_adv['predicted'] = best_m.predict(X_blind)
            df_adv['severity']  = df_adv['predicted'].apply(severity_label)
            df_adv['error_m']   = (df_adv['predicted'] - df_adv[TARGET]).abs()

            csv_path = f"{OUTPUT_DIR}/skenario_a_{name.replace(' ','_')}.csv"
            df_adv.to_csv(csv_path, index=False)
            print(f"  💾 Detail per titik: {csv_path}\n")

        except Exception as e:
            print(f"  ❌ {name}: {type(e).__name__}: {e}\n")

    if not results_a:
        print("  ⚠️  Tidak ada sumur baru yang berhasil diuji.")
        print("       Pastikan file sumur baru tersedia atau ganti ke Skenario B.\n")

    return results_a


# ==============================================================================
# SKENARIO B — Leave-One-Well-Out Cross Validation (retrain per fold)
# ==============================================================================

def _build_model(mname: str):
    """Buat instance model baru (fresh, belum di-train)."""
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
    return None


def run_scenario_b():
    print(f"\n{'═'*62}")
    print("  SKENARIO B — Leave-One-Well-Out Cross Validation (LOWO-CV)")
    print(f"{'═'*62}")
    print("  Setiap sumur dijadikan blind test sekali.")
    print("  Model dilatih ulang dari nol tanpa sumur tersebut.\n")

    # ── Load semua data sumur ─────────────────────────────────────────────────
    all_frames = []
    for sumur in PASANGAN_SUMUR:
        name   = sumur['nama']
        f_def  = sumur['def']
        f_plan = sumur['plan']
        if not os.path.exists(f_def) or not os.path.exists(f_plan):
            continue
        try:
            df_def  = smart_read_file(f_def)
            df_plan = smart_read_file(f_plan)
            df_feat = engineer_features(df_def, df_plan, name)
            if df_feat is not None and len(df_feat) >= MIN_ROWS_PER_WELL:
                all_frames.append(df_feat)
                print(f"  ✅ Loaded: {name} ({len(df_feat)} baris)")
        except Exception as e:
            print(f"  ❌ {name}: {e}")

    if len(all_frames) < 2:
        print("\n  ❌ Minimal 2 sumur diperlukan untuk LOWO-CV.")
        return

    # Cap outlier konsisten dengan training
    y_all  = pd.concat([f[TARGET] for f in all_frames])
    y_cap  = y_all.quantile(OUTLIER_QUANTILE)

    well_names  = [f['well_name'].iloc[0] for f in all_frames]
    lowo_results = {mname: {} for mname in LOWO_MODELS}

    print(f"\n  {'─'*58}")
    print(f"  {'WELL':<14} {'MODEL':<14} {'MAE':>7} {'RMSE':>7} {'R²':>7} {'n':>5}")
    print(f"  {'─'*58}")

    for i, blind_name in enumerate(well_names):
        train_frames = [f for f in all_frames if f['well_name'].iloc[0] != blind_name]
        blind_frame  = all_frames[i]

        df_train = pd.concat(train_frames, ignore_index=True)

        # Terapkan cap
        mask_tr   = df_train[TARGET] <= y_cap
        df_train  = df_train[mask_tr]
        mask_bl   = blind_frame[TARGET] <= y_cap
        df_blind  = blind_frame[mask_bl]

        X_tr  = df_train[BASE_FEATURES].values
        y_tr  = df_train[TARGET].values
        X_bl  = df_blind[BASE_FEATURES].values
        y_bl  = df_blind[TARGET].values

        if len(y_bl) < 3:
            continue

        scaler_lowo = RobustScaler()
        scaler_lowo.fit(X_tr)

        for mname in LOWO_MODELS:
            m = _build_model(mname)
            if m is None:
                continue

            if mname == 'ANN':
                from sklearn.neural_network import MLPRegressor
                m = MLPRegressor(
                    hidden_layer_sizes=(256, 128, 64, 32),
                    activation='relu', solver='adam', alpha=0.001,
                    batch_size=64, max_iter=600, early_stopping=True,
                    validation_fraction=0.1, n_iter_no_change=25,
                    random_state=RANDOM_STATE, verbose=False,
                )
                m.fit(scaler_lowo.transform(X_tr), y_tr)
                y_pred = m.predict(scaler_lowo.transform(X_bl))
            else:
                m.fit(X_tr, y_tr)
                y_pred = m.predict(X_bl)

            met = compute_metrics(y_bl, y_pred)
            lowo_results[mname][blind_name] = met

            print(f"  {blind_name:<14} {mname:<14} "
                  f"{met['MAE']:>7.4f} {met['RMSE']:>7.4f} {met['R2']:>7.4f} {met['n']:>5}")

    print(f"  {'─'*58}")

    # ── Agregasi hasil ─────────────────────────────────────────────────────────
    print(f"\n  RATA-RATA LOWO-CV PER MODEL")
    print(f"  {'─'*42}")
    print(f"  {'MODEL':<14} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print(f"  {'─'*42}")

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
        print(f"  {mname:<14} "
              f"{summary[mname]['MAE_mean']:>7.4f}±{summary[mname]['MAE_std']:.3f}  "
              f"{summary[mname]['RMSE_mean']:>7.4f}  "
              f"{summary[mname]['R2_mean']:>7.4f}±{summary[mname]['R2_std']:.3f}")

    print(f"  {'─'*42}")

    best_lowo = min(summary, key=lambda k: summary[k]['MAE_mean']) if summary else None
    if best_lowo:
        print(f"\n  🏆 Model terbaik (LOWO): {best_lowo}  "
              f"(MAE={summary[best_lowo]['MAE_mean']:.4f} ± {summary[best_lowo]['MAE_std']:.4f} m)")

    # ── Simpan hasil ───────────────────────────────────────────────────────────
    df_lowo_rows = []
    for mname, well_dict in lowo_results.items():
        for wname, met in well_dict.items():
            df_lowo_rows.append({'Model': mname, 'Well': wname, **met})
    if df_lowo_rows:
        df_lowo = pd.DataFrame(df_lowo_rows)
        csv_path = f'{OUTPUT_DIR}/lowo_cv_detail.csv'
        df_lowo.to_csv(csv_path, index=False)
        print(f"\n  💾 Detail per fold: {csv_path}")

    # ── Visualisasi ────────────────────────────────────────────────────────────
    _plot_lowo(lowo_results, summary, well_names)

    return lowo_results, summary


def _plot_lowo(lowo_results, summary, well_names):
    n_models = len([k for k in lowo_results if lowo_results[k]])
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Leave-One-Well-Out Cross Validation — Blind Test Results',
                 fontsize=13, fontweight='bold')

    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    x      = np.arange(len(well_names))
    width  = 0.8 / n_models

    # Plot 1: MAE per sumur per model
    ax1 = axes[0]
    for j, (mname, well_dict) in enumerate(lowo_results.items()):
        if not well_dict:
            continue
        maes = [well_dict.get(w, {}).get('MAE', 0) for w in well_names]
        ax1.bar(x + j * width - (n_models - 1) * width / 2,
                maes, width, label=mname, color=colors[j], edgecolor='white', alpha=0.85)

    # Advisory threshold lines
    thresholds = {'🟢 5m': 5, '🟡 15m': 15, '🟠 30m': 30}
    linestyles = ['--', '-.', ':']
    for (lbl, val), ls in zip(thresholds.items(), linestyles):
        ax1.axhline(val, color='gray', linestyle=ls, lw=1, alpha=0.7, label=lbl)

    ax1.set_xticks(x)
    ax1.set_xticklabels(well_names, rotation=20, ha='right', fontsize=9)
    ax1.set_ylabel('MAE (meter)')
    ax1.set_title('MAE per Sumur (Blind Test)', fontweight='bold')
    ax1.legend(fontsize=8)

    # Plot 2: R² per sumur per model
    ax2 = axes[1]
    for j, (mname, well_dict) in enumerate(lowo_results.items()):
        if not well_dict:
            continue
        r2s = [well_dict.get(w, {}).get('R2', 0) for w in well_names]
        ax2.bar(x + j * width - (n_models - 1) * width / 2,
                r2s, width, label=mname, color=colors[j], edgecolor='white', alpha=0.85)

    ax2.axhline(0.8, color='green', linestyle='--', lw=1, alpha=0.6, label='R²=0.8 target')
    ax2.axhline(0.6, color='orange', linestyle='--', lw=1, alpha=0.6, label='R²=0.6 baseline')
    ax2.set_xticks(x)
    ax2.set_xticklabels(well_names, rotation=20, ha='right', fontsize=9)
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² per Sumur (Blind Test)', fontweight='bold')
    ax2.set_ylim(bottom=min(0, ax2.get_ylim()[0]))
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plot_path = f'{OUTPUT_DIR}/lowo_cv_results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  📊 Plot: {plot_path}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        BLIND TEST — Drilling Trajectory Model           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Skenario aktif: {SKENARIO}\n")

    if SKENARIO in ('A', 'AB'):
        results_a = run_scenario_a()

    if SKENARIO in ('B', 'AB'):
        results_b = run_scenario_b()

    print(f"\n{'═'*62}")
    print(f"  ✅ Blind test selesai. Output di folder: {OUTPUT_DIR}/")
    print(f"{'═'*62}")

    # ── Catatan interpretasi ───────────────────────────────────────────────────
    print("""
  INTERPRETASI HASIL:
  ─────────────────────────────────────────────────────────
  MAE < 5m   → Model sangat andal, sesuai threshold ON TRACK
  MAE 5–15m  → Cukup baik untuk advisory level MONITOR
  MAE > 15m  → Model perlu lebih banyak data atau tuning

  R² > 0.8   → Model menjelaskan >80% pola deviasi lintasan
  R² 0.6–0.8 → Acceptable untuk advisory pendukung keputusan
  R² < 0.6   → Kurang andal, pertimbangkan fitur tambahan

  Jika MAE blind test >> MAE training: model overfit ke
  karakteristik sumur tertentu → tambahkan lebih banyak sumur
  atau perkecil kompleksitas model.
  ─────────────────────────────────────────────────────────
""")
