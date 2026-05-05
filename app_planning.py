import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize_scalar, minimize
from plotly.subplots import make_subplots
from io import StringIO
import uuid
import warnings
import re
import io
import xml.etree.ElementTree as ET
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader, PdfWriter
from datetime import datetime # Import datetime untuk Created On
import joblib
import os

warnings.filterwarnings('ignore')

st.set_page_config(page_title="DSS Well Master Ultimate", layout="wide", page_icon="🏗️")

def parse_casing_od(size_str):
    """Parse casing OD in inches from size strings like '20"', '9-5/8"', '13-3/8"'."""
    s = str(size_str).replace('"', '').replace("'", '').strip()
    if '-' in s:
        parts = s.split('-', 1)
        try:
            whole = float(parts[0])
            frac_str = parts[1]
            if '/' in frac_str:
                num, den = frac_str.split('/')
                return whole + float(num) / float(den)
        except Exception:
            pass
    if '/' in s:
        try:
            num, den = s.split('/')
            return float(num) / float(den)
        except Exception:
            pass
    try:
        return float(s)
    except Exception:
        return 10.0

SECTION_PALETTE = [
    {'color': '#555555', 'width': 16, 'label': 'Conductor'},
    {'color': '#9E9E9E', 'width': 12, 'label': 'Surface Csg'},
    {'color': '#1565C0', 'width': 9,  'label': 'Interm. Csg'},
    {'color': '#2E7D32', 'width': 6,  'label': 'Prod. Csg'},
    {'color': '#8B4513', 'width': 4,  'label': 'Open Hole'},
]

# Feature names must match exactly what the LSTM model was trained on
ML_FEATURES = [
    'measured_depth', 'inclination', 'azimuth', 'dogleg_severity',
    'delta_inc', 'delta_azi', 'inc_error', 'azi_error',
    'rolling_dls_mean', 'rolling_inc_std', 'depth_norm',
    'cumulative_inc', 'section_code', 'dls_vs_plan',
]

def build_ml_features(df_corr, df_plan):
    """Build the 14-feature DataFrame the LightGBM model expects."""
    df = df_corr.reset_index(drop=True)
    MD  = df['MD'].astype(float)
    Inc = df['Inc'].astype(float)
    Azi = df['Azi'].astype(float)
    DLS = df['DLS'].astype(float) if 'DLS' in df.columns else pd.Series(0.0, index=df.index)

    # Interpolate plan values at each correction MD
    p_md  = df_plan['MD'].values.astype(float)
    p_inc = np.interp(MD.values, p_md, df_plan['Inc'].values.astype(float))
    p_azi = np.interp(MD.values, p_md, df_plan['Azi'].values.astype(float))
    p_dls_arr = df_plan['DLS'].values.astype(float) if 'DLS' in df_plan.columns else np.zeros(len(df_plan))
    p_dls = np.interp(MD.values, p_md, p_dls_arr)

    delta_inc = Inc.diff().fillna(0.0)
    raw_dazi  = Azi.diff().fillna(0.0)
    # Handle 359→1 wrap-around
    delta_azi = raw_dazi.apply(lambda x: x - 360 if x > 180 else (x + 360 if x < -180 else x))

    rolling_dls_mean = DLS.rolling(5, min_periods=1).mean()
    rolling_inc_std  = Inc.rolling(5, min_periods=1).std().fillna(0.0)

    md_range   = MD.max() - MD.min()
    depth_norm = (MD - MD.min()) / (md_range if md_range > 1e-9 else 1.0)
    cum_inc    = Inc.cumsum()

    # Section code: 0=Vertical 1=Build 2=Hold 3=Drop
    g = delta_inc.rolling(3, min_periods=1).mean()
    sec = pd.Series(2, index=df.index)
    sec[Inc < 2.0] = 0
    sec[(Inc >= 2.0) & (g >  0.1)] = 1
    sec[(Inc >= 2.0) & (g < -0.1)] = 3

    feat = pd.DataFrame({
        'measured_depth':   MD.values,
        'inclination':      Inc.values,
        'azimuth':          Azi.values,
        'dogleg_severity':  DLS.values,
        'delta_inc':        delta_inc.values,
        'delta_azi':        delta_azi.values,
        'inc_error':        Inc.values - p_inc,
        'azi_error':        Azi.values - p_azi,
        'rolling_dls_mean': rolling_dls_mean.values,
        'rolling_inc_std':  rolling_inc_std.values,
        'depth_norm':       depth_norm.values,
        'cumulative_inc':   cum_inc.values,
        'section_code':     sec.values.astype(float),
        'dls_vs_plan':      DLS.values - p_dls,
    })[ML_FEATURES]  # enforce column order
    return feat.fillna(0.0)


def compute_distance_to_plan(df_corr, df_plan):
    """
    Compute actual 3D distance (metres) from each correction point to the
    nearest point on the plan curve.  Returns a numpy array, one value per row.
    """
    c = df_corr[['N', 'E', 'TVD']].values.astype(float)
    p = df_plan[['N', 'E', 'TVD']].values.astype(float)
    # Downsample correction to at most 200 pts for speed, then interpolate back
    step = max(1, len(c) // 200)
    c_s  = c[::step]
    dists_s = np.array([float(np.min(np.sqrt(np.sum((p - pt)**2, axis=1)))) for pt in c_s])
    # Interpolate back to full length
    idx_s  = np.arange(len(c_s)) * step
    idx_f  = np.arange(len(c))
    return np.interp(idx_f, idx_s, dists_s)


class LSTMAdvisoryModel:
    """Wraps the Keras LSTM artifact to expose the same predict(feats_df) interface."""

    model_type = 'LSTM'

    def __init__(self, keras_model, scaler, seq_len, n_features):
        self.model = keras_model
        self.scaler = scaler
        self.seq_len = seq_len
        self.n_features = n_features

    def predict(self, feats_df):
        X = feats_df.values.astype(np.float32)
        X_sc = self.scaler.transform(X)
        n = len(X_sc)
        seqs = np.zeros((n, self.seq_len, self.n_features), dtype=np.float32)
        for i in range(n):
            start = max(0, i - self.seq_len + 1)
            chunk = X_sc[start : i + 1]
            seqs[i, -len(chunk):] = chunk
        return self.model.predict(seqs, verbose=0).flatten()


class NumpyLSTMAdvisoryModel:
    """Pure-numpy LSTM inference — no TensorFlow required.

    Matches the DrillLSTM architecture from train_model_v2.py:
      LSTM(64, return_sequences=True) → Dropout → BN →
      LSTM(32, return_sequences=False) → Dropout → BN →
      Dense(32, relu) → Dense(16, relu) → Dense(1, linear)

    Weight index layout (from model.get_weights()):
      0-2  : LSTM1 kernel(n,256), recurrent(64,256), bias(256)
      3-6  : BN1 gamma, beta, moving_mean, moving_var
      7-9  : LSTM2 kernel(64,128), recurrent(32,128), bias(128)
      10-13: BN2 gamma, beta, moving_mean, moving_var
      14-15: Dense(32) kernel, bias
      16-17: Dense(16) kernel, bias
      18-19: Dense(1)  kernel, bias
    """

    model_type = 'LSTM'

    def __init__(self, weights, scaler, seq_len, n_features):
        self.scaler = scaler
        self.seq_len = seq_len
        self.n_features = n_features
        w = weights
        self.lstm1 = (w[0].astype(np.float32), w[1].astype(np.float32), w[2].astype(np.float32))
        self.bn1   = (w[3].astype(np.float32), w[4].astype(np.float32), w[5].astype(np.float32), w[6].astype(np.float32))
        self.lstm2 = (w[7].astype(np.float32), w[8].astype(np.float32), w[9].astype(np.float32))
        self.bn2   = (w[10].astype(np.float32), w[11].astype(np.float32), w[12].astype(np.float32), w[13].astype(np.float32))
        self.d1    = (w[14].astype(np.float32), w[15].astype(np.float32))
        self.d2    = (w[16].astype(np.float32), w[17].astype(np.float32))
        self.d3    = (w[18].astype(np.float32), w[19].astype(np.float32))

    @staticmethod
    def _sigmoid(x):
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

    @staticmethod
    def _lstm_fwd(seqs, kernel, rec_kernel, bias, units, return_seq):
        batch, seq_len, _ = seqs.shape
        h = np.zeros((batch, units), dtype=np.float32)
        c = np.zeros((batch, units), dtype=np.float32)
        sig = NumpyLSTMAdvisoryModel._sigmoid
        out_h = []
        for t in range(seq_len):
            z = seqs[:, t, :] @ kernel + h @ rec_kernel + bias
            i_g = sig(z[:, :units])
            f_g = sig(z[:, units:2*units])
            g_g = np.tanh(z[:, 2*units:3*units])
            o_g = sig(z[:, 3*units:])
            c = f_g * c + i_g * g_g
            h = o_g * np.tanh(c)
            out_h.append(h)
        return np.stack(out_h, axis=1) if return_seq else h

    @staticmethod
    def _bn(x, gamma, beta, mean, var, eps=1e-3):
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    def predict(self, feats_df):
        X_sc = self.scaler.transform(feats_df.values.astype(np.float32))
        n = len(X_sc)
        seqs = np.zeros((n, self.seq_len, self.n_features), dtype=np.float32)
        for i in range(n):
            start = max(0, i - self.seq_len + 1)
            chunk = X_sc[start:i + 1]
            seqs[i, -len(chunk):] = chunk

        x = self._lstm_fwd(seqs, *self.lstm1, units=64, return_seq=True)
        x = self._bn(x, *self.bn1)
        x = self._lstm_fwd(x, *self.lstm2, units=32, return_seq=False)
        x = self._bn(x, *self.bn2)
        x = np.maximum(0.0, x @ self.d1[0] + self.d1[1])
        x = np.maximum(0.0, x @ self.d2[0] + self.d2[1])
        x = x @ self.d3[0] + self.d3[1]
        return x.flatten()


@st.cache_resource
def load_drilling_model():
    base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, 'lstm_model.joblib')

    # Try TensorFlow-backed loading first (Python ≤ 3.12)
    try:
        import tensorflow as tf
        artifact = joblib.load(model_path)
        keras_model = tf.keras.models.model_from_json(artifact['model_config'])
        keras_model.set_weights(artifact['model_weights'])
        return LSTMAdvisoryModel(
            keras_model=keras_model,
            scaler=artifact['scaler'],
            seq_len=artifact['seq_len'],
            n_features=len(artifact['feature_names']),
        )
    except Exception:
        pass

    # Fallback: pure-numpy inference (Python 3.13+, no TensorFlow)
    try:
        artifact = joblib.load(model_path)
        return NumpyLSTMAdvisoryModel(
            weights=artifact['model_weights'],
            scaler=artifact['scaler'],
            seq_len=artifact['seq_len'],
            n_features=len(artifact['feature_names']),
        )
    except Exception as e:
        st.error(f"Failed to load LSTM model: {e}")
        return None

drilling_model = load_drilling_model()

# ==========================================
# 1. ENGINE & LOGIC (HI-RES UPGRADE)
# ==========================================
class UncertaintyEngine:
    def __init__(self):
        self.error_lat = 2.0; self.error_vert = 1.5
    def calculate_error_model(self, df):
        scale = df['MD'] / 1000.0
        df['Err_Lat'] = scale * self.error_lat
        df['Err_Vert'] = scale * self.error_vert
        df['Err_Rad'] = np.sqrt(df['Err_Lat']**2 + df['Err_Vert']**2)
        return df

class DrillingEngine:
    def __init__(self, unit, dls_ref):
        self.unit = unit
        self.dls_ref = dls_ref 
        
    def calculate_trajectory(self, md, inc, azi, start_n, start_e, start_tvd):
        # Ini adalah perhitungan standar MCM
        n = len(md)
        tvd = np.zeros(n); n_c = np.zeros(n); e_c = np.zeros(n)
        dls = np.zeros(n); bur = np.zeros(n); tr = np.zeros(n)
        
        tvd[0], n_c[0], e_c[0] = start_tvd, start_n, start_e
        inc_rad = np.radians(inc); azi_rad = np.radians(azi)
        
        for i in range(1, n):
            dL = md[i] - md[i-1]
            if dL <= 0: continue
            
            I1, I2 = inc_rad[i-1], inc_rad[i]
            A1, A2 = azi_rad[i-1], azi_rad[i]
            
            cos_beta = np.cos(I2-I1) - (np.sin(I1)*np.sin(I2)*(1-np.cos(A2-A1)))
            beta = np.arccos(np.clip(cos_beta, -1, 1))
            rf = (2/beta)*np.tan(beta/2) if abs(beta)>1e-6 else 1.0
            
            n_c[i] = n_c[i-1] + (dL/2)*(np.sin(I1)*np.cos(A1) + np.sin(I2)*np.cos(A2))*rf
            e_c[i] = e_c[i-1] + (dL/2)*(np.sin(I1)*np.sin(A1) + np.sin(I2)*np.sin(A2))*rf
            tvd[i] = tvd[i-1] + (dL/2)*(np.cos(I1) + np.cos(I2))*rf
            
            if dL > 0:
                dls[i] = np.degrees(beta) * (self.dls_ref / dL)
                bur[i] = np.degrees(I2 - I1) * (self.dls_ref / dL)
                d_azi = np.degrees(A2 - A1)
                if d_azi > 180: d_azi -= 360
                elif d_azi < -180: d_azi += 360
                tr[i] = d_azi * (self.dls_ref / dL)

        df = pd.DataFrame({'MD': md, 'Inc': inc, 'Azi': azi, 'TVD': tvd, 'N': n_c, 'E': e_c, 'DLS': dls, 'BUR': bur, 'TR': tr})
        df['VS'] = np.sqrt((df['N']-start_n)**2 + (df['E']-start_e)**2)
        return df

    def resample_and_smooth(self, df_input, step=1.0):
        """
        FITUR BARU: Mengambil data kasar (misal per 30m) dan interpolasi menjadi per 1m (halus).
        Menggunakan asumsi Constant DLS antara titik survey.
        """
        mds, incs, azis = [], [], []
        
        # Loop setiap interval survey asli
        for i in range(1, len(df_input)):
            md_prev, md_curr = df_input['MD'].iloc[i-1], df_input['MD'].iloc[i]
            inc_prev, inc_curr = df_input['Inc'].iloc[i-1], df_input['Inc'].iloc[i]
            azi_prev, azi_curr = df_input['Azi'].iloc[i-1], df_input['Azi'].iloc[i]
            
            dist = md_curr - md_prev
            if dist <= 0: continue
            
            # Hitung jumlah langkah interpolasi (per 1 meter)
            n_steps = int(dist / step)
            
            # Interpolasi Linear untuk Inc dan Azi (valid untuk Constant DLS)
            # Handle Azimuth Wrap (359 -> 1)
            d_azi = azi_curr - azi_prev
            if d_azi > 180: azi_prev += 360
            elif d_azi < -180: azi_curr += 360
            
            new_md = np.linspace(md_prev, md_curr, n_steps + 1)[:-1] # Exclude last point to avoid double
            new_inc = np.linspace(inc_prev, inc_curr, n_steps + 1)[:-1]
            new_azi = np.linspace(azi_prev, azi_curr, n_steps + 1)[:-1] % 360
            
            mds.extend(new_md); incs.extend(new_inc); azis.extend(new_azi)
            
        # Tambahkan titik terakhir
        mds.append(df_input['MD'].iloc[-1])
        incs.append(df_input['Inc'].iloc[-1])
        azis.append(df_input['Azi'].iloc[-1])
        
        # Hitung Ulang Koordinat dengan data rapat
        # Ambil koordinat awal dari input data (Surface / Tie-in)
        # Jika N/E/TVD belum ada, asumsikan 0 (akan dikoreksi di main app)
        s_n = df_input['N'].iloc[0] if 'N' in df_input.columns else 0
        s_e = df_input['E'].iloc[0] if 'E' in df_input.columns else 0
        s_tvd = df_input['TVD'].iloc[0] if 'TVD' in df_input.columns else 0
        
        return self.calculate_trajectory(mds, incs, azis, s_n, s_e, s_tvd)

class SmartPlanner:
    def __init__(self, surf_n, surf_e, rkb, unit_system='Metric'):
        self.surf_n = surf_n; self.surf_e = surf_e; self.rkb = rkb
        self.unit = unit_system; self.ft_to_m = 0.3048
        self.dls_ref = 30.0 if unit_system == 'Metric' else 100.0
        self.engine = DrillingEngine(unit_system, self.dls_ref)
        self.risk_engine = UncertaintyEngine()
        
    def solve_trajectory(self, target_n, target_e, target_tvdss, kop, dls, force_hold=None):
        # Unit conversion logic (simplified)
        f = self.ft_to_m if self.unit == 'Imperial' else 1.0
        rkb_m = self.rkb * f; kop_m = kop * f
        dls_m = dls * (30.0/(100.0*self.ft_to_m)) if self.unit=='Imperial' else dls
        
        tgt_tvd = (target_tvdss * f) + rkb_m
        s_n = self.surf_n * f; s_e = self.surf_e * f
        t_n = target_n * f; t_e = target_e * f
        
        delta_n = t_n - s_n; delta_e = t_e - s_e
        target_hd = np.sqrt(delta_n**2 + delta_e**2)
        tgt_azi = np.degrees(np.arctan2(delta_e, delta_n)) % 360
        
        # Solver J-Profile
        def err(h):
            r = np.radians(h); rad = (180/np.pi)*(30.0/dls_m)
            b_tvd = kop_m + (rad * np.sin(r))
            b_hd = rad * (1 - np.cos(r))
            rem_tvd = tgt_tvd - b_tvd
            return abs((b_hd + (rem_tvd * np.tan(r))) - target_hd) if rem_tvd > 0 else 1e6

        if force_hold and force_hold > 0.1: best_hold = force_hold
        else: best_hold = minimize_scalar(err, bounds=(0, 90), method='bounded').x
        
        # Generate Path dengan Resolusi Tinggi
        df, azi, hold = self._generate_path_hires(kop_m, dls_m, best_hold, tgt_azi, tgt_tvd, rkb_m, s_n, s_e)
        
        df = self.risk_engine.calculate_error_model(df)
        
        if self.unit == 'Imperial':
            for c in ['MD', 'TVD', 'TVDSS', 'VS', 'N', 'E', 'Err_Lat', 'Err_Vert', 'Err_Rad']:
                df[c] /= self.ft_to_m
                
        return df, azi, hold

    def _generate_path_hires(self, kop, dls, hold_inc, azi, target_tvd, rkb_val, surf_n, surf_e):
        # UPDATE: STEP SIZE 1.0 METER (HIGH RES)
        step = 1.0 
        mds, incs, azis = [0], [0], [azi]
        
        # Vertical Section
        mds.extend(np.arange(step, kop, step))
        mds.append(kop); incs.extend([0]*(len(mds)-1)); azis.extend([azi]*(len(mds)-1))
        
        # Build Section
        radius = (180/np.pi) * (30.0/dls)
        build_len = np.radians(hold_inc) * radius
        eob = kop + build_len
        
        build_mds = np.arange(kop + step, eob, step)
        mds.extend(build_mds)
        fracs = (build_mds - kop) / build_len
        incs.extend(fracs * hold_inc)
        azis.extend([azi]*len(build_mds))
        
        # EOB Point
        mds.append(eob); incs.append(hold_inc); azis.append(azi)
        
        # Hold Section
        tvd_eob = kop + (radius * np.sin(np.radians(hold_inc)))
        rem_tvd = target_tvd - tvd_eob
        if rem_tvd > 0:
            hold_len = rem_tvd / np.cos(np.radians(hold_inc))
            tgt_md = eob + hold_len
            
            hold_mds = np.arange(eob + step, tgt_md, step)
            mds.extend(hold_mds)
            incs.extend([hold_inc]*len(hold_mds))
            azis.extend([azi]*len(hold_mds))
            
            # TD Point
            mds.append(tgt_md); incs.append(hold_inc); azis.append(azi)
            
        df = self.engine.calculate_trajectory(mds, incs, azis, surf_n, surf_e, 0)
        df['Section'] = 'Plan'; df['TVDSS'] = df['TVD'] - rkb_val
        return df, azi, hold_inc

    def calculate_correction_path(self, actual_df, plan_df, model=None):
        """
        Find the correction path that MINIMISES predicted distance_to_plan.

        Architecture:
          1. Compute geometric direct-intercept as initial (bur0, tr0).
          2. If model available: optimise (bur, tr) so that the ML-predicted
             distance_to_plan is minimised across the entire correction path.
             The ML model IS the objective – geometry is only a constraint.
          3. Generate final 1 m-resolution path with best (bur, tr).
        """
        last = actual_df.iloc[-1]
        f    = self.ft_to_m if self.unit == 'Imperial' else 1.0

        N_c, E_c, TVD_c = float(last['N']), float(last['E']), float(last['TVD'])
        inc_c, azi_c, MD_c = float(last['Inc']), float(last['Azi']), float(last['MD'])

        tgt  = plan_df.iloc[-1]
        N_t, E_t, TVD_t = float(tgt['N']), float(tgt['E']), float(tgt['TVD'])

        # ── Geometric initial guess ───────────────────────────────────────────
        dN, dE, dTVD = N_t - N_c, E_t - E_c, TVD_t - TVD_c
        HD      = float(np.sqrt(dN**2 + dE**2))
        req_azi = float(np.degrees(np.arctan2(dE, dN)) % 360)
        req_inc = float(np.clip(np.degrees(np.arctan2(HD, max(dTVD, 0.1))), 0.0, 85.0))

        d_inc = req_inc - inc_c
        d_azi = float(((req_azi - azi_c + 180) % 360) - 180)

        dls_budget  = max(self.dls_ref * 0.4, 0.5)
        avg_inc_mid = float(np.sin(np.radians(inc_c + d_inc / 2.0)))
        build_len   = max(
            abs(d_inc)  / (dls_budget / 30.0),
            abs(d_azi * avg_inc_mid) / (dls_budget / 30.0),
            30.0,
        )
        bur0 = d_inc / max(build_len / 30.0, 1e-9)
        tr0  = d_azi / max(build_len / 30.0, 1e-9)

        # ── Candidate-path generator (coarse step for speed) ─────────────────
        def _gen(bur, tr, step=8.0):
            n_b  = max(1, int(build_len / step))
            di_b = (bur / 30.0) * step
            da_b = (tr  / 30.0) * step
            ms, ii, aa = [MD_c], [inc_c], [azi_c]
            for _ in range(n_b):
                ms.append(ms[-1] + step)
                ii.append(float(np.clip(ii[-1] + di_b, 0.0, 90.0)))
                aa.append((aa[-1] + da_b) % 360.0)

            # Re-aim from build end to final target and hold
            df_b  = self.engine.calculate_trajectory(ms, ii, aa, N_c, E_c, TVD_c)
            TVD_b = float(df_b['TVD'].iloc[-1])
            N_b   = float(df_b['N'].iloc[-1])
            E_b   = float(df_b['E'].iloc[-1])

            dN2, dE2, dTVD2 = N_t - N_b, E_t - E_b, TVD_t - TVD_b
            HD2   = float(np.sqrt(dN2**2 + dE2**2))
            h_azi = float(np.degrees(np.arctan2(dE2, dN2)) % 360)
            h_inc = float(np.clip(np.degrees(np.arctan2(HD2, max(dTVD2, 0.1))), 0.0, 85.0))
            h_dst = max(dTVD2 / max(np.cos(np.radians(h_inc)), 0.01), step)
            n_h   = max(1, int(h_dst / step))
            # smooth transition over first 30 m of hold
            ci2, ca2 = ii[-1], aa[-1]
            fi = (h_inc - ci2) / 30.0
            fa = float(((h_azi - ca2 + 180) % 360) - 180) / 30.0
            for k in range(n_h):
                ms.append(ms[-1] + step)
                if k < 30:
                    ci2 = float(np.clip(ci2 + fi, 0.0, 90.0))
                    ca2 = (ca2 + fa) % 360.0
                ii.append(ci2); aa.append(ca2)

            df_c = self.engine.calculate_trajectory(ms, ii, aa, N_c, E_c, TVD_c)
            df_c['TVDSS'] = df_c['TVD'] - (self.rkb * f)
            if self.unit == 'Imperial':
                df_c['TVDSS'] /= self.ft_to_m
            df_c['Section'] = 'Correction'
            return df_c

        # ── ML-guided optimisation (distance_to_plan is the objective) ────────
        if model is not None:
            def objective(params):
                b, t = float(params[0]), float(params[1])
                try:
                    df_c = _gen(b, t)
                    # Primary: minimise mean ML-predicted distance_to_plan
                    feats   = build_ml_features(df_c, plan_df)
                    preds   = np.clip(model.predict(feats), 0.0, None)
                    ml_cost = float(np.mean(preds))

                    # Constraint: DLS must stay within allowable limit
                    dls_val     = float(np.sqrt(b**2 + (t * np.sin(np.radians(inc_c)))**2))
                    dls_penalty = float(max(0.0, dls_val - self.dls_ref) ** 2) * 50.0

                    return ml_cost + dls_penalty
                except Exception:
                    return 1e9

            res = minimize(
                objective, [bur0, tr0],
                method='Nelder-Mead',
                bounds=[(-self.dls_ref, self.dls_ref),
                        (-self.dls_ref, self.dls_ref)],
                options={'maxiter': 400, 'xatol': 0.05, 'fatol': 0.05},
            )
            best_bur, best_tr = float(res.x[0]), float(res.x[1])
        else:
            best_bur, best_tr = bur0, tr0

        # ── Final hi-res path (1 m steps) ─────────────────────────────────────
        df = _gen(best_bur, best_tr, step=1.0)
        eff_len = float(df['MD'].iloc[-1]) - MD_c

        end = df.iloc[-1]
        dist_to_final = float(np.sqrt(
            (float(end['N'])   - N_t) ** 2 +
            (float(end['E'])   - E_t) ** 2 +
            (float(end['TVD']) - TVD_t) ** 2
        ))

        tot_dls = float(np.sqrt(best_bur**2 + (best_tr * np.sin(np.radians(inc_c)))**2))
        tf      = float(np.degrees(np.arctan2(best_tr, best_bur)) % 360)

        return df, tot_dls, best_bur, best_tr, tf, dist_to_final, eff_len

# ==========================================
# 2. UPDATED PARSER (WITH AUTO-SMOOTHING)
# ==========================================
def parse_trajectory_data(text_data, rkb, surf_n, surf_e, engine, azi_corr=0.0):
    if not text_data.strip(): return None
    try:
        data = StringIO(text_data)
        try: df = pd.read_csv(data, sep=None, engine='python') 
        except: data.seek(0); df = pd.read_csv(data, sep='\t')
        
        # Clean Headers
        df.columns = df.columns.str.upper().str.replace(r"[\(\[].*?[\)\]]", "", regex=True).str.strip()
        col_map = {'MEASURED DEPTH':'MD', 'INCLINATION':'Inc', 'AZIMUTH':'Azi', 'TRUE VERTICAL DEPTH':'TVD', 'NORTH':'N', 'EAST':'E'}
        for c in df.columns:
            for k, v in col_map.items():
                if k in c: df.rename(columns={c: v}, inplace=True); break
        
        req = ['MD', 'Inc', 'Azi']
        if not all(c in df.columns for c in req): return "MISSING_COLS"
        
        # Apply Azimuth Correction
        df['Azi'] = (df['Azi'] + azi_corr) % 360
        
        # Anchor to Surface if needed
        if df['MD'].iloc[0] > 0: 
            df = pd.concat([pd.DataFrame({'MD':[0],'Inc':[0],'Azi':[0], 'N':[surf_n], 'E':[surf_e], 'TVD':[0]}), df], ignore_index=True)
        
        # CRITICAL UPDATE: RESAMPLE & SMOOTH DATA (Interpolate to 1m)
        # Data impor biasanya jarang (misal per 30m atau 100ft), ini yang bikin patah.
        # Kita panggil engine.resample_and_smooth
        
        # Jika data belum punya koordinat, hitung dulu surface reference-nya
        df['N'] = df.get('N', surf_n)
        df['E'] = df.get('E', surf_e)
        df['TVD'] = df.get('TVD', 0) # Dummy start
        
        # Lakukan Resampling (Hi-Res Calculation)
        df_smooth = engine.resample_and_smooth(df, step=1.0)
        
        # Apply TVDSS
        df_smooth['TVDSS'] = df_smooth['TVD'] - rkb
        
        return df_smooth
        
    except Exception as e: return str(e)
# ==========================================
# 2. PARSERS & UTILS
# ==========================================
def calculate_economics(df):
    total_md = df['MD'].iloc[-1]
    cost = (total_md * 1500) + ((total_md/10/24) * 50000)
    return cost, total_md/240
def parse_hierarchical_case_data(uploaded_file):
    """
    Parser Cerdas: Membaca hubungan antara PROJECT -> WELL -> CASE.
    FIXED: Mengembalikan tuple (DataFrame, Error) agar sesuai dengan caller.
    """
    try:
        tree = ET.parse(uploaded_file)
        root = tree.getroot()
        
        # 1. Dictionary untuk menyimpan NAMA berdasarkan ID
        well_map = {}
        project_map = {}
        
        # 2. SCANNING TAHAP 1: Cari Nama Project & Well
        for child in root.findall(".//*"):
            tag = child.tag.split('}')[-1].upper()
            attr = {k.lower(): v for k, v in child.attrib.items()}
            
            if tag == 'CD_WELL':
                w_id = attr.get('well_id')
                w_name = attr.get('well_common_name', attr.get('well_name', w_id))
                if w_id: well_map[w_id] = w_name
                
            elif tag == 'CD_PROJECT':
                p_id = attr.get('project_id')
                p_name = attr.get('project_name', p_id)
                if p_id: project_map[p_id] = p_name

        # 3. SCANNING TAHAP 2: Ambil Case dan hubungkan dengan Nama Well
        cases = []
        for child in root.findall(".//*"):
            tag = child.tag.split('}')[-1].upper()
            
            if tag == 'CD_CASE':
                case_data = {k.lower(): v for k, v in child.attrib.items()}
                
                # Link ke Well Name
                w_id_ref = case_data.get('well_id')
                # Gunakan Unknown jika ID tidak ditemukan di map
                case_data['well_name_resolved'] = well_map.get(w_id_ref, f"Unknown Well ({w_id_ref})")
                
                # Bersihkan tanggal
                for key, val in case_data.items():
                    if isinstance(val, str) and val.startswith("{ts"):
                        case_data[key] = val.replace("{ts '", "").replace("'}", "")
                
                if 'case_name' not in case_data:
                    case_data['case_name'] = "Unnamed Case"
                    
                cases.append(case_data)
        
        if not cases:
            return None, "XML valid, tapi tidak ada tag <CD_CASE>."
            
        # --- PERBAIKAN UTAMA DI SINI ---
        # Kembalikan Tuple: (DataFrame, None)
        return pd.DataFrame(cases), None 

    except Exception as e:
        return None, str(e)
def parse_xml_file(uploaded_file):
    """
    Universal Parser: WITSML + Landmark EDM.
    Fitur Baru: Mengabaikan tag CD_ATTACHMENT (Gambar) agar data bersih.
    """
    try:
        tree = ET.parse(uploaded_file)
        root = tree.getroot()
        data = []
        
        # Keyword kolom yang valid (Data Survey)
        valid_keys = ['md', 'mdmn', 'measured_depth', 'inclination', 'azimuth', 'tvd', 'disp_ns', 'disp_ew', 'offset_north', 'offset_east']

        for child in root.findall(".//*"):
            # 1. CEK TAG: Jika ini adalah Attachment/Gambar, LEWATI (SKIP)
            tag_name = child.tag.split('}')[-1].upper()
            if 'ATTACHMENT' in tag_name or 'BLOB' in tag_name or 'IMAGE' in tag_name:
                continue 

            row_data = {}
            
            # A. Ambil Attributes (Style EDM)
            if child.attrib:
                for k, v in child.attrib.items():
                    row_data[k.lower()] = v 

            # B. Ambil Child Tags (Style WITSML)
            if len(child) > 0:
                for sub in child:
                    sub_tag = sub.tag.split('}')[-1].lower()
                    if sub.text:
                        row_data[sub_tag] = sub.text
            
            # C. VALIDASI: Apakah baris ini punya data MD/Inc/Azi?
            keys = row_data.keys()
            # Harus punya unsur Depth (MD)
            has_depth = any(k in keys for k in ['md', 'measured_depth', 'mdmn'])
            # Harus punya unsur Data (Inc/Azi/Coord)
            has_data = any(k in keys for k in ['inc', 'inclination', 'azimuth', 'offset_north', 'tvd'])
            
            # Filter tambahan: Jangan ambil baris yang cuma punya ID tapi ga ada angka
            if has_depth and has_data:
                data.append(row_data)

        if not data: 
            return "XML_CLEAN: No valid survey rows found. (Attachments ignored)."
        
        df = pd.DataFrame(data)
        
        # Clean numeric columns
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
            
        return df
        
    except Exception as e:
        return f"XML Error: {str(e)}"

def parse_case_data(uploaded_file):
    """
    Parser khusus untuk membaca tag <CD_CASE>.
    Mengambil daftar BHA/Case Design beserta detail operasionalnya.
    """
    try:
        tree = ET.parse(uploaded_file)
        root = tree.getroot()
        
        cases = []
        
        # Cari semua tag CD_CASE
        for child in root.findall(".//*"):
            tag = child.tag.split('}')[-1].upper()
            
            if tag == 'CD_CASE':
                # Ambil semua atribut secara otomatis
                case_data = {k.lower(): v for k, v in child.attrib.items()}
                
                # Bersihkan format tanggal (misal: {ts '2025...'} -> 2025...)
                for key, val in case_data.items():
                    if isinstance(val, str) and val.startswith("{ts"):
                        case_data[key] = val.replace("{ts '", "").replace("'}", "")
                
                # Pastikan ada nama case
                if 'case_name' not in case_data:
                    case_data['case_name'] = "Unnamed Case"
                    
                cases.append(case_data)
        
        if not cases:
            return None, "Tidak ditemukan tag <CD_CASE> dalam file XML ini."
            
        df = pd.DataFrame(cases)
        return df, None

    except Exception as e:
        return None, str(e)

def parse_trajectory_data(input_data, rkb, surf_n, surf_e, engine, azi_corr=0.0):
    try:
        # --- 1. LOAD DATA ---
        df = None
        if isinstance(input_data, pd.DataFrame): df = input_data
        elif isinstance(input_data, str) and input_data.strip():
            if any(ord(c) > 127 for c in input_data[:100]): return "ERROR_BINARY"
            data = StringIO(input_data)
            try: df = pd.read_csv(data, sep=None, engine='python') 
            except: data.seek(0); df = pd.read_csv(data, sep='\t')
        else: return None 

        # --- 2. MAPPING KOLOM (Compass/Landmark Support) ---
        df.columns = df.columns.str.upper().str.replace(r"[\(\[].*?[\)\]]", "", regex=True).str.strip()
        
        col_map = {
            'MD': 'MD', 'MEASURED DEPTH':'MD', 'DEPTH':'MD',
            'INC': 'Inc', 'INCLINATION':'Inc', 'ANGLE':'Inc',
            'AZI': 'Azi', 'AZIMUTH':'Azi', 'DIR':'Azi',
            'TVD': 'TVD',
            # Format XML Anda
            'OFFSET_NORTH': 'N', 'OFFSET_EAST': 'E',
            'DISP_NS': 'N', 'DISP_EW': 'E',
            'MAP_NORTH': 'N', 'MAP_EAST': 'E',
            'MDMN': 'MD', 'INCL': 'Inc', 'AZIM': 'Azi'
        }
        
        new_names = {}
        for c in df.columns:
            for k, v in col_map.items():
                if k == c: new_names[c] = v; break
                if k in c and len(c) > len(k): new_names[c] = v; break
        df.rename(columns=new_names, inplace=True)
        
        # Hapus kolom ganda hasil rename
        df = df.loc[:, ~df.columns.duplicated()]

        # --- 3. DATA CLEANING (CRITICAL FIX) ---
        req = ['MD', 'Inc', 'Azi'] 
        if not all(c in df.columns for c in req): 
            return f"MISSING_COLS: {list(df.columns)}"
        
        for c in req: df[c] = pd.to_numeric(df[c], errors='coerce')
        df.dropna(subset=req, inplace=True)

        # >>> FIX BENANG KUSUT DI SINI <<<
        # 1. Urutkan berdasarkan MD (dari surface ke TD)
        df = df.sort_values(by='MD', ascending=True)
        
        # 2. Hapus duplikat MD (jika ada overlap survey)
        df = df.drop_duplicates(subset=['MD'], keep='last')
        
        # 3. Reset Index biar rapi
        df = df.reset_index(drop=True)
        # >>> END FIX <<<

        # --- 4. KALKULASI & VISUALISASI ---
        df['Azi'] = (df['Azi'] + azi_corr) % 360
        
        # Anchor Surface (Jika data mulai dari kedalaman > 0)
        if df['MD'].iloc[0] > 0: 
            row0 = pd.DataFrame({'MD':[0],'Inc':[0],'Azi':[0], 'N':[surf_n], 'E':[surf_e], 'TVD':[0]})
            df = pd.concat([row0, df], ignore_index=True)
        else:
            if 'N' not in df.columns: df['N'] = surf_n
            else: df.loc[df['MD']==0, 'N'] = surf_n
            if 'E' not in df.columns: df['E'] = surf_e
            else: df.loc[df['MD']==0, 'E'] = surf_e
            if 'TVD' not in df.columns: df['TVD'] = 0
            else: df.loc[df['MD']==0, 'TVD'] = 0

        # Panggil Engine (Resample 1m agar mulus)
        df_smooth = engine.resample_and_smooth(df, step=1.0)
        df_smooth['TVDSS'] = df_smooth['TVD'] - rkb
        
        return df_smooth

    except Exception as e: return str(e)

def parse_scenario_bha_chain(xml_file):
    """
    Parser Berjenjang UPDATE (Stab Lookup Fix):
    1. Scan CD_BHA_COMP_STAB untuk mendapatkan 'stab_blade_od'.
    2. Logic Gauge: Prioritas UTAMA ambil dari stab_blade_od jika ID cocok.
    3. Logic Dimensi Lain: Tetap seperti sebelumnya (Fishneck ID, Feet-to-Meter).
    """
    try:
        xml_file.seek(0)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        scenarios = {}      
        cases = []          
        raw_cd_comps = []   
        mb_lookup = {}      
        
        # --- NEW: KAMUS KHUSUS STABILIZER ---
        # Format: { 'assembly_comp_id': 15.75 }
        stab_lookup = {} 

        # --- SCANNING XML ---
        for child in root.findall(".//*"):
            tag = child.tag.split('}')[-1].upper()
            attr = {k.lower(): v for k, v in child.attrib.items()}
            
            if tag == 'CD_SCENARIO':
                sid = attr.get('scenario_id')
                sname = attr.get('name', attr.get('scenario_name', sid))
                if sid: scenarios[sid] = sname
            
            elif tag == 'CD_CASE':
                sid_ref = attr.get('scenario_id')
                aid_ref = attr.get('assembly_id')
                cname = attr.get('case_name', 'Unnamed Case')
                desc = attr.get('case_description', '')
                if "Auto created" in desc: continue
                if "Casing String" in cname and "Run" not in cname: continue
                if sid_ref and aid_ref:
                    cases.append({'scenario_id': sid_ref, 'case_name': cname, 'assembly_id': aid_ref})

            elif tag == 'MB_ASSEMBLY_COMP':
                mb_id = attr.get('assembly_comp_id')
                mb_desc = attr.get('description')
                if mb_id and mb_desc: mb_lookup[mb_id] = mb_desc

            # --- DETEKSI DATA SPESIFIK STABILIZER ---
            elif tag == 'CD_BHA_COMP_STAB':
                s_id = attr.get('assembly_comp_id')
                # Ambil stab_blade_od
                s_od = float(attr.get('stab_blade_od', 0))
                
                if s_id and s_od > 0:
                    stab_lookup[s_id] = s_od

            elif tag in ['CD_ASSEMBLY_COMP', 'CD_ASSEMBLY_COMPONENT', 'CD_DRILL_STRING_COMP']:
                if attr.get('assembly_id'):
                    try: seq = int(attr.get('sequence_no', 0))
                    except: seq = 0 
                    attr['parsed_seq'] = seq 
                    raw_cd_comps.append(attr)

        # --- PROCESSING DATA ---
        final_components = []
        
        for attr in raw_cd_comps:
            # 1. Lookup Nama
            link_id = attr.get('assembly_comp_id')
            display_name = mb_lookup.get(link_id, attr.get('catalog_key_desc', attr.get('component_name', '-')))
            
            # 2. Ambil Dimensi Dasar
            val_od_body   = float(attr.get('od_body', attr.get('body_od', 0)))
            val_id_body   = float(attr.get('id_body', attr.get('body_id', 0)))
            val_fish_od   = float(attr.get('fishneck_od', 0))
            val_fish_len  = float(attr.get('fishneck_length', 0))
            type_code     = attr.get('sect_type_code', '').upper()

            # --- LOGIC OD (Tabel) ---
            # Prioritas: Fishneck OD -> Body OD
            if val_fish_od > 0:
                final_od = val_fish_od
            else:
                final_od = val_od_body

            # --- LOGIC ID (Tabel) ---
            # Prioritas: Fishneck Length -> ID Body
            if val_fish_len > 0:
                final_id = val_fish_len
            else:
                final_id = val_id_body

            # --- LOGIC GAUGE (Critical Update) ---
            final_gauge = None
            
            # CEK 1: Apakah ID komponen ini punya data di tabel CD_BHA_COMP_STAB?
            if link_id in stab_lookup:
                final_gauge = stab_lookup[link_id] # Pakai 15.75 dari tabel khusus
                
            # CEK 2: Jika tidak, cek apakah dia BIT (PDC/Tri-Cone)
            elif 'BIT' in type_code or type_code == 'PDC':
                final_gauge = val_od_body # Bit Gauge = Body OD
            
            # CEK 3: Fallback untuk Stabilizer lama (jika tidak ada di tabel STAB)
            elif type_code in ['IBS', 'SBS', 'STAB', 'REZ']:
                # Coba cari di atribut biasa (od_blade / blade_od)
                val_blade_od = float(attr.get('od_blade', attr.get('blade_od', attr.get('od_max', 0))))
                if val_blade_od > 0:
                    final_gauge = val_blade_od
                else:
                    final_gauge = val_od_body

            # --- LOGIC LENGTH (Feet to Meter) ---
            raw_len_ft = float(attr.get('element_length', attr.get('length', 0)))
            len_meter = raw_len_ft * 0.3048

            final_components.append({
                'assembly_id': attr.get('assembly_id'),
                'Sequence': attr['parsed_seq'] + 1,
                'Description': display_name,
                'Top Connection': attr.get('connection_name', attr.get('connection_type', '-')),
                
                'OD (in)': final_od,
                'ID (in)': final_id,
                'Gauge (in)': final_gauge, # Nilai Gauge sudah benar sekarang
                
                'Length': len_meter, 
                'Weight (ppf)': float(attr.get('approximate_weight', 0)) 
            })

        df_comps = pd.DataFrame(final_components)
        
        if not df_comps.empty:
            # Sort Descending (Besar ke Kecil)
            df_comps = df_comps.sort_values(by=['assembly_id', 'Sequence'], ascending=[True, False])
            
            # Reset Index
            df_comps = df_comps.reset_index(drop=True)
            df_comps.insert(0, 'No', df_comps.index + 1)
            df_comps['No'] = df_comps['No'].astype(str)
            
            # ❌ HAPUS BARIS INI:
            # df_comps['Total (m)'] = df_comps['Length'].cumsum()  <-- HAPUS INI

        return scenarios, pd.DataFrame(cases), df_comps

    except Exception as e:
        return {}, pd.DataFrame(), pd.DataFrame()
def get_scenarios_dual_keys(xml_file):
    """
    Parser Dual Key:
    1. Ambil Offset dari 'survey_header_id' (Plan Header).
    2. Ambil Data Link dari 'def_survey_header_id' (Station Link).
    """
    try:
        xml_file.seek(0)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        scenarios = {} 
        header_offsets = {} 
        
        # 1. MAPPING HEADER OFFSET (Cari md_min di semua header)
        for child in root.findall(".//*"):
            if child.tag.endswith("SURVEY_HEADER"): 
                attr = {k.lower(): v for k, v in child.attrib.items()}
                
                # Bisa jadi survey_header_id atau def_survey_header_id
                hid = attr.get('survey_header_id', attr.get('def_survey_header_id'))
                md_min = float(attr.get('md_min', attr.get('tie_on_depth', 0)))
                
                if hid:
                    header_offsets[hid] = abs(md_min) if md_min < 0 else 0

        # 2. SCENARIO DUAL LINKING
        for child in root.findall(".//*"):
            if child.tag.endswith("CD_SCENARIO"):
                attr = {k.lower(): v for k, v in child.attrib.items()}
                sid = attr.get('scenario_id')
                sname = attr.get('name', attr.get('scenario_name', sid))
                
                # KUNCI 1: UNTUK OFFSET (Plan ID)
                offset_id = attr.get('survey_header_id')
                
                # KUNCI 2: UNTUK DATA POINTS (Definitive ID)
                # Jika def_id kosong, pakai survey_header_id sebagai fallback
                station_id = attr.get('def_survey_header_id')
                if not station_id: station_id = offset_id
                
                if sid:
                    # Ambil Offset menggunakan Kunci 1
                    val_offset = header_offsets.get(offset_id, 0)
                    
                    scenarios[sid] = {
                        'name': sname,
                        'id': sid,
                        'station_link_id': station_id, # ID "E5x0p" (Untuk cari titik)
                        'offset_val': val_offset       # Nilai 143.xx (Dari ID "upQyP")
                    }

        return scenarios

    except Exception as e:
        return {}

def generate_bha_pdf(template_path, header_data, df_bha):
    """
    Fixed Version: Header Tabel Muncul & Sequence Mulai dari 1
    """
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=A4)
    
    # ==========================================
    # 1. HEADER HALAMAN (Customer, Well, dll)
    # ==========================================
    c.setFont("Helvetica-Bold", 9)
    
    # Created On (Pojok Kanan Atas)
    current_date = datetime.now().strftime("%d-%b-%Y")
    c.setFillColorRGB(1, 1, 1) # Putih
    c.drawString(523, 770, current_date)
    c.setFillColorRGB(0, 0, 0) # Hitam
    
    # Well Info
    x_val = 355 
    c.drawString(x_val, 752, str(header_data.get('customer', '-')))
    c.drawString(x_val, 737, str(header_data.get('well_name', '-')))
    c.drawString(x_val, 722, str(header_data.get('job_number', '-')))
    c.drawString(x_val, 707, str(header_data.get('rig_name', '-')))
    c.drawString(x_val, 692, str(header_data.get('field_name', '-')))
    c.drawString(x_val, 677, str(header_data.get('country', 'Indonesia')))

    # ==========================================
    # 2. HEADER TABEL (BAGIAN YANG HILANG KEMARIN)
    # ==========================================
    # Kita gambar ulang judul kolom agar pasti muncul
    y_header = 615 
    
    # Definisi Garis Vertikal (Sama untuk Header & Data)
    v_lines = [25, 50, 300, 350, 400, 450, 550] 
    
    # A. Gambar Garis Header
    c.setStrokeColorRGB(0.5, 0.5, 0.5)
    c.setLineWidth(0.5)
    
    # Garis Horizontal ATAS Header
    c.line(25, y_header + 10, 550, y_header + 10)
    
    # Garis Horizontal BAWAH Header
    c.line(25, y_header - 4, 550, y_header - 4)
    
    # Garis Vertikal Header
    for vx in v_lines:
        c.line(vx, y_header + 10, vx, y_header - 4)
    
    c.setFont("Helvetica-Bold", 8)
    c.setFillColorRGB(0, 0, 0) # Hitam
    
    # Judul Kolom Manual (Sesuaikan posisi X dengan kolom grid)
    # Gunakan drawCentredString agar rapi di tengah kolom
    c.drawCentredString(37.5, y_header, "No")
    c.drawString(55, y_header, "Description") # Rata kiri
    c.drawCentredString(325, y_header, "OD (in)")
    c.drawCentredString(375, y_header, "ID (in)")
    c.drawCentredString(425, y_header, "Weight (ppf)")
    c.drawCentredString(500, y_header, "Cum. Len")
    
    # Garis Pembatas Header (Bawah Judul)
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(1)
    c.line(25, y_header - 5, 550, y_header - 5)

    # ==========================================
    # 3. ISI DATA TABEL
    # ==========================================
    y_position = 600 
    row_height = 15
    
    c.setFont("Helvetica", 8)
    cum_len = 0
    v_lines = [25, 50, 300, 350, 400, 450, 550] 
    
    # Reset index agar iterasi mulai dari 0 lagi untuk penomoran
    for index, row in df_bha.reset_index(drop=True).iterrows():
        if y_position < 50:
            c.drawString(50, y_position, "... (Data berlanjut) ...")
            break
        
        # --- PERBAIKAN SEQUENCE ---
        # Paksa sequence menggunakan index loop + 1.
        # Jadi baris pertama pasti "1", baris kedua "2", dst.
        seq = str(index + 1)
        
        desc = str(row.get('Description', '-'))[:45]
        od = f"{row.get('OD (in)', 0):.3f}"
        id_pipe = f"{row.get('ID (in)', 0):.3f}"
        length = row.get('Length', 0)
        cum_len += length
        
        # Gambar Garis Grid
        c.setStrokeColorRGB(0.5, 0.5, 0.5)
        c.setLineWidth(0.5)
        c.line(25, y_position - 4, 550, y_position - 4) # Garis Horizontal
        
        y_top = y_position + 10
        y_bot = y_position - 4
        for vx in v_lines:
            c.line(vx, y_top, vx, y_bot) # Garis Vertikal
            
        # Isi Teks
        c.setFillColorRGB(0, 0, 0) 
        c.drawCentredString(37.5, y_position, seq)      # No Urut Baru (1, 2, 3...)
        c.drawString(55, y_position, desc)
        c.drawCentredString(325, y_position, od)
        c.drawCentredString(375, y_position, id_pipe)
        c.drawCentredString(425, y_position, f"{length:.2f}")
        c.drawCentredString(500, y_position, f"{cum_len:.2f}")
        
        y_position -= row_height

    c.save()
    packet.seek(0)

    # ==========================================
    # 4. MERGE DENGAN TEMPLATE
    # ==========================================
    new_pdf = PdfReader(packet)
    existing_pdf = PdfReader(template_path)
    output = PdfWriter()

    page = existing_pdf.pages[0]
    page.merge_page(new_pdf.pages[0])
    output.add_page(page)

    out_stream = io.BytesIO()
    output.write(out_stream)
    out_stream.seek(0)
    
    return out_stream

# ==========================================
# 3. STATE & UI
# ==========================================
if 'layers' not in st.session_state: st.session_state['layers'] = {} 
if 'meta' not in st.session_state: st.session_state['meta'] = {}

default_surf_n = 9000000.0; default_surf_e = 400000.0
default_tgt_n = 9000400.0; default_tgt_e = 400400.0; default_tgt_tvdss = 2200.0
default_kop = 500.0; default_dls = 3.0; default_hold_ovr = 0.0; do_override = False

if 'autofill_data' in st.session_state:
    data = st.session_state['autofill_data']
    default_surf_n = data['surf_n']; default_surf_e = data['surf_e']
    default_tgt_n = data['tgt_n']; default_tgt_e = data['tgt_e']
    default_tgt_tvdss = data['tgt_tvdss']
    default_kop = data['kop']; default_hold_ovr = data['hold']
    do_override = True
    st.toast("Parameters Auto-Filled from Data!")
    del st.session_state['autofill_data']

st.sidebar.title("🎛️ DSS Command Center")

# --- SIDEBAR: QUICK IMPORT (INTEGRATED WITH DrillingEngine) ---
with st.sidebar.expander("⚡ Quick Import (Master)", expanded=True):
    st.caption("Mode: Auto-Fix using DrillingEngine Class")
    
    qi_file = st.file_uploader("Drop XML File", type=['xml'], key="qi_master")
    selected_scenario_data = None
    
    if qi_file:
        st.session_state['shared_xml_file'] = qi_file
        
        # 1. PARSE SCENARIO (Tetap pakai fungsi parser dual key kita)
        # (Pastikan get_scenarios_dual_keys sudah ada di Utils)
        scenarios_found = get_scenarios_dual_keys(qi_file)
        
        if scenarios_found:
            opts = {v['name']: k for k, v in scenarios_found.items()}
            sorted_opts = sorted(list(opts.keys()))
            
            sel_label = st.selectbox("Pilih Scenario / Plan:", sorted_opts)
            sel_sid = opts[sel_label]
            selected_scenario_data = scenarios_found[sel_sid]
            
            # Simpan State
            st.session_state['selected_scenario_id'] = sel_sid
            st.session_state['selected_scenario_name'] = selected_scenario_data['name']
            
            # Info Offset
            off = selected_scenario_data['offset_val']
            st.info(f"📏 Offset: {off:.2f} | Link ID: {selected_scenario_data['station_link_id']}")
        else:
            st.error("Scenario tidak ditemukan.")

    st.markdown("---")
    file_unit = st.radio("Satuan XML:", ["Meter", "Feet"], index=1, horizontal=True)

    if st.button("🚀 Load Visuals"):
        if not selected_scenario_data:
            st.error("Pilih Scenario dulu.")
        else:
            try:
                # Setup
                target_station_id = selected_scenario_data['station_link_id']
                offset_to_add = selected_scenario_data['offset_val']
                
                # 2. PARSE STATIONS (Ambil MD, Inc, Azi saja)
                qi_file.seek(0)
                tree = ET.parse(qi_file)
                root = tree.getroot()
                
                data_points = []
                station_tags = ['CD_DEFINITIVE_SURVEY_STATION', 'CD_TRAJECTORY_STATION', 'CD_SURVEY_STATION']
                
                for child in root.findall(".//*"):
                    tag = child.tag.split('}')[-1].upper()
                    if tag in station_tags:
                        attr = {k.lower(): v for k, v in child.attrib.items()}
                        
                        pid1 = attr.get('def_survey_header_id')
                        pid2 = attr.get('survey_header_id')
                        pid3 = attr.get('trajectory_id')
                        
                        if target_station_id in [pid1, pid2, pid3]:
                            data_points.append({
                                'MD': float(attr.get('md', 0)),
                                'Inc': float(attr.get('inc', attr.get('inclination', 0))),
                                'Azi': float(attr.get('azi', attr.get('azimuth', 0)))
                            })
                
                if data_points:
                    # Sort data berdasarkan MD
                    df_raw = pd.DataFrame(data_points).sort_values(by='MD').reset_index(drop=True)
                    
                    # 3. USE DrillingEngine UNTUK HITUNG ULANG (MCM)
                    # Kita panggil class Anda di sini
                    # Unit 'Metric' dan DLS Ref 30 (Standar)
                    engine = DrillingEngine('Metric', 30.0)
                    
                    # Siapkan input list
                    mds = df_raw['MD'].tolist()
                    incs = df_raw['Inc'].tolist()
                    azis = df_raw['Azi'].tolist()
                    
                    # Hitung (Start N/E/TVD = 0)
                    df_clean = engine.calculate_trajectory(mds, incs, azis, 0, 0, 0)
                    
                    # 4. TERAPKAN OFFSET (RKB 143)
                    if offset_to_add > 0:
                        st.toast(f"Adding Offset: {offset_to_add:.2f}")
                        df_clean['MD'] = df_clean['MD'] + offset_to_add
                        df_clean['TVD'] = df_clean['TVD'] + offset_to_add
                        df_clean['TVDSS'] = df_clean['TVD'] # Asumsi RKB di 0 relative to offset
                    
                    # 5. KONVERSI UNIT (FEET -> METER)
                    is_feet = (file_unit == "Feet") or (df_clean['MD'].max() > 5000)
                    if is_feet:
                        # Konversi semua kolom spasial
                        for c in ['MD', 'TVD', 'TVDSS', 'N', 'E', 'VS']:
                            if c in df_clean.columns: df_clean[c] *= 0.3048

                    # 6. UPDATE APP STATE
                    s_n = df_clean['N'].iloc[0]; s_e = df_clean['E'].iloc[0]
                    t_n = df_clean['N'].iloc[-1]; t_e = df_clean['E'].iloc[-1]; t_tvd = df_clean['TVD'].iloc[-1]
                    
                    st.session_state['autofill_data'] = {'surf_n': s_n, 'surf_e': s_e, 'tgt_n': t_n, 'tgt_e': t_e, 'tgt_tvdss': t_tvd, 'kop': 0, 'hold': 0}
                    
                    if 'Plan' in st.session_state['layers']: del st.session_state['layers']['Plan']
                    st.session_state['layers']['Plan'] = {
                        'df': df_clean, 'color': '#0052CC', 'show': True, 'type': 'plan', 'name': selected_scenario_data['name']
                    }
                    
                    st.session_state['meta'] = {'rkb': 0, 'surf_n': s_n, 'surf_e': s_e, 'unit': 'Metric', 'planner': SmartPlanner(s_n, s_e, 0, 'Metric')}
                    
                    st.success(f"✅ Success using DrillingEngine! Max MD: {df_clean['MD'].max():.2f}")
                    st.rerun()
                else:
                    st.error(f"Station kosong untuk ID: {target_station_id}")

            except Exception as e: st.error(f"Error: {e}")
# --- UNIT SELECTION ---
st.sidebar.markdown("---")
unit_sys = st.sidebar.radio("Units", ["Metric", "Imperial"], horizontal=True)
u_label = "m" if unit_sys == "Metric" else "ft"
dls_label = "deg/30m" if unit_sys == "Metric" else "deg/100ft"

with st.sidebar.form("plan_form"):
    st.header("1. Well Planning Parameters")
    plan_mode = st.radio("Plan Source", ["Calculator (J-Profile)", "Import Plan (Compass Data)"])
    
    c1, c2 = st.columns(2)
    r_floor = c1.number_input(f"Rotary Table ({u_label})", value=6.1)
    r_elev = c2.number_input(f"Cellar Elev", value=19.46)
    
    surf_n = c1.number_input("Surf N", value=default_surf_n, format="%.2f", key='sn')
    surf_e = c2.number_input("Surf E", value=default_surf_e, format="%.2f", key='se')
    
    st.markdown("---")
    
    if plan_mode == "Calculator (J-Profile)":
        tgt_n = c1.number_input("Target N", value=default_tgt_n, format="%.2f", key='tn')
        tgt_e = c2.number_input("Target E", value=default_tgt_e, format="%.2f", key='te')
        tgt_tvdss = st.number_input(f"Target TVDSS ({u_label})", value=default_tgt_tvdss, key='tt')
        kop = c1.number_input(f"KOP ({u_label})", value=default_kop, key='kp')
        dls = c2.number_input(f"DLS", value=default_dls, key='dl')
        
        force_hold = st.checkbox("Override Hold Angle?", value=do_override, key='chk_override')
        manual_hold = st.number_input("Force Hold (deg)", value=default_hold_ovr if do_override else 0.0, key='num_hold') if force_hold else None
        imported_plan_txt = ""
    else:
        st.info("Paste Data from Compass")
        imported_plan_txt = st.text_area("Paste Plan Data:", height=150)
        tgt_n, tgt_e, tgt_tvdss, kop, dls, manual_hold = 0,0,0,0,0,None
    
    plan_submit = st.form_submit_button("🚀 UPDATE PLAN", type="primary")

if plan_submit:
    r_rkb = r_floor + r_elev
    planner = SmartPlanner(surf_n, surf_e, r_rkb, unit_sys)
    
    if plan_mode == "Calculator (J-Profile)":
        df_plan, azi, hold = planner.solve_trajectory(tgt_n, tgt_e, tgt_tvdss, kop, dls, manual_hold)
        st.success(f"Calculated! Azi: {azi:.2f}°, Hold: {hold:.2f}°")
    else:
        eng = planner.engine
        df_plan = parse_trajectory_data(imported_plan_txt, r_rkb, surf_n, surf_e, eng)
        if isinstance(df_plan, pd.DataFrame):
            st.success("Plan Imported!")
            df_plan['Section'] = 'Plan'
        else: st.error(df_plan); df_plan = None

    if df_plan is not None:
        st.session_state['layers']['Plan'] = {'df': df_plan, 'color': '#0000FF', 'show': True, 'type': 'plan'}
        st.session_state['meta'] = {'rkb': r_rkb, 'surf_n': surf_n, 'surf_e': surf_e, 'unit': unit_sys, 'planner': planner}

# --- CASING & FORMATION ---
with st.sidebar.expander("🛠️ Casing & Formation"):
    casing_init = pd.DataFrame([{"Size": "20\"", "Depth": 50, "Type": "MD"}, {"Size": "9-5/8\"", "Depth": 1200, "Type": "MD"}])
    edited_casing = st.data_editor(casing_init, num_rows="dynamic")
    form_text = st.text_area("Formation (Name, Depth)", "Top GUF, 446.5\nTop TAF, 558.0")

# --- ACTUAL & PRESCRIPTIVE ---
with st.sidebar.expander("📉 Actual / Correction"):
    # ML model status indicator
    if drilling_model is not None:
        st.success(f"✅ LSTM model loaded ({drilling_model.n_features} features, seq_len={drilling_model.seq_len})")
    else:
        st.error("❌ ML model NOT loaded — predictions unavailable")

    if st.button("🎲 Demo Actual"):
        st.session_state['act_txt'] = "MD Inc Azi\n0 0 0\n500 0 0\n600 2 45\n1000 12 48"
    act_txt = st.text_area("Actual Data:", value=st.session_state.get('act_txt', ''))
    st.caption("Path length to target is computed automatically by the ML model.")

    if st.button("🎯 Run Correction to Target", type="primary"):
        if 'Plan' in st.session_state['layers']:
            meta = st.session_state['meta']
            eng = meta['planner'].engine
            df_act = parse_trajectory_data(act_txt, meta['rkb'], meta['surf_n'], meta['surf_e'], eng)

            if isinstance(df_act, pd.DataFrame):
                st.session_state['layers']['Actual'] = {'df': df_act, 'color': '#FF0000', 'show': True}
                planner = meta['planner']
                df_plan = st.session_state['layers']['Plan']['df']

                with st.spinner("Computing ML-guided path to target…"):
                    df_corr, total_dls, req_bur, best_turn, tf, dist_to_final, corr_len = \
                        planner.calculate_correction_path(df_act, df_plan, model=drilling_model)

                st.session_state['layers']['Correction'] = {'df': df_corr, 'color': '#00FF00', 'show': True}

                # Advisory: geometric distance-to-plan (accurate, interpretable)
                # ML is used inside the optimizer to FIND the best path;
                # the advisory display uses direct 3D geometry.
                advisory_score = None
                max_advisory_score = None
                if not df_corr.empty and not df_plan.empty:
                    try:
                        dtp = compute_distance_to_plan(df_corr, df_plan)
                        advisory_score     = float(np.mean(dtp))
                        max_advisory_score = float(np.max(dtp))
                    except Exception as e:
                        st.warning(f"⚠️ Distance-to-plan computation error: {e}")

                st.session_state['prescription'] = {
                    'bur': req_bur, 'turn': best_turn, 'dls': total_dls,
                    'len': round(corr_len, 1), 'tf': tf,
                    'advisory': advisory_score, 'max_advisory': max_advisory_score,
                    'dist_to_final': dist_to_final
                }
                st.success(f"Path computed: {corr_len:.0f} {u_label} → {dist_to_final:.1f} {u_label} from target")
            else:
                st.error(df_act)

# --- OFFSET WELLS ---
with st.sidebar.expander("🛡️ Offset Wells"):
    off_name = st.text_input("Offset Name", "Offset-01")
    c1, c2, c3 = st.columns(3)
    off_n = c1.number_input("Off N", 0.0, format="%.2f")
    off_e = c2.number_input("Off E", 0.0, format="%.2f")
    azi_corr = c3.number_input("Azi Corr", 0.0)
    off_txt = st.text_area("Offset Data", height=100)
    
    if st.button("Add Offset"):
        meta = st.session_state.get('meta', {})
        eng = meta.get('planner').engine if 'planner' in meta else None
        if eng:
            use_n = off_n if off_n != 0 else meta.get('surf_n', 0)
            use_e = off_e if off_e != 0 else meta.get('surf_e', 0)
            df_off = parse_trajectory_data(off_txt, meta.get('rkb', 0), use_n, use_e, eng, azi_corr)
            
            if isinstance(df_off, pd.DataFrame):
                if 'Offsets' not in st.session_state['layers']: st.session_state['layers']['Offsets'] = []
                st.session_state['layers']['Offsets'].append({'id': str(uuid.uuid4()), 'name': off_name, 'df': df_off, 'color': '#808080', 'show': True})
                st.success(f"Added {off_name}")
            else: st.error(df_off)

# --- VISUAL MANAGER ---
st.sidebar.subheader("🎨 Layers")
if 'Plan' in st.session_state['layers']:
    l = st.session_state['layers']['Plan']
    l['show'] = st.sidebar.checkbox("Plan", l['show'])
    l['color'] = st.sidebar.color_picker(" ", l['color'])
if 'Actual' in st.session_state['layers']:
    l = st.session_state['layers']['Actual']
    l['show'] = st.sidebar.checkbox("Actual", l['show'])
    l['color'] = st.sidebar.color_picker(" ", l['color'])
if 'Correction' in st.session_state['layers']:
    l = st.session_state['layers']['Correction']
    l['show'] = st.sidebar.checkbox("Correction", l['show'])
    l['color'] = st.sidebar.color_picker(" ", l['color'])
if 'Offsets' in st.session_state['layers']:
    st.sidebar.markdown("**Offset Wells:**")
    del_list = []
    for i, off in enumerate(st.session_state['layers']['Offsets']):
        c1, c2, c3 = st.sidebar.columns([0.2, 0.6, 0.2])
        off['show'] = c1.checkbox("👁️", off['show'], key=f"v_{off['id']}")
        off['color'] = c2.color_picker(off['name'], off['color'], key=f"c_{off['id']}")
        if c3.button("🗑️", key=f"d_{off['id']}"): del_list.append(i)
    for i in sorted(del_list, reverse=True): st.session_state['layers']['Offsets'].pop(i); st.rerun()

# ==========================================
# 4. DASHBOARD RENDER (COMMERCIAL UPGRADE)
# ==========================================
st.title("🏗️ DSS Well Master Ultimate")

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 2px solid #007bff; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; color: #0f52ba; }
</style>
""", unsafe_allow_html=True)

if 'Plan' in st.session_state['layers']:
    df_plan = st.session_state['layers']['Plan']['df']
    cost, time = calculate_economics(df_plan)
    
    # --- TOP KPI METRICS ---
    with st.container():
        # 1. SIAPKAN DATA
        # Ambil data KPI dari XML Header (jika ada)
        kpi = st.session_state.get('current_kpi', {})
        xml_max_dls = kpi.get('max_dls', 0)
        xml_tort = kpi.get('tortuosity', 0)
        
        # Ambil data Plan (Trajectory)
        final_md = df_plan['MD'].iloc[-1] if not df_plan.empty else 0
        max_inc = df_plan['Inc'].max() if not df_plan.empty else 0
        
        # Hitung Dummy Cost (Jika variabel cost belum didefinisikan, kita buat dummy)
        # Asumsi cost $1000 per meter
        cost = final_md * 1000 
        
        # 2. HITUNG MIN SEPARATION (Logika Lama Anda)
        min_sep = 9999.0
        if 'Offsets' in st.session_state['layers']:
            active = [o for o in st.session_state['layers']['Offsets'] if o['show']]            
            if active:
                try:
                    p1 = df_plan[['N', 'E', 'TVD']].values
                    for o in active:
                        p2 = o['df'][['N', 'E', 'TVD']].values
                        # Downsample for speed
                        if len(p2) > 1000: p2 = p2[::10] 
                        if len(p1) > 1000: p1_s = p1[::10]
                        else: p1_s = p1
                        
                        # Euclidean distance check (Approx)
                        # Hitung jarak terdekat antar titik
                        from scipy.spatial.distance import cdist
                        dists = cdist(p1_s, p2)
                        current_min = np.min(dists)
                        min_sep = min(min_sep, current_min)
                except Exception as e:
                    pass # Ignore error calculation for UI stability

        # 3. TAMPILKAN 5 KOLOM
        c1, c2, c3, c4, c5 = st.columns(5)
        
        # C1: Total Depth
        c1.metric(
            "Total Depth (MD)", 
            f"{final_md:,.0f} {u_label}", 
            delta="Target Reached" if final_md > 0 else None
        )
        
        # C2: Max Inclination
        c2.metric("Max Inclination", f"{max_inc:.2f}°")
        
        # C3: Est. Cost (Kode Lama)
        c3.metric("Est. Cost", f"${cost/1000:,.0f} K")
        
        # C4: Max DLS / Tortuosity (DATA XML BARU)
        # Kita gabung infonya biar hemat tempat
        c4.metric(
            "Max DLS (Header)", 
            f"{xml_max_dls:.2f}",
            delta=f"Tort: {xml_tort:.3f}",
            delta_color="off" # Warna abu-abu netral
        )
        
        # C5: Min Separation (Kode Lama)
        sep_val = f"{min_sep:.1f} m" if min_sep != 9999.0 else "N/A"
        sep_state = "CRITICAL" if min_sep < 10 else "Safe"
        sep_color = "inverse" if min_sep < 10 else "normal"
        
        c5.metric(
            "Min Separation", 
            sep_val, 
            delta=sep_state, 
            delta_color=sep_color
        )
        
    st.markdown("---")

    # --- PRESCRIPTION ALERT BANNER ---
    if 'prescription' in st.session_state and st.session_state['layers'].get('Correction', {}).get('show'):
        p = st.session_state['prescription']
        
        # 1. Ambil nilai rata-rata (Advisory) dan nilai tertinggi (Max Advisory)
        advisory_val = p.get('advisory', None)
        max_advisory_val = p.get('max_advisory', advisory_val) # Jika max tidak ada, pakai rata-rata
        
        # 2. Logika Keputusan (Traffic Light DSS) berdasarkan NILAI TERTINGGI (Max DTP)
        status_teks = ""
        warna_bg = "#e6fffa" # Default Hijau Terang (Aman)
        warna_border = "#00b894" # Default Hijau Gelap
        
        if advisory_val is not None and advisory_val != "N/A":
            if isinstance(advisory_val, str):
                try: advisory_val = float(advisory_val)
                except: advisory_val = 0.0
            if isinstance(max_advisory_val, str):
                try: max_advisory_val = float(max_advisory_val)
                except: max_advisory_val = 0.0
                
            # Distance-to-plan thresholds (metres, geometric)
            # avg = how far off the correction path deviates on average
            # max = worst single point deviation along the path
            # At path END (dist_to_final) it should be near 0 = success
            if max_advisory_val < 5.0:
                status    = "🟢 On Track"
                advice    = "Correction path stays tightly on the planned trajectory."
                warna_bg  = "#f0fdf4"; warna_border = "#16a34a"
            elif max_advisory_val < 15.0:
                status    = "🟡 Minor Deviation"
                advice    = "Slight divergence during correction — within acceptable steering tolerance."
                warna_bg  = "#fefce8"; warna_border = "#ca8a04"
            elif max_advisory_val < 40.0:
                status    = "🟠 Moderate Deviation"
                advice    = "Correction path deviates noticeably mid-curve. Verify BUR and toolface orientation."
                warna_bg  = "#fff7ed"; warna_border = "#ea580c"
            elif max_advisory_val < 80.0:
                status    = "🔴 Significant Deviation"
                advice    = "Large mid-path deviation detected. Consider re-running correction with adjusted DLS."
                warna_bg  = "#fef2f2"; warna_border = "#dc2626"
            else:
                status    = "⚠️ High Deviation"
                advice    = "Correction path swings far from plan before reaching target. Review actual survey data and re-plan."
                warna_bg  = "#fdf4ff"; warna_border = "#9333ea"

            teks_max = (f" &nbsp;|&nbsp; Peak: <b style='color:{warna_border};'>{max_advisory_val:.1f} {u_label}</b>"
                        if max_advisory_val != advisory_val else "")
            status_teks = (
                f"📏 Avg Distance-to-Plan along correction: <b>{advisory_val:.1f} {u_label}</b>{teks_max}<br>"
                f"<span style='font-size:0.95rem; color:#555;'>{advice}</span><br>"
                f"🚦 <b>{status}</b>"
            )
        else:
            status_teks = "📏 Distance-to-Plan: computing…"

        # Distance-to-final-target (end of correction → plan TD)
        dist_final = p.get('dist_to_final', None)
        if dist_final is not None:
            if dist_final < 10:
                dtf_color = "#16a34a"; dtf_icon = "🎯"
                dtf_label = "Target reached"
            elif dist_final < 50:
                dtf_color = "#ca8a04"; dtf_icon = "📍"
                dtf_label = "Near target"
            else:
                dtf_color = "#dc2626"; dtf_icon = "📍"
                dtf_label = "Still offset"
            dist_teks = (f"{dtf_icon} End-of-correction distance to target: "
                         f"<b style='color:{dtf_color};'>{dist_final:.1f} {u_label}</b> — {dtf_label}")
        else:
            dist_teks = ""

        # Banner
        st.markdown(f"""
        <div style="padding:16px 20px; background-color:{warna_bg};
                    border-left:5px solid {warna_border}; border-radius:6px;
                    margin-bottom:20px; box-shadow:0 1px 4px rgba(0,0,0,0.08);">
            <h4 style="margin:0 0 8px 0; color:{warna_border};">💡 AI Prescriptive Correction</h4>
            <p style="margin:0; font-size:1.05rem; line-height:1.6;">
                <b>{'BUILD' if p.get('bur',0)>0 else 'DROP'}</b> {abs(p.get('bur',0)):.2f} {dls_label} &nbsp;|&nbsp;
                <b>TURN {'RIGHT' if p.get('turn',0)>0 else 'LEFT'}</b> {abs(p.get('turn',0)):.2f} {dls_label} &nbsp;|&nbsp;
                Toolface <b>{p.get('tf',0):.0f}°</b> &nbsp;|&nbsp;
                Correction length <b>{p.get('len',0):.0f} {u_label}</b>
            </p>
            <div style="margin-top:10px; padding-top:10px;
                        border-top:1px solid {warna_border}33;
                        font-size:1.0rem; color:#333; line-height:1.7;">
                {status_teks}
            </div>
            {f'<div style="margin-top:8px; font-size:0.95rem; color:#444;">{dist_teks}</div>' if dist_teks else ''}
        </div>
        """, unsafe_allow_html=True)

    tab1, tab2, tab3,tab4,tab5= st.tabs(["🌍 3D Trajectory Analysis", "📐 2D Engineering Plots", "📋 Raw Survey Data","📈 Drilling Mechanics Logs","🗂️ BHA & Case Manager"])
    
    # --- COLOR PALETTE ---
    palette = {
        'Plan': '#0052CC',      # Professional Engineering Blue
        'Actual': '#D32F2F',    # Alert Red
        'Correction': '#00C853',# Success Green
        'Offset': '#9E9E9E'     # Neutral Grey
    }

    with tab1:
        # Commercial 3D Plot
        fig3d = go.Figure()
        layers = st.session_state['layers']
        
        # --- 1. HELPER TO DRAW WELL PATHS (Uses Scatter3d) ---
       # --- FUNGSI HELPER PLOT 3D (UPDATED) ---
        def add_3d_trace(df, name, color, width=5, dash='solid', opacity=1.0):
            # 1. CEK KOLOM DEPTH (Robust Check)
            # Prioritas: TVDSS -> TVD -> Skip jika tidak ada
            if 'TVDSS' in df.columns:
                z_data = df['TVDSS']
                z_label = "TVDSS"
            elif 'TVD' in df.columns:
                z_data = df['TVD']
                z_label = "TVD"
            else:
                st.warning(f"Skipping trace '{name}': No Depth column (TVD/TVDSS) found.")
                return

            # 2. CONFIG HOVER TEMPLATE
            # Kita gunakan z_label agar tooltip sesuai datanya (TVD atau TVDSS)
            hover_temp = (
                "<b>" + name + "</b><br>" +
                "MD: %{text:.1f}<br>" +
                "Inc: %{customdata[0]:.2f}°<br>" +
                "Azi: %{customdata[1]:.2f}°<br>" +
                f"{z_label}: %{{z:.1f}}"
            )
            
            # 3. CRITICAL: Use Scatter3d for Lines
            fig3d.add_trace(go.Scatter3d(
                x=df['E'], y=df['N'], z=z_data, # <--- Gunakan variabel z_data yg aman
                mode='lines', 
                name=name,
                line=dict(color=color, width=width, dash=dash),
                opacity=opacity,
                text=df['MD'],
                customdata=np.stack((df['Inc'], df['Azi']), axis=-1),
                hovertemplate=hover_temp
            ))
            
            # 4. ADD CONE AT THE BIT (Last Point)
            if name in ['Plan', 'Actual', 'Correction']:
                # Ambil koordinat terakhir
                last_idx = df.index[-1]
                last_e = df.loc[last_idx, 'E']
                last_n = df.loc[last_idx, 'N']
                last_z = z_data.iloc[-1] # <--- Ambil Z terakhir dari data yang aman

                fig3d.add_trace(go.Scatter3d(
                    x=[last_e], y=[last_n], z=[last_z],
                    mode='markers', name=f"{name} TD",
                    marker=dict(size=5, color=color, symbol='diamond'),
                    showlegend=False, hoverinfo='skip'
                ))

        # --- 2. SURFACE PLANE (Uses Mesh3d) ---
        if 'Plan' in layers:
            center_n = layers['Plan']['df']['N'].mean()
            center_e = layers['Plan']['df']['E'].mean()
        else:
            center_n, center_e = 0, 0

        # CRITICAL: Use Mesh3d for the Surface Plane
        fig3d.add_trace(go.Mesh3d(
            x=[center_e-500, center_e+500, center_e+500, center_e-500],
            y=[center_n-500, center_n-500, center_n+500, center_n+500],
            z=[0, 0, 0, 0],
            color='lightblue', 
            opacity=0.1, 
            name='Sea Level', 
            showlegend=True
        ))

        # --- 3. DRAW LINES (section-coloured for Plan; standard for others) ---

        def add_casing_shoe_ring_3d(df_traj, shoe_md, ring_r, ring_color, ring_label):
            """Draw a horizontal ring at a casing shoe depth on the 3D trajectory."""
            z_col = 'TVDSS' if 'TVDSS' in df_traj.columns else 'TVD'
            idx = (df_traj['MD'] - shoe_md).abs().idxmin()
            row = df_traj.iloc[idx]
            theta = np.linspace(0, 2 * np.pi, 24)
            fig3d.add_trace(go.Scatter3d(
                x=row['E'] + ring_r * np.cos(theta),
                y=row['N'] + ring_r * np.sin(theta),
                z=[row[z_col]] * 24,
                mode='lines',
                line=dict(color=ring_color, width=3),
                name=ring_label,
                showlegend=True,
                hovertemplate=f"{ring_label}<br>MD: {shoe_md:.0f}<extra></extra>"
            ))

        if layers['Plan']['show']:
            df_p = layers['Plan']['df']
            # Build sorted casing list from casing table
            try:
                csg_sorted = edited_casing.copy()
                csg_sorted['_depth'] = pd.to_numeric(csg_sorted['Depth'], errors='coerce').fillna(0)
                csg_sorted = csg_sorted.sort_values('_depth').reset_index(drop=True)
            except Exception:
                csg_sorted = pd.DataFrame()

            # ── Frenet-Serret tube mesh for 3-D casing visualization ──────────────
            def _tube_3d(df_seg, radius, color, label, n_sides=8):
                pts_e = df_seg['E'].values.astype(float)
                pts_n = df_seg['N'].values.astype(float)
                _zcol = 'TVDSS' if 'TVDSS' in df_seg.columns else 'TVD'
                pts_z = df_seg[_zcol].values.astype(float)
                pts = np.column_stack([pts_e, pts_n, pts_z])
                n_pts = len(pts)
                if n_pts < 2:
                    return None
                if n_pts > 80:
                    idx_ds = np.round(np.linspace(0, n_pts - 1, 80)).astype(int)
                    pts = pts[idx_ds]; n_pts = len(pts)
                # Tangents
                tg = np.zeros_like(pts)
                tg[:-1] = pts[1:] - pts[:-1]; tg[-1] = tg[-2]
                nrm = np.linalg.norm(tg, axis=1, keepdims=True)
                tg /= np.where(nrm < 1e-10, 1.0, nrm)
                # Parallel-transport normals (Frenet-Serret)
                ref = np.array([1., 0., 0.]) if abs(tg[0, 0]) < 0.9 else np.array([0., 1., 0.])
                nv = np.zeros_like(pts)
                nv[0] = np.cross(tg[0], ref); nv[0] /= np.linalg.norm(nv[0])
                for ii in range(1, n_pts):
                    proj = nv[ii-1] - np.dot(nv[ii-1], tg[ii]) * tg[ii]
                    pn = np.linalg.norm(proj)
                    nv[ii] = proj / pn if pn > 1e-10 else nv[ii-1]
                bv = np.cross(tg, nv)
                theta = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
                ct, st = np.cos(theta), np.sin(theta)
                vx, vy, vz = [], [], []
                for ii in range(n_pts):
                    for jj in range(n_sides):
                        p = pts[ii] + radius * (ct[jj] * nv[ii] + st[jj] * bv[ii])
                        vx.append(p[0]); vy.append(p[1]); vz.append(p[2])
                ti_l, tj_l, tk_l = [], [], []
                for ii in range(n_pts - 1):
                    for jj in range(n_sides):
                        jn = (jj + 1) % n_sides
                        v00 = ii*n_sides + jj;  v01 = ii*n_sides + jn
                        v10 = (ii+1)*n_sides + jj; v11 = (ii+1)*n_sides + jn
                        ti_l += [v00, v00]; tj_l += [v01, v10]; tk_l += [v10, v11]
                return go.Mesh3d(
                    x=vx, y=vy, z=vz, i=ti_l, j=tj_l, k=tk_l,
                    color=color, opacity=0.85,
                    name=label, showlegend=True, hoverinfo='name',
                    lighting=dict(ambient=0.7, diffuse=0.9),
                )

            if not csg_sorted.empty:
                shoes = list(csg_sorted['_depth'].values)
                td_md = float(df_p['MD'].max())
                boundaries = [0.0] + shoes + [td_md]

                for i in range(len(boundaries) - 1):
                    md_lo, md_hi = boundaries[i], boundaries[i + 1]
                    seg_df = df_p[(df_p['MD'] >= md_lo) & (df_p['MD'] <= md_hi)]
                    if seg_df.empty:
                        continue
                    style = SECTION_PALETTE[min(i, len(SECTION_PALETTE) - 1)]
                    sec_label = style['label']
                    if i < len(csg_sorted):
                        sec_label = f"{csg_sorted.iloc[i]['Size']} – {style['label']}"
                    if i < len(csg_sorted):
                        od_i = parse_casing_od(csg_sorted.iloc[i]['Size'])
                    else:
                        od_i = (parse_casing_od(csg_sorted.iloc[-1]['Size']) * 0.85
                                if not csg_sorted.empty else 8.0)
                    tube_r = od_i * 0.0254 * 40   # ×40 visual exaggeration
                    mesh = _tube_3d(seg_df, tube_r, style['color'], sec_label)
                    if mesh is not None:
                        fig3d.add_trace(mesh)

                # Shoe rings at section transitions
                for idx_r, row_r in csg_sorted.iterrows():
                    od_in = parse_casing_od(row_r['Size'])
                    ring_r = od_in * 0.0254 * 40
                    style_r = SECTION_PALETTE[min(idx_r, len(SECTION_PALETTE) - 1)]
                    add_casing_shoe_ring_3d(df_p, float(row_r['_depth']),
                                            ring_r, style_r['color'],
                                            f"Shoe {row_r['Size']} @{row_r['_depth']:.0f}{u_label}")
            else:
                add_3d_trace(df_p, 'Plan', layers['Plan']['color'], width=6)

        if 'Actual' in layers and layers['Actual']['show']:
            add_3d_trace(layers['Actual']['df'], 'Actual', layers['Actual']['color'], width=7)

        if 'Correction' in layers and layers['Correction']['show']:
            add_3d_trace(layers['Correction']['df'], 'Correction',
                         layers['Correction']['color'], width=6, dash='dash')
            # Mark the end of the correction path clearly
            df_corr3d = layers['Correction']['df']
            z_col3d = 'TVDSS' if 'TVDSS' in df_corr3d.columns else 'TVD'
            ep = df_corr3d.iloc[-1]
            fig3d.add_trace(go.Scatter3d(
                x=[ep['E']], y=[ep['N']], z=[ep[z_col3d]],
                mode='markers+text',
                marker=dict(size=8, color='#00C853', symbol='diamond'),
                text=['End Correction'], textposition='top center',
                name='Correction End', showlegend=False
            ))

        if 'Offsets' in layers:
            for o in layers['Offsets']:
                if o['show']:
                    add_3d_trace(o['df'], o['name'], o['color'], width=3, dash='solid', opacity=0.6)

        # --- 4. LAYOUT SETTINGS ---
        fig3d.update_layout(
            scene=dict(
                xaxis=dict(title='EAST (+/-)', backgroundcolor="rgb(240, 240, 240)", gridcolor="white", showbackground=True, zerolinecolor="white"),
                yaxis=dict(title='NORTH (+/-)', backgroundcolor="rgb(240, 240, 240)", gridcolor="white", showbackground=True, zerolinecolor="white"),
                zaxis=dict(title='TVD / TVDSS', autorange="reversed", backgroundcolor="rgb(230, 230, 240)", gridcolor="white", showbackground=True, zerolinecolor="white"),
                aspectmode='data' 
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=700,
            legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.05, bgcolor="rgba(255,255,255,0.8)")
        )
        st.plotly_chart(fig3d, use_container_width=True)

    with tab2:
        # --- IMPROVED 2D SECTION VIEW LOGIC ---
        
        # 1. Tentukan Arah Vertical Section (VS Azimuth)
        # Secara default, arahkan ke Target. Jika tidak ada target, pakai arah akhir sumur.
        if 'Plan' in layers:
            last_pt = layers['Plan']['df'].iloc[-1]
            # Hitung arah dari surface ke koordinat terakhir (Target)
            delta_n_tgt = last_pt['N'] - layers['Plan']['df']['N'].iloc[0]
            delta_e_tgt = last_pt['E'] - layers['Plan']['df']['E'].iloc[0]
            default_vs_azi = np.degrees(np.arctan2(delta_e_tgt, delta_n_tgt)) % 360
        else:
            default_vs_azi = 0.0

        # UI Control untuk VS Azimuth (Seperti di Compass)
        c_ctrl, c_view = st.columns([1, 4])
        with c_ctrl:
            st.markdown("##### 📐 View Settings")
            vs_azimuth = st.number_input("VS Azimuth (deg)", value=default_vs_azi, min_value=0.0, max_value=360.0, step=1.0, help="Arah irisan vertikal (Projected Plane). Ubah ini untuk melihat section dari sudut pandang berbeda.")
            
        # Fungsi menghitung Projected VS (Rumus Compass)
        def get_projected_vs(df, origin_n, origin_e, azimuth_deg):
            az_rad = np.radians(azimuth_deg)
            # Rumus Proyeksi: DeltaN * cos(az) + DeltaE * sin(az)
            return (df['N'] - origin_n) * np.cos(az_rad) + (df['E'] - origin_e) * np.sin(az_rad)

        col_plan, col_sec = st.columns(2)
        f_plan = go.Figure()
        f_sec = go.Figure()
        
        # --- PLOTTING LOOP ---
        for k, v in layers.items():
            # Tentukan Origin (Surface Location)
            # Untuk Offset wells, kita asumsikan origin relatif terhadap plan utama 
            # atau hitung VS relatif terhadap surface well tersebut jika diinginkan.
            # Di sini kita pakai surface Plan utama sebagai referensi (0,0) VS.
            
            origin_n = layers['Plan']['df']['N'].iloc[0] if 'Plan' in layers else 0
            origin_e = layers['Plan']['df']['E'].iloc[0] if 'Plan' in layers else 0
            
            dfs_to_plot = []
            if k == 'Offsets':
                for o in v:
                    if o['show']: dfs_to_plot.append((o['df'], o['name'], o['color'], 'solid'))
            elif v['show']:
                d = 'dot' if k == 'Correction' else 'solid'
                dfs_to_plot.append((v['df'], k, v['color'], d))
            
            for df_plot, name, color, dash in dfs_to_plot:
                # Hitung VS Baru (Projected)
                vs_proj = get_projected_vs(df_plot, origin_n, origin_e, vs_azimuth)
                
                # Plot Plan View (N vs E)
                f_plan.add_trace(go.Scatter(
                    x=df_plot['E'], y=df_plot['N'], mode='lines', name=name, 
                    line=dict(color=color, width=2, dash=dash),
                    hovertemplate="N: %{y:.1f}<br>E: %{x:.1f}"
                ))
                
                # Plot Section View (TVDSS vs Projected VS)
                f_sec.add_trace(go.Scatter(
                    x=vs_proj, y=df_plot['TVDSS'], mode='lines', name=name, 
                    line=dict(color=color, width=2, dash=dash),
                    hovertemplate="VS: %{x:.1f}<br>TVD: %{y:.1f}"
                ))

        # --- SECTOR LINE DI PLAN VIEW ---
        # Menambahkan garis putus-putus di Plan View yang menunjukkan arah VS Azimuth
        if 'Plan' in layers:
            max_disp = layers['Plan']['df']['MD'].max() * 0.5
            rad_az = np.radians(vs_azimuth)
            f_plan.add_shape(type="line",
                x0=origin_e, y0=origin_n,
                x1=origin_e + np.sin(rad_az)*max_disp,
                y1=origin_n + np.cos(rad_az)*max_disp,
                line=dict(color="grey", width=1, dash="dot"),
                name="VS Direction"
            )

        # Casing ribbons + shoe markers in Section View
        if not edited_casing.empty and 'Plan' in layers:
            df_p = layers['Plan']['df']
            vs_p_proj = get_projected_vs(df_p, origin_n, origin_e, vs_azimuth)

            try:
                csg_2d = edited_casing.copy()
                csg_2d['_depth'] = pd.to_numeric(csg_2d['Depth'], errors='coerce').fillna(0)
                csg_2d = csg_2d.sort_values('_depth').reset_index(drop=True)

                td_md_2d = float(df_p['MD'].max())
                boundaries_2d = [0.0] + list(csg_2d['_depth'].values) + [td_md_2d]

                # Visual scale: exaggerate OD so casing is visible against TVD axis
                # Auto-scale: use 3 % of total VS range per inch of OD
                vs_range = float(vs_p_proj.max() - vs_p_proj.min())
                vs_scale = max(vs_range * 0.015, 5.0)   # metres per inch of OD

                # ── Connected tapered wellbore outline ─────────────────────────────
                # OD per section: [casing_0, ..., casing_n, open_hole]
                sec_ods_2d = [parse_casing_od(csg_2d.iloc[i2]['Size'])
                              for i2 in range(len(csg_2d))]
                oh_od_2d = sec_ods_2d[-1] * 0.85 if sec_ods_2d else 10.0
                sec_ods_2d.append(oh_od_2d)
                half_ws_2d = [od * vs_scale / 2.0 for od in sec_ods_2d]

                left_vs_pts, left_tvd_pts = [], []
                right_vs_pts, right_tvd_pts = [], []
                prev_hw2d = None
                center_traces_2d = []

                for i2 in range(len(boundaries_2d) - 1):
                    lo2, hi2 = boundaries_2d[i2], boundaries_2d[i2 + 1]
                    mask2 = (df_p['MD'] >= lo2) & (df_p['MD'] <= hi2)
                    seg2 = df_p[mask2]
                    if len(seg2) < 1:
                        continue
                    hw2 = half_ws_2d[min(i2, len(half_ws_2d) - 1)]
                    vs_seg = vs_p_proj[mask2].values
                    tvd_seg = seg2['TVDSS'].values
                    style2 = SECTION_PALETTE[min(i2, len(SECTION_PALETTE) - 1)]
                    sec_name2 = (f"{csg_2d.iloc[i2]['Size']} – {style2['label']}"
                                 if i2 < len(csg_2d) else style2['label'])

                    if prev_hw2d is not None:
                        # Horizontal step-in at this shoe: from prev width to new width
                        sv, st_tvd = vs_seg[0], tvd_seg[0]
                        left_vs_pts.append(sv - hw2);  left_tvd_pts.append(st_tvd)
                        right_vs_pts.append(sv + hw2); right_tvd_pts.append(st_tvd)
                        vs_iter = vs_seg[1:]; tvd_iter = tvd_seg[1:]
                    else:
                        vs_iter = vs_seg; tvd_iter = tvd_seg

                    for v, t in zip(vs_iter, tvd_iter):
                        left_vs_pts.append(v - hw2);  left_tvd_pts.append(t)
                        right_vs_pts.append(v + hw2); right_tvd_pts.append(t)

                    center_traces_2d.append((vs_seg, tvd_seg, style2['color'], sec_name2))
                    prev_hw2d = hw2

                if left_vs_pts:
                    # Single closed polygon: left wall down + right wall up
                    poly_vs  = left_vs_pts  + right_vs_pts[::-1]  + [left_vs_pts[0]]
                    poly_tvd = left_tvd_pts + right_tvd_pts[::-1] + [left_tvd_pts[0]]
                    f_sec.add_trace(go.Scatter(
                        x=poly_vs, y=poly_tvd,
                        fill='toself',
                        fillcolor='rgba(180,180,180,0.28)',
                        line=dict(color='#444444', width=1.5),
                        mode='lines',
                        name='Wellbore Outline',
                        showlegend=True,
                        hoverinfo='skip',
                    ))

                # Section centerlines (colored per section, with hover)
                for vs_s, tvd_s, col_s, name_s in center_traces_2d:
                    f_sec.add_trace(go.Scatter(
                        x=vs_s, y=tvd_s,
                        mode='lines',
                        line=dict(color=col_s, width=2),
                        name=name_s,
                        showlegend=True,
                        hovertemplate=f"{name_s}<br>VS: %{{x:.1f}}<br>TVD: %{{y:.1f}}<extra></extra>"
                    ))

                # Casing shoe horizontal markers
                for _, row_s in csg_2d.iterrows():
                    try:
                        d_s = float(row_s['_depth'])
                        idx_s = (df_p['MD'] - d_s).abs().idxmin()
                        vs_s  = float(vs_p_proj.loc[idx_s])
                        tvd_s = float(df_p.loc[idx_s, 'TVDSS'])
                        od_s  = parse_casing_od(row_s['Size'])
                        hw_s  = od_s * vs_scale / 2.0
                        f_sec.add_trace(go.Scatter(
                            x=[vs_s - hw_s, vs_s + hw_s],
                            y=[tvd_s, tvd_s],
                            mode='lines+text',
                            line=dict(color='black', width=2),
                            text=[None, f" {row_s['Size']} shoe"],
                            textposition='middle right',
                            showlegend=False,
                            hovertemplate=f"Shoe {row_s['Size']}<br>Depth: {d_s:.0f}{u_label}<extra></extra>"
                        ))
                    except Exception:
                        pass
            except Exception:
                # Fallback: simple triangle markers (original behaviour)
                for _, row_f in edited_casing.iterrows():
                    try:
                        d_f = float(row_f['Depth'])
                        idx_f = (df_p['MD'] - d_f).abs().argsort()[:1]
                        if not idx_f.empty:
                            vs_f  = vs_p_proj.iloc[idx_f].values[0]
                            tvd_f = df_p['TVDSS'].iloc[idx_f].values[0]
                            f_sec.add_trace(go.Scatter(
                                x=[vs_f], y=[tvd_f], mode='markers+text',
                                marker=dict(symbol='triangle-left', size=10, color='black'),
                                text=[row_f['Size']], textposition='middle left', showlegend=False
                            ))
                    except Exception:
                        pass
        
        # Formation Lines
        try:
            for line in form_text.split('\n'):
                if ',' in line:
                    p = line.split(',')
                    f_sec.add_hline(y=float(p[1]), line_dash="dash", line_color="orange", annotation_text=p[0], annotation_position="bottom right")
        except: pass

        # --- LAYOUT FIXES (CRITICAL) ---
        f_plan.update_layout(
            title=f"Plan View", 
            xaxis_title="East", yaxis_title="North", 
            height=600, 
            yaxis_scaleanchor="x", # KUNCI: Biar bulat tetap bulat
            paper_bgcolor='white', plot_bgcolor='white',
            xaxis=dict(gridcolor='#eee', zeroline=False), yaxis=dict(gridcolor='#eee', zeroline=False)
        )
        
        f_sec.update_layout(
            title=f"Section View – Wellbore Schematic (Azimuth: {vs_azimuth:.1f}°)",
            xaxis_title=f"Vertical Section at {vs_azimuth:.1f}° ({u_label})",
            yaxis_title=f"TVDSS ({u_label})",
            height=700,
            yaxis_autorange="reversed",
            paper_bgcolor='white', plot_bgcolor='#f8f9fa',
            xaxis=dict(gridcolor='#dee2e6', zeroline=True, zerolinecolor='black'),
            yaxis=dict(gridcolor='#dee2e6', zeroline=False),
            legend=dict(bgcolor='rgba(255,255,255,0.85)', bordercolor='#ccc', borderwidth=1)
        )
        
        c_view.plotly_chart(f_plan, use_container_width=True)
        c_view.plotly_chart(f_sec, use_container_width=True)
        
    with tab3:
        st.markdown("### 📋 Trajectory Data Export")
        if 'Plan' in layers:
            csv = layers['Plan']['df'].to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Plan CSV", csv, "plan.csv", "text/csv")
            st.dataframe(layers['Plan']['df'], height=500, use_container_width=True)

    with tab4:
    # ENGINEERING CHARTS (Directly inspired by Image 3)
        st.subheader("📈 Drilling Mechanics Logs")
        
        if 'Plan' in st.session_state['layers']:
            df = st.session_state['layers']['Plan']['df']
            
            # Calculate derived engineering metrics
            # (Usually these come from physics models, here we approximate for visualization)
            df['Build_Rate'] = df['Inc'].diff().fillna(0) * (30 / 10) # deg/30m
            df['Turn_Rate'] = df['Azi'].diff().fillna(0) * (30 / 10) # deg/30m
            # Simple Toolface proxy
            df['Toolface'] = np.degrees(np.arctan2(df['Turn_Rate'], df['Build_Rate'])).fillna(0)
            
            # Create 4-Row Subplot like Image 3
            fig_eng = make_subplots(
                rows=4, cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=("Build Rate (°/30m)", "Turn Rate (°/30m)", "Toolface Angle (°)", "Dogleg Severity (°/30m)")
            )
            
            # 1. Build Rate (Green Line)
            fig_eng.add_trace(go.Scatter(x=df['MD'], y=df['Build_Rate'], line=dict(color='#2E7D32', width=2), name='Build Rate'), row=1, col=1)
            
            # 2. Turn Rate (Blue Line)
            fig_eng.add_trace(go.Scatter(x=df['MD'], y=df['Turn_Rate'], line=dict(color='#1565C0', width=2), name='Turn Rate'), row=2, col=1)
            
            # 3. Toolface (Purple Scatter)
            fig_eng.add_trace(go.Scatter(x=df['MD'], y=df['Toolface'], mode='markers', marker=dict(size=4, color='#6A1B9A'), name='Toolface'), row=3, col=1)
            
            # 4. DLS (Red Area)
            fig_eng.add_trace(go.Scatter(x=df['MD'], y=df['DLS'], fill='tozeroy', line=dict(color='#C62828', width=1), name='DLS'), row=4, col=1)
            
            fig_eng.update_layout(
                height=900, 
                paper_bgcolor='white', # The outer margin area
                plot_bgcolor='white',  # The inner grid area
                showlegend=False
            )
            
            st.plotly_chart(fig_eng, use_container_width=True)
        else:
            st.info("Generate a trajectory to view engineering logs.")

    with tab5:
        st.header("🔧 BHA & Assembly Viewer")
        st.caption("Auto-link dengan file dari Quick Import.")
        
        col_bha_1, col_bha_2 = st.columns([1, 2])
        
        selected_comps_df = None
        sel_case_name = ""
        sel_scen_label = ""
        
        # --- LOGIKA SHARED FILE ---
        # Cek apakah ada file dari Sidebar?
        active_file = st.session_state.get('shared_xml_file', None)
        
        with col_bha_1:
            # Jika tidak ada file dari sidebar, tampilkan uploader manual (Fallback)
            if active_file is None:
                st.info("Belum ada file di Sidebar. Upload manual di sini:")
                active_file = st.file_uploader("Upload XML", type=['xml'], key="bha_manual_up")
            else:
                st.success(f"📂 Menggunakan file: {active_file.name}")
            
            if active_file:
                # 1. PARSE DATA
                scen_dict, cases_df, comps_df = parse_scenario_bha_chain(active_file)
                
                if scen_dict and not cases_df.empty:
                    st.markdown("---")
                    
                    # 2. AUTO-SELECT SCENARIO DARI SIDEBAR
                    pre_selected_id = st.session_state.get('selected_scenario_id')
                    
                    scen_opts = {name: sid for sid, name in scen_dict.items()}
                    sorted_scen_names = sorted(list(scen_opts.keys()))
                    
                    # Cari index scenario yang cocok dengan Sidebar
                    default_idx = 0
                    if pre_selected_id:
                        target_name = next((name for sid, name in scen_dict.items() if sid == pre_selected_id), None)
                        if target_name and target_name in sorted_scen_names:
                            default_idx = sorted_scen_names.index(target_name)
                    
                    sel_scen_label = st.selectbox("1️⃣ Pilih Scenario / Plan:", sorted_scen_names, index=default_idx)
                    sel_scen_id = scen_opts[sel_scen_label]
                    
                    # 3. FILTER & SORT CASE
                    filtered_cases = cases_df[cases_df['scenario_id'] == sel_scen_id].copy()
                    
                    if not filtered_cases.empty:
                        filtered_cases = filtered_cases.sort_values(by='case_name', ascending=True)
                        
                        case_opts = dict(zip(filtered_cases['case_name'], filtered_cases['assembly_id']))
                        
                        sel_case_name = st.selectbox("2️⃣ Pilih BHA Run:", list(case_opts.keys()))
                        sel_assembly_id = case_opts[sel_case_name]
                        
                        # 4. FILTER KOMPONEN
                        if not comps_df.empty:
                            selected_comps_df = comps_df[comps_df['assembly_id'] == sel_assembly_id].copy()
                            if not selected_comps_df.empty:
                                selected_comps_df = selected_comps_df.drop(columns=['assembly_id'])
                        else:
                            st.warning("Data komponen kosong.")
                    else:
                        st.warning("Scenario ini tidak memiliki BHA Run.")
                else:
                    st.error("Struktur XML tidak valid.")

        with col_bha_2:
            if selected_comps_df is not None and not selected_comps_df.empty:
                st.subheader(f"📋 {sel_case_name}")
                
                # --- 1. PERSIAPAN DATA LOKAL ---
                df_display = selected_comps_df.copy()
                
                # Reset Nomor Urut agar mulai dari 1
                df_display = df_display.reset_index(drop=True)
                df_display['No'] = (df_display.index + 1).astype(str)
                
                # --- 2. HITUNG TOTAL (CUMULATIVE SUM) DISINI ---
                # Rumus ini sekarang hanya berjalan pada data BHA yang dipilih saja
                df_display['Total (m)'] = df_display['Length'].cumsum()
                
                # --- 3. HITUNG GRAND TOTAL UNTUK FOOTER ---
                total_len_val = df_display['Length'].sum()
                
                # --- 4. BUAT BARIS FOOTER ---
                footer_data = {
                    'No': '', 
                    'Sequence': 9999, 
                    'Description': 'Total:', 
                    'Top Connection': '',
                    'OD (in)': None, 
                    'ID (in)': None, 
                    'Gauge (in)': None,
                    'Length': None,       
                    'Weight (ppf)': None,
                    'Total (m)': total_len_val # Total Akhir
                }
                
                footer_row = pd.DataFrame([footer_data])
                
                # Gabungkan
                df_final_view = pd.concat([df_display, footer_row], ignore_index=True)
                
                # Rapikan kolom No
                cols = list(df_final_view.columns)
                if 'No' in cols:
                    cols.insert(0, cols.pop(cols.index('No')))
                    df_final_view = df_final_view[cols]

                # --- 5. TAMPILKAN TABEL ---
                st.dataframe(
                    df_final_view,
                    column_config={
                        "No": st.column_config.TextColumn("No", width="small"),
                        "Sequence": None, 
                        "assembly_id": None, 
                        "Description": st.column_config.TextColumn("Description", width="large"),
                        "Top Connection": st.column_config.TextColumn("Conn", width="medium"),
                        "OD (in)": st.column_config.NumberColumn("OD (in)", format="%.3f"),
                        "ID (in)": st.column_config.NumberColumn("ID (in)", format="%.3f"),
                        "Gauge (in)": st.column_config.NumberColumn("Gauge (in)", format="%.3f"),
                        "Weight (ppf)": st.column_config.NumberColumn("Weight (ppf)", format="%.2f"),
                        
                        # Length & Total (3 Desimal)
                        "Length": st.column_config.NumberColumn("Length (m)", format="%.3f"),
                        "Total (m)": st.column_config.NumberColumn("Total (m)", format="%.3f"),
                    },
                    use_container_width=True,
                    hide_index=True,
                    height=500
                )
                # TOTAL LENGTH
                if 'Length' in selected_comps_df.columns:
                    t_len = selected_comps_df['Length'].sum()
                    st.info(f"📏 Total Length: **{t_len:.2f}**")

                # VISUALISASI 2D (STICK PLOT)
                with st.expander("Lihat Visualisasi 2D", expanded=True):
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    depth = 0
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                    
                    for idx, row in selected_comps_df.iterrows():
                        l = row.get('Length', 1)
                        if l <= 0: l = 0.5
                        od = row['OD (in)']
                        name = row['Description']
                        
                        fig.add_trace(go.Scatter(
                            x=[-od/2, od/2, od/2, -od/2, -od/2],
                            y=[depth, depth, depth+l, depth+l, depth],
                            fill="toself",
                            line=dict(color='black', width=1),
                            fillcolor=colors[idx % len(colors)],
                            name=name,
                            text=f"<b>{name}</b><br>OD: {od}<br>L: {l}",
                            hoverinfo='text'
                        ))
                        depth += l
                    
                    fig.update_layout(
                        yaxis=dict(autorange="reversed", title="Cumulative Length"),
                        xaxis=dict(visible=False),
                        showlegend=False,
                        height=500,
                        margin=dict(t=20, b=20),
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # --- BAGIAN INTEGRASI PDF REPORT ---
                st.markdown("---")
                st.subheader("🖨️ Generate PDF Report")

                col_pr1, col_pr2 = st.columns([1, 2])

                with col_pr1:
                    st.caption("1. Upload Template (PDF)")
                    template_file = st.file_uploader("Template (Background)", type="pdf", key="tpl_up_tab5")

                with col_pr2:
                    st.caption("2. Isi Header Laporan")
                    h_cust = st.text_input("Customer", value="Pertamina Hulu Rokan", key="h_cust")
                    h_well = st.text_input("Well Name", value="Sumur-X", key="h_well")
                    h_job = st.text_input("Job Number", value="JOB-2025-001", key="h_job")
                    
                    c1, c2 = st.columns(2)
                    with c1: h_rig = st.text_input("Rig Name", value="Rig-01", key="h_rig")
                    with c2: h_field = st.text_input("Field Name", value="Minas", key="h_field")
                
                # Tombol Generate (Full Width)
                if template_file and st.button("🚀 Generate & Download PDF", key="btn_gen_pdf"):
                    try:
                        # Bungkus header
                        header_info = {
                            "customer": h_cust,
                            "well_name": h_well,
                            "job_number": h_job,
                            "rig_name": h_rig,
                            "field_name": h_field,
                            "country": "Indonesia"
                        }
                        
                        # Panggil Fungsi Generator (Pastikan sudah diimport)
                        pdf_bytes = generate_bha_pdf(template_file, header_info, selected_comps_df)
                        
                        st.success("✅ PDF Berhasil Dibuat!")
                        st.download_button(
                            label="📥 Download PDF",
                            data=pdf_bytes,
                            file_name=f"BHA_Report_{h_well}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"Gagal membuat PDF: {e}")

            elif selected_comps_df is not None:
                st.info("BHA ini tidak memiliki detail komponen.")
            else:
                st.markdown("""
                <div style='text-align: center; color: grey; padding: 50px;'>
                    <h3>⬅️ Ready</h3>
                    <p>Silakan pilih Scenario dan Case di sebelah kiri untuk melihat data.</p>
                </div>
                """, unsafe_allow_html=True)