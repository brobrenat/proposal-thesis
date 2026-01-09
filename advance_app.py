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

warnings.filterwarnings('ignore')

st.set_page_config(page_title="DSS Well Master Ultimate", layout="wide", page_icon="🏗️")

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

    def calculate_correction_path(self, actual_df, plan_df, correction_len):
        # ... (Logika solver correction sama seperti sebelumnya, tapi kita akan resample outputnya)
        last = actual_df.iloc[-1]
        tgt_md = last['MD'] + correction_len
        seg = plan_df[plan_df['MD'] >= tgt_md]
        tgt_row = seg.iloc[0] if not seg.empty else plan_df.iloc[-1]
        
        # Solver (Low Res for Speed)
        tgt_pos = np.array([tgt_row['N'], tgt_row['E'], tgt_row['TVD']])
        start_pos = np.array([last['N'], last['E'], last['TVD']])
        def sim(p):
            b, t = p; s = correction_len/5.0
            cn, ce, ct = start_pos; ci, ca = last['Inc'], last['Azi']
            di, da = (b/self.dls_ref)*s, (t/self.dls_ref)*s
            for _ in range(5):
                ai=np.radians(ci+di/2); aa=np.radians(ca+da/2)
                ct+=s*np.cos(ai); cn+=s*np.sin(ai)*np.cos(aa); ce+=s*np.sin(ai)*np.sin(aa)
                ci+=di; ca+=da
            return np.sum((np.array([cn,ce,ct])-tgt_pos)**2)
        
        res = minimize(sim, [0,0], bounds=[(-15,15),(-15,15)], method='Nelder-Mead')
        best_b, best_t = res.x
        
        # Generate Detail Path (HI-RES)
        # UPDATE: STEP SIZE 1.0
        step = 1.0; n_steps = max(1, int(correction_len/step))
        mds, incs, azis = [last['MD']], [last['Inc']], [last['Azi']]
        di = (best_b/self.dls_ref)*step
        da = (best_t/self.dls_ref)*step
        
        for _ in range(n_steps):
            mds.append(mds[-1]+step); incs.append(incs[-1]+di); azis.append(azis[-1]+da)
            
        df = self.engine.calculate_trajectory(mds, incs, azis, last['N'], last['E'], last['TVD'])
        
        f = self.ft_to_m if self.unit=='Imperial' else 1.0
        df['TVDSS'] = df['TVD'] - (self.rkb * f)
        if self.unit == 'Imperial': df['TVDSS'] /= self.ft_to_m
        df['Section'] = 'Correction'
        
        avg_inc = np.radians(last['Inc'])
        tot_dls = np.sqrt(best_b**2 + (best_t * np.sin(avg_inc))**2)
        tf = np.degrees(np.arctan2(best_t, best_b)) % 360
        return df, tot_dls, best_b, best_t, tf

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

def parse_trajectory_data(text_data, rkb, surf_n, surf_e, engine, azi_corr=0.0):
    if not text_data.strip(): return None
    try:
        # 1. SMART PARSER (Detect separator)
        data = StringIO(text_data)
        try: df = pd.read_csv(data, sep=None, engine='python') # Auto-detect sep
        except: 
            data.seek(0); df = pd.read_csv(data, sep='\t') # Fallback tab

        # 2. CLEAN HEADERS
        df.columns = df.columns.str.upper().str.replace(r"[\(\[].*?[\)\]]", "", regex=True).str.strip()
        col_map = {
            'MEASURED DEPTH':'MD', 'INCLINATION':'Inc', 'AZIMUTH':'Azi', 
            'TRUE VERTICAL DEPTH':'TVD', 'VERTICAL SECTION':'VS',
            '+N/S-':'N', '+E/W-':'E', 'NORTH':'N', 'EAST':'E',
            'MD': 'MD', 'INC': 'Inc', 'AZI': 'Azi'
        }
        for c in df.columns:
            for k, v in col_map.items():
                if k in c: df.rename(columns={c: v}, inplace=True); break

        req = ['MD', 'Inc', 'Azi'] 
        if not all(c in df.columns for c in req): 
            return "MISSING_COLS: Required at least MD, Inc, Azi"
        
        # --- AUTO-CALCULATE COORDINATES IF MISSING ---
        if 'N' not in df.columns or 'E' not in df.columns or 'TVD' not in df.columns:
            # Apply Azimuth Correction ONLY if re-calculating
            df['Azi'] = (df['Azi'] + azi_corr) % 360
            
            df = df.sort_values('MD').reset_index(drop=True)
            
            # AUTO-ANCHOR TO SURFACE (Tie-in 0)
            if df['MD'].iloc[0] > 0:
                row0 = pd.DataFrame({'MD': [0], 'Inc': [0], 'Azi': [0]})
                df = pd.concat([row0, df], ignore_index=True)
            
            # RE-COMPUTE using Engine (MCM)
            df_calc = engine.calculate_trajectory(
                df['MD'].values, df['Inc'].values, df['Azi'].values,
                start_n=surf_n, start_e=surf_e, start_tvd=0
            )
            
            df_calc['TVDSS'] = df_calc['TVD'] - rkb
            return df_calc
        
        # If data already has coords, use them but fix headers if needed
        if 'TVDSS' not in df.columns: df['TVDSS'] = df['TVD'] - rkb
        if 'VS' not in df.columns: df['VS'] = np.sqrt((df['N'] - surf_n)**2 + (df['E'] - surf_e)**2)
        
        if 'DLS' not in df.columns:
            md_arr = df['MD'].values
            inc_rad = np.radians(df['Inc'].values)
            azi_rad = np.radians(df['Azi'].values)
            dls_arr = np.zeros(len(df))
            ref_len = 30.0 
            for i in range(1, len(df)):
                dL = md_arr[i] - md_arr[i-1]
                if dL > 0:
                    arg = np.cos(inc_rad[i]-inc_rad[i-1]) - \
                          (np.sin(inc_rad[i-1])*np.sin(inc_rad[i])*(1-np.cos(azi_rad[i]-azi_rad[i-1])))
                    arg = np.clip(arg, -1.0, 1.0)
                    dls_val = np.degrees(np.arccos(arg)) * (ref_len / dL)
                    dls_arr[i] = dls_val
            df['DLS'] = dls_arr
            
        return df
    except Exception as e: return str(e)

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

# --- AUTO-FILL EXPANDER ---
with st.sidebar.expander("⚡ Quick Import (Auto-Fill)", expanded=False):
    st.caption("Paste full data to extract parameters. **Requires: N, E, TVD**")
    paste_data = st.text_area("Paste Data:", height=100)
    if st.button("⚡ Extract & Apply"):
        try:
            temp_engine = DrillingEngine('Metric', 30.0)
            df_temp = parse_trajectory_data(paste_data, 0, 0, 0, temp_engine) 
            if isinstance(df_temp, pd.DataFrame) and {'N', 'E', 'TVD'}.issubset(df_temp.columns):
                s_n = df_temp['N'].iloc[0]; s_e = df_temp['E'].iloc[0]
                t_n = df_temp['N'].iloc[-1]; t_e = df_temp['E'].iloc[-1]; t_tvd = df_temp['TVD'].iloc[-1]
                kop_idx = df_temp[df_temp['Inc'] > 0.5].index
                e_kop = df_temp['MD'].iloc[kop_idx[0]] if len(kop_idx) > 0 else 0
                e_hold = df_temp['Inc'].max()
                st.session_state['autofill_data'] = {
                    'surf_n': s_n, 'surf_e': s_e, 'tgt_n': t_n, 'tgt_e': t_e, 'tgt_tvdss': t_tvd,
                    'kop': e_kop, 'hold': e_hold
                }
                st.rerun()
            else:
                st.error("Missing N/E/TVD for parameter extraction.")
        except Exception as e: st.error(f"Extraction Failed: {e}")

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
    if st.button("🎲 Demo Actual"):
        st.session_state['act_txt'] = "MD Inc Azi\n0 0 0\n500 0 0\n600 2 45\n1000 12 48"
    act_txt = st.text_area("Actual Data:", value=st.session_state.get('act_txt', ''))
    corr_len = st.number_input("Correction Len", value=300.0)
    
    if st.button("Run Prescription"):
        if 'Plan' in st.session_state['layers']:
            meta = st.session_state['meta']
            eng = meta['planner'].engine
            df_act = parse_trajectory_data(act_txt, meta['rkb'], meta['surf_n'], meta['surf_e'], eng)
            
            if isinstance(df_act, pd.DataFrame):
                st.session_state['layers']['Actual'] = {'df': df_act, 'color': '#FF0000', 'show': True}
                planner = meta['planner']
                df_plan = st.session_state['layers']['Plan']['df']
                # UNPACK 5 VARIABLES: df_corr, total_dls, req_bur, best_turn, toolface
                df_corr, total_dls, req_bur, best_turn, tf = planner.calculate_correction_path(df_act, df_plan, corr_len)
                st.session_state['layers']['Correction'] = {'df': df_corr, 'color': '#00FF00', 'show': True}
                st.session_state['prescription'] = {'bur': req_bur, 'turn': best_turn, 'dls': total_dls, 'len': corr_len, 'tf': tf}
            else: st.error(df_act)

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
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Depth (MD)", f"{df_plan['MD'].iloc[-1]:,.0f} {u_label}", delta="Target Reached" if df_plan['MD'].iloc[-1] > 0 else None)
        c2.metric("Max Inclination", f"{df_plan['Inc'].max():.2f}°")
        c3.metric("Est. Cost", f"${cost/1000:,.0f} K")
        
        min_sep = 9999.0
        if 'Offsets' in st.session_state['layers']:
            active = [o for o in st.session_state['layers']['Offsets'] if o['show']]
            if active:
                p1 = df_plan[['N', 'E', 'TVD']].values
                for o in active:
                    p2 = o['df'][['N', 'E', 'TVD']].values
                    # Downsample for speed if huge
                    if len(p2)>1000: p2=p2[::5]
                    # Simple Euclidean distance check
                    d = np.min(np.linalg.norm(p1[:, None] - p2[None, :], axis=2))
                    min_sep = min(min_sep, d)
        
        c5.metric("Min Separation", f"{min_sep:.1f} m" if min_sep!=9999 else "N/A", 
                  delta="CRITICAL ALERT" if min_sep<10 else "Safe Zone", 
                  delta_color="inverse")

    # --- PRESCRIPTION ALERT BANNER ---
    if 'prescription' in st.session_state and st.session_state['layers'].get('Correction', {}).get('show'):
        p = st.session_state['prescription']
        st.markdown(f"""
        <div style="padding:15px; background-color:#e6fffa; border-left:5px solid #00b894; border-radius:5px; margin-bottom:20px;">
            <h4 style="margin:0; color:#00695c;">💡 AI Prescriptive Correction</h4>
            <p style="margin:0;">Steering Command: <b>{'BUILD' if p['bur']>0 else 'DROP'} {abs(p['bur']):.2f} {dls_label}</b> | 
            <b>TURN {'RIGHT' if p['turn']>0 else 'LEFT'} {abs(p['turn']):.2f} {dls_label}</b> | 
            Toolface: <b>{p['tf']:.0f}°</b> over next <b>{p['len']} {u_label}</b></p>
        </div>
        """, unsafe_allow_html=True)

    tab1, tab2, tab3,tab4 = st.tabs(["🌍 3D Trajectory Analysis", "📐 2D Engineering Plots", "📋 Raw Survey Data","📈 Drilling Mechanics Logs"])
    
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
        def add_3d_trace(df, name, color, width=5, dash='solid', opacity=1.0):
            # Improved Hover Template
            hover_temp = (
                "<b>" + name + "</b><br>" +
                "MD: %{text:.1f}<br>" +
                "Inc: %{customdata[0]:.2f}°<br>" +
                "Azi: %{customdata[1]:.2f}°<br>" +
                "TVDSS: %{z:.1f}"
            )
            
            # CRITICAL: Use Scatter3d for Lines
            fig3d.add_trace(go.Scatter3d(
                x=df['E'], y=df['N'], z=df['TVDSS'],
                mode='lines', 
                name=name,
                line=dict(color=color, width=width, dash=dash),
                opacity=opacity,
                text=df['MD'],
                customdata=np.stack((df['Inc'], df['Azi']), axis=-1),
                hovertemplate=hover_temp
            ))
            
            # Add Cone at the Bit (Last Point)
            if name in ['Plan', 'Actual', 'Correction']:
                last = df.iloc[-1]
                fig3d.add_trace(go.Scatter3d(
                    x=[last['E']], y=[last['N']], z=[last['TVDSS']],
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

        # CRITICAL: Use Mesh3d for the Surface Plane (No 'mode' argument here)
        fig3d.add_trace(go.Mesh3d(
            x=[center_e-500, center_e+500, center_e+500, center_e-500],
            y=[center_n-500, center_n-500, center_n+500, center_n+500],
            z=[0, 0, 0, 0],
            color='lightblue', 
            opacity=0.1, 
            name='Sea Level', 
            showlegend=True
        ))

        # --- 3. DRAW LINES ---
        if layers['Plan']['show']: 
            add_3d_trace(layers['Plan']['df'], 'Plan', layers['Plan']['color'], width=6)
            
        if 'Actual' in layers and layers['Actual']['show']: 
            add_3d_trace(layers['Actual']['df'], 'Actual', layers['Actual']['color'], width=7)
            
        if 'Correction' in layers and layers['Correction']['show']: 
            add_3d_trace(layers['Correction']['df'], 'Correction', layers['Correction']['color'], width=6, dash='dash')
            
        if 'Offsets' in layers:
            for o in layers['Offsets']:
                if o['show']: add_3d_trace(o['df'], o['name'], o['color'], width=3, dash='solid', opacity=0.6)

        # --- 4. LAYOUT SETTINGS ---
        fig3d.update_layout(
            scene=dict(
                xaxis=dict(title='EAST (+/-)', backgroundcolor="rgb(240, 240, 240)", gridcolor="white", showbackground=True, zerolinecolor="white"),
                yaxis=dict(title='NORTH (+/-)', backgroundcolor="rgb(240, 240, 240)", gridcolor="white", showbackground=True, zerolinecolor="white"),
                zaxis=dict(title='TVDSS (Depth)', autorange="reversed", backgroundcolor="rgb(230, 230, 240)", gridcolor="white", showbackground=True, zerolinecolor="white"),
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

        # Casing & Formation (Logic sama, disesuaikan koordinatnya)
        if not edited_casing.empty and 'Plan' in layers:
            df_p = layers['Plan']['df']
            # Recalculate VS for Plan just for casing lookup
            vs_p_proj = get_projected_vs(df_p, origin_n, origin_e, vs_azimuth)
            
            for _, row in edited_casing.iterrows():
                try:
                    d = float(row['Depth'])
                    idx = (df_p['MD']-d).abs().argsort()[:1]
                    if not idx.empty:
                        # Ambil VS dari hasil proyeksi yang sudah dihitung
                        vs_val = vs_p_proj.iloc[idx].values[0]
                        tvd_val = df_p['TVDSS'].iloc[idx].values[0]
                        f_sec.add_trace(go.Scatter(
                            x=[vs_val], y=[tvd_val], mode='markers+text',
                            marker=dict(symbol='triangle-left', size=10, color='black'),
                            text=[row['Size']], textposition='middle left', showlegend=False
                        ))
                except: pass
        
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
            title=f"Section View (Azimuth: {vs_azimuth:.1f}°)", 
            xaxis_title=f"Vertical Section at {vs_azimuth:.1f}°", 
            yaxis_title="TVDSS", 
            height=600, 
            yaxis_autorange="reversed", 
            yaxis_scaleanchor="x", # KUNCI: Biar 100m Depth terlihat sama panjang dengan 100m VS
            paper_bgcolor='white', plot_bgcolor='white',
            xaxis=dict(gridcolor='#eee', zeroline=True, zerolinecolor='black'), 
            yaxis=dict(gridcolor='#eee', zeroline=False)
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