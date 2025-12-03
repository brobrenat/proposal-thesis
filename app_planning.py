import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize_scalar, minimize
from io import StringIO
import uuid
import warnings
import re

warnings.filterwarnings('ignore')

st.set_page_config(page_title="DSS Well Master Ultimate", layout="wide", page_icon="🏗️")

# ==========================================
# 1. ENGINE & LOGIC
# ==========================================
class SmartPlanner:
    def __init__(self, surf_n, surf_e, rkb, unit_system='Metric'):
        self.surf_n = surf_n
        self.surf_e = surf_e
        self.rkb = rkb
        self.unit = unit_system
        self.ft_to_m = 0.3048
        self.dls_ref = 30.0 if unit_system == 'Metric' else 100.0
        self.engine = DrillingEngine(unit_system, self.dls_ref)
        
    def solve_trajectory(self, target_n, target_e, target_tvdss, kop, dls, force_hold=None):
        # Unit Handling
        if self.unit == 'Imperial':
            rkb_m = self.rkb * self.ft_to_m
            tgt_tvdss_m = target_tvdss * self.ft_to_m
            kop_m = kop * self.ft_to_m
            dls_m = dls * (30.0 / (100.0 * self.ft_to_m))
            surf_n_m = self.surf_n * self.ft_to_m
            surf_e_m = self.surf_e * self.ft_to_m
            tgt_n_m = target_n * self.ft_to_m
            tgt_e_m = target_e * self.ft_to_m
        else:
            rkb_m = self.rkb; tgt_tvdss_m = target_tvdss; kop_m = kop; dls_m = dls
            surf_n_m = self.surf_n; surf_e_m = self.surf_e
            tgt_n_m = target_n; tgt_e_m = target_e

        delta_n = tgt_n_m - surf_n_m
        delta_e = tgt_e_m - surf_e_m
        target_tvd_m = tgt_tvdss_m + rkb_m
        target_hd_m = np.sqrt(delta_n**2 + delta_e**2)
        
        tgt_azi_rad = np.arctan2(delta_e, delta_n)
        tgt_azi_deg = np.degrees(tgt_azi_rad) % 360
        
        def error_func(hold_angle):
            rad = np.radians(hold_angle)
            radius = (180/np.pi) * (30.0/dls_m)
            build_tvd = kop_m + (radius * np.sin(rad))
            build_hd = radius * (1 - np.cos(rad))
            rem_tvd = target_tvd_m - build_tvd
            if rem_tvd < 0: return 1e6 
            rem_hd = rem_tvd * np.tan(rad)
            return abs((build_hd + rem_hd) - target_hd_m)

        if force_hold is not None and force_hold > 0.1:
            best_hold = force_hold
        else:
            res = minimize_scalar(error_func, bounds=(0, 90), method='bounded')
            best_hold = res.x if res.success else 0
        
        df_metric, azi, hold = self._generate_path_mcm(kop_m, dls_m, best_hold, tgt_azi_deg, target_tvd_m, rkb_m, surf_n_m, surf_e_m)
        
        if self.unit == 'Imperial':
            for col in ['MD', 'TVD', 'TVDSS', 'VS', 'N', 'E']:
                df_metric[col] = df_metric[col] / self.ft_to_m
            
        return df_metric, azi, hold

    def _generate_path_mcm(self, kop, dls, hold_inc, azi, target_tvd, rkb_val, surf_n, surf_e):
        # Generate Survey Points
        step = 10.0 
        mds, incs, azis = [0], [0], [azi]
        
        # Vertical
        curr_md = 0
        while curr_md < kop:
            curr_md += step
            if curr_md > kop: curr_md = kop
            mds.append(curr_md); incs.append(0); azis.append(azi)
            if curr_md == kop: break
            
        # Build
        radius = (180/np.pi) * (30.0/dls)
        build_len = np.radians(hold_inc) * radius
        eob_md = kop + build_len
        
        curr_md = kop
        while curr_md < eob_md:
            curr_md += step
            if curr_md > eob_md: curr_md = eob_md
            frac = (curr_md - kop) / build_len
            mds.append(curr_md); incs.append(frac * hold_inc); azis.append(azi)
            if curr_md == eob_md: break
            
        # Hold
        tvd_eob = kop + (radius * np.sin(np.radians(hold_inc)))
        rem_tvd = target_tvd - tvd_eob
        if rem_tvd > 0:
            hold_len = rem_tvd / np.cos(np.radians(hold_inc))
            target_md = eob_md + hold_len
            
            curr_md = eob_md
            while curr_md < target_md:
                curr_md += step
                if curr_md > target_md: curr_md = target_md
                mds.append(curr_md); incs.append(hold_inc); azis.append(azi)
                if curr_md == target_md: break
        
        # Hitung Koordinat pakai MCM Engine
        df = self.engine.calculate_trajectory(mds, incs, azis, surf_n, surf_e, 0)
        
        df['Section'] = 'Plan'
        df['TVDSS'] = df['TVD'] - rkb_val
        return df, azi, hold_inc

    def calculate_correction_path(self, actual_df, plan_df, correction_len):
        last_act = actual_df.iloc[-1]
        target_md = last_act['MD'] + correction_len
        
        # --- FIX: Variable name corrected here ---
        plan_segment = plan_df[plan_df['MD'] >= target_md]
        
        if plan_segment.empty: 
            target_row = plan_df.iloc[-1] 
        else: 
            target_row = plan_segment.iloc[0]
        # -----------------------------------------
        
        target_pos = np.array([target_row['N'], target_row['E'], target_row['TVD']])
        start_pos = np.array([last_act['N'], last_act['E'], last_act['TVD']])
        start_inc = last_act['Inc']
        start_azi = last_act['Azi']
        
        def simulate(params):
            bur, tr = params
            sim_step = correction_len / 5.0
            curr_n, curr_e, curr_tvd = start_pos
            curr_inc, curr_azi = start_inc, start_azi
            
            d_inc = (bur / self.dls_ref) * sim_step
            d_azi = (tr / self.dls_ref) * sim_step
            
            for _ in range(5):
                avg_inc = np.radians(curr_inc + d_inc/2)
                avg_azi = np.radians(curr_azi + d_azi/2)
                curr_tvd += sim_step * np.cos(avg_inc)
                curr_n += sim_step * np.sin(avg_inc) * np.cos(avg_azi)
                curr_e += sim_step * np.sin(avg_inc) * np.sin(avg_azi)
                curr_inc += d_inc; curr_azi += d_azi
                
            return np.sum((np.array([curr_n, curr_e, curr_tvd]) - target_pos)**2)

        res = minimize(simulate, [0, 0], bounds=[(-15, 15), (-15, 15)], method='Nelder-Mead')
        best_bur, best_turn = res.x
        
        # Generate Detail Path
        step = 10.0
        n_steps = int(correction_len / step)
        if n_steps < 1: n_steps = 1
        
        mds, incs, azis = [last_act['MD']], [last_act['Inc']], [last_act['Azi']]
        d_inc = (best_bur / self.dls_ref) * step
        d_azi = (best_turn / self.dls_ref) * step
        
        for _ in range(n_steps):
            mds.append(mds[-1] + step)
            incs.append(incs[-1] + d_inc)
            azis.append(azis[-1] + d_azi)
            
        df_corr = self.engine.calculate_trajectory(mds, incs, azis, last_act['N'], last_act['E'], last_act['TVD'])
        df_corr['TVDSS'] = df_corr['TVD'] - (self.rkb * (self.ft_to_m if self.unit=='Imperial' else 1))
        df_corr['Section'] = 'Correction'
        
        avg_inc = np.radians(last_act['Inc'])
        total_dls = np.sqrt(best_bur**2 + (best_turn * np.sin(avg_inc))**2)
        
        # Toolface Calc
        toolface = np.degrees(np.arctan2(best_turn, best_bur)) % 360

        return df_corr, total_dls, best_bur, best_turn, toolface

    def calculate_extension(self, current_df, add_length, target_inc, target_azi):
        last = current_df.iloc[-1]
        step = 10.0
        n_steps = int(add_length / step)
        if n_steps < 1: n_steps = 1
        
        mds, incs, azis = [last['MD']], [last['Inc']], [last['Azi']]
        d_inc = (target_inc - last['Inc']) / n_steps
        d_azi = (target_azi - last['Azi']) / n_steps
        
        for _ in range(n_steps):
            mds.append(mds[-1] + step)
            incs.append(incs[-1] + d_inc)
            azis.append(azis[-1] + d_azi)
            
        df_ext = self.engine.calculate_trajectory(mds, incs, azis, last['N'], last['E'], last['TVD'])
        
        conv_rkb = self.rkb * (self.ft_to_m if self.unit=='Imperial' else 1) if self.unit=='Imperial' else self.rkb
        df_ext['TVDSS'] = df_ext['TVD'] - conv_rkb
        df_ext['Section'] = 'Extension'
        return pd.concat([current_df, df_ext.iloc[1:]], ignore_index=True)


class DrillingEngine:
    """ 
    Core Minimum Curvature Algorithm (API RP 78)
    """
    def __init__(self, unit, dls_ref):
        self.unit = unit
        self.dls_ref = dls_ref
        
    def calculate_trajectory(self, md, inc, azi, start_n, start_e, start_tvd):
        n = len(md)
        tvd = np.zeros(n); n_c = np.zeros(n); e_c = np.zeros(n); dls = np.zeros(n)
        tvd[0], n_c[0], e_c[0] = start_tvd, start_n, start_e
        
        inc_rad = np.radians(inc)
        azi_rad = np.radians(azi)
        
        for i in range(1, n):
            dL = md[i] - md[i-1]
            I1, I2 = inc_rad[i-1], inc_rad[i]
            A1, A2 = azi_rad[i-1], azi_rad[i]
            
            cos_beta = np.cos(I2-I1) - (np.sin(I1)*np.sin(I2)*(1-np.cos(A2-A1)))
            beta = np.arccos(np.clip(cos_beta, -1, 1))
            
            if abs(beta) < 1e-6: rf = 1.0
            else: rf = (2/beta) * np.tan(beta/2)
            
            n_c[i] = n_c[i-1] + (dL/2)*(np.sin(I1)*np.cos(A1) + np.sin(I2)*np.cos(A2))*rf
            e_c[i] = e_c[i-1] + (dL/2)*(np.sin(I1)*np.sin(A1) + np.sin(I2)*np.sin(A2))*rf
            tvd[i] = tvd[i-1] + (dL/2)*(np.cos(I1) + np.cos(I2))*rf
            
            if dL > 0:
                dls[i] = np.degrees(beta) * (self.dls_ref / dL)
            else:
                dls[i] = 0
                
        df = pd.DataFrame({'MD': md, 'Inc': inc, 'Azi': azi, 'TVD': tvd, 'N': n_c, 'E': e_c, 'DLS': dls})
        df['VS'] = np.sqrt((df['N']-start_n)**2 + (df['E']-start_e)**2)
        return df

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
# 4. DASHBOARD RENDER
# ==========================================
st.title("🏗️ DSS Well Master Ultimate")

if 'Plan' in st.session_state['layers']:
    df_plan = st.session_state['layers']['Plan']['df']
    cost, time = calculate_economics(df_plan)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Depth", f"{df_plan['MD'].iloc[-1]:.0f}")
    c2.metric("Max Inc", f"{df_plan['Inc'].max():.2f}°")
    c3.metric("Cost", f"${cost/1000:.0f}K")
    
    min_sep = 9999.0
    if 'Offsets' in st.session_state['layers']:
        active = [o for o in st.session_state['layers']['Offsets'] if o['show']]
        if active:
            p1 = df_plan[['N', 'E', 'TVD']].values
            for o in active:
                p2 = o['df'][['N', 'E', 'TVD']].values
                if len(p2)>1000: p2=p2[::5]
                d = np.min(np.linalg.norm(p1[:, None] - p2[None, :], axis=2))
                min_sep = min(min_sep, d)
    
    c5.metric("Separation", f"{min_sep:.1f}m" if min_sep!=9999 else "N/A", delta="CRITICAL" if min_sep<10 else "Safe", delta_color="inverse" if min_sep<10 else "normal")

    if 'prescription' in st.session_state and st.session_state['layers'].get('Correction', {}).get('show'):
        p = st.session_state['prescription']
        st.info(f"💡 **PRESCRIPTION:** {('BUILD' if p['bur']>0 else 'DROP')} {abs(p['bur']):.2f} {dls_label} , TURN {('RIGHT' if p['turn']>0 else 'LEFT')} {abs(p['turn']):.2f} {dls_label}, Toolface: {p['tf']:.0f}°")

    tab1, tab2, tab3 = st.tabs(["🌍 3D View", "📐 2D Views", "📋 Data"])
    
    with tab1:
        fig3d = go.Figure()
        layers = st.session_state['layers']
        
        def plot_3d(df, name, col, dash='solid'):
            txt = [f"MD:{m:.0f}<br>I:{i:.1f}<br>A:{a:.1f}" for m,i,a in zip(df['MD'], df['Inc'], df['Azi'])]
            fig3d.add_trace(go.Scatter3d(x=df['E'], y=df['N'], z=df['TVDSS'], mode='lines', name=name, line=dict(color=col, width=5, dash=dash), text=txt, hoverinfo='text'))
        
        if layers['Plan']['show']: plot_3d(layers['Plan']['df'], 'Plan', layers['Plan']['color'])
        if 'Actual' in layers and layers['Actual']['show']: plot_3d(layers['Actual']['df'], 'Actual', layers['Actual']['color'])
        if 'Correction' in layers and layers['Correction']['show']: plot_3d(layers['Correction']['df'], 'Correction', layers['Correction']['color'], 'dash')
        if 'Offsets' in layers:
            for o in layers['Offsets']:
                if o['show']: plot_3d(o['df'], o['name'], o['color'], 'dot')
        
        fig3d.update_layout(scene=dict(zaxis=dict(autorange="reversed"), aspectmode='data'), height=600, uirevision='constant')
        st.plotly_chart(fig3d, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        f_plan = go.Figure()
        f_sec = go.Figure()
        
        def plot_2d(df, name, col, dash='solid'):
            f_plan.add_trace(go.Scatter(x=df['E'], y=df['N'], mode='lines', name=name, line=dict(color=col, dash=dash)))
            f_sec.add_trace(go.Scatter(x=df['VS'], y=df['TVDSS'], mode='lines', name=name, line=dict(color=col, dash=dash)))
            
        if layers['Plan']['show']: plot_2d(layers['Plan']['df'], 'Plan', layers['Plan']['color'])
        if 'Actual' in layers and layers['Actual']['show']: plot_2d(layers['Actual']['df'], 'Actual', layers['Actual']['color'])
        if 'Correction' in layers and layers['Correction']['show']: plot_2d(layers['Correction']['df'], 'Correction', layers['Correction']['color'], 'dash')
        if 'Offsets' in layers:
            for o in layers['Offsets']:
                if o['show']: plot_2d(o['df'], o['name'], o['color'], 'dot')
        
        if not edited_casing.empty:
            for _, row in edited_casing.iterrows():
                try:
                    d = float(row['Depth'])
                    pt = df_plan.iloc[(df_plan['MD']-d).abs().argsort()[:1]]
                    f_sec.add_trace(go.Scatter(x=pt['VS'], y=pt['TVDSS'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='black'), name=f"Csg {row['Size']}"))
                except: pass
    try:
        for line in form_text.split('\n'):
            p = line.split(',')
            nm, dp = p[0], float(p[1])
            f_sec.add_hline(y=dp, line_dash="dash", line_color="grey", annotation_text=nm)
    except: pass

    f_plan.update_layout(title="Plan View", xaxis_title="East", yaxis_title="North", yaxis_scaleanchor="x", uirevision='constant')
    f_sec.update_layout(title="Section View", xaxis_title="Vertical Section", yaxis_title="TVDSS", yaxis_autorange="reversed", uirevision='constant')
    
    with c1: st.plotly_chart(f_plan, use_container_width=True)
    with c2: st.plotly_chart(f_sec, use_container_width=True)

    with tab3:
        st.dataframe(df_plan)