import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize_scalar, minimize
from io import StringIO
import uuid
import warnings

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
        
    def solve_trajectory(self, target_n, target_e, target_tvdss, kop, dls):
        # Unit Handling
        if self.unit == 'Imperial':
            rkb_m = self.rkb * self.ft_to_m
            tgt_tvdss_m = target_tvdss * self.ft_to_m
            kop_m = kop * self.ft_to_m
            dls_m = dls * (30.0 / (100.0 * self.ft_to_m))
        else:
            rkb_m = self.rkb; tgt_tvdss_m = target_tvdss; kop_m = kop; dls_m = dls

        delta_n = target_n - self.surf_n
        delta_e = target_e - self.surf_e
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

        res = minimize_scalar(error_func, bounds=(0, 90), method='bounded')
        if not res.success: return None, 0, 0
        
        df_metric, azi, hold = self._generate_path(kop_m, dls_m, res.x, tgt_azi_deg, target_tvd_m, rkb_m)
        
        if self.unit == 'Imperial':
            for col in ['MD', 'TVD', 'TVDSS', 'VS']:
                df_metric[col] = df_metric[col] / self.ft_to_m
            
        return df_metric, azi, hold

    def _generate_path(self, kop, dls, hold_inc, azi, target_tvd, rkb_val):
        step = 5.0
        path = []
        md, tvd, hd, inc = 0, 0, 0, 0
        radius = (180/np.pi) * (30.0/dls)
        inc_step = (dls/30.0) * step
        
        while tvd < target_tvd:
            section = "Hold"
            if md < kop: 
                inc = 0; section = "Vertical"
            elif inc < hold_inc: 
                inc += inc_step; section = "Build"
                if inc > hold_inc: inc = hold_inc
            
            rad_inc = np.radians(inc)
            d_tvd = step * np.cos(rad_inc)
            d_hd = step * np.sin(rad_inc)
            md += step; tvd += d_tvd; hd += d_hd
            path.append([md, inc, azi, tvd, hd, section])
            
        df = pd.DataFrame(path, columns=['MD', 'Inc', 'Azi', 'TVD', 'VS', 'Section'])
        rad_azi = np.radians(azi)
        df['N'] = self.surf_n + (df['VS'] * np.cos(rad_azi))
        df['E'] = self.surf_e + (df['VS'] * np.sin(rad_azi))
        df['TVDSS'] = df['TVD'] - rkb_val
        return df, azi, hold_inc

    def calculate_correction_path(self, actual_df, plan_df, correction_len):
        """
        PRESCRIPTIVE ANALYTICS ENGINE (OPTIMIZER):
        Finds optimal Build Rate and Turn Rate to hit the Plan XYZ coordinate
        at the end of the correction length.
        """
        last_act = actual_df.iloc[-1]
        
        # 1. Target Point on Plan
        target_md = last_act['MD'] + correction_len
        
        # Find closest point on plan
        plan_segment = plan_df[plan_df['MD'] >= target_md]
        if plan_segment.empty: 
            target_row = plan_df.iloc[-1] # Fallback to TD
        else: 
            target_row = plan_segment.iloc[0]
            
        target_pos = np.array([target_row['N'], target_row['E'], target_row['TVD']])
        start_pos = np.array([last_act['N'], last_act['E'], last_act['TVD']])
        start_inc = last_act['Inc']
        start_azi = last_act['Azi']
        
        # 2. Simulator Function for Optimizer
        def simulate_section(params):
            # params = [build_rate, turn_rate] (deg per unit length)
            build_rate, turn_rate = params
            
            # Simple Minimum Curvature Stepping
            sim_step = 10.0
            n_steps = int(correction_len / sim_step)
            if n_steps == 0: n_steps = 1
            
            curr_n, curr_e, curr_tvd = start_pos
            curr_inc, curr_azi = start_inc, start_azi
            
            # Rate per step
            d_inc = (build_rate / self.dls_ref) * sim_step
            d_azi = (turn_rate / self.dls_ref) * sim_step
            
            for _ in range(n_steps):
                next_inc = curr_inc + d_inc
                next_azi = curr_azi + d_azi
                
                avg_inc = np.radians((curr_inc + next_inc)/2)
                avg_azi = np.radians((curr_azi + next_azi)/2)
                
                curr_tvd += sim_step * np.cos(avg_inc)
                curr_n += sim_step * np.sin(avg_inc) * np.cos(avg_azi)
                curr_e += sim_step * np.sin(avg_inc) * np.sin(avg_azi)
                
                curr_inc = next_inc
                curr_azi = next_azi
                
            final_pos = np.array([curr_n, curr_e, curr_tvd])
            # Distance squared to target
            return np.sum((final_pos - target_pos)**2)

        # 3. Optimize to find BUR and TR
        # Initial guess: 0 build, 0 turn
        initial_guess = [0.0, 0.0]
        
        # Bounds for realistic DLS (e.g. max +/- 10 deg/30m)
        bnds = ((-10, 10), (-10, 10))
        
        res = minimize(simulate_section, initial_guess, bounds=bnds, method='Nelder-Mead')
        
        best_bur, best_turn = res.x
        
        # 4. Generate Detailed Path with Optimized Params
        path = []
        step = 5.0
        n_steps = int(correction_len / step)
        if n_steps == 0: n_steps = 1
        
        d_inc = (best_bur / self.dls_ref) * step
        d_azi = (best_turn / self.dls_ref) * step
        
        curr_md = last_act['MD']
        curr_n, curr_e, curr_tvd = start_pos
        curr_inc, curr_azi = start_inc, start_azi
        
        for _ in range(n_steps):
            next_inc = curr_inc + d_inc
            next_azi = curr_azi + d_azi
            
            avg_inc = np.radians((curr_inc + next_inc)/2)
            avg_azi = np.radians((curr_azi + next_azi)/2)
            
            curr_md += step
            curr_tvd += step * np.cos(avg_inc)
            curr_n += step * np.sin(avg_inc) * np.cos(avg_azi)
            curr_e += step * np.sin(avg_inc) * np.sin(avg_azi)
            
            curr_vs = np.sqrt((curr_n - self.surf_n)**2 + (curr_e - self.surf_e)**2)
            curr_tvdss = curr_tvd - self.rkb
            
            path.append([curr_md, curr_inc, curr_azi, curr_tvd, curr_tvdss, curr_n, curr_e, curr_vs, "Correction"])
            
            curr_inc = next_inc
            curr_azi = next_azi
            
        df_corr = pd.DataFrame(path, columns=['MD', 'Inc', 'Azi', 'TVD', 'TVDSS', 'N', 'E', 'VS', 'Section'])
        
        # Calculate resulting Dogleg
        # DLS approx = sqrt(BUR^2 + (TR*sin(Inc))^2)
        avg_inc_rad = np.radians(start_inc + (curr_inc - start_inc)/2)
        total_dls = np.sqrt(best_bur**2 + (best_turn * np.sin(avg_inc_rad))**2)
        
        return df_corr, total_dls, best_bur, best_turn

def calculate_economics(df):
    total_md = df['MD'].iloc[-1]
    cost_per_meter = 1500; rop_avg = 10; rig_cost_per_day = 50000
    drilling_days = (total_md / rop_avg) / 24
    total_cost = (total_md * cost_per_meter) + (drilling_days * rig_cost_per_day)
    return total_cost, drilling_days

def parse_trajectory_data(text_data, rkb, surf_n=0, surf_e=0):
    if not text_data.strip(): return None
    try:
        data = StringIO(text_data)
        line1 = text_data.strip().split('\n')[0]
        sep = ',' if ',' in line1 else '\t' if '\t' in line1 else r'\s+'
        df = pd.read_csv(data, sep=sep, engine='python')
        df.columns = df.columns.str.replace(r"[\(\[].*?[\)\]]", "", regex=True).str.strip()
        col_map = {
            'Measured Depth': 'MD', 'Inclination': 'Inc', 'Azimuth': 'Azi',
            'True Vertical Depth': 'TVD', 'Vertical Section': 'VS',
            '+N/S-': 'N', '+E/W-': 'E', 'North': 'N', 'East': 'E'
        }
        df.rename(columns=col_map, inplace=True)
        req = ['MD', 'TVD', 'N', 'E'] # Relaxed requirement
        if not all(c in df.columns for c in req): 
            return "MISSING_COLS: Require MD, TVD, N, E (plus INC, AZI for detail)"
        
        if 'Inc' not in df.columns: df['Inc'] = 0
        if 'Azi' not in df.columns: df['Azi'] = 0
        if 'TVDSS' not in df.columns: df['TVDSS'] = df['TVD'] - rkb
        
        # LOGIC FOR RELATIVE OFFSET DATA
        # If the first point is (0,0,0) or close to it, we assume it's relative.
        # We shift it by the provided surf_n and surf_e
        # However, if surf_n/e are 0, we assume the user didn't set offset surface, or it's same as plan.
        # Check first row
        first_n = df['N'].iloc[0]
        first_e = df['E'].iloc[0]
        
        # Heuristic: If coordinates are small (< 10000) but Surf N/E are large (> 10000), SHIFT.
        if (abs(first_n) < 10000 and abs(surf_n) > 10000):
            df['N'] = df['N'] + surf_n
        
        if (abs(first_e) < 10000 and abs(surf_e) > 10000):
            df['E'] = df['E'] + surf_e
            
        if 'VS' not in df.columns:
            # Re-calculate VS (might be wrong if offset surface is vastly different, but best guess)
            # Using Plotting Surface Reference
            # Note: For offsets, VS is tricky. Usually relative to its own wellhead or plan wellhead.
            # Here we calc relative to Plan Surface for consistent plotting
            df['VS'] = np.sqrt((df['N'] - surf_n)**2 + (df['E'] - surf_e)**2)
            
        return df
    except Exception as e: return str(e)

# ==========================================
# 2. STATE & LAYER MANAGEMENT
# ==========================================
if 'layers' not in st.session_state:
    st.session_state['layers'] = {} 
if 'meta_data' not in st.session_state:
    st.session_state['meta_data'] = {'rkb': 0, 'surf_n':0, 'surf_e':0}

# ==========================================
# 3. UI SIDEBAR
# ==========================================
st.sidebar.title("🎛️ DSS Command Center")

# --- TAB 1: PLANNING (ENGINEERING) ---
with st.sidebar.form("plan_form"):
    st.header("1. Well Planning Parameters")
    
    unit_sys = st.radio("Units", ["Metric", "Imperial"], horizontal=True)
    u_label = "m" if unit_sys == "Metric" else "ft"
    dls_label = "deg/30m" if unit_sys == "Metric" else "deg/100ft"
    
    c1, c2 = st.columns(2)
    r_floor = c1.number_input(f"Rotary Table ({u_label})"); r_elev = c2.number_input(f"Cellar Elev")
    surf_n = c1.number_input("Surf N"); surf_e = c2.number_input("Surf E")
    
    st.markdown("---")
    tgt_n = c1.number_input("Target N"); tgt_e = c2.number_input("Target E")
    tgt_tvdss = st.number_input(f"Target TVDSS ({u_label})")
    
    c3, c4 = st.columns(2)
    kop = c3.number_input(f"KOP ({u_label})"); dls = c4.number_input(f"DLS")
    
    plan_submit = st.form_submit_button("🚀 CALCULATE PLAN", type="primary")

if plan_submit:
    r_rkb = r_floor + r_elev
    planner = SmartPlanner(surf_n, surf_e, r_rkb, unit_sys)
    df_plan, azi, hold = planner.solve_trajectory(tgt_n, tgt_e, tgt_tvdss, kop, dls)
    
    if df_plan is not None:
        st.session_state['layers']['Plan'] = {'df': df_plan, 'color': '#0000FF', 'show': True, 'type': 'plan'}
        st.session_state['meta_data'] = {'rkb': r_rkb, 'surf_n': surf_n, 'surf_e': surf_e, 'unit': unit_sys, 'planner': planner}
        st.success(f"Plan Updated! Azi: {azi:.2f}°, Hold: {hold:.2f}°")
    else:
        st.error("Plan Calculation Failed.")

# --- TAB 2: CASING MANAGER ---
with st.sidebar.expander("🛠️ Casing & Formation Manager", expanded=False):
    st.caption("Edit Casing Program")
    casing_init = pd.DataFrame([
        {"Size": "20\"", "Depth": 50, "Type": "MD"},
        {"Size": "13-3/8\"", "Depth": 446.5, "Type": "TVDSS"},
        {"Size": "9-5/8\"", "Depth": 1200, "Type": "MD"}
    ])
    edited_casing = st.data_editor(casing_init, num_rows="dynamic", key="casing_editor")
    
    st.caption(f"Formation Tops (Name, Depth)")
    form_text = st.text_area("Format: Name, Depth", "Top GUF, 446.5\nTop TAF, 558.0\nTop LAF, 1379.8", key="form_editor")

# --- TAB 3: ACTUAL & PRESCRIPTIVE ---
with st.sidebar.expander("📉 Actual & Prescription (Correction)", expanded=False):
    st.caption("Paste Actual Survey (MD Inc Azi TVD N E)")
    actual_txt = st.text_area("Actual Data:", height=100)
    corr_len = st.number_input(f"Correction Length ({u_label})")
    
    if st.button("Run Prescriptive Analysis"):
        if 'Plan' in st.session_state['layers']:
            meta = st.session_state['meta_data']
            # Pass surf_n/e to parser for calculation if needed
            df_act = parse_trajectory_data(actual_txt, meta['rkb'], meta['surf_n'], meta['surf_e'])
            
            if isinstance(df_act, pd.DataFrame):
                st.session_state['layers']['Actual'] = {'df': df_act, 'color': '#FF0000', 'show': True, 'type': 'actual'}
                
                planner = meta['planner']
                df_plan = st.session_state['layers']['Plan']['df']
                df_corr, total_dls, best_bur, best_turn = planner.calculate_correction_path(df_act, df_plan, corr_len)
                
                st.session_state['layers']['Correction'] = {'df': df_corr, 'color': '#00FF00', 'show': True, 'type': 'corr'}
                st.session_state['prescription'] = {'bur': best_bur, 'turn': best_turn, 'dls': total_dls, 'len': corr_len}
                st.success("Analysis Complete!")
            else:
                st.error(f"Error Parsing: {df_act}")
        else:
            st.warning("Generate Plan First!")

# --- TAB 4: OFFSET WELLS ---
with st.sidebar.expander("🛡️ Offset Wells", expanded=False):
    off_name = st.text_input("Well Name", "Offset-01")
    
    # Input for Offset Surface Location
    st.markdown("**Offset Surface Location (Optional)**")
    c_off1, c_off2 = st.columns(2)
    off_surf_n = c_off1.number_input("Off N", 0.0)
    off_surf_e = c_off2.number_input("Off E", 0.0)
    st.caption("If data starts at 0,0, these coordinates will be added.")
    
    off_txt = st.text_area("Offset Data (MD TVD N E)", height=100)
    
    if st.button("Add Offset"):
        meta = st.session_state['meta_data']
        rkb_offset = meta.get('rkb', 25)
        
        # Use provided offset surface, or default to plan surface if 0
        use_n = off_surf_n if off_surf_n != 0 else meta.get('surf_n', 0)
        use_e = off_surf_e if off_surf_e != 0 else meta.get('surf_e', 0)
        
        df_off = parse_trajectory_data(off_txt, rkb_offset, use_n, use_e)
        
        if isinstance(df_off, pd.DataFrame):
            if 'Offsets' not in st.session_state['layers']: st.session_state['layers']['Offsets'] = []
            st.session_state['layers']['Offsets'].append({
                'id': str(uuid.uuid4()), 'name': off_name, 'df': df_off,
                'color': '#808080', 'show': True
            })
            st.success(f"Added {off_name}")
        else:
            st.error(f"Data Error: {df_off}")

# --- TAB 5: VISUAL CONTROL ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Layer Manager")

if 'Plan' in st.session_state['layers']:
    c1, c2 = st.sidebar.columns([0.2, 0.8])
    st.session_state['layers']['Plan']['show'] = c1.checkbox("👁️", True, key="show_plan")
    st.session_state['layers']['Plan']['color'] = c2.color_picker("Plan", "#0000FF", key="col_plan")

if 'Actual' in st.session_state['layers']:
    c1, c2 = st.sidebar.columns([0.2, 0.8])
    st.session_state['layers']['Actual']['show'] = c1.checkbox("👁️", True, key="show_act")
    st.session_state['layers']['Actual']['color'] = c2.color_picker("Actual", "#FF0000", key="col_act")

if 'Correction' in st.session_state['layers']:
    c1, c2 = st.sidebar.columns([0.2, 0.8])
    st.session_state['layers']['Correction']['show'] = c1.checkbox("👁️", True, key="show_corr")
    st.session_state['layers']['Correction']['color'] = c2.color_picker("Correction", "#00FF00", key="col_corr")

if 'Offsets' in st.session_state['layers']:
    st.sidebar.markdown("**Offset Wells:**")
    offsets_to_remove = []
    for i, off in enumerate(st.session_state['layers']['Offsets']):
        c1, c2, c3 = st.sidebar.columns([0.2, 0.6, 0.2])
        off['show'] = c1.checkbox("👁️", off['show'], key=f"v_{off['id']}")
        off['color'] = c2.color_picker(off['name'], off['color'], key=f"c_{off['id']}")
        if c3.button("🗑️", key=f"d_{off['id']}"):
            offsets_to_remove.append(i)
    
    # Remove logic
    for i in sorted(offsets_to_remove, reverse=True):
        st.session_state['layers']['Offsets'].pop(i)
        st.rerun()

# ==========================================
# 4. MAIN DASHBOARD RENDER
# ==========================================
st.title("🏗️ DSS Well Master Ultimate")

# --- GLOBAL KPIs & ECONOMICS ---
if 'Plan' in st.session_state['layers']:
    df_plan = st.session_state['layers']['Plan']['df']
    meta_info = st.session_state['meta_data']
    cost, time = calculate_economics(df_plan)
    
    # KPIs
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Depth", f"{df_plan['MD'].iloc[-1]:.0f}")
    m2.metric("Max Inclination", f"{df_plan['Inc'].max():.2f}°")
    m3.metric("Est. Cost", f"${cost/1000:.1f} K")
    m4.metric("Est. Days", f"{time:.1f} Days")
    
    # Global Collision Check
    min_sep = 9999.0
    if 'Offsets' in st.session_state['layers']:
        active_offs = [w for w in st.session_state['layers']['Offsets'] if w['show']]
        if active_offs:
            p1 = df_plan[['N', 'E', 'TVD']].values
            for off in active_offs:
                p2 = off['df'][['N', 'E', 'TVD']].values
                if len(p2) > 1000: p2 = p2[::5] # Downsample
                dists = np.min(np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=2), axis=1)
                min_sep = min(min_sep, np.min(dists))
    
    sep_col = "normal"
    if min_sep < 10: sep_col = "inverse"
    m5.metric("Min Separation", f"{min_sep:.1f}" if min_sep!=9999.0 else "N/A", delta="CRITICAL" if min_sep < 10 else "Safe", delta_color=sep_col)

# --- PRESCRIPTIVE INSIGHTS ---
if 'prescription' in st.session_state and st.session_state['layers'].get('Correction', {}).get('show'):
    pres = st.session_state['prescription']
    turn_dir = "RIGHT" if pres['turn'] > 0 else "LEFT"
    build_drop = 'BUILD' if pres['bur'] > 0 else 'DROP'
    
    st.info(f"""
    💡 **PRESCRIPTIVE ACTION (RETURN TO PLAN):**
    
    Agar kembali ke jalur dalam jarak **{pres['len']} {u_label}**, Driller harus:
    1.  Melakukan **{build_drop}** dengan Rate **{abs(pres['bur']):.2f} {dls_label}**.
    2.  Melakukan **TURN {turn_dir}** sebesar **{abs(pres['turn']):.2f} {dls_label}**.
    3.  Resultan Dogleg Severity: **{pres['dls']:.2f} {dls_label}**.
    """)

# --- VISUALIZATION TABS ---
tab1, tab2, tab3 = st.tabs(["🌍 3D View", "📐 2D Views", "📋 Data"])

with tab1:
    fig3d = go.Figure()
    layers = st.session_state['layers']
    
    # Helper 3D
    def add_3d_trace(df, name, color, dash='solid'):
        # Hover data standard
        h_text = [f"MD: {m:.1f}<br>Inc: {i:.2f}<br>TVD: {t:.1f}" 
                  for m, i, t in zip(df['MD'], df.get('Inc', [0]*len(df)), df['TVD'])]
        
        fig3d.add_trace(go.Scatter3d(
            x=df['E'], y=df['N'], z=df['TVDSS'], mode='lines', name=name,
            line=dict(color=color, width=5, dash=dash),
            text=h_text, hoverinfo='text'
        ))

    if 'Plan' in layers and layers['Plan']['show']:
        add_3d_trace(layers['Plan']['df'], 'Plan', layers['Plan']['color'])
    if 'Actual' in layers and layers['Actual']['show']:
        add_3d_trace(layers['Actual']['df'], 'Actual', layers['Actual']['color'])
    if 'Correction' in layers and layers['Correction']['show']:
        add_3d_trace(layers['Correction']['df'], 'Correction', layers['Correction']['color'], 'dash')
    
    if 'Offsets' in layers:
        for off in layers['Offsets']:
            if off['show']: add_3d_trace(off['df'], off['name'], off['color'], 'dot')

    fig3d.update_layout(scene=dict(zaxis=dict(autorange="reversed"), aspectmode='data'), height=700, uirevision='constant')
    st.plotly_chart(fig3d, use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    fig_plan = go.Figure()
    fig_sec = go.Figure()
    
    # --- HELPER FIX VS CALCULATION (KEY ERROR FIX HERE) ---
    def add_2d_traces(fig, x_col, y_col, dash='solid'):
        surf_n = st.session_state['meta_data']['surf_n']
        surf_e = st.session_state['meta_data']['surf_e']
        
        for key, layer in layers.items():
            if key == 'Offsets':
                for off in layer:
                    if off['show']:
                        df_off = off['df']
                        # CALC VS IF MISSING (The Fix)
                        if x_col == 'VS' and 'VS' not in df_off.columns:
                            df_off = df_off.copy()
                            if 'N' in df_off and 'E' in df_off:
                                df_off['VS'] = np.sqrt((df_off['N'] - surf_n)**2 + (df_off['E'] - surf_e)**2)
                            else:
                                continue # Cannot plot VS without coords
                        
                        x = df_off[x_col] if x_col in df_off else df_off['E']
                        fig.add_trace(go.Scatter(x=x, y=df_off[y_col], mode='lines', name=off['name'], line=dict(color=off['color'], dash='dot')))
            elif layer['show']:
                df = layer['df']
                # Safety check for other layers too
                if x_col == 'VS' and 'VS' not in df.columns:
                     df['VS'] = np.sqrt((df['N'] - surf_n)**2 + (df['E'] - surf_e)**2)
                
                fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='lines', name=key, line=dict(color=layer['color'], dash=dash if key=='Correction' else 'solid')))

    # Plan View
    add_2d_traces(fig_plan, 'E', 'N')
    fig_plan.update_layout(title="Plan View", xaxis_title="E", yaxis_title="N", yaxis_scaleanchor="x", uirevision='constant')
    
    # Section View
    add_2d_traces(fig_sec, 'VS', 'TVDSS')
    
    # Plot Casing
    if 'Plan' in layers:
        plan_df = layers['Plan']['df']
        if not edited_casing.empty:
            for _, row in edited_casing.iterrows():
                try:
                    c_depth = float(row['Depth'])
                    c_type = row['Type']
                    limit_md = c_depth
                    if c_type == 'TVDSS':
                        limit_md = plan_df.iloc[(plan_df['TVDSS']-c_depth).abs().argsort()[:1]]['MD'].values[0]
                    
                    csg_df = plan_df[plan_df['MD'] <= limit_md]
                    if not csg_df.empty:
                        fig_sec.add_trace(go.Scatter(x=csg_df['VS'], y=csg_df['TVDSS'], mode='lines', 
                                                   line=dict(color='black', width=4), opacity=0.5, name=f"Csg {row['Size']}"))
                except: pass
                
    # Plot Formation
    try:
        for line in form_text.split('\n'):
            p = line.split(',')
            nm, dp = p[0], float(p[1])
            fig_sec.add_hline(y=dp, line_dash="dash", line_color="grey", annotation_text=nm)
    except: pass

    fig_sec.update_layout(title="Section View", xaxis_title="VS", yaxis_title="TVDSS", yaxis_autorange="reversed", uirevision='constant')
    
    with c1: st.plotly_chart(fig_plan, use_container_width=True)
    with c2: st.plotly_chart(fig_sec, use_container_width=True)

with tab3:
    if 'Plan' in st.session_state['layers']:
        st.dataframe(st.session_state['layers']['Plan']['df'])