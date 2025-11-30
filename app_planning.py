import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize_scalar
from io import StringIO
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Advanced DSS Well Planner", layout="wide", page_icon="🏗️")

# ==========================================
# 1. ENGINE & LOGIC
# ==========================================
class SmartPlanner:
    def __init__(self, surf_n, surf_e, rkb, unit_system='Metric'):
        self.surf_n = surf_n
        self.surf_e = surf_e
        self.rkb = rkb
        self.unit = unit_system
        # Standar DLS reference: 30m untuk Metric, 100ft untuk Imperial
        self.dls_ref = 30.0 if unit_system == 'Metric' else 100.0
        
    def solve_trajectory(self, target_n, target_e, target_tvdss, kop, dls):
        # 1. Hitung Delta Geometri
        delta_n = target_n - self.surf_n
        delta_e = target_e - self.surf_e
        target_tvd = target_tvdss + self.rkb
        
        tgt_azi_rad = np.arctan2(delta_e, delta_n)
        tgt_azi_deg = np.degrees(tgt_azi_rad) % 360
        target_hd = np.sqrt(delta_n**2 + delta_e**2)
        
        # 2. Solver Matematika
        def error_func(hold_angle):
            rad = np.radians(hold_angle)
            # Radius Curvature calculation based on Unit System
            radius = (180/np.pi) * (self.dls_ref/dls)
            
            # Build Section
            build_tvd = kop + (radius * np.sin(rad))
            build_hd = radius * (1 - np.cos(rad))
            
            # Hold Section
            rem_tvd = target_tvd - build_tvd
            if rem_tvd < 0: return 1e6 # Penalty
            rem_hd = rem_tvd * np.tan(rad)
            
            return abs((build_hd + rem_hd) - target_hd)

        res = minimize_scalar(error_func, bounds=(0, 90), method='bounded')
        if not res.success: return None, 0, 0
        
        return self._generate_path(kop, dls, res.x, tgt_azi_deg, target_tvd)

    def _generate_path(self, kop, dls, hold_inc, azi, target_tvd):
        step = 5.0 # High resolution step
        path = []
        md, tvd, hd, inc = 0, 0, 0, 0
        radius = (180/np.pi) * (self.dls_ref/dls)
        inc_step = (dls/self.dls_ref) * step
        
        # Simulasi Pengeboran (MD Loop)
        while tvd < target_tvd:
            section = "Hold"
            current_dls = 0.0
            
            if md < kop: 
                inc = 0; section = "Vertical"
            elif inc < hold_inc: 
                inc += inc_step; section = "Build"
                current_dls = dls
                if inc > hold_inc: inc = hold_inc
            
            rad_inc = np.radians(inc)
            d_tvd = step * np.cos(rad_inc)
            d_hd = step * np.sin(rad_inc)
            
            md += step; tvd += d_tvd; hd += d_hd
            path.append([md, inc, azi, tvd, hd, section, current_dls])
            
        df = pd.DataFrame(path, columns=['MD', 'Inc', 'Azi', 'TVD', 'VS', 'Section', 'DLS_Calc'])
        rad_azi = np.radians(azi)
        df['N'] = self.surf_n + (df['VS'] * np.cos(rad_azi))
        df['E'] = self.surf_e + (df['VS'] * np.sin(rad_azi))
        df['TVDSS'] = df['TVD'] - self.rkb
        return df, azi, hold_inc

    def calculate_extension(self, current_df, add_length, target_inc, target_azi):
        """
        Fitur Extension: Melanjutkan pengeboran ke target baru (Inc/Azi baru)
        Menggunakan Minimum Curvature Method step-by-step
        """
        last_row = current_df.iloc[-1]
        start_md = last_row['MD']
        start_tvd = last_row['TVD']
        start_n = last_row['N']
        start_e = last_row['E']
        start_vs = last_row['VS']
        start_inc = last_row['Inc']
        start_azi = last_row['Azi']
        
        path = []
        step = 5.0 # Step size simulasi
        
        # Hitung perubahan sudut per step (Build/Turn Rate)
        # Asumsi perubahan linear sepanjang section (Constant DLS)
        steps_count = int(add_length / step)
        if steps_count == 0: steps_count = 1
        
        inc_step = (target_inc - start_inc) / steps_count
        azi_step = (target_azi - start_azi) / steps_count
        
        curr_md = start_md
        curr_tvd = start_tvd
        curr_n = start_n
        curr_e = start_e
        curr_vs = start_vs
        curr_inc = start_inc
        curr_azi = start_azi
        
        remaining_len = add_length
        
        while remaining_len > 0:
            d_md = min(step, remaining_len)
            remaining_len -= d_md
            
            next_inc = curr_inc + (inc_step * (d_md/step))
            next_azi = curr_azi + (azi_step * (d_md/step))
            
            # Minimum Curvature Calculation for this small step
            I1, I2 = np.radians(curr_inc), np.radians(next_inc)
            A1, A2 = np.radians(curr_azi), np.radians(next_azi)
            
            # Dogleg angle (Beta)
            cos_beta = np.cos(I2 - I1) - (np.sin(I1) * np.sin(I2) * (1 - np.cos(A2 - A1)))
            # Safety check for floating point errors
            if cos_beta > 1.0: cos_beta = 1.0
            if cos_beta < -1.0: cos_beta = -1.0
            
            beta = np.arccos(cos_beta)
            
            # Ratio Factor
            if abs(beta) < 0.0001:
                rf = 1.0 # Limit kalau lurus
            else:
                rf = (2 / beta) * np.tan(beta / 2)
            
            d_n = (d_md/2) * (np.sin(I1)*np.cos(A1) + np.sin(I2)*np.cos(A2)) * rf
            d_e = (d_md/2) * (np.sin(I1)*np.sin(A1) + np.sin(I2)*np.sin(A2)) * rf
            d_tvd = (d_md/2) * (np.cos(I1) + np.cos(I2)) * rf
            
            curr_md += d_md
            curr_tvd += d_tvd
            curr_n += d_n
            curr_e += d_e
            
            # Update VS (approximate projection)
            # Re-calculate VS based on new N, E relative to surface
            # Note: VS biasanya diproyeksikan ke arah target awal, tapi untuk extension kita pakai Euclidean distance dari surface utk simplifikasi visual
            curr_vs = np.sqrt((curr_n - self.surf_n)**2 + (curr_e - self.surf_e)**2)
            
            # Hitung DLS lokal untuk step ini
            dls_step = np.degrees(beta) * (self.dls_ref / d_md)
            
            curr_inc = next_inc
            curr_azi = next_azi
            
            path.append([curr_md, curr_inc, curr_azi, curr_tvd, curr_vs, "Extension", dls_step])
            
        df_ext = pd.DataFrame(path, columns=['MD', 'Inc', 'Azi', 'TVD', 'VS', 'Section', 'DLS_Calc'])
        df_ext['N'] = [p[0] for p in zip(path)] # N tidak tersimpan di list path di atas, perlu fix
        
        # FIX: Simpan N dan E di path list
        # Re-create list with N and E
        # Agar efisien, kita update dataframe N dan E langsung dari kalkulasi loop tadi?
        # Lebih baik update structure path di atas.
        
        # Re-running loop logic cleanly:
        return self._generate_extension_df(current_df, add_length, target_inc, target_azi)

    def _generate_extension_df(self, current_df, add_length, target_inc, target_azi):
        # Helper function bersih untuk extension
        last_row = current_df.iloc[-1]
        # ... (Sama seperti logika di atas) ...
        # Mari kita implementasi ulang yang rapi
        
        path = []
        step = 10.0
        
        curr_md = last_row['MD']
        curr_inc = last_row['Inc']
        curr_azi = last_row['Azi']
        curr_tvd = last_row['TVD']
        curr_n = last_row['N']
        curr_e = last_row['E']
        
        # Kalkulasi step change
        steps = int(np.ceil(add_length / step))
        d_md = add_length / steps
        d_inc = (target_inc - curr_inc) / steps
        d_azi = (target_azi - curr_azi) / steps
        
        for _ in range(steps):
            next_inc = curr_inc + d_inc
            next_azi = curr_azi + d_azi
            
            # Min Curve
            I1, I2 = np.radians(curr_inc), np.radians(next_inc)
            A1, A2 = np.radians(curr_azi), np.radians(next_azi)
            
            cos_beta = np.cos(I2 - I1) - (np.sin(I1) * np.sin(I2) * (1 - np.cos(A2 - A1)))
            beta = np.arccos(np.clip(cos_beta, -1, 1))
            
            if abs(beta) < 1e-6: rf = 1.0
            else: rf = (2 / beta) * np.tan(beta / 2)
            
            delta_n = (d_md/2) * (np.sin(I1)*np.cos(A1) + np.sin(I2)*np.cos(A2)) * rf
            delta_e = (d_md/2) * (np.sin(I1)*np.sin(A1) + np.sin(I2)*np.sin(A2)) * rf
            delta_tvd = (d_md/2) * (np.cos(I1) + np.cos(I2)) * rf
            
            curr_md += d_md
            curr_tvd += delta_tvd
            curr_n += delta_n
            curr_e += delta_e
            
            # Recalc DLS
            dls_val = np.degrees(beta) * (self.dls_ref / d_md)
            
            # VS Calculation
            curr_vs = np.sqrt((curr_n - self.surf_n)**2 + (curr_e - self.surf_e)**2)
            
            curr_inc = next_inc
            curr_azi = next_azi
            
            path.append([curr_md, curr_inc, curr_azi, curr_tvd, curr_vs, "Extension", dls_val, curr_n, curr_e])
            
        df_ext = pd.DataFrame(path, columns=['MD', 'Inc', 'Azi', 'TVD', 'VS', 'Section', 'DLS_Calc', 'N', 'E'])
        df_ext['TVDSS'] = df_ext['TVD'] - self.rkb
        
        return pd.concat([current_df, df_ext], ignore_index=True)

# Fungsi Parsing Offset Well (Copy-Paste)
def parse_offset_well(text_data):
    if not text_data.strip():
        return None
    try:
        # Ganti koma dengan tab jika user copy dari CSV, atau biarkan jika dari Excel
        data = StringIO(text_data)
        df = pd.read_csv(data, sep='\t') # Asumsi copy dari Excel (Tab separated)
        if df.shape[1] < 2: # Kalau gagal, coba koma
            data = StringIO(text_data)
            df = pd.read_csv(data, sep=',')
            
        # Standarisasi kolom (Case Insensitive)
        df.columns = df.columns.str.upper().str.strip()
        
        # Pastikan kolom wajib ada
        required = ['TVD', 'N', 'E'] # MD opsional untuk plot visual
        if not all(col in df.columns for col in required):
            return "MISSING_COLS"
            
        if 'TVDSS' not in df.columns:
            df['TVDSS'] = df['TVD'] # Asumsi sementara jika tidak ada RKB offset
            
        return df
    except Exception as e:
        return str(e)

# ==========================================
# 2. UI SIDEBAR (FORM INPUT)
# ==========================================
st.sidebar.title("🎛️ Planning Control")

# UNIT SWITCHER
unit_sys = st.sidebar.radio("Unit System", ["Metric (m)", "Imperial (ft)"], horizontal=True)
u_label = "m" if "Metric" in unit_sys else "ft"
dls_label = "deg/30m" if "Metric" in unit_sys else "deg/100ft"

with st.sidebar.form("planning_form"):
    st.markdown("### 1. Surface & Rig")
    r_floor = st.number_input(f"Rotary Table ({u_label})", value=6.1)
    r_elev = st.number_input(f"Cellar Elev ({u_label})", value=19.46)
    surf_n = st.number_input(f"Surface Y / North ({u_label})", value=9000000.0)
    surf_e = st.number_input(f"Surface X / East ({u_label})", value=400000.0)

    st.markdown("### 2. Target Coordinates")
    tgt_n = st.number_input(f"Target Y / North ({u_label})", value=9000400.0)
    tgt_e = st.number_input(f"Target X / East ({u_label})", value=400400.0)
    tgt_tvdss = st.number_input(f"Target Z / TVDSS ({u_label})", value=2200.0)

    st.markdown("### 3. Engineering Params")
    kop = st.number_input(f"KOP (MD {u_label})", value=500.0)
    dls = st.number_input(f"Dogleg Severity ({dls_label})", value=3.0, step=0.1)

    st.markdown("---")
    submitted = st.form_submit_button("🚀 GENERATE BASE PLAN", type="primary")

# --- EXPANDER: DATA TAMBAHAN (CASING & OFFSET) ---
with st.sidebar.expander("🛠️ Casing & Formation Manager", expanded=False):
    st.caption("Edit Casing Program")
    casing_data = pd.DataFrame([
        {"Size": "20\"", "Depth": 50, "Type": "MD"},
        {"Size": "13-3/8\"", "Depth": 446.5, "Type": "TVDSS"},
        {"Size": "9-5/8\"", "Depth": 1200, "Type": "MD"}
    ])
    edited_casing = st.data_editor(casing_data, num_rows="dynamic")
    
    st.caption("Formation Tops (Name, TVDSS)")
    form_text = st.text_area("Format: Name, Depth", "Top GUF, 446.5\nTop TAF, 558.0\nTop LAF, 1379.8")

with st.sidebar.expander("⚠️ Offset Well (Anti-Collision)", expanded=False):
    st.caption(f"Paste data from Excel (Header: MD, TVD, N, E)")
    offset_input = st.text_area("Paste Data Here:", height=150, 
                                placeholder="MD\tTVD\tN\tE\n0\t0\t9000100\t400100\n100\t100\t9000100\t400100...")
    
with st.sidebar.expander("⏬ Trajectory Extension (Next Target)", expanded=True):
    st.info("Simulasi pengeboran lanjutan (Section Baru)")
    ext_len = st.number_input(f"Extension Length ({u_label})", value=0.0, step=50.0)
    
    # Default values will be updated after first run, initially dummy
    col_e1, col_e2 = st.columns(2)
    ext_inc = col_e1.number_input("Target Inclination (deg)", value=90.0, step=0.1)
    ext_azi = col_e2.number_input("Target Azimuth (deg)", value=0.0, step=0.1)

# ==========================================
# 3. MAIN PROCESS
# ==========================================
st.title("🏗️ Advanced DSS Well Planner")
st.markdown(f"**Unit System:** {unit_sys} | **Objective:** Safe & Cost-Effective Trajectory")

# Session State Management
if 'plan_data' not in st.session_state:
    st.session_state['plan_data'] = None

if submitted:
    r_rkb = r_floor + r_elev
    planner = SmartPlanner(surf_n, surf_e, r_rkb, unit_sys)
    df_plan, azi, hold = planner.solve_trajectory(tgt_n, tgt_e, tgt_tvdss, kop, dls)
    
    if df_plan is not None:
        st.session_state['plan_data'] = df_plan
        st.session_state['meta'] = {'azi': azi, 'hold': hold, 'rkb': r_rkb}
    else:
        st.error("❌ Trajectory Calculation Failed. Check Target Parameters.")

# --- RENDER DASHBOARD ---
if st.session_state['plan_data'] is not None:
    df_final = st.session_state['plan_data'].copy()
    meta = st.session_state['meta']
    r_rkb = meta['rkb']
    planner = SmartPlanner(surf_n, surf_e, r_rkb, unit_sys) # Re-init for visual helpers

    # Update Default Extension Inputs if user hasn't changed them yet? 
    # Streamlit widgets retain state, so we can display info
    last_inc = df_final['Inc'].iloc[-1]
    last_azi = df_final['Azi'].iloc[-1]
    st.sidebar.caption(f"Last Survey: Inc {last_inc:.2f}°, Azi {last_azi:.2f}°")

    # --- A. EXTENSION MODULE (MULTI TARGET SIMULATOR) ---
    if ext_len > 0:
        # User defined Extension
        df_final = planner.calculate_extension(df_final, ext_len, ext_inc, ext_azi)
        st.toast(f"✅ Extension added: +{ext_len} {u_label} @ Inc {ext_inc}°")

    # --- B. OFFSET WELL PROCESSING ---
    df_offset = None
    if offset_input:
        parsed = parse_offset_well(offset_input)
        if isinstance(parsed, pd.DataFrame):
            df_offset = parsed
            # Adjust TVDSS if only TVD provided
            if 'TVDSS' not in df_offset.columns:
                df_offset['TVDSS'] = df_offset['TVD'] - r_rkb # Approx
        elif parsed == "MISSING_COLS":
            st.warning("⚠️ Offset data must have columns: TVD, N, E")

    # --- C. ADVANCED DSS ANALYSIS ---
    
    # 1. Collision Analysis
    min_sep = 9999.0
    collision_depth = 0
    if df_offset is not None:
        # Simplifikasi perhitungan jarak titik-ke-titik terdekat
        # Menggunakan cdist atau simple loop broadcasting
        p1 = df_final[['N', 'E', 'TVD']].values
        p2 = df_offset[['N', 'E', 'TVD']].values
        
        # Cari jarak minimum
        # (Untuk performa, kita sampling jika data terlalu besar)
        dists = np.min(np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=2), axis=1)
        min_sep = np.min(dists)
        collision_depth = df_final.iloc[np.argmin(dists)]['MD']

    # 2. Engineering Limits
    max_inc = df_final['Inc'].max()
    max_dls_calc = df_final['DLS_Calc'].max()
    td_final = df_final['MD'].iloc[-1]

    # --- D. KPI METRICS (HEADER) ---
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f"Total Depth ({u_label})", f"{td_final:,.0f}")
    m2.metric("Target Azimuth", f"{meta['azi']:.2f}°")
    m3.metric("Max Inclination", f"{max_inc:.2f}°")
    m4.metric("Max DLS Actual", f"{max_dls_calc:.2f}", help=f"Maximum Dogleg in {dls_label}")
    
    # Logic Pewarnaan Status Collision
    sep_color = "normal"
    if min_sep < 10: sep_color = "inverse" # Merah jika streamlit theme default
    m5.metric("Min Separation", f"{min_sep:.1f} {u_label}" if df_offset is not None else "N/A", 
              delta="-CRITICAL" if min_sep < 10 else "Safe", delta_color=sep_color)

    # --- E. DSS RECOMMENDATION BOX ---
    st.markdown("### 🧠 Decision Support Analysis")
    
    col_dss1, col_dss2 = st.columns([2, 1])
    
    with col_dss1:
        # Rule Based System
        risk_msgs = []
        status = "APPROVED"
        color = "success"
        
        # Rule 1: Collision
        if min_sep < 10:
            risk_msgs.append(f"🔴 **COLLISION RISK:** Jarak ke Offset Well < 10 {u_label} di kedalaman {collision_depth:.0f}.")
            status = "REJECTED"
            color = "error"
        elif min_sep < 30:
            risk_msgs.append(f"🟠 **WARNING:** Jarak ke Offset Well cukup dekat (< 30 {u_label}).")
            if status != "REJECTED": status = "WARNING"
            if color != "error": color = "warning"
            
        # Rule 2: High DLS (Tortuosity)
        if max_dls_calc > (dls + 1.0): # Toleransi 1 derajat dari plan
            risk_msgs.append(f"🔴 **HIGH DLS:** Kelengkungan {max_dls_calc:.2f} melebihi batas aman operasional.")
            status = "REJECTED"
            color = "error"
            
        # Rule 3: Deepening Cost
        if ext_len > 1000:
            risk_msgs.append(f"🟠 **COST ALERT:** Extension > 1000 {u_label} akan meningkatkan biaya signfikan.")
            
        if status == "APPROVED":
            st.success(f"### ✅ STATUS: {status}\nPlan aman untuk dieksekusi.")
        elif status == "WARNING":
            st.warning(f"### ⚠️ STATUS: {status}\nPerlu mitigasi risiko.")
        else:
            st.error(f"### ⛔ STATUS: {status}\nPlan berbahaya. Revisi parameter diperlukan.")
            
        for msg in risk_msgs:
            st.markdown(msg)

    with col_dss2:
        st.info("💡 **Cost Estimation**")
        # Simple Cost Model
        cost_base = 250000 # Fixed
        cost_var = td_final * (150 if u_label == 'ft' else 500) # Variable per depth
        st.write(f"Estimated Cost: **${(cost_base + cost_var):,.0f}**")
        st.write(f"Rig Days: **{(td_final/500):.1f} Days**")

    # --- F. VISUALIZATION TABS ---
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["🌍 3D Interactive", "📐 Engineering View", "📋 Data Table"])

    # --- PLOTLY HELPERS ---
    def plot_casing_lines(fig, view_type='section'):
        # Parse Edited Casing DF
        if not edited_casing.empty:
            for index, row in edited_casing.iterrows():
                try:
                    c_size = row['Size']
                    c_depth = float(row['Depth'])
                    c_type = row['Type']
                    
                    # Convert to plotting depth
                    limit_md = c_depth
                    if c_type == 'TVDSS': # Convert TVDSS to MD approx
                        tvd_target = c_depth + r_rkb
                        # Find closest MD
                        idx = (df_final['TVD'] - tvd_target).abs().idxmin()
                        limit_md = df_final.loc[idx, 'MD']
                    
                    df_csg = df_final[df_final['MD'] <= limit_md]
                    if df_csg.empty: continue
                    
                    if view_type == 'section':
                        fig.add_trace(go.Scatter(
                            x=df_csg['VS'], y=df_csg['TVDSS'], mode='lines',
                            line=dict(color='black', width=5), opacity=0.5,
                            name=f"Csg {c_size}", hoverinfo='skip'
                        ))
                        # Shoe Marker
                        fig.add_trace(go.Scatter(
                            x=[df_csg['VS'].iloc[-1]], y=[df_csg['TVDSS'].iloc[-1]],
                            mode='markers+text', marker=dict(symbol='triangle-down', color='black', size=10),
                            text=[c_size], textposition='bottom center', name='Shoe'
                        ))
                except: pass

    with tab1:
        fig3d = go.Figure()
        
        # Hover Data setup
        hover_text = [f"MD: {m:.1f}<br>Inc: {i:.2f}<br>TVD: {t:.1f}" 
                      for m, i, t in zip(df_final['MD'], df_final['Inc'], df_final['TVD'])]
        
        # Main Well
        fig3d.add_trace(go.Scatter3d(
            x=df_final['E'], y=df_final['N'], z=df_final['TVDSS'],
            mode='lines', name='Planned Well',
            line=dict(color=df_final['Inc'], colorscale='Jet', width=5, colorbar=dict(title='Inc')),
            text=hover_text, hoverinfo='text'
        ))
        
        # Offset Well
        if df_offset is not None:
            fig3d.add_trace(go.Scatter3d(
                x=df_offset['E'], y=df_offset['N'], z=df_offset['TVDSS'],
                mode='lines', name='Offset Well',
                line=dict(color='red', width=3, dash='dash')
            ))
            
        # Target
        fig3d.add_trace(go.Scatter3d(
            x=[tgt_e], y=[tgt_n], z=[tgt_tvdss],
            mode='markers', name='Target', marker=dict(size=8, color='gold', symbol='diamond')
        ))

        fig3d.update_layout(
            scene=dict(
                xaxis_title=f'East ({u_label})', yaxis_title=f'North ({u_label})', zaxis_title=f'TVDSS ({u_label})',
                zaxis=dict(autorange="reversed"), aspectmode='data'
            ),
            height=700, margin=dict(l=0, r=0, b=0, t=0)
        )
        st.plotly_chart(fig3d, use_container_width=True)

    with tab2:
        col_plan, col_sec = st.columns(2)
        
        with col_plan:
            st.subheader("Plan View (N vs E)")
            fig_plan = go.Figure()
            fig_plan.add_trace(go.Scatter(x=df_final['E'], y=df_final['N'], mode='lines', name='Plan', line=dict(color='blue')))
            if df_offset is not None:
                fig_plan.add_trace(go.Scatter(x=df_offset['E'], y=df_offset['N'], mode='lines', name='Offset', line=dict(color='red', dash='dot')))
            
            fig_plan.add_trace(go.Scatter(x=[tgt_e], y=[tgt_n], mode='markers', marker=dict(size=12, color='gold', symbol='star'), name='Target'))
            fig_plan.update_layout(xaxis_title=f"East ({u_label})", yaxis_title=f"North ({u_label})", yaxis_scaleanchor="x", height=600)
            st.plotly_chart(fig_plan, use_container_width=True)
            
        with col_sec:
            st.subheader("Section View (VS vs TVDSS)")
            fig_sec = go.Figure()
            fig_sec.add_trace(go.Scatter(x=df_final['VS'], y=df_final['TVDSS'], mode='lines', name='Well Path', line=dict(color='blue')))
            
            # Plot Casing
            plot_casing_lines(fig_sec, 'section')
            
            # Plot Formation
            try:
                for line in form_text.split('\n'):
                    p = line.split(',')
                    nm, dp = p[0], float(p[1])
                    fig_sec.add_hline(y=dp, line_dash="dash", line_color="grey", annotation_text=nm)
            except: pass
            
            fig_sec.update_layout(xaxis_title=f"Vertical Section ({u_label})", yaxis_title=f"TVDSS ({u_label})", yaxis_autorange="reversed", height=600)
            st.plotly_chart(fig_sec, use_container_width=True)

    with tab3:
        st.dataframe(df_final)