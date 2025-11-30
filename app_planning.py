import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize_scalar
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="DSS Well Master", layout="wide", page_icon="🏗️")

# ==========================================
# 1. ENGINE & LOGIC (BACKEND SYSTEM)
# ==========================================
class SmartPlanner:
    def __init__(self, surf_n, surf_e, rkb):
        self.surf_n = surf_n
        self.surf_e = surf_e
        self.rkb = rkb
        
    def solve_trajectory(self, target_n, target_e, target_tvdss, kop, dls):
        # 1. Hitung Delta Geometri
        delta_n = target_n - self.surf_n
        delta_e = target_e - self.surf_e
        target_tvd = target_tvdss + self.rkb
        
        tgt_azi_rad = np.arctan2(delta_e, delta_n)
        tgt_azi_deg = np.degrees(tgt_azi_rad) % 360
        target_hd = np.sqrt(delta_n**2 + delta_e**2)
        
        # 2. Solver Matematika (Mencari Sudut Hold Terbaik)
        def error_func(hold_angle):
            rad = np.radians(hold_angle)
            radius = (180/np.pi) * (30/dls)
            # Build Section
            build_tvd = kop + (radius * np.sin(rad))
            build_hd = radius * (1 - np.cos(rad))
            # Hold Section
            rem_tvd = target_tvd - build_tvd
            if rem_tvd < 0: return 1e6
            rem_hd = rem_tvd * np.tan(rad)
            return abs((build_hd + rem_hd) - target_hd)

        res = minimize_scalar(error_func, bounds=(0, 90), method='bounded')
        if not res.success: return None, 0, 0
        
        return self._generate_path(kop, dls, res.x, tgt_azi_deg, target_tvd)

    def _generate_path(self, kop, dls, hold_inc, azi, target_tvd):
        # Resolusi diperhalus (step=5m) agar visualisasi tidak kaku/patah-patah
        step = 5.0 
        path = []
        md, tvd, hd, inc = 0, 0, 0, 0
        radius = (180/np.pi) * (30/dls)
        inc_step = (dls/30.0) * step
        
        # Simulasi Pengeboran
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
        df['TVDSS'] = df['TVD'] - self.rkb
        return df, azi, hold_inc

# Fungsi Estimasi Biaya & Waktu (Fitur IS / Manajemen)
def calculate_economics(df):
    total_md = df['MD'].iloc[-1]
    # Asumsi Biaya & Kecepatan (Business Logic)
    cost_per_meter = 1500 # USD/meter
    rop_avg = 10 # meter/jam (Rate of Penetration)
    rig_cost_per_day = 50000 # USD/day
    
    drilling_hours = total_md / rop_avg
    drilling_days = drilling_hours / 24
    
    service_cost = total_md * cost_per_meter
    rig_cost = drilling_days * rig_cost_per_day
    total_cost = service_cost + rig_cost
    
    return total_cost, drilling_days

# ==========================================
# 2. UI SIDEBAR DENGAN FORM (Mencegah Auto-Run)
# ==========================================
st.sidebar.title("🎛️ Control Panel")

# --- INI KUNCI PERBAIKANNYA: st.form ---
with st.sidebar.form("planning_form"):
    st.subheader("1. Surface Configuration")
    r_floor = st.number_input("Rotary Table (m)")
    r_elev = st.number_input("Cellar Elev (m)")
    r_rkb = r_floor + r_elev
    surf_n = st.number_input("Surface N (Y)")
    surf_e = st.number_input("Surface E (X)")

    st.subheader("2. Target Coordinates")
    tgt_n = st.number_input("Target N (Y)")
    tgt_e = st.number_input("Target E (X)")
    tgt_tvdss = st.number_input("Target TVDSS (Z)")

    st.subheader("3. Engineering Params")
    kop = st.number_input("KOP (mMD)")
    dls = st.slider("DLS (deg/30m)")

    st.subheader("4. Risk Simulation (Anti-Collision)")
    offset_dist = st.slider("Jarak Sumur Tetangga (m)")
    
    st.markdown("---")
    # Tombol Submit di dalam Form
    submitted = st.form_submit_button("🚀 GENERATE PLAN", type="primary")

# ==========================================
# 3. MAIN DASHBOARD LOGIC
# ==========================================
st.title("🏗️ DSS Well Planning & Risk Analyzer")
st.markdown("Sistem Pendukung Keputusan untuk perencanaan sumur otomatis, estimasi biaya, dan analisis risiko tabrakan.")

# Inisialisasi Session State agar data tidak hilang saat refresh visual
if 'data_generated' not in st.session_state:
    st.session_state['data_generated'] = False

# Jika Tombol Ditekan -> Lakukan Kalkulasi
if submitted:
    planner = SmartPlanner(surf_n, surf_e, r_rkb)
    df_plan, azi, hold = planner.solve_trajectory(tgt_n, tgt_e, tgt_tvdss, kop, dls)
    
    if df_plan is not None:
        # Simpan hasil ke Session State
        st.session_state['df_plan'] = df_plan
        st.session_state['azi'] = azi
        st.session_state['hold'] = hold
        st.session_state['offset_dist'] = offset_dist
        st.session_state['surf_n'] = surf_n
        st.session_state['surf_e'] = surf_e
        st.session_state['tgt_n'] = tgt_n
        st.session_state['tgt_e'] = tgt_e
        st.session_state['tgt_tvdss'] = tgt_tvdss
        st.session_state['r_rkb'] = r_rkb
        st.session_state['data_generated'] = True
    else:
        st.error("❌ TARGET TIDAK TERJANGKAU! Sudut Build (DLS) terlalu kecil. Mohon naikkan DLS.")
        st.session_state['data_generated'] = False

# --- RENDER VISUALISASI (Hanya jika data sudah ada di Session State) ---
if st.session_state['data_generated']:
    
    # Ambil data dari Session State
    df_plan = st.session_state['df_plan']
    azi = st.session_state['azi']
    hold = st.session_state['hold']
    dist_val = st.session_state['offset_dist']
    
    # --- A. BUSINESS LOGIC (COST & TIME) ---
    cost_est, time_est = calculate_economics(df_plan)

    # --- B. ANTI-COLLISION LOGIC ---
    df_offset = df_plan.copy()
    df_offset['N'] = df_offset['N'] + (dist_val * 0.7) 
    df_offset['E'] = df_offset['E'] + (dist_val * 0.7)
    noise = np.linspace(0, 5, len(df_offset))
    df_offset['N'] += noise

    dist_vector = np.sqrt((df_plan['N']-df_offset['N'])**2 + (df_plan['E']-df_offset['E'])**2 + (df_plan['TVD']-df_offset['TVD'])**2)
    min_sep = dist_vector.min()

    # --- C. DECISION SUPPORT PANEL (KPI) ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target Azimuth", f"{azi:.2f}°", "Direction")
    c2.metric("Est. Total Cost", f"${cost_est/1000:.1f} K", "USD")
    c3.metric("Est. Drilling Days", f"{time_est:.1f} Days", "Time")
    c4.metric("Min Separation", f"{min_sep:.1f} m", 
              "-CRITICAL" if min_sep < 10 else "Safe", 
              delta_color="inverse")

    # --- D. ALERT SYSTEM ---
    if min_sep < 10:
        st.error(f"⛔ **KEPUTUSAN: DILARANG MENGEBOR!** Risiko tabrakan sangat tinggi (Jarak < 10m).")
    elif min_sep < 20:
        st.warning(f"⚠️ **KEPUTUSAN: WASPADA.** Jarak antar sumur dekat ({min_sep:.1f}m).")
    else:
        st.success(f"✅ **KEPUTUSAN: APPROVE.** Rencana pengeboran aman dan efisien.")

    # --- E. VISUALISASI ---
    tab1, tab2, tab3 = st.tabs(["🌍 3D Interactive", "📐 Engineering View", "📊 Data Table"])

    with tab1:
        fig3d = go.Figure()
        
        # Setup Hover Data
        hover_data = np.stack((
            df_plan['MD'], df_plan['Inc'], df_plan['Azi'], df_plan['Section'], df_plan['TVD']
        ), axis=-1)
        
        # 1. Planned Well
        fig3d.add_trace(go.Scatter3d(
            x=df_plan['E'], y=df_plan['N'], z=df_plan['TVDSS'],
            mode='markers+lines', name='Planned Well',
            line=dict(color='gray', width=2), 
            customdata=hover_data,
            hovertemplate="<b>MD: %{customdata[0]:.2f}</b><br>Inc: %{customdata[1]:.2f}°<br>Status: %{customdata[3]}<extra></extra>",
            marker=dict(size=4, color=df_plan['Inc'], colorscale='Jet', showscale=True, colorbar=dict(title="Inc"))
        ))
        
        # 2. Offset Well
        fig3d.add_trace(go.Scatter3d(
            x=df_offset['E'], y=df_offset['N'], z=df_offset['TVDSS'],
            mode='lines', name='Offset Well',
            line=dict(color='red', width=5, dash='dot'), hoverinfo='name'
        ))
        
        # 3. Markers
        fig3d.add_trace(go.Scatter3d(x=[st.session_state['surf_e']], y=[st.session_state['surf_n']], z=[-st.session_state['r_rkb']], mode='markers', name='Surface', marker=dict(size=8, color='black')))
        fig3d.add_trace(go.Scatter3d(x=[st.session_state['tgt_e']], y=[st.session_state['tgt_n']], z=[st.session_state['tgt_tvdss']], mode='markers', name='Target', marker=dict(size=10, color='gold', symbol='diamond')))

        fig3d.update_layout(
            scene=dict(xaxis_title='E', yaxis_title='N', zaxis_title='TVDSS', zaxis=dict(autorange="reversed"), aspectmode='data'),
            height=700, margin=dict(l=0, r=0, b=0, t=0)
        )
        st.plotly_chart(fig3d, use_container_width=True)

    with tab2:
        c_left, c_right = st.columns(2)
        with c_left:
            st.subheader("Plan View (Peta)")
            fig_plan = go.Figure()
            fig_plan.add_trace(go.Scatter(x=df_plan['E'], y=df_plan['N'], mode='lines', name='Plan', line=dict(color='blue')))
            fig_plan.add_trace(go.Scatter(x=df_offset['E'], y=df_offset['N'], mode='lines', name='Offset', line=dict(color='red', dash='dot')))
            fig_plan.add_trace(go.Scatter(x=[st.session_state['tgt_e']], y=[st.session_state['tgt_n']], mode='markers', marker=dict(size=15, color='gold', symbol='star'), name='Target'))
            fig_plan.update_layout(xaxis_title="East", yaxis_title="North", height=500, yaxis_scaleanchor="x")
            st.plotly_chart(fig_plan, use_container_width=True)
            
        with c_right:
            st.subheader("Section View (Samping)")
            fig_sec = go.Figure()
            fig_sec.add_trace(go.Scatter(x=df_plan['VS'], y=df_plan['TVDSS'], mode='lines', name='Plan', line=dict(color='blue', width=3)))
            
            # Casing Dummy
            casing_depth = 1000 
            df_csg = df_plan[df_plan['MD'] < casing_depth]
            
            # FIXED LINE OPACITY SYNTAX HERE
            fig_sec.add_trace(go.Scatter(
                x=df_csg['VS'], y=df_csg['TVDSS'], mode='lines', name='Casing', 
                opacity=0.5, line=dict(color='black', width=6)
            ))
            
            fig_sec.update_layout(xaxis_title="Vertical Section", yaxis_title="TVDSS", yaxis_autorange="reversed", height=500)
            st.plotly_chart(fig_sec, use_container_width=True)

    with tab3:
        st.dataframe(df_plan)

else:
    st.info("👈 Silakan masukkan parameter di sebelah kiri dan tekan tombol **GENERATE PLAN**.")