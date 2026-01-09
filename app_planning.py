import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize_scalar, minimize
from io import StringIO
import uuid
import json
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="WellDesign Pro | Cloud", layout="wide", page_icon="🏗️")

# ==========================================
# 1. ADVANCED ENGINE (MATH CORE)
# ==========================================
class RiskEngine:
    """
    Handles Anti-Collision, Error Modelling (Covariance), and Separation Factors.
    Simplified ISCWSA Model: Linear error growth for demo purposes.
    """
    def __init__(self):
        # Error factors (1 sigma)
        self.error_lateral = 0.005 # 5m error per 1000m MD
        self.error_highside = 0.003
        self.error_along = 0.001
        self.sf_threshold = 1.5

    def calculate_uncertainty(self, df):
        """Calculates 3x3 Covariance Matrix for every point"""
        cov_matrices = []
        ellipses = []
        
        for idx, row in df.iterrows():
            md, inc, azi = row['MD'], np.radians(row['Inc']), np.radians(row['Azi'])
            
            # Local instrument errors (H, L, A) converted to NEV (North, East, Vertical)
            # Simplified rotation matrix for demo (Full ISCWSA is complex)
            
            sigma_l = self.error_lateral * md
            sigma_h = self.error_highside * md
            sigma_a = self.error_along * md
            
            # Covariance Matrix (Local)
            cov_local = np.diag([sigma_h**2, sigma_l**2, sigma_a**2])
            
            # Rotation (Alignment) Matrix
            # This rotates Highside/Lateral/Along to North/East/Vertical
            Rb = np.array([
                [np.cos(inc)*np.cos(azi), -np.sin(azi),  np.sin(inc)*np.cos(azi)],
                [np.cos(inc)*np.sin(azi),  np.cos(azi),  np.sin(inc)*np.sin(azi)],
                [-np.sin(inc),             0,            np.cos(inc)]
            ])
            
            # Covariance (Global) = R * Cov_Local * R.T
            cov_global = Rb @ cov_local @ Rb.T
            
            cov_matrices.append(cov_global.tolist())
            
            # Ellipse Radii (Major axis) for visualization
            ellipses.append(np.sqrt(np.diag(cov_global)) * 2) # 2 Sigma scaling
            
        df['Covariance'] = cov_matrices
        df['Error_Rad'] = [e[1] for e in ellipses] # Use Lateral error for simple radius
        return df

    def scan_collision(self, plan_df, offset_dfs):
        """Calculates Center-to-Center (CtC) and Separation Factor (SF)"""
        risks = []
        if plan_df.empty: return risks
        
        # Downsample for speed
        p_path = plan_df[['N', 'E', 'TVD', 'Error_Rad']].values[::5] # Check every 5th point
        
        for off in offset_dfs:
            if not off['show']: continue
            o_path = off['df'][['N', 'E', 'TVD']].values
            
            min_sf = 99.0
            min_dist = 9999.0
            crit_md = 0
            
            # Brute force check (KDTree would be better for production)
            for i, p_pt in enumerate(p_path):
                # Distance to all points in offset
                dists = np.linalg.norm(o_path - p_pt[:3], axis=1)
                d_min = np.min(dists)
                
                # Simple SF: Distance / (Radius1 + Radius2)
                # Assuming Offset has similar error radius to Plan at that depth
                sf = d_min / (p_pt[3] * 2) # *2 assumes offset has same error
                
                if sf < min_sf:
                    min_sf = sf
                    min_dist = d_min
                    crit_md = plan_df.iloc[i*5]['MD']
            
            risks.append({
                'Offset': off['name'],
                'Min_SF': min_sf,
                'Min_Dist': min_dist,
                'At_MD': crit_md,
                'Status': 'CRITICAL' if min_sf < self.sf_threshold else 'Safe'
            })
        return pd.DataFrame(risks)

class SmartPlanner:
    def __init__(self, surf_n, surf_e, rkb, unit_system='Metric'):
        self.surf_n = surf_n
        self.surf_e = surf_e
        self.rkb = rkb
        self.unit = unit_system
        self.ft_to_m = 0.3048
        self.dls_ref = 30.0 if unit_system == 'Metric' else 100.0
        self.engine = DrillingEngine(unit_system, self.dls_ref)
        
    def solve_j_profile(self, target_n, target_e, target_tvdss, kop, dls, force_hold=None):
        # ... (Same as before, abbreviated for brevity) ...
        # NOTE: In full code, paste the previous solve_trajectory logic here.
        # For this upgraded snippet, I will include the NEW S-Profile logic.
        return self._solve_generic_profile('J', target_n, target_e, target_tvdss, kop, dls, force_hold)

    def solve_s_profile(self, target_n, target_e, target_tvdss, kop, dls, build_hold_angle, drop_dls):
        """S-Profile: Vertical -> Build -> Tangent -> Drop -> Vertical(Target)"""
        # Simplified S-Profile Solver for Demo
        # 1. Calculate geometry to hit target
        # This is a complex geometric solver, simplified here to:
        # Build to angle X, Hold for Y, Drop to vertical at Target
        
        # Placeholder for complex solver:
        # We will create a path that builds, holds 500m, then drops.
        
        # Reuse Generic Solver for the Build Section
        df, azi, hold = self._solve_generic_profile('S', target_n, target_e, target_tvdss, kop, dls, build_hold_angle)
        
        # Add Drop Section
        last = df.iloc[-1]
        drop_len = (last['Inc'] / drop_dls) * 30.0
        
        # Extend with Drop
        return self.engine.calculate_drop_section(df, drop_dls, drop_len)

    def _solve_generic_profile(self, type, t_n, t_e, t_tvdss, kop, dls, hold_val):
        # Logic from previous response goes here to generate J-Curve
        # Re-implementing core J-Curve for completeness of this file
        
        # Unit Conversion
        u_mult = self.ft_to_m if self.unit == 'Imperial' else 1.0
        surf_n_m, surf_e_m = self.surf_n * u_mult, self.surf_e * u_mult
        tgt_n_m, tgt_e_m = t_n * u_mult, t_e * u_mult
        tgt_tvd_m = (t_tvdss * u_mult) + (self.rkb * u_mult)
        kop_m = kop * u_mult
        dls_m = dls * (30.0 / (100.0 * self.ft_to_m)) if self.unit == 'Imperial' else dls

        delta_n = tgt_n_m - surf_n_m
        delta_e = tgt_e_m - surf_e_m
        tgt_azi = np.degrees(np.arctan2(delta_e, delta_n)) % 360
        target_hd = np.sqrt(delta_n**2 + delta_e**2)
        
        # Solver
        def error_func(h_ang):
            rad = np.radians(h_ang)
            r = (180/np.pi) * (30.0/dls_m)
            build_tvd = kop_m + (r * np.sin(rad))
            build_hd = r * (1 - np.cos(rad))
            rem_tvd = tgt_tvd_m - build_tvd
            if rem_tvd < 0: return 1e6
            rem_hd = rem_tvd * np.tan(rad)
            return abs((build_hd + rem_hd) - target_hd)

        if hold_val and hold_val > 0: best_hold = hold_val
        else:
            res = minimize_scalar(error_func, bounds=(0, 90), method='bounded')
            best_hold = res.x
            
        # Generate Points
        mds, incs, azis = [0], [0], [tgt_azi]
        # Vert
        mds.append(kop_m); incs.append(0); azis.append(tgt_azi)
        # Build
        r = (180/np.pi) * (30.0/dls_m)
        blen = np.radians(best_hold) * r
        mds.append(kop_m + blen); incs.append(best_hold); azis.append(tgt_azi)
        # Hold
        tvd_eob = kop_m + (r * np.sin(np.radians(best_hold)))
        rem_tvd = tgt_tvd_m - tvd_eob
        hold_len = rem_tvd / np.cos(np.radians(best_hold))
        mds.append(mds[-1] + hold_len); incs.append(best_hold); azis.append(tgt_azi)
        
        df = self.engine.calculate_trajectory(mds, incs, azis, surf_n_m, surf_e_m, 0)
        
        # Convert back if Imperial
        if self.unit == 'Imperial':
            for c in ['MD','TVD','N','E','VS']: df[c] /= self.ft_to_m
            
        df['TVDSS'] = df['TVD'] - self.rkb
        df['Section'] = 'Plan'
        return df, tgt_azi, best_hold


class DrillingEngine:
    def __init__(self, unit, dls_ref):
        self.unit = unit
        self.dls_ref = dls_ref
    
    def calculate_trajectory(self, md, inc, azi, start_n, start_e, start_tvd):
        # Minimum Curvature Method (High Performance)
        # Interpolate between key points for smoothness (every 10m)
        full_md, full_inc, full_azi = [], [], []
        step = 10.0
        
        for i in range(1, len(md)):
            d_md = md[i] - md[i-1]
            n_steps = int(d_md / step)
            if n_steps < 1: n_steps = 1
            full_md.extend(np.linspace(md[i-1], md[i], n_steps, endpoint=False))
            full_inc.extend(np.linspace(inc[i-1], inc[i], n_steps, endpoint=False))
            # Handle Azimuth crossover
            a1, a2 = azi[i-1], azi[i]
            if abs(a2-a1) > 180: 
                if a2 > a1: a1 += 360
                else: a2 += 360
            azis_interp = np.linspace(a1, a2, n_steps, endpoint=False) % 360
            full_azi.extend(azis_interp)
            
        # Add last point
        full_md.append(md[-1]); full_inc.append(inc[-1]); full_azi.append(azi[-1])
        
        # Calculation Loop
        n = len(full_md)
        tvd, n_c, e_c, dls = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        tvd[0], n_c[0], e_c[0] = start_tvd, start_n, start_e
        
        inc_r, azi_r = np.radians(full_inc), np.radians(full_azi)
        
        for i in range(1, n):
            dL = full_md[i] - full_md[i-1]
            I1, I2 = inc_r[i-1], inc_r[i]
            A1, A2 = azi_r[i-1], azi_r[i]
            
            cos_beta = np.cos(I2-I1) - (np.sin(I1)*np.sin(I2)*(1-np.cos(A2-A1)))
            beta = np.arccos(np.clip(cos_beta, -1, 1))
            rf = (2/beta)*np.tan(beta/2) if abs(beta)>1e-6 else 1.0
            
            n_c[i] = n_c[i-1] + (dL/2)*(np.sin(I1)*np.cos(A1) + np.sin(I2)*np.cos(A2))*rf
            e_c[i] = e_c[i-1] + (dL/2)*(np.sin(I1)*np.sin(A1) + np.sin(I2)*np.sin(A2))*rf
            tvd[i] = tvd[i-1] + (dL/2)*(np.cos(I1) + np.cos(I2))*rf
            
            dls[i] = np.degrees(beta) * (self.dls_ref / dL) if dL > 0 else 0
            
        df = pd.DataFrame({'MD': full_md, 'Inc': full_inc, 'Azi': full_azi, 'TVD': tvd, 'N': n_c, 'E': e_c, 'DLS': dls})
        df['VS'] = np.sqrt((df['N']-start_n)**2 + (df['E']-start_e)**2)
        return df

    def calculate_drop_section(self, df_curr, dls, length):
        # Append a drop section to existing DF
        last = df_curr.iloc[-1]
        step = 10.0
        n_steps = int(length / step)
        
        new_md = [last['MD'] + (i+1)*step for i in range(n_steps)]
        drop_per_m = dls / 30.0
        new_inc = [max(0, last['Inc'] - (i+1)*step*drop_per_m) for i in range(n_steps)]
        new_azi = [last['Azi']] * n_steps
        
        # Calculate coords for extension
        df_ext = self.calculate_trajectory(
            [last['MD']] + new_md, 
            [last['Inc']] + new_inc, 
            [last['Azi']] + new_azi,
            last['N'], last['E'], last['TVD']
        )
        return pd.concat([df_curr, df_ext.iloc[1:]], ignore_index=True)

# ==========================================
# 2. UTILS & PARSING
# ==========================================
def parse_csv(text):
    try:
        df = pd.read_csv(StringIO(text), sep=None, engine='python')
        df.columns = df.columns.str.upper().str.strip()
        # Normalization logic here...
        rename_map = {'MEASURED DEPTH':'MD', 'INCLINATION':'Inc', 'AZIMUTH':'Azi', 'TRUE VERTICAL DEPTH':'TVD', 'NORTH':'N', 'EAST':'E'}
        for k,v in rename_map.items():
            cols = [c for c in df.columns if k in c]
            if cols: df.rename(columns={cols[0]: v}, inplace=True)
        return df
    except: return None

# ... [Keep Imports and Classes RiskEngine, SmartPlanner, DrillingEngine from previous code] ...

# ==========================================
# 3. UI & APP LOGIC (ENHANCED)
# ==========================================
if 'layers' not in st.session_state: st.session_state['layers'] = {}
risk_engine = RiskEngine()

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2061/2061168.png", width=50)
st.sidebar.title("💎 WellDesign Clone")

# --- SIDEBAR: PROJECT TREE (Like Image 4) ---
with st.sidebar:
    st.markdown("### 🗂️ Project Explorer")
    with st.expander("🏗️ Site: Alpha Pad", expanded=True):
        st.caption("Coordinate Ref: UTM Zone 31N")
        
        # Target Configuration (Like Image 2)
        st.markdown("**🎯 Target Definition**")
        tgt_tvd = st.number_input("Target TVD", value=2200.0)
        c1, c2 = st.columns(2)
        tol_v = c1.number_input("Vert Tol (+/-)", value=10.0)
        tol_h = c2.number_input("Horiz Tol (+/-)", value=25.0)
        
        # Formation Config (Like Image 2)
        st.markdown("**🪨 Geological Tops**")
        form_top = st.number_input("Reservoir Top TVD", value=2150.0)
        
        # Visibility Toggles (Like Image 4)
        st.markdown("**👁️ Visibility**")
        show_plan = st.checkbox("Show Plan", True)
        show_uncert = st.checkbox("Show Uncertainty", True)
        
    st.divider()
    
    # Parametric Inputs
    st.subheader("📐 Trajectory Parameters")
    profile_type = st.selectbox("Profile", ["S-Type", "J-Type"])
    kop = st.number_input("KOP", 500.0)
    dls = st.number_input("Build DLS", 3.0)
    tgt_n = 9000400.0; tgt_e = 400400.0 # Fixed for demo
    
    if st.button("✨ CALC TRAJECTORY", type="primary"):
        planner = SmartPlanner(9000000, 400000, 25, 'Metric')
        # Simple Logic switch for demo
        df, _, _ = planner.solve_j_profile(tgt_n, tgt_e, tgt_tvd, kop, dls)
        df = risk_engine.calculate_uncertainty(df)
        st.session_state['layers']['Plan'] = {'df': df, 'color': '#FF6D00', 'show': True} # Orange like Image 2

# ==========================================
# 4. MAIN DASHBOARD (POWERFUL VIEWS)
# ==========================================
from plotly.subplots import make_subplots

# Custom CSS to mimic the clean look of Image 3
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #ddd; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# TABS matching the workflow in Image 3 (Trajectory, Editor, Reports)
tab_3d, tab_eng, tab_data = st.tabs(["🌍 3D Digital Twin", "📈 Engineering Logs", "📋 Survey Data"])

with tab_3d:
    # 3D View imitating Image 1 & 2
    fig = go.Figure()
    
    if 'Plan' in st.session_state['layers'] and show_plan:
        df = st.session_state['layers']['Plan']['df']
        
        # 1. The Well Path (Orange Tube style)
        fig.add_trace(go.Scatter3d(
            x=df['E'], y=df['N'], z=df['TVD'],
            mode='lines', name='Active Wellbore',
            line=dict(color='#FF6D00', width=8), # Oliasoft Orange
            hovertemplate="MD: %{text}<br>TVD: %{z}", text=df['MD']
        ))
        
        # 2. Target Box (The "Green Zone" from Image 2)
        # Create a 3D box around the last point
        last = df.iloc[-1]
        x_c, y_c, z_c = last['E'], last['N'], last['TVD']
        
        # Mesh Cube Logic
        x_box = [x_c-tol_h, x_c+tol_h, x_c+tol_h, x_c-tol_h, x_c-tol_h, x_c+tol_h, x_c+tol_h, x_c-tol_h]
        y_box = [y_c-tol_h, y_c-tol_h, y_c+tol_h, y_c+tol_h, y_c-tol_h, y_c-tol_h, y_c+tol_h, y_c+tol_h]
        z_box = [z_c-tol_v, z_c-tol_v, z_c-tol_v, z_c-tol_v, z_c+tol_v, z_c+tol_v, z_c+tol_v, z_c+tol_v]
        
        fig.add_trace(go.Mesh3d(
            x=x_box, y=y_box, z=z_box,
            i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color='rgba(0, 255, 0, 0.3)', # Translucent Green
            name='Target Tolerance', flatshading=True
        ))
        
        # 3. Formation Plane (The Yellow Layers from Image 2)
        center_n, center_e = df['N'].mean(), df['E'].mean()
        fig.add_trace(go.Mesh3d(
            x=[center_e-1000, center_e+1000, center_e+1000, center_e-1000],
            y=[center_n-1000, center_n-1000, center_n+1000, center_n+1000],
            z=[form_top, form_top, form_top, form_top],
            color='orange', opacity=0.4, name='Top Reservoir'
        ))
        
        # 4. Uncertainty "Clouds" (Image 4)
        if show_uncert and 'Error_Rad' in df.columns:
            sub = df.iloc[::15] # Downsample
            fig.add_trace(go.Scatter3d(
                x=sub['E'], y=sub['N'], z=sub['TVD'],
                mode='markers', 
                marker=dict(size=sub['Error_Rad'], color='gray', opacity=0.2, symbol='circle'),
                name='Cone of Uncertainty'
            ))

    # Grid & Layout mimicking Image 1
    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="#f0f0f0", gridcolor="white", title="Easting (m)"),
            yaxis=dict(backgroundcolor="#f0f0f0", gridcolor="white", title="Northing (m)"),
            zaxis=dict(backgroundcolor="#e0e0e0", gridcolor="white", title="TVD (m)", autorange="reversed"),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=700
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_eng:
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

with tab_data:
    st.dataframe(st.session_state['layers'].get('Plan', {}).get('df', pd.DataFrame()), use_container_width=True)