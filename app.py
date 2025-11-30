# =============================================================================
# SKRIPSI DSS: MONITORING & ANALISIS TRAJEKTORI SUMUR (STREAMLIT APP)
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="DSS Drilling Monitor",
    page_icon="🛢️",
    layout="wide"
)

# --- JUDUL & HEADER ---
st.title("🛢️ DSS Trajectory Analysis & Monitoring System")
st.markdown("""
**Sistem Pendukung Keputusan** untuk visualisasi 3D dan deteksi dini anomali (Early Warning) 
pada trajektori pengeboran berarah.
""")

# --- SIDEBAR: INPUT USER ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload File Excel Trajectory", type=["xlsx", "csv"])

st.sidebar.header("2. Parameter Keselamatan")
# Input Threshold DLS (Batas Aman)
dls_limit = st.sidebar.slider("Batas Aman DLS (deg/30m)", 1.0, 5.0, 3.0, 0.1)

st.sidebar.info(
    """
    **Format Kolom Wajib:**
    - MD (Measured Depth)
    - Inc (Inclination)
    - Azi (Azimuth)
    - TVD (True Vertical Depth)
    - +N/S- (Koordinat Utara/Selatan)
    - +E/W- (Koordinat Timur/Barat)
    """
)

# --- FUNGSI HITUNG DOGLEG ---
def calculate_dls(df):
    md = df['MD'].values
    inc = np.radians(df['Inc'].values)
    azi = np.radians(df['Azi'].values)
    
    dls_list = [0]
    status_list = ["Normal"]
    
    for i in range(1, len(md)):
        L = md[i] - md[i-1]
        if L == 0:
            dls_list.append(0)
            status_list.append("Normal")
            continue
            
        # Rumus Minimum Curvature untuk DLS
        arg = np.sin(inc[i-1]) * np.sin(inc[i]) * np.cos(azi[i] - azi[i-1]) + \
              np.cos(inc[i-1]) * np.cos(inc[i])
        arg = np.clip(arg, -1.0, 1.0) # Safety clip
        
        dogleg_rad = np.arccos(arg)
        dls_val = np.degrees(dogleg_rad) * (30.0 / L) # deg/30m
        
        dls_list.append(dls_val)
        
        if dls_val > dls_limit:
            status_list.append("CRITICAL")
        elif dls_val > (dls_limit * 0.7):
            status_list.append("WARNING")
        else:
            status_list.append("Normal")
            
    df['DLS'] = dls_list
    df['Status'] = status_list
    
    # Hitung Vertical Section (Jarak Datar dari Pusat)
    df['VS'] = np.sqrt(df['+N/S-']**2 + df['+E/W-']**2)
    
    return df

# --- LOGIKA UTAMA APLIKASI ---
if uploaded_file is not None:
    try:
        # Load Data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        # Validasi Kolom
        required_cols = ['MD', 'Inc', 'Azi', 'TVD', '+N/S-', '+E/W-']
        if not all(col in df.columns for col in required_cols):
            st.error(f"File Excel tidak valid! Pastikan memiliki kolom: {', '.join(required_cols)}")
        else:
            # --- PROSES ANALISIS (ENGINE) ---
            df_processed = calculate_dls(df)
            
            # --- KPI DASHBOARD (METRICS) ---
            col1, col2, col3, col4 = st.columns(4)
            
            max_dls = df_processed['DLS'].max()
            total_depth = df_processed['MD'].max()
            critical_points = df_processed[df_processed['Status'] == 'CRITICAL'].shape[0]
            
            col1.metric("Total Depth (MD)", f"{total_depth:,.0f} ft")
            col2.metric("Max Inclination", f"{df_processed['Inc'].max():.2f}°")
            col3.metric("Max Dogleg Severity", f"{max_dls:.2f}°/30m", delta_color="inverse")
            col4.metric("Titik Kritis (Bahaya)", f"{critical_points} Titik", 
                        delta="-Bad" if critical_points > 0 else "normal")

            # --- TABS VISUALISASI ---
            tab1, tab2, tab3 = st.tabs(["📈 3D Visualization", "📐 2D Engineering View", "⚠️ Risk Analysis"])
            
            # TAB 1: 3D INTERAKTIF
            with tab1:
                st.subheader("Visualisasi 3D Trajektori Interaktif")
                
                # Warna jalur berdasarkan DLS
                fig_3d = px.scatter_3d(
                    df_processed, x='+E/W-', y='+N/S-', z='TVD',
                    color='DLS', color_continuous_scale='Jet',
                    hover_data=['MD', 'Inc', 'Azi', 'Status'],
                    size_max=5, opacity=0.8
                )
                
                # Update layout agar TVD (Depth) terbalik (0 di atas)
                fig_3d.update_layout(
                    scene=dict(
                        zaxis=dict(autorange="reversed"),
                        aspectmode='data'
                    ),
                    margin=dict(l=0, r=0, b=0, t=0),
                    height=600
                )
                st.plotly_chart(fig_3d, use_container_width=True)
                
            # TAB 2: 2D PLAN & SECTION VIEW
            with tab2:
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("### 🗺️ Plan View (Peta Atas)")
                    fig_plan = px.line(df_processed, x='+E/W-', y='+N/S-', markers=False)
                    fig_plan.add_scatter(x=[0], y=[0], mode='markers', marker=dict(size=12, color='red'), name='Surface')
                    fig_plan.update_layout(xaxis_title="East/West (ft)", yaxis_title="North/South (ft)", yaxis_scaleanchor="x")
                    st.plotly_chart(fig_plan, use_container_width=True)
                    
                with col_right:
                    st.markdown("### 📉 Section View (Tampak Samping)")
                    fig_sec = px.line(df_processed, x='VS', y='TVD', markers=False)
                    fig_sec.update_layout(yaxis_autorange="reversed", xaxis_title="Vertical Section (ft)", yaxis_title="TVD (ft)")
                    # Tambahkan Garis Casing (Contoh dummy logic, nanti bisa diinput)
                    st.plotly_chart(fig_sec, use_container_width=True)

            # TAB 3: ANALISIS RISIKO
            with tab3:
                st.subheader("Analisis Dogleg Severity (DLS)")
                
                # Grafik Garis DLS vs Depth
                fig_risk = go.Figure()
                fig_risk.add_trace(go.Scatter(x=df_processed['MD'], y=df_processed['DLS'], mode='lines', name='Actual DLS'))
                
                # Garis Batas Merah
                fig_risk.add_hrect(y0=dls_limit, y1=max(5, max_dls+1), line_width=0, fillcolor="red", opacity=0.1, annotation_text="Zona Bahaya")
                fig_risk.add_hline(y=dls_limit, line_dash="dash", line_color="red", annotation_text=f"Limit: {dls_limit}")
                
                fig_risk.update_layout(xaxis_title="Measured Depth (ft)", yaxis_title="DLS (deg/30m)")
                st.plotly_chart(fig_risk, use_container_width=True)
                
                # Tabel Data Risiko
                st.markdown("### 📋 Daftar Titik Anomali (High Risk)")
                risk_df = df_processed[df_processed['Status'] == 'CRITICAL'][['MD', 'Inc', 'Azi', 'TVD', 'DLS', 'Status']]
                
                if not risk_df.empty:
                    st.error(f"Terdeteksi {len(risk_df)} titik dengan DLS melebihi batas {dls_limit} deg/30m!")
                    st.dataframe(risk_df.style.format("{:.2f}"))
                else:
                    st.success("Tidak ada anomali terdeteksi. Trajektori Aman.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
        st.info("Pastikan format Excel sesuai dengan standar Landmark/Compass.")

else:
    # Tampilan Awal (Belum Upload)
    st.info("👆 Silakan upload file Excel Trajektori pada panel sebelah kiri.")
    st.write("---")
    st.markdown("### Fitur Sistem:")
    st.markdown("""
    1.  **Visualisasi 3D Otomatis**: Mengubah data angka menjadi model sumur interaktif.
    2.  **Deteksi DLS**: Menghitung *Dogleg Severity* secara otomatis di setiap titik kedalaman.
    3.  **Risk Monitoring**: Memberikan peringatan jika kelengkungan sumur melebihi batas aman.
    4.  **Standard View**: Menampilkan Plan View dan Section View sesuai standar industri.
    """)