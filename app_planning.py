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
import xml.etree.ElementTree as ET

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
    Parser Berjenjang dengan LOOKUP Logic & SORTING:
    1. Scan CD_SCENARIO & CD_CASE.
    2. Scan MB_ASSEMBLY_COMP (buat kamus Referensi Nama).
    3. Scan CD_ASSEMBLY_COMP (simpan sementara + ambil sequence_no).
    4. Gabungkan (Join) Nama.
    5. SORTING berdasarkan Sequence Number.
    """
    try:
        xml_file.seek(0)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 1. WADAH DATA
        scenarios = {}      
        cases = []          
        
        # Temp Storage
        raw_cd_comps = []   
        mb_lookup = {}      # Kamus {assembly_comp_id: Description_Text}
        
        # 2. SCANNING XML
        for child in root.findall(".//*"):
            tag = child.tag.split('}')[-1].upper()
            attr = {k.lower(): v for k, v in child.attrib.items()}
            
            # A. SCENARIO
            if tag == 'CD_SCENARIO':
                sid = attr.get('scenario_id')
                sname = attr.get('name', attr.get('scenario_name', sid))
                if sid: scenarios[sid] = sname
            
            # B. CASE
            elif tag == 'CD_CASE':
                sid_ref = attr.get('scenario_id')
                aid_ref = attr.get('assembly_id')
                cname = attr.get('case_name', 'Unnamed Case')
                desc = attr.get('case_description', '')
                
                if "Auto created" in desc: continue
                if "Casing String" in cname and "Run" not in cname: continue

                if sid_ref and aid_ref:
                    cases.append({
                        'scenario_id': sid_ref,
                        'case_name': cname,
                        'assembly_id': aid_ref
                    })

            # C. MASTER TABLE (MB_) UNTUK LOOKUP NAMA
            elif tag == 'MB_ASSEMBLY_COMP':
                mb_id = attr.get('assembly_comp_id')
                mb_desc = attr.get('description')
                if mb_id and mb_desc:
                    mb_lookup[mb_id] = mb_desc

            # D. DATA KOMPONEN MENTAH (CD_)
            elif tag in ['CD_ASSEMBLY_COMP', 'CD_ASSEMBLY_COMPONENT', 'CD_DRILL_STRING_COMP']:
                if attr.get('assembly_id'):
                    # --- NEW: AMBIL SEQUENCE NO ---
                    # Kita ubah ke integer agar urutannya benar (1, 2, 10 bukan 1, 10, 2)
                    try:
                        seq = int(attr.get('sequence_no', 0))
                    except ValueError:
                        seq = 0 # Default jika error/kosong
                    
                    # Simpan sequence ke dalam dictionary raw
                    attr['parsed_seq'] = seq 
                    raw_cd_comps.append(attr)

        # 3. PROSES PENGGABUNGAN & FINISHING
        final_components = []
        
        for attr in raw_cd_comps:
            # Logic Lookup Nama (MB Table)
            link_id = attr.get('assembly_comp_id')
            if link_id and link_id in mb_lookup:
                display_name = mb_lookup[link_id] 
            else:
                display_name = attr.get('catalog_key_desc', attr.get('component_name', '-'))

            final_components.append({
                'assembly_id': attr.get('assembly_id'),
                'Sequence': attr['parsed_seq'],  # <--- Field Baru untuk Sorting
                'Description': display_name,
                'Connection': attr.get('connection_name', attr.get('connection_type', '-')),
                'OD (in)': float(attr.get('od_body', attr.get('body_od', attr.get('od', 0)))),
                'ID (in)': float(attr.get('id_body', attr.get('body_id', attr.get('id', 0)))),
                'Length': float(attr.get('element_length', attr.get('length', 0))) 
            })

        # Konversi ke DataFrame
        df_comps = pd.DataFrame(final_components)
        
        # 4. SORTING LOGIC
        if not df_comps.empty:
            # Urutkan berdasarkan Assembly ID dulu, baru Sequence Number
            df_comps = df_comps.sort_values(by=['assembly_id', 'Sequence'], ascending=[True, True])

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
            active = [o for o in st.session_state['layers']['Offsets'].values() if o['show']]
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
        st.markdown(f"""
        <div style="padding:15px; background-color:#e6fffa; border-left:5px solid #00b894; border-radius:5px; margin-bottom:20px;">
            <h4 style="margin:0; color:#00695c;">💡 AI Prescriptive Correction</h4>
            <p style="margin:0;">Steering Command: <b>{'BUILD' if p['bur']>0 else 'DROP'} {abs(p['bur']):.2f} {dls_label}</b> | 
            <b>TURN {'RIGHT' if p['turn']>0 else 'LEFT'} {abs(p['turn']):.2f} {dls_label}</b> | 
            Toolface: <b>{p['tf']:.0f}°</b> over next <b>{p['len']} {u_label}</b></p>
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
                        # 1. PARSE DATA (Gunakan fungsi parser BHA yang sudah kita buat)
                        # (Pastikan fungsi 'parse_scenario_bha_chain' sudah ada di utils)
                        scen_dict, cases_df, comps_df = parse_scenario_bha_chain(active_file)
                        
                        if scen_dict and not cases_df.empty:
                            st.markdown("---")
                            
                            # 2. AUTO-SELECT SCENARIO DARI SIDEBAR
                            # Ambil ID Scenario yang dipilih di Sidebar (jika ada)
                            pre_selected_id = st.session_state.get('selected_scenario_id')
                            
                            scen_opts = {name: sid for sid, name in scen_dict.items()}
                            sorted_scen_names = sorted(list(scen_opts.keys()))
                            
                            # Cari index scenario yang cocok dengan Sidebar
                            default_idx = 0
                            if pre_selected_id:
                                # Cari nama scenario berdasarkan ID
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
                        
                        # TABEL COMPONENT
                        st.dataframe(
                            selected_comps_df,
                            column_config={
                                "Description": st.column_config.TextColumn("Description", width="large"),
                                "Connection": st.column_config.TextColumn("Conn", width="small"),
                                "OD (in)": st.column_config.NumberColumn("OD", format="%.3f"),
                                "ID (in)": st.column_config.NumberColumn("ID", format="%.3f"),
                                "Length": st.column_config.NumberColumn("Len", format="%.2f"),
                            },
                            use_container_width=True,
                            hide_index=True
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

                    elif selected_comps_df is not None:
                        st.info("BHA ini tidak memiliki detail komponen.")
                    else:
                        st.markdown("""
                        <div style='text-align: center; color: grey; padding: 50px;'>
                            <h3>⬅️ Ready</h3>
                            <p>Silakan pilih Scenario dan Case.</p>
                        </div>
                        """, unsafe_allow_html=True)