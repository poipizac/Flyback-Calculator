import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math
import json
from io import BytesIO
from fpdf import FPDF
import os
import re
from datetime import datetime

# Page Config
st.set_page_config(page_title="Flyback Dynamic Power Calculator", layout="wide")

# Custom JSON Encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# MOSFET Component Database (Primary PWM)
MOSFET_DB = pd.DataFrame([
    {"Model": "Infineon IPP60R190P6", "Vds": 600, "Rdson": 0.190, "Coss": 440, "Qg": 33},
    {"Model": "Infineon IPP65R045C7", "Vds": 650, "Rdson": 0.045, "Coss": 1100, "Qg": 93},
    {"Model": "ST STF13N60M2", "Vds": 600, "Rdson": 0.380, "Coss": 240, "Qg": 16},
    {"Model": "ST STF24N60M2", "Vds": 600, "Rdson": 0.168, "Coss": 580, "Qg": 32},
    {"Model": "Toshiba TK12A60W", "Vds": 600, "Rdson": 0.230, "Coss": 310, "Qg": 35},
    {"Model": "ON Semi FCPF190N60E", "Vds": 600, "Rdson": 0.190, "Coss": 490, "Qg": 31},
    {"Model": "Custom (Manual)", "Vds": 0, "Rdson": 0.21, "Coss": 180, "Qg": 20}
])

# SR MOSFET Database (Secondary Rectification)
SR_MOSFET_DB = pd.DataFrame([
    {"Model": "Infineon IPP023N10N5", "Vds": 100, "Rdson": 0.0023, "Coss": 990, "Qg": 112},
    {"Model": "Infineon IPP052N08N5", "Vds": 80, "Rdson": 0.0052, "Coss": 630, "Qg": 45},
    {"Model": "ST STP110N10F7", "Vds": 100, "Rdson": 0.0085, "Coss": 650, "Qg": 50},
    {"Model": "Toshiba TK100E10N1", "Vds": 100, "Rdson": 0.0034, "Coss": 950, "Qg": 105},
    {"Model": "ON Semi NTMFS5C628NL", "Vds": 60, "Rdson": 0.0024, "Coss": 1400, "Qg": 52},
    {"Model": "Custom (Manual)", "Vds": 0, "Rdson": 0.01, "Coss": 500, "Qg": 30}
])

st.title("Flyback 動態功耗計算機")
st.markdown("互動式損耗評估與效率估算工具 (Flyback Topology)")

def calculate_flyback_performance(p, v_ac_rms, load_pct=1.0):
    """
    p: dictionary of all system/component parameters
    v_ac_rms: actual AC input voltage
    load_pct: fraction of rated I_out (0.0 to 1.0)
    Returns: (efficiency, duty_cycle, total_loss, warnings, other_internals)
    """
    # 1. Base Variables
    i_out_step = p['i_out'] * load_pct
    p_out_step = p['v_out'] * i_out_step
    
    # Target values for mode calculation (uses system target efficiency)
    p_in_target = p_out_step / p['eta_target']
    v_ro = (p['n_p'] / p['n_s']) * p['v_out']
    n_ratio = p['n_p'] / p['n_s']
    
    # 2. Rectifier & Input
    i_in_rms = p_in_target / (p['bridge_pf'] * v_ac_rms)
    i_in_avg = (2.0 / math.pi) * (math.sqrt(2) * i_in_rms)
    p_bridge = 0.5 * i_in_avg * (2.0 * p['bridge_vf'])
    
    v_in_dc = v_ac_rms * math.sqrt(2)
    # Simplified bulk cap loss (using proportional RMS current)
    v_bulk = max(p['v_bulk_min'], v_in_dc * 0.8) # Simple approximation for trace plots
    i_bulk_avg = p_in_target / (0.5 * (v_in_dc + v_bulk))
    pf_factor = max(0.0, (1.0 / (2.0 * (p['bridge_pf']**2))) - 0.5)
    i_bulk_rms = i_bulk_avg * math.sqrt(pf_factor)
    p_bulk_loss = (i_bulk_rms**2) * p['esr_bulk']
    
    p_lf_loss = (i_in_rms**2) * p['dcr_lf1']
    if p.get('lf_qty', 1) == 2:
        p_lf_loss += (i_in_rms**2) * p.get('dcr_lf2', 0.0)

    # 3. Switching Mode Calculation
    f_sw = p['f_sw_input'] * 1000.0
    l_m = p['l_m_uH'] * 1e-6
    d_max = 0.0
    i_p_pk = 0.0
    i_s_pk = 0.0
    ip_rms = 0.0
    is_rms = 0.0
    warnings = []
    
    op_mode = p['op_mode']
    if op_mode == "CRM":
        d_max = v_ro / (v_bulk + v_ro)
        i_p_pk = (v_bulk * d_max) / (l_m * f_sw)
        i_s_pk = n_ratio * i_p_pk
        ip_rms = math.sqrt(d_max / 3.0) * i_p_pk
        is_rms = math.sqrt((1.0 - d_max) / 3.0) * i_s_pk
    elif op_mode == "DCM":
        d_max = math.sqrt((2.0 * l_m * f_sw * p_in_target) / (v_bulk**2))
        d2 = (v_bulk * d_max) / v_ro
        i_p_pk = (v_bulk * d_max) / (l_m * f_sw)
        i_s_pk = n_ratio * i_p_pk
        ip_rms = i_p_pk * math.sqrt(d_max / 3.0)
        is_rms = i_s_pk * math.sqrt(d2 / 3.0)
    elif op_mode == "CCM":
        d_max = v_ro / (v_bulk + v_ro)
        i_in_avg_dc = p_in_target / v_bulk
        delta_i = (v_bulk * d_max) / (l_m * f_sw)
        i_p_pk = (i_in_avg_dc / d_max) + (delta_i / 2.0)
        i_p_valley = i_p_pk - delta_i
        ip_rms = math.sqrt(d_max * (i_p_pk**2 + i_p_pk * i_p_valley + i_p_valley**2) / 3.0)
        i_s_pk = n_ratio * i_p_pk
        i_s_valley = n_ratio * i_p_valley
        is_rms = math.sqrt((1.0 - d_max) * (i_s_pk**2 + i_s_pk * i_s_valley + i_s_valley**2) / 3.0)
    elif op_mode == "QR":
        c_oss = p['c_oss_eff_pf'] * 1e-12
        t_res = math.pi * math.sqrt(l_m * c_oss)
        a = 0.5 * l_m
        b = -p_in_target * l_m * (1.0/v_bulk + 1.0/v_ro)
        c = -p_in_target * t_res
        try:
            i_p_pk = (-b + math.sqrt(max(0, b**2 - 4*a*c))) / (2*a)
            t_on = (l_m * i_p_pk) / v_bulk
            t_off = (l_m * i_p_pk) / v_ro
            f_sw = 1.0 / (t_on + t_off + t_res)
            d_max = t_on * f_sw
            d2 = t_off * f_sw
            i_s_pk = n_ratio * i_p_pk
            ip_rms = i_p_pk * math.sqrt(d_max / 3.0)
            is_rms = i_s_pk * math.sqrt(d2 / 3.0)
        except:
            d_max = 0.0

    # 4. Component Losses
    # MOSFET
    is_valley = (op_mode in ["QR", "CRM"])
    v_sw_final = max(0.0, v_bulk - v_ro) if is_valley else (v_bulk + v_ro)
    p_pwm_cond = (ip_rms**2) * p['r_ds_on_sw']
    p_pwm_sw = 0.5 * (p['c_oss_eff_pf'] * 1e-12) * (v_sw_final**2) * f_sw
    p_mos_total = p_pwm_cond + p_pwm_sw
    
    # RCD
    p_rcd = 0.5 * (p['lk_uH']*1e-6) * (i_p_pk**2) * f_sw if p['rcd_enable'] else 0.0
    
    # Transformer
    p_t_core = p['p_cv'] * 1000.0 * (p['ve_mm3'] * 1e-9)
    p_t_prim = (ip_rms**2) * p['dcr_tp']
    p_t_sec = (is_rms**2) * p['dcr_ts']
    p_xfmr_total = p_t_core + p_t_prim + p_t_sec
    
    # SR MOSFET
    p_sr_cond = (is_rms**2) * p['r_ds_on_sr']
    p_sr_dead = p['v_f_sr'] * i_s_pk * (p['t_on_delay_sr_ns']*1e-9) * f_sw
    p_sr_total = p_sr_cond + p_sr_dead
    
    # Output & Other
    p_blocking = (i_out_step**2) * p['r_ds_on_blocking']
    p_lf51 = (is_rms**2) * p['dcr_lf51']
    i_outcap_rms = math.sqrt(max(0.0, is_rms**2 - i_out_step**2))
    p_outcap = (i_outcap_rms**2) * p['esr_outcap']
    
    total_loss = p_bridge + p_bulk_loss + p_lf_loss + p_mos_total + p_rcd + p_xfmr_total + p_sr_total + p_blocking + p_lf51 + p_outcap + p['p_other']
    p_in_total = p_out_step + total_loss
    efficiency = (p_out_step / p_in_total * 100) if p_in_total > 0 else 0
    
    results = {
        "efficiency": efficiency,
        "duty_cycle": d_max,
        "total_loss": total_loss,
        "p_out": p_out_step,
        "p_in": p_in_total,
        "f_sw_khz": f_sw / 1000.0,
        "i_p_pk": i_p_pk,
        "ip_rms": ip_rms,
        "is_rms": is_rms,
        "p_mos_total": p_mos_total,
        "p_pwm_cond": p_pwm_cond,
        "p_pwm_sw": p_pwm_sw,
        "p_rcd": p_rcd,
        "p_xfmr_total": p_xfmr_total,
        "p_t_core": p_t_core,
        "p_t_prim": p_t_prim,
        "p_t_sec": p_t_sec,
        "p_sr_total": p_sr_total,
        "p_sr_cond": p_sr_cond,
        "p_sr_dead": p_sr_dead,
        "p_bridge": p_bridge,
        "p_bulk_loss": p_bulk_loss,
        "p_lf_loss": p_lf_loss,
        "p_blocking": p_blocking,
        "p_output_filter_total": p_lf51 + p_outcap,
        "v_sw_final": v_sw_final,
        "i_outcap_rms": i_outcap_rms
    }
    return results

# Sidebar Inputs
st.sidebar.header("專案管理 (Project Management)")
uploaded_file = st.sidebar.file_uploader("載入專案 (Import JSON)", type=["json"])

# Initialize session state for inputs if not present
if "config" not in st.session_state:
    st.session_state.config = {}

if uploaded_file is not None:
    try:
        st.session_state.config = json.load(uploaded_file)
        st.sidebar.success("專案載入成功！")
    except Exception as e:
        st.sidebar.error(f"載入失敗: {e}")

# Function to get value from session state or default
def get_val(key, default):
    return st.session_state.config.get(key, default)

def clean_text_for_pdf(text):
    """Remove non-ASCII characters and special symbols for FPDF compatibility."""
    if not isinstance(text, str):
        text = str(text)
    # Keep only ASCII printable characters
    return re.sub(r'[^\x00-\x7F]+', '', text).strip()

def create_pdf_report(params, results, chart_img=None):
    class PDF(FPDF):
        def footer(self):
            self.set_y(-15)
            # Use English for footer to avoid font issues if not loaded
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")
            self.line(10, self.get_y(), 200, self.get_y())

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)

    # 1. Enterprise Header (Deep Blue)
    pdf.set_fill_color(41, 128, 185)
    pdf.rect(0, 0, 210, 40, "F")
    pdf.set_text_color(255, 255, 255)
    
    title_text = "Flyback Design Evaluation Report"
    pdf.set_y(15)
    pdf.cell(0, 10, txt=title_text, ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.set_y(45)

    def draw_section_header(title_en):
        pdf.set_fill_color(236, 240, 241)
        pdf.set_draw_color(41, 128, 185)
        pdf.set_line_width(0.5)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, txt=f" {title_en}", ln=True, fill=True, border="TB")
        pdf.ln(3)

    # Section 1: System Specifications
    draw_section_header("1. System Specifications")
    
    specs = [
        ("Mode", params['op_mode']),
        ("MOSFET", params.get('selected_mosfet', 'N/A')),
        ("Vac_min", f"{params['v_ac_min']} Vrms"),
        ("Vac_max", f"{params['v_ac_max']} Vrms"),
        ("Vout", f"{params['v_out']} V"),
        ("Iout", f"{params['i_out']} A"),
        ("fsw", f"{params.get('f_sw_input', '--')} kHz"),
        ("Target Eta", f"{params['eta_target_percent']}%")
    ]
    
    fill = False
    for label, val in specs:
        pdf.set_fill_color(248, 249, 250)
        pdf.set_font("Helvetica", "B", 10)
        # Apply cleaning to ensure no Unicode issues
        clean_label = clean_text_for_pdf(label)
        clean_val = clean_text_for_pdf(val)
        pdf.cell(95, 8, txt=f" {clean_label}:", border=0, fill=fill)
        pdf.set_font("Helvetica", size=11)
        pdf.cell(0, 8, txt=clean_val, border=0, ln=True, fill=fill)
        pdf.set_draw_color(230, 230, 230)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        fill = not fill
    
    pdf.ln(8)

    # Section 2: Power Loss & Efficiency Evaluation
    draw_section_header("2. Power Loss & Efficiency")
    
    perf = [
        ("Pout", f"{results['p_out']:.2f} W", False),
        ("Total Loss", f"{results['total_loss']:.3f} W", True),
        ("Pin", f"{results['p_in']:.3f} W", False),
        ("Efficiency", f"{results['efficiency']:.2f} %", True)
    ]

    fill = False
    for label, val, highlight in perf:
        pdf.set_fill_color(248, 249, 250)
        if highlight:
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(41, 128, 185)
        else:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(0, 0, 0)
            
        clean_label = clean_text_for_pdf(label)
        clean_val = clean_text_for_pdf(val)
        pdf.cell(95, 10, txt=f" {clean_label}:", border=0, fill=fill)
        
        if highlight:
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(30, 132, 73) 
        else:
            pdf.set_font("Helvetica", size=11)
            
        pdf.cell(0, 10, txt=clean_val, border=0, ln=True, fill=fill)
        pdf.set_text_color(0, 0, 0)
        pdf.set_draw_color(230, 230, 230)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        fill = not fill

    pdf.ln(10)

    # 3. Characteristic Curves
    draw_section_header("3. Characteristic Curves")
    if chart_img:
        try:
            pdf.image(chart_img, x=20, w=170)
        except:
            pdf.cell(0, 10, txt="Image error", ln=True)
    else:
        pdf.cell(0, 10, txt="Chart not available", ln=True)

    pdf.set_y(-30)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100)
    footer_text = f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    pdf.cell(0, 5, txt=footer_text, ln=True, align="L")
    ref_text = "Professional engineering data. For design reference only."
    pdf.cell(0, 5, txt=ref_text, ln=True, align="L")

    return pdf.output()

st.sidebar.header("系統規格 (System Specifications)")

op_mode = st.sidebar.selectbox(
    "運作模式 (Operation Mode)",
    ["CRM", "DCM", "CCM", "QR"],
    index=["CRM", "DCM", "CCM", "QR"].index(get_val("op_mode", "CRM"))
)

v_ac_min = st.sidebar.number_input("最小交流輸入電壓 V_AC_min (Vrms)", value=float(get_val("v_ac_min", 90.0)), step=1.0)
v_ac_max = st.sidebar.number_input("最大交流輸入電壓 V_AC_max (Vrms)", value=float(get_val("v_ac_max", 264.0)), step=1.0)
v_out = st.sidebar.number_input("輸出電壓 V_out (V)", value=float(get_val("v_out", 12.0)), step=0.1)
i_out = st.sidebar.number_input("額定負載 I_out (A)", value=float(get_val("i_out", 5.0)), step=0.1)
f_sw_input = st.sidebar.number_input("工作頻率 f_sw (kHz)", value=float(get_val("f_sw_input", 65.0)), step=1.0, disabled=(op_mode == "QR"))
eta_target = st.sidebar.slider("估計效率 η_target (%)", 50, 99, int(get_val("eta_target_percent", 85))) / 100.0
f_line_min = st.sidebar.number_input("輸入電源頻率 f_line_min (Hz)", value=float(get_val("f_line_min", 47.0)), step=1.0)

st.sidebar.header("Flyback 核心參數 (Core Parameters)")
col1, col2 = st.sidebar.columns(2)
n_p = col1.number_input("Np (匝)", value=int(get_val("n_p", 24)), step=1)
n_s = col2.number_input("Ns (匝)", value=int(get_val("n_s", 4)), step=1)
n_ratio = n_p / n_s

l_m_uH = st.sidebar.number_input("激磁電感 Lm (µH)", value=float(get_val("l_m_uH", 230.0)), step=1.0)
l_m = l_m_uH * 1e-6

# Advanced Settings in Expanders
with st.sidebar.expander("橋式整流器 (Bridge Rectifier)"):
    bridge_vf = st.number_input("本體二極體壓降 Vf_Bridge (V)", value=float(get_val("bridge_vf", 0.69)), step=0.01)
    bridge_n = st.number_input("並聯數量 N_Bridge", value=int(get_val("bridge_n", 1)), step=1)
    bridge_pf = st.number_input("功率因數 PF", value=float(get_val("bridge_pf", 0.58)), step=0.01)

with st.sidebar.expander("輸入大電容 (Bulk Cap)"):
    v_bulk_min = st.number_input("目標谷底電壓 V_Bulk_min (V)", value=float(get_val("v_bulk_min", 70.0)), step=1.0)
    esr_bulk = st.number_input("大電容等效電阻 ESR_Bulk (Ω)", value=float(get_val("esr_bulk", 2.05)), step=0.001)

with st.sidebar.expander("PWM MOSFET (主開關)"):
    model_list = MOSFET_DB["Model"].tolist()
    default_model = get_val("selected_mosfet", "Infineon IPP60R190P6")
    default_idx = model_list.index(default_model) if default_model in model_list else 0
    
    selected_model = st.selectbox("選擇 MOSFET 型號", model_list, index=int(default_idx))
    mosfet_data = MOSFET_DB[MOSFET_DB["Model"] == selected_model].iloc[0]
    
    if selected_model == "Custom (Manual)":
        r_ds_on_sw = st.number_input("導通電阻 Rds_on_SW (Ω)", value=float(get_val("r_ds_on_sw", 0.21)), step=0.01)
        c_oss_eff_pf = st.number_input("等效輸出電容 Coss_eff (pF)", value=float(get_val("c_oss_eff_pf", 180.0)), step=1.0)
    else:
        # Use database values
        r_ds_on_sw = mosfet_data["Rdson"]
        c_oss_eff_pf = mosfet_data["Coss"]
        st.info(f"Vds: {mosfet_data['Vds']}V | Rdson: {r_ds_on_sw}Ω | Coss: {c_oss_eff_pf}pF")
        
    c_oss_eff = c_oss_eff_pf * 1e-12

with st.sidebar.expander("RCD 吸收損耗"):
    rcd_enable = st.checkbox("啟用 RCD 計算", value=bool(get_val("rcd_enable", True)))
    lk_uH = st.number_input("變壓器一次側漏感 Lk (µH)", value=float(get_val("lk_uH", 4.5)), step=0.1)
    l_k = lk_uH * 1e-6

with st.sidebar.expander("二極體/SR MOSFET"):
    sr_model_list = SR_MOSFET_DB["Model"].tolist()
    default_sr_model = get_val("selected_sr_mosfet", "Infineon IPP023N10N5")
    default_sr_idx = sr_model_list.index(default_sr_model) if default_sr_model in sr_model_list else 0
    
    selected_sr_model = st.selectbox("選擇 SR MOSFET 型號", sr_model_list, index=int(default_sr_idx))
    sr_mosfet_data = SR_MOSFET_DB[SR_MOSFET_DB["Model"] == selected_sr_model].iloc[0]
    
    if selected_sr_model == "Custom (Manual)":
        r_ds_on_sr = st.number_input("導通電阻 Rds_on_SR (Ω)", value=float(get_val("r_ds_on_sr", 0.01)), step=0.001)
        c_oss_sr_pf = st.number_input("輸出電容 Coss_SR (pF)", value=float(get_val("c_oss_sr_pf", 500.0)), step=10.0)
    else:
        r_ds_on_sr = sr_mosfet_data["Rdson"]
        c_oss_sr_pf = sr_mosfet_data["Coss"]
        st.info(f"Vds: {sr_mosfet_data['Vds']}V | Rdson: {r_ds_on_sr}Ω | Coss: {c_oss_sr_pf}pF")

    v_f_sr = st.number_input("本體二極體壓降 Vf_SR (V)", value=float(get_val("v_f_sr", 1.2)), step=0.1)
    t_on_delay_sr_ns = st.number_input("導通延遲時間 ton_delay_SR (ns)", value=float(get_val("t_on_delay_sr_ns", 20.0)), step=1.0)
    t_on_delay_sr = t_on_delay_sr_ns * 1e-9

with st.sidebar.expander("變壓器 (Transformer)"):
    dcr_tp = st.number_input("一次側直流阻抗 DCR_T_prim (Ω)", value=float(get_val("dcr_tp", 0.3)), step=0.001)
    dcr_ts = st.number_input("二次側直流阻抗 DCR_T_sec (Ω)", value=float(get_val("dcr_ts", 0.0082)), step=0.0001)
    ve_mm3 = st.number_input("磁芯體積 Ve (mm³)", value=float(get_val("ve_mm3", 4258.3)), step=0.1)
    p_cv = st.number_input("單位體積鐵損 Pcv (kW/m³)", value=float(get_val("p_cv", 300.0)), step=1.0)

with st.sidebar.expander("其他元件 (EMI / VBUS / Etc)"):
    dcr_lf1 = st.number_input("LF1 直流阻抗 (Ω)", value=float(get_val("dcr_lf1", 0.132)), step=0.001)
    lf_qty = st.selectbox("電感數量", [1, 2], index=[1, 2].index(get_val("lf_qty", 1)))
    dcr_lf2 = st.number_input("LF2 直流阻抗 (Ω)", value=float(get_val("dcr_lf2", 0.1)), step=0.001) if lf_qty == 2 else 0.0
    r_ds_on_blocking = st.number_input("Blocking MOSFET Rdson (Ω)", value=float(get_val("r_ds_on_blocking", 0.016)), step=0.001)
    dcr_lf51 = st.number_input("LF51 直流阻抗 (Ω)", value=float(get_val("dcr_lf51", 0.008)), step=0.001)
    esr_outcap = st.number_input("輸出電容 ESR (Ω)", value=float(get_val("esr_outcap", 0.02)), step=0.001)
    p_other = st.number_input("其他綜合損耗 (W)", value=float(get_val("p_other", 0.5)), step=0.1)

# Assemble current parameters into a dict
sys_params = {
    "v_out": v_out, "i_out": i_out, "eta_target": eta_target, "n_p": n_p, "n_s": n_s, "op_mode": op_mode,
    "v_ac_min": v_ac_min, "v_ac_max": v_ac_max, "f_sw_input": f_sw_input, "l_m_uH": l_m_uH,
    "bridge_vf": bridge_vf, "bridge_pf": bridge_pf, "v_bulk_min": v_bulk_min, "esr_bulk": esr_bulk,
    "selected_mosfet": selected_model, "r_ds_on_sw": r_ds_on_sw, "c_oss_eff_pf": c_oss_eff_pf, 
    "selected_sr_mosfet": selected_sr_model, "r_ds_on_sr": r_ds_on_sr, "c_oss_sr_pf": c_oss_sr_pf,
    "rcd_enable": rcd_enable, "lk_uH": lk_uH, "v_f_sr": v_f_sr, "t_on_delay_sr_ns": t_on_delay_sr_ns,
    "dcr_tp": dcr_tp, "dcr_ts": dcr_ts, "ve_mm3": ve_mm3, "p_cv": p_cv,
    "dcr_lf1": dcr_lf1, "dcr_lf2": dcr_lf2, "lf_qty": lf_qty, "r_ds_on_blocking": r_ds_on_blocking,
    "dcr_lf51": dcr_lf51, "esr_outcap": esr_outcap, "p_other": p_other
}

# Run primary calculation for single operating point
main_res = calculate_flyback_performance(sys_params, v_ac_min, load_pct=1.0)

# Unpack for UI (Legacy names for compatibility)
p_out = main_res["p_out"]
total_loss = main_res["total_loss"]
p_in = main_res["p_in"]
efficiency = main_res["efficiency"]
d_max = main_res["duty_cycle"]
ip_rms = main_res["ip_rms"]
is_rms = main_res["is_rms"]
i_p_pk = main_res["i_p_pk"]
p_mos_total = main_res["p_mos_total"]
p_pwm_cond = main_res["p_pwm_cond"]
p_pwm_sw = main_res["p_pwm_sw"]
p_rcd = main_res["p_rcd"]
p_xfmr_total = main_res["p_xfmr_total"]
p_t_core = main_res["p_t_core"]
p_t_prim = main_res["p_t_prim"]
p_t_sec = main_res["p_t_sec"]
p_sr_total = main_res["p_sr_total"]
p_sr_cond = main_res["p_sr_cond"]
p_sr_dead = main_res["p_sr_dead"]
p_bridge = main_res["p_bridge"]
p_bulk_loss = main_res["p_bulk_loss"]
p_lf_loss = main_res["p_lf_loss"]
p_blocking = main_res["p_blocking"]
p_output_filter_total = main_res["p_output_filter_total"]
v_sw_final = main_res["v_sw_final"]
i_outcap_rms = main_res["i_outcap_rms"]
f_sw = main_res["f_sw_khz"] * 1000.0
i_s_pk = main_res["i_p_pk"] * (n_p/n_s) # Derived
v_ro = (n_p/n_s) * v_out
warnings = [] # Warnings logic could be added to helper
if op_mode == "QR":
    st.sidebar.info(f"QR 計算頻率: {main_res['f_sw_khz']:.2f} kHz")
# Results already unpacked from main_res above

# --- Display Results ---
for w in warnings:
    st.warning(w)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("輸出功率 (Pout)", f"{p_out:.1f} W")
col_m2.metric("總功耗 (Total Loss)", f"{total_loss:.3f} W")
col_m3.metric("輸入功率 (Pin)", f"{p_in:.3f} W")
col_m4.metric("效率 (Efficiency)", f"{efficiency:.2f} %")

# --- Interactive Comparison Charts ---
st.subheader("📊 進階特性分析 (Advanced Analysis)")

v_range_plot = np.linspace(v_ac_min, v_ac_max, 20)
load_pcts = [1.0, 0.75, 0.5, 0.25]
comparison_data = []

for lp in load_pcts:
    load_label = f"{int(lp*100)}% Load"
    for vac in v_range_plot:
        res = calculate_flyback_performance(sys_params, vac, load_pct=lp)
        comparison_data.append({
            "Input Voltage (V)": vac,
            "Duty Cycle (D)": res["duty_cycle"],
            "Efficiency (%)": res["efficiency"],
            "Load Condition": load_label
        })

df_comp = pd.DataFrame(comparison_data)

# 1. Duty Cycle Comparison
fig_d_comp = px.line(df_comp, x="Input Voltage (V)", y="Duty Cycle (D)", color="Load Condition",
                     title="Duty Cycle vs. Input Voltage (Multiple Loads)",
                     template="plotly_dark", markers=True)
fig_d_comp.update_layout(xaxis=dict(tickvals=[90, 115, 230, 264]))
st.plotly_chart(fig_d_comp, use_container_width=True)

# 2. Efficiency vs. Load (Standard Datasheet View)
v_test_list = [v_ac_min, v_ac_max]
load_range = np.linspace(0.1, 1.0, 20)
eff_load_data = []

for vac in v_test_list:
    vac_label = f"{vac:.1f} Vac"
    for lp in load_range:
        res_step = calculate_flyback_performance(sys_params, vac, load_pct=lp)
        eff_load_data.append({
            "Load (%)": lp * 100,
            "Efficiency (%)": res_step["efficiency"],
            "Input Voltage": vac_label
        })

df_eff_load = pd.DataFrame(eff_load_data)
fig_eff_load = px.line(df_eff_load, x="Load (%)", y="Efficiency (%)", color="Input Voltage",
                        title="Efficiency vs. Load (High/Low Line Comparison)",
                        template="plotly_dark", markers=True)
st.plotly_chart(fig_eff_load, use_container_width=True)

# --- Selected MOSFET Specs ---
st.subheader("🔌 選用元件規格 (Selected Component Specs)")
# Localize and Merge specs for display
specs_display = []

# PWM MOSFET Row
pwm_row = mosfet_data.copy()
pwm_row["Position"] = "Primary (PWM)"
specs_display.append(pwm_row)

# SR MOSFET Row
sr_row = sr_mosfet_data.copy()
sr_row["Position"] = "Secondary (SR)"
specs_display.append(sr_row)

specs_df = pd.DataFrame(specs_display)
specs_df = specs_df.rename(columns={
    "Position": "位置",
    "Model": "型號",
    "Vds": "耐壓 Vds (V)",
    "Rdson": "導通電阻 Rdson (Ω)",
    "Coss": "輸出電容 Coss (pF)",
    "Qg": "閘極電荷 Qg (nC)"
})
# Reorder columns
cols = ["位置", "型號", "耐壓 Vds (V)", "導通電阻 Rdson (Ω)", "輸出電容 Coss (pF)", "閘極電荷 Qg (nC)"]
st.dataframe(specs_df[cols], use_container_width=True, hide_index=True)

tab1, tab2 = st.tabs(["📊 損耗分配 (Loss Breakdown)", "📝 公式推導 (Derivations)"])

with tab1:
    loss_data = {
        "組件 (Component)": [
            "橋式整流器 (Bridge)", 
            "大電容 (Bulk Cap)", 
            "輸入電感 (Input EMI)",
            "主開關 (PWM MOSFET)",
            "RCD 吸收器",
            "變壓器 (Transformer)",
            "同步整流 (SR MOSFET)",
            "Blocking MOSFET",
            "輸出濾波 (Output Filter)",
            "其他 (Other)"
        ],
        "損耗 (Loss) [W]": [
            p_bridge, p_bulk_loss, p_lf_loss, p_mos_total, p_rcd, p_xfmr_total, p_sr_total, p_blocking, p_output_filter_total, p_other
        ]
    }
    df = pd.DataFrame(loss_data)
    df["百分比 (%)"] = (df["損耗 (Loss) [W]"] / total_loss * 100).round(2)
    
    st.dataframe(df, use_container_width=True)
    
    # Plotly Donut Chart
    fig = go.Figure(data=[go.Pie(
        labels=df["組件 (Component)"], 
        values=df["損耗 (Loss) [W]"], 
        hole=.5,
        marker=dict(colors=px.colors.qualitative.Prism)
    )])
    fig.update_layout(
        title_text="功耗比例分佈 (Loss Distribution)",
        annotations=[dict(text='Total Loss', x=0.5, y=0.5, font_size=16, showarrow=False)],
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("核心計算公式")
    st.latex(r"P_{out} = V_{out} \cdot I_{out} = " + f"{v_out} \cdot {i_out} = {p_out:.1f} \\text{{ W}}")
    st.latex(f"\\text{{Target Mode: }} \\mathbf{{{op_mode}}}")
    
    if op_mode == "CRM":
        st.latex(r"D_{max} = \frac{n \cdot V_{out}}{V_{Bulk\_min} + n \cdot V_{out}} = " + f"{d_max:.3f}")
        st.latex(r"I_{p\_rms} = \sqrt{\frac{D_{max}}{3}} \cdot I_{p\_pk} = " + f"{ip_rms:.3f} \text{{ A}}")
    elif op_mode == "QR":
        st.latex(r"t_{res} = \pi \sqrt{L_m C_{oss}} = " + f"{t_res*1e6:.2f} \mu s")
        st.latex(r"f_{sw} = " + f"{f_sw/1000:.2f} \text{{ kHz}}")

    st.subheader("損耗公式摘要")
    st.latex(r"P_{SW\_cond} = I_{p\_rms}^2 \cdot R_{ds\_on\_SW} = " + f"{ip_rms:.3f}^2 \\cdot {r_ds_on_sw} = {p_pwm_cond:.3f} \\text{{ W}}")
    st.latex(r"P_{SW\_sw} = \frac{1}{2} C_{oss} V_{sw}^2 f_{sw} = 0.5 \\cdot " + f"{c_oss_eff*1e12:.1f}p \\cdot {v_sw_final:.1f}^2 \\cdot {f_sw/1000:.1f}k = {p_pwm_sw:.3f} \\text{{ W}}")
    
    st.divider()
    st.markdown("##### 磁性元件損耗 (Transformer)")
    st.latex(r"P_{T\_pri} = I_{p\_rms}^2 \cdot DCR_{T\_pri} = " + f"{ip_rms:.3f}^2 \\cdot {dcr_tp} = {p_t_prim:.3f} \\text{{ W}}")
    st.latex(r"P_{T\_sec} = I_{s\_rms}^2 \cdot DCR_{T\_sec} = " + f"{is_rms:.3f}^2 \\cdot {dcr_ts} = {p_t_sec:.3f} \\text{{ W}}")
    st.latex(r"P_{T\_core} = P_{cv} \cdot V_e = " + f"{p_cv} \\cdot {ve_mm3*1e-9:.6f} = {p_t_core:.3f} \\text{{ W}}")

    st.divider()
    st.markdown("##### 二次側同步整流 (SR MOSFET)")
    st.latex(r"P_{SR\_cond} = I_{s\_rms}^2 \cdot R_{ds\_on\_SR} = " + f"{is_rms:.3f}^2 \\cdot {r_ds_on_sr} = {p_sr_cond:.3f} \\text{{ W}}")
    st.latex(r"P_{dead} = V_{f\_SR} \cdot I_{s\_pk} \cdot t_{delay} \cdot f_{sw} = " + f"{v_f_sr} \\cdot {i_s_pk:.2f} \\cdot {t_on_delay_sr_ns}n \\cdot {f_sw/1000:.1f}k = {p_sr_dead:.3f} \\text{{ W}}")

# --- Monte Carlo Yield Analysis ---
st.divider()
with st.expander("🎲 蒙地卡羅量產良率分析 (Monte Carlo Yield Analysis)", expanded=False):
    st.markdown("模擬零組件公差對滿載效率的影響 (Worst Case: Vac_min, 100% Load)")
    
    col_mc1, col_mc2 = st.columns(2)
    lm_tol = col_mc1.slider("Lm 公差 (%)", 0, 20, 10)
    rdson_tol = col_mc2.slider("Rds_on 公差 (%)", 0, 20, 10)
    
    if st.button("執行 1,000 次蒙地卡羅模擬"):
        N_sim = 1000
        eff_results = []
        
        # Nominal values
        lm_nom = sys_params['l_m_uH']
        rdson_nom = sys_params['r_ds_on_sw']
        
        # Generate normal distributions (3-sigma)
        np.random.seed(42) # For reproducibility
        lm_samples = np.random.normal(lm_nom, lm_nom * (lm_tol/100)/3, N_sim)
        rdson_samples = np.random.normal(rdson_nom, rdson_nom * (rdson_tol/100)/3, N_sim)
        
        # Simulation Loop
        progress_bar = st.progress(0)
        for i in range(N_sim):
            # Create a shallow copy and update drifted params
            sim_params = sys_params.copy()
            sim_params['l_m_uH'] = lm_samples[i]
            sim_params['r_ds_on_sw'] = rdson_samples[i]
            
            # Calculate at worst case: V_ac_min, 100% Load
            res = calculate_flyback_performance(sim_params, v_ac_min, load_pct=1.0)
            eff_results.append(res['efficiency'])
            
            if i % 100 == 0:
                progress_bar.progress((i + 1) / N_sim)
        progress_bar.empty()
        
        # Plot Histogram
        df_mc = pd.DataFrame({"Efficiency (%)": eff_results})
        fig_mc = px.histogram(df_mc, x="Efficiency (%)", 
                              title="Efficiency Distribution (1000 Monte Carlo Runs @ Worst Case)",
                              nbins=30, template="plotly_dark",
                              color_discrete_sequence=['#1f77b4'])
        fig_mc.update_layout(bargap=0.1)
        st.plotly_chart(fig_mc, use_container_width=True)
        
        # Summary statistics
        st.info(f"📊 統計摘要: 平均效率: {np.mean(eff_results):.2f}% | 最小值: {np.min(eff_results):.2f}% | 最大值: {np.max(eff_results):.2f}%")

# Prepare export data
config_to_export = {
    "op_mode": op_mode,
    "v_ac_min": v_ac_min,
    "v_ac_max": v_ac_max,
    "v_out": v_out,
    "i_out": i_out,
    "f_sw_input": f_sw_input,
    "eta_target_percent": eta_target * 100,
    "f_line_min": f_line_min,
    "n_p": n_p,
    "n_s": n_s,
    "l_m_uH": l_m_uH,
    "bridge_vf": bridge_vf,
    "bridge_n": bridge_n,
    "bridge_pf": bridge_pf,
    "v_bulk_min": v_bulk_min,
    "esr_bulk": esr_bulk,
    "selected_mosfet": selected_model,
    "r_ds_on_sw": r_ds_on_sw,
    "c_oss_eff_pf": c_oss_eff_pf,
    "rcd_enable": rcd_enable,
    "lk_uH": lk_uH,
    "r_ds_on_sr": r_ds_on_sr,
    "v_f_sr": v_f_sr,
    "t_on_delay_sr_ns": t_on_delay_sr_ns,
    "dcr_tp": dcr_tp,
    "dcr_ts": dcr_ts,
    "ve_mm3": ve_mm3,
    "p_cv": p_cv,
    "dcr_lf1": dcr_lf1,
    "lf_qty": lf_qty,
    "dcr_lf2": dcr_lf2,
    "r_ds_on_blocking": r_ds_on_blocking,
    "dcr_lf51": dcr_lf51,
    "esr_outcap": esr_outcap,
    "p_other": p_other
}

export_json = json.dumps(config_to_export, cls=NumpyEncoder, indent=2)
st.sidebar.download_button(
    label="匯出專案 (Export JSON)",
    data=export_json,
    file_name="flyback_config.json",
    mime="application/json"
)

# PDF Report Generation
results_for_report = {
    "p_out": p_out,
    "total_loss": total_loss,
    "p_in": p_in,
    "efficiency": efficiency
}

# Capture Plotly Chart for PDF (Efficiency vs. Load)
chart_img_buffer = None
try:
    img_bytes = fig_eff_load.to_image(format="png", engine="kaleido")
    chart_img_buffer = BytesIO(img_bytes)
except Exception as e:
    # Silent warning in sidebar, the PDF will show the failure text
    st.sidebar.warning(f"Note: Plotly snapshot failed ({e}). Check 'kaleido'.")

pdf_output = create_pdf_report(config_to_export, results_for_report, chart_img_buffer)
pdf_bytes = bytes(pdf_output)  # Ensure data is bytes for Streamlit
st.sidebar.download_button(
    label="📄 下載 PDF 測試報告",
    data=pdf_bytes,
    file_name="flyback_test_report.pdf",
    mime="application/pdf"
)
