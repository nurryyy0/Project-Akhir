import os
import random
import html
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from fpdf import FPDF 
import io
from datetime import datetime
import streamlit.components.v1 as components
import base64
import nbformat
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, balanced_accuracy_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from matplotlib.patches import Patch
import plotly.graph_objects as go

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Smart Soil Fertility - Klasifikasi Lahan",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; box-sizing: border-box; }
.stApp { background-color: #0f1715; color: #e8efe9; }
section[data-testid="stSidebar"] { background-color: #0a1411; }
section[data-testid="stSidebar"] * { color: #e8efe9 !important; }
.block-container { padding-top: 2rem; }
h1, h2, h3, h4 { color: #e8efe9; }
.metric-card {
    background: #16201c;
    border: 1px solid #1f2d27;
    border-radius: 12px;
    padding: 18px 20px;
}
.metric-card .label { color: #9bb0a4; font-size: 0.85rem; }
.metric-card .value { color: #e8efe9; font-size: 2rem; font-weight: 700; }
.hero {
    background: linear-gradient(135deg,#1f6b3a,#0f3d22);
    padding: 28px 32px; border-radius: 14px; color: #fff;
   
}
.hero h1 { color:#fff; margin:0 0 8px 0; }
.hero p  { color:#d6efd9; margin:0; }
.pill {
    display:inline-block; padding:4px 12px; border-radius:999px;
    font-size:0.85rem; margin-right:8px; margin-bottom:6px;
}
.pill-red    { background:#3a1a1a; color:#ff8b8b; border:1px solid #5a2424; }
.pill-yellow { background:#3a2f12; color:#ffcd6b; border:1px solid #5a4720; }
.pill-green  { background:#13331f; color:#7ee29a; border:1px solid #1f5c34; }
.pill-gray{ background:#e5e7eb; color:#374151; border:1px solid #9ca3af;}

.feat-card {
    background:#16201c; border:1.5px solid #1f2d27;
    border-radius:10px; padding:14px 16px; height:100%;
}
.feat-card .name { font-weight:600; color:#e8efe9; }
.feat-card .unit { float:right; color:#9bb0a4; font-size:0.8rem; }
.feat-card .desc { color:#9bb0a4; font-size:0.85rem; margin-top:4px; }
div[role='radiogroup'] label {
    display: flex !important;
    align-items: center;
    padding: 0.6rem 0.75rem;
    border-radius: 8px;
    font-size: 0.95rem;
    cursor: pointer;
    transition: background .2s;
    border: 0.5px solid #1e4030;
}
div[role='radiogroup'] label:hover {
    background: rgba(45,106,45,0.15);
}
div[role='radiogroup'] input[type='radio'] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "axes.facecolor": "#16201c",
    "figure.facecolor": "#16201c",
    "axes.edgecolor": "#2a3a33",
    "axes.labelcolor": "#cfd8d2",
    "xtick.color": "#cfd8d2",
    "ytick.color": "#cfd8d2",
    "text.color": "#e8efe9",
    "axes.grid": True,
    "grid.color": "#243029",
})

GREEN   = "#2fa05a"
ORANGE  = "#e8a13a"
RED     = "#d96a6a"
PALETTE = {0: RED, 1: ORANGE, 2: GREEN}
LABEL_NAME = {0: "Kurang Subur", 1: "Cukup Subur", 2: "Sangat Subur"}

# ============================================================
# DATA & MODEL — hanya didefinisikan SEKALI
# ============================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset1.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "RF.joblib")
    return joblib.load(model_path)

@st.cache_data
def evaluate_model(_model):
    """
    Evaluasi pakai split yang sama dengan training di Jupyter
    (random_state=42, test_size=0.2, stratify=y).
    Model TIDAK dilatih ulang — langsung predict.
    """
    df  = load_data()
    X   = df.drop("Output", axis=1)
    y   = df["Output"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = _model.predict(X_test)

    return {
        "test_acc"    : accuracy_score(y_test, y_pred),
        "bal_acc"     : balanced_accuracy_score(y_test, y_pred),
        "f1"          : f1_score(y_test, y_pred, average="macro"),
        "report"      : classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "cm"          : confusion_matrix(y_test, y_pred, labels=[0, 1, 2]),
    }

# ============================================================
# SIDEBAR — Hanya Judul
# ============================================================
def get_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
with st.sidebar:
    logo_b64 = get_image_base64(os.path.join(os.path.dirname(__file__), "ChatGPT Image 10 Mei 2026, 10.30.51.png"))

with st.sidebar:
    logo_b64 = get_image_base64(os.path.join(os.path.dirname(__file__), "ChatGPT Image 10 Mei 2026, 10.30.51.png"))
    st.markdown(f"""
        <div style='text-align:center; padding: 1rem 0 0.5rem;'>
            <img src="data:image/png;base64,{logo_b64}"
                style="width:150px; height:90px; object-fit:contain;
                       display:block; margin:0 auto 8px auto;
                       mix-blend-mode: lighten;">
            <div style='font-weight:700; font-size:1.1rem; color:#4ade80;'>Soil Fertility Predictor</div>
            <div style='font-size:0.75rem; color:#888; margin-top:2px;'>Klasifikasi Kesuburan Tanah</div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    ## 📖 Petunjuk Penggunaan Aplikasi
    01. Pergi ke tab Klasifikasi Lahan untuk memulai prediksi kesuburan tanah.
    02. Input Parameter Tanah, masukkan nilai parameter seperti N, P, K, pH, EC, OC, dan unsur mikro lainnya.
    03. Klik Prediksi, model akan memproses data dan mengklasifikasi kesuburan secara instan.
    04. Hasil indeks kesuburan dan rekomendasi tindakan akan ditampilkan secara lengkap.
        """)
    st.divider()
    st.caption("Machine Learning Project by Nurry")

df    = load_data()
model = load_model() 

st.markdown("""
    <div class="hero">
    <h1>🌿 Smart Soil Fertility Prediction System</h1>
    <p>Sistem cerdas untuk menentukan tingkat kesuburan tanah berdasarkan berbagai
       parameter kimia tanah secara cepat, praktis, dan mudah digunakan.</p>
    </div>
""", unsafe_allow_html=True)

# ============================================================
# TABS NAVIGASI — di bawah judul aplikasi
# ============================================================

st.markdown("""
<style>
.stTabs {
    margin-top: 30px;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 18px;
    padding-bottom: 10px;
    border-bottom: 1.5px solid rgba(120, 180, 170, 0.15);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Sora', sans-serif;
    font-size: 50px;
    font-weight: 600;
    color: #7fa39a;
    padding: 16px 28px;
    border-radius: 16px;
    background: transparent;
    transition: all 0.25s ease;
    border: 1px solid transparent;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #d8fff3 !important;
    background: rgba(14,240,184,0.06);
    transform: translateY(-2px);
}
.stTabs [aria-selected="true"] {
    color: #0ef0b8 !important;
    background:
        linear-gradient(
            135deg,
            rgba(14,240,184,0.14),
            rgba(14,240,184,0.05)
        ) !important;
    border: 1px solid rgba(14,240,184,0.22) !important;
    box-shadow:
        0 4px 18px rgba(14,240,184,0.08);
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}
.stTabs [data-baseweb="tab-panel"] {
    margin-top: 12px;
    padding: 1.8rem;
    border-radius: 18px;
    background: rgba(10, 24, 24, 0.55);
    border: 1px solid rgba(120, 180, 170, 0.12);
}
@media (max-width: 768px) {
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        padding: 12px 18px;
    }
}
</style>
""", unsafe_allow_html=True)
tab1, tab2, tab3,tab4 = st.tabs([ "📊 Prediksi Lahan","🏠 Informasi", "🔍 Analisis Dataset",  "📋 Tentang"])

#================================ 
# PAGE INFORMASI
#================================
def page_beranda():

    # ── Hero ─────────────────────────────────────────────────────
    st.markdown("""
        <h1 style="font-size:45px; font-weight:800; color:white; line-height:1.2; margin-bottom:8px;">
            Prediksi Tingkat<br><span style="color:#6fcf97;">Kesuburan Tanah</span>
        </h1>
        <p style="font-size:16px; color:white; line-height:1.6; margin-bottom:28px;">
            Dukung pertanian yang lebih modern dengan memanfaatkan teknologi untuk 
            mengetahui kondisi tanah secara lebih mudah dan efisien.
        </p>
    """, unsafe_allow_html=True)
    # ── Tentang Aplikasi ─────────────────────────────────────────
    st.markdown("""
        <p style="font-size:30px; font-weight:700; color:white; margin-bottom:4px;">📱 Tentang Aplikasi</p>
        <div style="background:#1a2e22; border:1px solid rgba(111,207,151,0.15);
                    border-radius:14px; padding:20px 22px; margin-bottom:16px;
                    font-size:16px; color:white; line-height:1.8;">
            Aplikasi berbasis Streamlit yang dirancang untuk memprediksi tingkat kesuburan tanah 
            menggunakan algoritma Machine Learning. Aplikasi ini menganalisis berbagai parameter kimia tanah 
            seperti Nitrogen (N), Fosfor (P), Kalium (K), pH, Electrical Conductivity (EC), Organic Carbon (OC), 
            Sulfur (S), Zinc (Zn), Iron (Fe), Copper (Cu), Manganese (Mn), dan Boron (B) untuk menentukan 
            kategori kesuburan tanah menjadi
            <strong style="color:#6fcf97;">Kurang Subur</strong>,
            <strong style="color:#6fcf97;">Cukup Subur</strong>, atau
            <strong style="color:#6fcf97;">Sangat Subur</strong>.
        </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.subheader("🧪 Penjelasan Fitur")
    feats = [
        ("Nitrogen (N)",              "kg/ha", "Nutrisi utama untuk pertumbuhan daun & batang (fase vegetatif)"),
        ("Phosphorus (P)",            "kg/ha", "Penting untuk Mendukung perkembangan akar, biji, dan pembungaan"),
        ("Kalium (K)",                "kg/ha", "Meningkatkan ketahanan tanaman terhadap penyakit"),
        ("pH Tanah",                  "-",     "Skala keasaman tanah; mempengaruhi ketersediaan semua nutrisi"),
        ("Electrical Conductivity",   "dS/m",  "Mengukur kadar garam terlarut dalam tanah"),
        ("Organic Carbon (OC)",       "%",     "Kandungan karbon organik dalam tanah sebagai Indikator kesuburan & kesehatan biologis tanah"),
        ("Sulfur (S)",                "ppm",   "Komponen asam amino unsur penting untuk sintesis protein"),
        ("Zinc (Zn)",                 "ppm",   "Mikronutrien esensial untuk enzim tanaman"),
        ("Iron (Fe)",                 "ppm",   "Diperlukan untuk pembentukan klorofil"),
        ("Copper (Cu)",               "ppm",   "Berperan dalam fotosintesis, respirasi, dan lignifikasi"),
        ("Manganese (Mn)",            "ppm",   "Penting untuk mendukung metabolisme nitrogen & aktivasi enzim"),
        ("Boron (B)",                 "ppm",   "Esensial untuk pembentukan dinding sel & transportasi gula"),
    ]
    cols = st.columns(3)
    for i, (n, u, d) in enumerate(feats):
        with cols[i % 3]:
            st.markdown(
                f"<div class='feat-card'><span class='unit'>{u}</span>"
                f"<div class='name'>{n}</div><div class='desc'>{d}</div></div>",
                unsafe_allow_html=True,
            )
            st.write("")
    # ── Tujuan Aplikasi ──────────────────────────────────────────
    tujuan = [
        ("🌿", "Membantu pengguna memahami kondisi kesuburan tanah secara mudah dan cepat."),
        ("📊", "Memberikan prediksi tingkat kesuburan berdasarkan parameter tanah yang diinputkan."),
        ("📇", "Menyediakan visualisasi data agar informasi lebih mudah dipahami pengguna."),
        ("📈", "Membantu meningkatkan produktivitas pertanian melalui analisis tanah yang lebih akurat."),
    ]

    items_html = ""
    for icon, text in tujuan:
        items_html += f"""
        <div style="display:flex; align-items:flex-start; gap:12px;
                    padding:14px 0; border-bottom:1px solid rgba(255,255,255,0.06);">
            <div style="width:32px; height:32px; background:rgba(111,207,151,0.15);
                        border-radius:8px; display:flex; align-items:center;
                        justify-content:center; font-size:16px; flex-shrink:0;">
                {icon}
            </div>
            <div style="font-size:16px; color:white;
                        line-height:1.6; padding-top:6px;">
                {text}
            </div>
        </div>
        """

    st.markdown(f"""
        <p style="font-size:30px; font-weight:700; color:white; margin-bottom:4px;"> 📲 Tujuan Aplikasi</p>
        <div style="background:#1a2e22; border:1px solid rgba(111,207,151,0.15);
                    border-radius:14px; padding:8px 22px; margin-bottom:16px;">
            {items_html}
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div style='text-align:center; padding: 0.5rem 0;'>
        <div style='font-size:15px; color:#3a5244;'>© 2026 - Soil Fertility Predictor</div>
        <div style='font-size:17px; color:#556b5f; margin-top:2px; font-weight:600;'>Machine Learning Project - Nurry</div>
    </div>
""", unsafe_allow_html=True)
#=============================================================
# PAGE: KLASIFIKASI LAHAN # PDF di notepad
# ============================================================
def buat_pdf_laporan(predicted_label, conf, warna_hex, deskripsi,
                     rows, kurang, berlebih, prob_df, fig):
    pdf = FPDF()
    prob_df = None
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def hex2rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    # ── HEADER ────────────────────────────────────────────────────────────────
    pdf.set_fill_color(30, 42, 39)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.set_y(8)
    pdf.cell(0, 10, "Laporan Prediksi Kesuburan Tanah", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(180, 200, 190)
    pdf.cell(0, 6, f"Digenerate: {datetime.now().strftime('%d %B %Y  %H:%M')}", ln=True, align="C")
    pdf.ln(6)

    # ── 1. HASIL PREDIKSI ─────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(60, 60, 60)
    pdf.set_fill_color(240, 245, 242)
    pdf.cell(0, 8, "  1. Hasil Prediksi", ln=True, fill=True)
    pdf.ln(3)

    r, g, b = hex2rgb(warna_hex)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(r, g, b)
    pdf.cell(0, 12, predicted_label, ln=True, align="C")

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, f"Keyakinan Model: {conf:.2f}%", ln=True, align="C")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 7, "Deskripsi:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(70, 70, 70)
    for baris in deskripsi.replace("<br>", "\n").split("\n"):
        baris = baris.strip().lstrip("-").strip()
        if baris:
            pdf.set_x(15)
            pdf.multi_cell(0, 6, f"  -  {baris}")
    pdf.ln(4)

    # ── 2. GRAFIK PROBABILITAS ────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(60, 60, 60)
    pdf.set_fill_color(240, 245, 242)
    pdf.cell(0, 8, "  2. Probabilitas per Kelas", ln=True, fill=True)
    pdf.ln(3)

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", dpi=120, bbox_inches="tight", facecolor="#FFFFFF")
    img_buf.seek(0)
    pdf.image(img_buf, x=30, w=150)
    pdf.ln(4)

    # ── 3. TABEL PERBANDINGAN ─────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(240, 245, 242)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, "  3. Perbandingan Input vs Data Ideal", ln=True, fill=True)
    pdf.ln(3)

    col_w  = [52, 38, 38, 28, 34]
    headers = ["Fitur", "Nilai Input", "Nilai Ideal", "Selisih", "Status"]
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(50, 80, 65)
    pdf.set_text_color(255, 255, 255)
    for w, h in zip(col_w, headers):
        pdf.cell(w, 7, h, border=1, fill=True, align="C")
    pdf.ln()

    STATUS_COLOR = {
        " Optimal" : (220, 245, 220),
        " Kurang"  : (255, 230, 230),
        " Berlebih": (255, 240, 210),
    }

    def bersihkan_teks(t):
        import re
        return re.sub(r'[^\x00-\x7F\u00C0-\u024F]', '', str(t)).strip()

    STATUS_PDF = {
        "✅ Optimal" : " Optimal",
        "⬇️ Kurang"  : " Kurang",
        "⬆️ Berlebih": " Berlebih",
    }

    pdf.set_font("Helvetica", "", 9)
    for row in rows:
        status_bersih = STATUS_PDF.get(row["Status"], row["Status"])
        vals = [
            bersihkan_teks(row["Fitur"]),
            bersihkan_teks(row["Nilai Input"]),
            bersihkan_teks(row["Nilai Ideal"]),
            bersihkan_teks(row["Selisih"]),
            bersihkan_teks(status_bersih),
        ]
        r2, g2, b2 = STATUS_COLOR.get(status_bersih, (245, 245, 245))
        pdf.set_fill_color(r2, g2, b2)
        pdf.set_text_color(40, 40, 40)
        for w, v in zip(col_w, vals):
            pdf.cell(w, 6, v, border=1, fill=True, align="C")
        pdf.ln()
    pdf.ln(4)

    # ── 4. REKOMENDASI ────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(240, 245, 242)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, "  4. Rekomendasi Perbaikan Lahan", ln=True, fill=True)
    pdf.ln(3)

    if not kurang and not berlebih:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(50, 150, 80)
        pdf.cell(0, 8, "Semua parameter sudah optimal!", ln=True, align="C")
    else:
        if kurang:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(229, 57, 53)
            pdf.cell(0, 7, "Parameter yang Perlu Ditingkatkan:", ln=True)
            for no, item in enumerate(kurang, 1):
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(50, 50, 50)
                pdf.set_x(20)
                pdf.cell(0, 6, f"{no}. {item['fitur']}", ln=True)

                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(80, 80, 80)

                pdf.set_x(25)
                pdf.multi_cell(160, 6, item['saran'])

                pdf.ln(2)

        if berlebih:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(255, 167, 38)
            pdf.cell(0, 7, "Parameter yang Perlu Dikurangi:", ln=True)
            for no, item in enumerate(berlebih, 1):
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(50, 50, 50)
                pdf.set_x(20)
                pdf.cell(0, 6, f"{no}. {item['fitur']}", ln=True)

                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(80, 80, 80)

                pdf.set_x(25)
                pdf.multi_cell(160, 6, item['saran'])

    return bytes(pdf.output())
#============================
# PAGE KLASIFIKASI 
#============================
def page_klasifikasi():
    st.title("Klasifikasi Lahan")
    st.subheader("Masukkan parameter tanah untuk memprediksi tingkat kesuburan")

    features = df.drop("Output", axis=1).columns.tolist()
    ranges   = {f: (float(df[f].min()), float(df[f].max())) for f in features}

    if "reset_counter" not in st.session_state:
        st.session_state.reset_counter = 0

    default_values = {
        "N":  264,  "P":  12.1, "K":  560,  "pH": 7.9,
        "EC": 0.63, "OC": 0.39, "S":  4.22, "Zn": 0.33,
        "Fe": 3.22, "Cu": 0.75, "Mn": 11.60,"B":  0.30,
    }

    NAMA_FITUR = {
        "N":  "Nitrogen (N)",            "P":  "Fosfor (P)",
        "K":  "Kalium (K)",              "pH": "pH Tanah",
        "EC": "Electrical Conductivity", "OC": "Organic Carbon (OC)",
        "S":  "Sulfur (S)",              "Zn": "Zinc (Zn)",
        "Fe": "Iron (Fe)",               "Cu": "Copper (Cu)",
        "Mn": "Manganese (Mn)",          "B":  "Boron (B)",
    }

    SATUAN = {
        "N":"mg/kg","P":"mg/kg","K":"mg/kg","pH":"",
        "EC":"dS/m","OC":"%",  "S":"mg/kg","Zn":"mg/kg",
        "Fe":"mg/kg","Cu":"mg/kg","Mn":"mg/kg","B":"mg/kg",
    }

    DATA_IDEAL = {
        "N":314.0,"P":83.3, "K":486.0,"pH":7.41,
        "EC":0.80,"OC":0.68,"S":7.54, "Zn":0.51,
        "Fe":7.63,"Cu":1.36,"Mn":12.06,"B":0.25,
    }

    SARAN_FITUR = {
        "N" :("Tambahkan pupuk Urea atau ZA untuk mempercepat pertumbuhan vegetatif & meningkatkan Nitrogen.",
              "Kurangi pupuk Nitrogen; kelebihan bisa membuat tanaman terlalu rimbun, sukulen, dan mudah roboh (lodging)."),
        "P" :("Tambahkan pupuk SP-36 atau TSP untuk penguatan akar dan bunga.",
              "Kurangi pupuk fosfat, kelebihan P dapat mengikat dan menghambat penyerapan Seng (Zn) dan Besi (Fe)."),
        "K" :("Tambahkan pupuk KCl atau K2SO4 untuk daya tahan penyakit dan kualitas buah.",
              "Kurangi pupuk kalium, kelebihan K  mengganggu penyerapan Magnesium (Mg) dan Kalsium (Ca)."),
        "pH":("Lakukan pengapuran dengan Dolomit atau kapur pertanian untuk menetralkan keasaman.",
              "Tambahkan belerang (sulfur) atau bahan organik asam (seperti gambut) untuk menurunkan pH."),
        "EC":("Perbaiki struktur tanah dan tambahkan bahan organik untuk meningkatkan EC.",
              "Lakukan penggenangan dan pengaliran air (leaching) untuk mencuci tumpukan garam keluar dari zona akar."),
        "OC":("Tambahkan kompos atau pupuk kandang untuk meningkatkan kuailtas humus.",
              "Kurangi frekuensi pembajakan tanah agar mikroorganisme tidak merombak bahan organik terlalu cepat."),
        "S" :("Tambahkan pupuk ZA atau gipsum untuk meningkatkan kandungan Sulfur.",
              "Kurangi pupuk berbasis sulfur agar tidak terjadi pengasaman tanah yang ekstrem."),
        "Zn":("Semprotkan larutan Zinc Sulfate (ZnSO4) untuk mengatasi defisiensi Zinc.",
              "Kurangi aplikasi Zinc, kelebihan Zn dapat meracuni tanaman dan menghambat penyerapan Besi (Fe) dan Mn."),
        "Fe":("Tambahkan chelated iron (Fe-EDTA) atau pupuk mikro mengandung besi.",
              "Perbaiki sistem drainase tanah karena kelebihan Besi biasanya terjadi karena tanah terlalu basah atau tergenang."),
        "Cu":("Semprotkan Copper Sulfate (CuSO4) untuk mengatasi defisiensi tembaga.",
              "Kurangi aplikasi tembaga, kelebihan Tembaga bersifat sangat toksik (racun) bagi sel tanaman."),
        "Mn":("Tambahkan Manganese Sulfate pada media tanam untuk meningkatkan kadar Mangan.",
              "Perbaiki drainase dan Naikkan pH tanah melalui pengapuran untuk mengendapkan kelebihan Mangan agar tidak diserap tanaman."),
        "B" :("Semprotkan Borax atau pupuk mikro khusus Boron dengan dosis sangat rendah.",
              "Hentikan total aplikasi; Boron memiliki rentang aman yang sangat sempit, sedikit saja berlebih akan langsung membuat tepi daun terbakar (toksisitas)."),
    }

    WARNA_KELAS = {
        "Kurang Subur": "#E53935",
        "Cukup Subur":  "#e8a13a",
        "Sangat Subur": "#43A047",
    }

    WARNA_BAR = {
        "Kurang Subur": "#E53935",
        "Cukup Subur":  "#e8a13a",
        "Sangat Subur": "#43A047",
    }

    DESKRIPSI = {
        "Kurang Subur": (
            "- Kandungan unsur hara sangat rendah, tidak mencukupi kebutuhan tanaman.<br>"
            "- Struktur tanah buruk dan pH tidak seimbang.<br>"
            "- Memerlukan perbaikan menyeluruh sebelum dapat ditanami."
        ),
        "Cukup Subur": (
            "- Kandungan unsur hara cukup untuk mendukung pertumbuhan tanaman.<br>"
            "- Kondisi tanah sudah layak tanam dengan pemeliharaan rutin.<br>"
            "- Produktivitas dapat ditingkatkan dengan pemupukan berimbang."
        ),
        "Sangat Subur": (
            "- Kandungan unsur hara lengkap dan optimal untuk pertumbuhan tanaman.<br>"
            "- Struktur tanah baik, pH seimbang, dan mikronutrien mencukupi.<br>"
            "- Lahan siap untuk produksi pertanian intensif."
        ),
    }

    REKOMENDASI = {
        "Kurang Subur": (
            "🌱 **Rekomendasi Tindakan:**\n"
            "- Lakukan pengapuran jika pH terlalu asam (< 5.5)\n"
            "- Tambahkan pupuk organik (kompos/pupuk kandang) untuk memperbaiki struktur tanah\n"
            "- Aplikasi pupuk NPK dengan dosis tinggi sebelum tanam\n"
            "- Pertimbangkan tanaman penutup tanah (cover crop) untuk memulihkan kesuburan"
        ),
        "Cukup Subur": (
            "🌾 **Rekomendasi Tindakan:**\n"
            "- Pertahankan pemupukan berimbang (N, P, K sesuai kebutuhan tanaman)\n"
            "- Lakukan uji tanah berkala setiap musim tanam\n"
            "- Tambahkan bahan organik secara rutin untuk menjaga struktur tanah\n"
            "- Optimalkan irigasi agar unsur hara terserap efisien"
        ),
        "Sangat Subur": (
            "🏆 **Rekomendasi Tindakan:**\n"
            "- Pertahankan kondisi tanah dengan rotasi tanaman\n"
            "- Hindari penggunaan pupuk kimia berlebihan agar tidak merusak keseimbangan\n"
            "- Manfaatkan lahan untuk budidaya tanaman bernilai ekonomi tinggi\n"
            "- Dokumentasikan praktik pertanian sebagai acuan lahan lain"
        ),
    }

    # ── Input Form ────────────────────────────────────────────────────────────
    if "reset_counter" not in st.session_state:
        st.session_state.reset_counter = 0
    if "just_reset" not in st.session_state:
        st.session_state.just_reset = False

    counter  = st.session_state.reset_counter
    is_reset = st.session_state.get("just_reset", False)
    if is_reset:
        st.session_state.just_reset = False

    cols   = st.columns(4)
    inputs = {}
    for i, f in enumerate(features):
        lo, hi = ranges[f]
        with cols[i % 4]:
            inputs[f] = st.number_input(
                NAMA_FITUR.get(f, f),
                value=0.0 if is_reset else float(default_values.get(f, (lo + hi) / 2)),
                step=0.1,
                format="%.2f",
                placeholder="...",
                key=f"input_{f}_{counter}",
            )

    st.write("")
    b1, b2  = st.columns(2)
    do_pred = b1.button("🔍 Prediksi Kesuburan", use_container_width=True, type="primary")
    if b2.button("🔄 Reset", use_container_width=True):
        st.session_state.reset_counter += 1
        st.session_state.just_reset = True
        st.rerun()

    if do_pred:
        kosong = [NAMA_FITUR.get(f, f) for f in features if inputs[f] == 0.0]
        if kosong:
            st.error("🚫 Harap lengkapi semua fitur berikut sebelum melakukan prediksi:")
            for nama in kosong:
                st.warning(f"⚠️ **{nama}** wajib diisi, parameter harus lebih dari 0,00).")
            st.stop()
        else:
            pass  # lanjut ke logika prediksi
    # ── Prediksi ──────────────────────────────────────────────────────────────
    if do_pred:
        empty = [f for f in features if inputs[f] is None]
        if empty:
            st.warning(f"Harap isi semua input terlebih dahulu: {', '.join(empty)}")
        else:
            x     = pd.DataFrame([[inputs[f] for f in features]], columns=features)
            pred  = int(model.predict(x)[0])
            proba = model.predict_proba(x)[0]
            conf  = proba[list(model.classes_).index(pred)] * 100

            predicted_label = LABEL_NAME[pred]
            warna           = WARNA_KELAS[predicted_label]

            ICON_BESAR = {
                "Kurang Subur": "⚠️",
                "Cukup Subur":  "🌱",
                "Sangat Subur": "🏆",
            }
            GRADE = {
                "Kurang Subur": "1",
                "Cukup Subur":  "2",
                "Sangat Subur": "3",
            }

            icon_besar  = ICON_BESAR[predicted_label]
            grade_label = GRADE[predicted_label]

            # ── Divider & header ──────────────────────────────────────────────
            st.markdown("---")
            st.markdown(
                "<h2 style='text-align:center; letter-spacing:1px; "
                "font-size:40px; color:#9bb0a4; margin-bottom:4px;'>"
                "📋 HASIL ANALISIS KESUBURAN TANAH"
                "</h2>",
                unsafe_allow_html=True,
            )

            # ── 1. Hero Card ──────────────────────────────────────────────────
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #0d1f18 0%, #132b20 60%, #1a3828 100%);
                    border: 2px solid {warna}40;
                    border-left: 6px solid {warna};
                    border-radius: 16px;
                    padding: 32px 36px;
                    margin: 18px 0 10px 0;
                    box-shadow: 0 8px 32px {warna}22, 0 2px 8px #00000066;
                    position: relative;
                    overflow: hidden;
                ">
                    <div style="
                        position:absolute; top:-40px; right:-40px;
                        width:160px; height:160px;
                        background: radial-gradient(circle, {warna}30 0%, transparent 70%);
                        border-radius:50%; pointer-events:none;
                    "></div>
                    <div style="display:flex; align-items:center; gap:28px; flex-wrap:wrap;">
                        <div style="
                            font-size:3.8rem;
                            width:90px; height:90px;
                            background: {warna}18;
                            border: 2px solid {warna}55;
                            border-radius:50%;
                            display:flex; align-items:center; justify-content:center;
                            flex-shrink:0;
                        ">{icon_besar}</div>
                        <div style="flex:1; min-width:200px;">
                            <div style="
                                font-size:0.78rem; font-weight:700;
                                letter-spacing:2.5px; color:{warna}cc;
                                text-transform:uppercase; margin-bottom:4px;
                            ">STATUS KESUBURAN TANAH</div>
                            <div style="
                                font-size:2.6rem; font-weight:900;
                                color:{warna}; line-height:1.1;
                                letter-spacing:-0.5px;
                            ">{predicted_label}</div>
                            <div style="color:#9bb0a4; font-size:0.88rem; margin-top:6px;">
                                {DESKRIPSI[predicted_label]}
                            </div>
                        </div>
                        <div style="
                            text-align:center;
                            background: {warna}18;
                            border: 2px solid {warna}44;
                            border-radius:14px;
                            padding: 14px 24px;
                            flex-shrink:0;
                        ">
                            <div style="font-size:0.72rem; color:#9bb0a4;
                                        letter-spacing:2px; text-transform:uppercase;">KELAS</div>
                            <div style="font-size:3rem; font-weight:900;
                                        color:{warna}; line-height:1.1;">{grade_label}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── 2. Metrics row ────────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                    <div style='background:#0a1a12; border:1px solid #1f3028; border-radius:10px; padding:14px 16px;'>
                        <div style='font-size:0.7rem; color:#556b5f; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;'>🎯 Keyakinan Model</div>
                        <div style='font-size:1.6rem; font-weight:700; color:#e8efe9;'>{conf:.1f}%</div>
                        <div style='font-size:0.75rem; margin-top:4px; color:{"#4ade80" if conf >= 75 else ("#e8a13a" if conf >= 50 else "#E53935")};'>
                            {"▲ Tinggi" if conf >= 75 else ("● Sedang" if conf >= 50 else "▼ Rendah")}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                    <div style='background:#0a1a12; border:1px solid #1f3028; border-radius:10px; padding:14px 16px;'>
                        <div style='font-size:0.7rem; color:#556b5f; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;'>🌿 Status Lahan</div>
                        <div style='font-size:1.6rem; font-weight:700; color:#e8efe9;'>{predicted_label}</div>
                        <div style='font-size:0.75rem; margin-top:4px; color:#3a5244;'>Hasil Klasifikasi</div>
                    </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                    <div style='background:#0a1a12; border:1px solid #1f3028; border-radius:10px; padding:14px 16px;'>
                        <div style='font-size:0.7rem; color:#556b5f; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;'>📊 Prediksi Kelas</div>
                        <div style='font-size:1.6rem; font-weight:700; color:#e8efe9;'>{grade_label}</div>
                        <div style='font-size:0.75rem; margin-top:4px; color:#3a5244;'>Grade Lahan</div>
                    </div>
                """, unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                    <div style='background:#0a1a12; border:1px solid #1f3028; border-radius:10px; padding:14px 16px;'>
                        <div style='font-size:0.7rem; color:#556b5f; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;'>🧪 Parameter Diuji</div>
                        <div style='font-size:1.6rem; font-weight:700; color:#e8efe9;'>{len(features)} Fitur</div>
                        <div style='font-size:0.75rem; margin-top:4px; color:#3a5244;'>Total Input</div>
                    </div>
                """, unsafe_allow_html=True)
            # ── 3. Probabilitas per Kelas ─────────────────────────────────────
            st.markdown("---")
            st.subheader("📈 Distribusi Probabilitas Kelas")

            labels_prob = [LABEL_NAME[c] for c in model.classes_]
            values_prob = [round(p * 100, 2) for p in proba]
            colors_prob = [WARNA_BAR.get(l, "#888") for l in labels_prob]

            fig_prob = go.Figure()
            for lbl, val, clr in zip(labels_prob, values_prob, colors_prob):
                fig_prob.add_trace(go.Bar(
                    y=[lbl], x=[val], orientation='h',
                    marker=dict(color=clr, line=dict(color=clr, width=1)),
                    text=f"<b>{val:.2f}%</b>",
                    textposition='outside',
                    textfont=dict(size=14, color=clr),
                    hovertemplate=f"<b>{lbl}</b><br>Probabilitas: {val:.2f}%<extra></extra>",
                    name=lbl,
                ))

            fig_prob.update_layout(
                plot_bgcolor="#0d1f18", paper_bgcolor="#0d1f18",
                height=200,
                margin=dict(l=10, r=80, t=10, b=10),
                xaxis=dict(
                    range=[0, 115], showgrid=True, gridcolor="#1f3028",
                    ticksuffix="%", tickfont=dict(color="#9bb0a4", size=11),
                    zeroline=False,
                ),
                yaxis=dict(tickfont=dict(color="#e8efe9", size=13), showgrid=False),
                showlegend=False, bargap=0.35,
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            # ── 4. Gauge ──────────────────────────────────────────────────────
            
            proba = model.predict_proba(x)[0]

            indeks_kesuburan = round(
                proba[0] * 20 +   # Kurang Subur
                proba[1] * 60 +   # Cukup Subur
                proba[2] * 100,   # Sangat Subur
                1
            )

            # Label indeks
            if indeks_kesuburan >= 66:
                label_indeks = "Sangat Subur"
                warna_indeks = "#43A047"
            elif indeks_kesuburan >= 40:
                label_indeks = "Cukup Subur"
                warna_indeks = "#e8a13a"
            else:
                label_indeks = "Kurang Subur"
                warna_indeks = "#E53935"

            GAUGE_COLOR = {
                "Kurang Subur": ["#E53935", "#b71c1c"],
                "Cukup Subur":  ["#e8a13a", "#f57f17"],
                "Sangat Subur": ["#43A047", "#1b5e20"],
            }
            bar_clr, ref_clr = GAUGE_COLOR[predicted_label]

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=indeks_kesuburan,
                number={"suffix": "", "font": {"size": 40, "color": "#e8efe9"}},
                
                title={
                    "text": f"Indeks Kesuburan Tanah<br><span style='color:{warna_indeks}; font-size:16px;'>▶ {label_indeks}</span>",
                    "font": {"size": 25, "color": "white"}
                },
                gauge={
                    "axis": {
                        "range": [0, 100], "tickwidth": 1,
                        "tickcolor": "#9bb0a4",
                        "tickvals": [0, 20, 40, 60, 80, 100],
                        "ticktext": ["0", "20", "40", "60", "80", "100"],
                        "tickfont": {"color": "#9bb0a4", "size": 10},
                    },
                    "bar": {"color": bar_clr, "thickness": 0.28},
                    "bgcolor": "#0d1f18", "borderwidth": 0,
                    "steps": [
                        {"range": [0,  33], "color": "#1a1a1a"},
                        {"range": [33, 66], "color": "#1f2d27"},
                        {"range": [66, 100], "color": "#213327"},
                    ],
                    "threshold": {
                        "line": {"color": ref_clr, "width": 3},
                        "thickness": 0.8, "value": indeks_kesuburan,
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0d1f18",
                font={"color": "#e8efe9"},
                height=390,
                margin=dict(l=20, r=20, t=30, b=10),
            )

            fig_gauge.add_annotation(
                x=0.0, y=0.0,
                xref="paper", yref="paper",
                text="<b style='color:#E53935'>⬤</b> <span style='color:#888'>Kurang Subur</span>  0 – 39",
                showarrow=False,
                font={"size": 10, "color": "#888"},
                align="left",
                xanchor="left",
                yanchor="bottom",
                bgcolor="#1a1a1a",
                bordercolor="#333",
                borderwidth=1,
                borderpad=5,
            )

            fig_gauge.add_annotation(
                x=0.5, y=0.0,
                xref="paper", yref="paper",
                text="<b style='color:#e8a13a'>⬤</b> <span style='color:#9bb0a4'>Cukup Subur</span>  40 – 65",
                showarrow=False,
                font={"size": 10, "color": "#9bb0a4"},
                align="center",
                xanchor="center",
                yanchor="bottom",
                bgcolor="#1f2d27",
                bordercolor="#2e4a38",
                borderwidth=1,
                borderpad=5,
            )

            fig_gauge.add_annotation(
                x=1.0, y=0.0,
                xref="paper", yref="paper",
                text="<b style='color:#43A047'>⬤</b> <span style='color:#9bb0a4'>Sangat Subur</span>  66 – 100",
                showarrow=False,
                font={"size": 10, "color": "#9bb0a4"},
                align="right",
                xanchor="right",
                yanchor="bottom",
                bgcolor="#213327",
                bordercolor="#2e5e3a",
                borderwidth=1,
                borderpad=5,
            )
            # ── 5. Radar Chart ────────────────────────────────────────────────
            radar_features = list(DATA_IDEAL.keys())
            radar_labels   = [NAMA_FITUR[f] for f in radar_features]

            def norm(f, val):
                lo, hi = ranges[f]
                if hi == lo:
                    return 0.5
                return max(0.0, min(1.0, (val - lo) / (hi - lo)))

            input_norm = [norm(f, float(inputs[f])) for f in radar_features]
            ideal_norm = [norm(f, DATA_IDEAL[f])    for f in radar_features]

            input_norm_closed = input_norm + [input_norm[0]]
            ideal_norm_closed = ideal_norm + [ideal_norm[0]]
            labels_closed     = radar_labels + [radar_labels[0]]

            def hex_to_rgba(hex_color, alpha=0.13):
                h = hex_color.lstrip('#')
                r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                return f"rgba({r},{g},{b},{alpha})"

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=ideal_norm_closed, theta=labels_closed,
                fill='toself', name='Data Ideal',
                line=dict(color='#43A047', width=2, dash='dot'),
                fillcolor='rgba(67,160,71,0.10)',
                marker=dict(size=5, color='#43A047'),
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=input_norm_closed, theta=labels_closed,
                fill='toself', name='Input Anda',
                line=dict(color=warna, width=2.5),
                fillcolor=hex_to_rgba(warna, 0.18),
                marker=dict(size=6, color=warna),
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="#0d1f18",
                    radialaxis=dict(
                        visible=True, range=[0, 1],
                        showticklabels=False,
                        gridcolor="#1f3028", linecolor="#1f3028",
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=10, color="#9bb0a4"),
                        gridcolor="#1f3028", linecolor="#1f3028",
                    ),
                ),
                paper_bgcolor="#0d1f18", plot_bgcolor="#0d1f18",
                font=dict(color="#e8efe9"),
                legend=dict(
                    font=dict(color="#e8efe9", size=11),
                    bgcolor="#132b20", bordercolor="#1f3028",
                    borderwidth=1, x=0.82, y=1.15,
                ),
                margin=dict(l=30, r=30, t=40, b=30),
                height=390,
                title=dict(
                    text="Profil Parameter: Input vs Ideal",
                    font=dict(size=25, color="white"),
                    x=0.3,
                   
                ),
            )

            col_g, col_r = st.columns([1, 1.6])
            with col_g:
                st.plotly_chart(fig_gauge, use_container_width=True)
            with col_r:
                st.plotly_chart(fig_radar, use_container_width=True)

            # ── 6. Tabel Perbandingan ─────────────────────────────────────────
            st.markdown("---")
            st.subheader("📊 Perbandingan Input vs Data Ideal")

            # mengambil label kelas tertinggi (Sangat Subur)
            _kelas_sangat_subur = max(df["Output"].unique())
            # Hitung standar deviasi tiap fitur di kelas Sangat Subur
            # Standar deviasi = rata-rata jarak tiap nilai dari rata-ratanya
            # → semakin besar std, semakin lebar sebaran data di kelas itu
            # → dipakai sebagai toleransi karena mencerminkan "batas wajar" data asli
            _std = df[df["Output"] == _kelas_sangat_subur][list(DATA_IDEAL.keys())].std()

            # Ubah ke dictionary agar bisa dipanggil per fitur: TOLERANSI_ABSOLUT["N"], dll
            # Tujuan: input yang memang Sangat Subur tidak akan kena rekomendasi kurang/lebih
            TOLERANSI_ABSOLUT = _std.to_dict()

            rows     = []
            kurang   = []
            berlebih = []

            for fitur, ideal_val in DATA_IDEAL.items():
                input_val = float(inputs[fitur])
                selisih   = input_val - ideal_val
                toleransi = TOLERANSI_ABSOLUT.get(fitur, ideal_val * 0.10)

                if abs(selisih) <= toleransi:
                    status = "✅ Optimal"
                elif selisih < 0:
                    status = "⬇️ Kurang"
                    kurang.append({
                        "fitur": NAMA_FITUR[fitur],
                        "saran": SARAN_FITUR[fitur][0],
                    })
                else:
                    status = "⬆️ Berlebih"
                    berlebih.append({
                        "fitur": NAMA_FITUR[fitur],
                        "saran": SARAN_FITUR[fitur][1],
                    })

                rows.append({
                    "Fitur"      : NAMA_FITUR[fitur],
                    "Nilai Input": f"{input_val:.2f} {SATUAN[fitur]}".strip(),
                    "Nilai Ideal": f"{ideal_val:.2f} {SATUAN[fitur]}".strip(),
                    "Selisih"    : f"{selisih:+.2f}",
                    "Status"     : status,
                })

            total_param   = len(rows)
            optimal_count = sum(1 for r in rows if "Optimal"  in r["Status"])
            kurang_count  = sum(1 for r in rows if "Kurang"   in r["Status"])
            lebih_count   = sum(1 for r in rows if "Berlebih" in r["Status"])

            ts1, ts2, ts3 = st.columns(3)
            ts1.metric("✅ Optimal",  f"{optimal_count}/{total_param} param")
            ts2.metric("⬇️ Kurang",   f"{kurang_count}/{total_param} param")
            ts3.metric("⬆️ Berlebih", f"{lebih_count}/{total_param} param")

            df_tabel = pd.DataFrame(rows)

            def warnai_status(val):
                if "Optimal" in str(val):
                    return "background-color:#1b4332; color:#52b788; font-weight:700; border-radius:6px;"
                elif "Kurang" in str(val):
                    return "background-color:#3b1414; color:#E53935; font-weight:700; border-radius:6px;"
                elif "Berlebih" in str(val):
                    return "background-color:#3b2a00; color:#FFA726; font-weight:700; border-radius:6px;"
                return ""

            def warnai_selisih(val):
                try:
                    v = float(val)
                    if v > 0:
                        return "color:#FFA726; font-weight:700;"
                    elif v < 0:
                        return "color:#E53935; font-weight:700;"
                    else:
                        return "color:#52b788; font-weight:700;"
                except:
                    return ""
            styled_df = (
                df_tabel.style
                .map(warnai_status,  subset=["Status"])
                .map(warnai_selisih, subset=["Selisih"])
                .set_properties(**{
                    "background-color": "#0a1a12",
                    "color": "#c8dcd0",
                    "border": "none",
                    "border-bottom": "1px solid #132b20",
                    "font-size": "0.85rem",
                    "padding": "10px 14px",
                })
                .set_table_styles([
                    {"selector": "th", "props": [
                        ("background-color", "#0a1a12"),
                        ("color", "#4ade80"),
                        ("font-size", "0.72rem"),
                        ("letter-spacing", "1.5px"),
                        ("text-transform", "uppercase"),
                        ("padding", "12px 14px"),
                        ("border-bottom", "1px solid #1f3028"),
                        ("font-weight", "600"),
                    ]},
                    {"selector": "tr:hover td", "props": [
                        ("background-color", "#0f2318 !important"),
                        ("transition", "background 0.2s ease"),
                    ]},
                    {"selector": "table", "props": [
                        ("border-collapse", "collapse"),
                        ("width", "100%"),
                    ]},
                    {"selector": "td", "props": [
                        ("border-left", "none"),
                        ("border-right", "none"),
                    ]},
                ])
            )
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            # ── 7. Rekomendasi Perbaikan ──────────────────────────────────────
            st.markdown("---")
            st.subheader("💡 Rekomendasi Perbaikan Lahan")

            if not kurang and not berlebih:
                st.markdown(
                    """
                    <div style="
                        background: linear-gradient(135deg,#1b4332,#1f5c3a);
                        border: 1.5px solid #52b78855;
                        border-radius: 14px;
                        padding: 24px 28px;
                        text-align: center;
                    ">
                        <div style="font-size:2.4rem; margin-bottom:8px;">🎉</div>
                        <div style="font-size:1.15rem; font-weight:700;
                                    color:#52b788; margin-bottom:6px;">
                            Semua Parameter Optimal!
                        </div>
                        <div style="color:#9bb0a4; font-size:0.93rem;">
                            Lahan kamu dalam kondisi terbaik dan siap untuk produksi pertanian intensif.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                # ── PERBAIKAN BUG: gunakan for-loop biasa, bukan generator ──
                if kurang:
                    items_html = ""
                    for item in kurang:
                        items_html += f"""
                        <div style="
                            background:#1a1a2e;
                            border-left:4px solid #E53935;
                            border-radius:0 10px 10px 0;
                            padding:12px 16px;
                            margin-bottom:10px;
                        ">
                            <div style="
                                display:flex; align-items:center; gap:8px;
                                font-weight:700; color:#ff6b6b;
                                font-size:0.95rem; margin-bottom:4px;
                            ">
                                <span style="
                                    background:#E5393522; border:1px solid #E5393555;
                                    border-radius:6px; padding:2px 8px; font-size:0.78rem;
                                    letter-spacing:1px;
                                ">⬇ KURANG</span>
                                {item['fitur']}
                            </div>
                            <div style="color:#cccccc; font-size:0.88rem; line-height:1.6;">
                                {item['saran']}
                            </div>
                        </div>
                        """

                    st.markdown(
                        f"""
                        <div style="
                            background:#130d1a;
                            border:1.5px solid #E5393540;
                            border-radius:14px;
                            padding:20px 22px;
                            margin-bottom:16px;
                        ">
                            <div style="
                                font-size:1rem; font-weight:800;
                                color:#E53935; letter-spacing:0.5px;
                                margin-bottom:14px;
                                display:flex; align-items:center; gap:8px;
                            ">
                                ⬇️ Parameter yang Perlu <span style="text-decoration:underline">Ditingkatkan</span>
                                <span style="
                                    background:#E5393530; color:#ff6b6b;
                                    border-radius:20px; padding:1px 10px;
                                    font-size:0.8rem; font-weight:600;
                                ">{len(kurang)} item</span>
                            </div>
                            {items_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if berlebih:
                    items_html = ""
                    for item in berlebih:
                        items_html += f"""
                        <div style="
                            background:#1e1a0a;
                            border-left:4px solid #FFA726;
                            border-radius:0 10px 10px 0;
                            padding:12px 16px;
                            margin-bottom:10px;
                        ">
                            <div style="
                                display:flex; align-items:center; gap:8px;
                                font-weight:700; color:#FFC94D;
                                font-size:0.95rem; margin-bottom:4px;
                            ">
                                <span style="
                                    background:#FFA72622; border:1px solid #FFA72655;
                                    border-radius:6px; padding:2px 8px; font-size:0.78rem;
                                    letter-spacing:1px;
                                ">⬆ BERLEBIH</span>
                                {item['fitur']}
                            </div>
                            <div style="color:#cccccc; font-size:0.88rem; line-height:1.6;">
                                {item['saran']}
                            </div>
                        </div>
                        """

                    st.markdown(
                        f"""
                        <div style="
                            background:#1a1400;
                            border:1.5px solid #FFA72640;
                            border-radius:14px;
                            padding:20px 22px;
                            margin-bottom:16px;
                        ">
                            <div style="
                                font-size:1rem; font-weight:800;
                                color:#FFA726; letter-spacing:0.5px;
                                margin-bottom:14px;
                                display:flex; align-items:center; gap:8px;
                            ">
                                ⬆️ Parameter yang Perlu <span style="text-decoration:underline">Dikurangi</span>
                                <span style="
                                    background:#FFA72630; color:#FFC94D;
                                    border-radius:20px; padding:1px 10px;
                                    font-size:0.8rem; font-weight:600;
                                ">{len(berlebih)} item</span>
                            </div>
                            {items_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # ── Download PDF (setelah semua rekomendasi) ──────────────────────
            st.markdown("---")
            fig_pdf, ax_pdf = plt.subplots(figsize=(6, 2.5))
            fig_pdf.patch.set_facecolor("#FFFFFF")
            ax_pdf.set_facecolor("#FFFFFF")
            bar_colors_pdf = [WARNA_BAR.get(l, "#888") for l in labels_prob]
            ax_pdf.barh(labels_prob, values_prob, color=bar_colors_pdf)
            ax_pdf.set_xlabel("Probabilitas (%)", color="#333333")
            ax_pdf.tick_params(colors="#333333")
            for spine in ax_pdf.spines.values():
                spine.set_color("#cccccc")
            ax_pdf.set_xlim(0, 115)
            for i, v in enumerate(values_prob):
                ax_pdf.text(v + 1, i, f"{v:.2f}%", va="center", color="#333333", fontsize=9)
            plt.tight_layout()
            pdf_bytes = buat_pdf_laporan(
                predicted_label = predicted_label,
                conf            = conf,
                warna_hex       = warna,
                deskripsi       = DESKRIPSI[predicted_label],
                rows            = rows,
                kurang          = kurang,
                berlebih        = berlebih,
                prob_df         = pd.DataFrame(),
                fig             = fig_pdf,
            )
            plt.close(fig_pdf)
            st.download_button(
                label              = "📄 Download Laporan PDF",
                data               = pdf_bytes,
                file_name          = "laporan_kesuburan_tanah.pdf",
                mime               = "application/pdf",
                use_container_width= True,
                type               = "primary",
            )
                   

# ============================================================
# PAGE: ANALISIS DATASEST
# ============================================================
def page_eksplorasi():
    GREEN  = "#2fa05a"
    ORANGE = "#e8a13a"
    RED    = "#d96a6a"

    st.title("Analisis Dataset")
    nb_path = os.path.join(os.path.dirname(__file__), "KLASIFIKASI_LAHAN.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

  

    tab1, tab2 = st.tabs(["📁 Tentang Dataset", "💻 Kode Jupyter"])

    with tab1:
        st.title("📑 Sumber Dataset")

        st.markdown("""
        <div style="background:#16201c; padding:15px; border-radius:15px; color:white; border: 0.5px solid #1e4030;">
            Sumber dataset yang digunakan berasal dari platform Kaggle yang berisi 11 data terkait kondisi tanah dan berbagai parameter  
            untuk mengklasifikasikan tingkat kesuburan lahan menjadi 3 kelas 
            Data ini telah melalui proses pengolahan dan digunakan sebagai dasar dalam membantu menentukan tingkat kesuburan tanah.
        </div>
        """, unsafe_allow_html=True)
        st.write("")

        c1, c2, c3, c4,c5 = st.columns(5)
        counts = df["Output"].value_counts()
        cards = [
            ("Jumlah Sampel",  len(df)),
            ("Jumlah Fitur",   df.shape[1] - 1),
            ("Jumlah Kelas",   df["Output"].nunique()),
            ("Missing Values", int(df.isna().sum().sum())),
            ("duplicated Values", int(df.duplicated().sum())),

        ]
        for col, (label, val) in zip([c1, c2, c3, c4,c5], cards):
            col.markdown(
                f"<div class='metric-card'><div class='value'>{val}</div>"
                f"<div class='label'>{label}</div></div>",
                unsafe_allow_html=True,
            )
        st.write("   ")
        st.subheader("Preview Data")

        col1, col2 = st.columns([3, 1])
        with col1:
            jumlah = st.slider("Jumlah baris", min_value=5, max_value=50, value=10, step=5)
        with col2:
            st.write("")
            acak = st.button("🔄 Acak", use_container_width=True)

        if "seed" not in st.session_state or acak:
            st.session_state.seed = random.randint(0, 9999)

        st.dataframe(df.sample(jumlah, random_state=st.session_state.seed), use_container_width=True)
        st.caption(f"Menampilkan {jumlah} dari {len(df)} baris")

        st.write("")
        st.divider()
        st.title("📤 Eksplorasi Data")
        st.subheader("Visualisasi dan analisis karakteristik dataset kesuburan tanah")

        st.subheader(" Distribusi Fitur")



        nama_fitur = {
            "N"  : "Nitrogen (N)",
            "P"  : "Fosfor (P)",
            "K"  : "Kalium (K)",
            "pH" : "pH Tanah",
            "EC" : "Electrical Conductivity (EC)",
            "OC" : "Organic Carbon (OC)",
            "S"  : "Sulfur (S)",
            "Zn" : "Seng (Zn)",
            "Fe" : "Besi (Fe)",
            "Cu" : "Tembaga (Cu)",
            "Mn" : "Mangan (Mn)",
            "B"  : "Boron (B)",
        }

        fitur_list    = [col for col in df.columns if col != 'Output']
        fitur_dipilih = st.selectbox(
            "Pilih parameter tanah:",
            fitur_list,
            format_func=lambda x: nama_fitur.get(x, x)
        )

        nama_dipilih = nama_fitur.get(fitur_dipilih, fitur_dipilih)
        series       = df[fitur_dipilih]

        counts, bin_edges = np.histogram(series, bins=20)
        bin_centers       = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_labels        = [f"{e:.2f}" for e in bin_centers]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=bin_centers,
            y=counts,
            name=nama_dipilih,
            marker_color="#1D9E75",
            marker_line_color="#0d1f18",
            marker_line_width=1,
            hovertemplate=(
                "<b>Nilai:</b> %{x:.2f}<br>"
                "<b>Jumlah:</b> %{y} data<extra></extra>"
            ),
        ))

        fig.update_layout(
            paper_bgcolor="#16201c",
            plot_bgcolor="#1e2e22",
            font=dict(color="#e8efe9"),
            title=dict(
                text=f"Distribusi {nama_dipilih}",
                font=dict(size=14, color="#e8efe9"),
                x=0.5,
            ),
            xaxis=dict(
                title=nama_dipilih,
                title_font=dict(size=11, color="#9bb0a4"),
                tickfont=dict(size=10, color="#9bb0a4"),
                gridcolor="rgba(255,255,255,0.06)",
                linecolor="rgba(255,255,255,0.1)",
                showgrid=True,
                gridwidth=0.5,
            ),
            yaxis=dict(
                title="Jumlah Data",
                title_font=dict(size=11, color="#9bb0a4"),
                tickfont=dict(size=10, color="#9bb0a4"),
                gridcolor="rgba(255,255,255,0.06)",
                linecolor="rgba(255,255,255,0.1)",
                showgrid=True,
                gridwidth=0.5,
            ),
            bargap=0.04,
            margin=dict(l=40, r=20, t=50, b=40),
            height=320,
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)

        def metric_box(col, label, value):
            col.markdown(
                f"""<div style="background:#1e2e22; border:0.5px solid #2a3a33;
                    border-radius:10px; padding:17px 19px; text-align:center;">
                    <div style="font-size:12px; color:rgba(255,255,255,0.5);
                        margin-bottom:6px; font-weight:500;">{label}</div>
                    <div style="font-size:33px; font-weight:600;
                        color:white;">{value}</div>
                </div>""",
                unsafe_allow_html=True
            )

        metric_box(col1, "Rata-rata", f"{series.mean():.2f}")
        metric_box(col2, "Median",    f"{series.median():.2f}")
        metric_box(col3, "Min",       f"{series.min():.2f}")
        metric_box(col4, "Max",       f"{series.max():.2f}")

        st.write("")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Distribusi Output")

            counts = df["Output"].value_counts().sort_index()
            labels = [LABEL_NAME[c] for c in counts.index]
            values = counts.values.tolist()
            colors = [PALETTE[c] for c in counts.index]
            total  = sum(values)

            fig = go.Figure(go.Pie(
                labels=labels,
                values=values,
                hole=0.68,
                marker=dict(colors=colors, line=dict(color="#0d1f18", width=3)),
                textinfo="none",
                hovertemplate="<b>%{label}</b><br>Jumlah: %{value}<br>Persentase: %{percent}<extra></extra>",
                sort=False,
            ))

            fig.add_annotation(
                text=f"<b>{total}</b><br><span style='font-size:11px'>total</span>",
                x=0.5, y=0.5,
                font=dict(size=18, color="#e8efe9"),
                showarrow=False,
            )

            fig.update_layout(
                paper_bgcolor="#16201c",
                margin=dict(l=10, r=10, t=10, b=10),
                height=280,
                legend=dict(
                    font=dict(size=11, color="#e8efe9"),
                    bgcolor="rgba(0,0,0,0)",
                ),
            )

            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            for col_m, label, val, color in zip([c1, c2, c3], labels, values, colors):
                pct = val / total * 150
                col_m.markdown(
                    f"""<div style="display:inline-flex; flex-direction:column; align-items:center;
                        background:{color}22; border:1.5px solid {color};
                        border-radius:20px; padding:18px 10px; text-align:center; width:100%;
                        box-sizing:border-box;">
                        <div style="font-size:18px; color:{color};
                            letter-spacing:0.6px; margin-bottom:4px;">{label}</div>
                        <div style="font-size:40px; font-weight:700;
                            color:white; line-height:1;">{val}</div>
                        <div style="font-size:18px; color:rgba(255,255,255,0.5);
                            margin-top:4px;">{pct:.1f}%</div>
                    </div>""",
                    unsafe_allow_html=True
                )
        with col2:
            st.subheader("Rata-rata NPK per Kelas")

            avg = df.groupby("Output")[["N", "P", "K"]].mean().reset_index()
            avg["Label"] = avg["Output"].map(LABEL_NAME)

            fig = go.Figure()

            fig.add_trace(go.Bar(
                name="N",
                x=avg["Label"],
                y=avg["N"].round(2),
                marker_color="#2fa05a",
                marker_line_color="#0d1f18",
                marker_line_width=1.5,
                hovertemplate="<b>%{x}</b><br>Nitrogen (N): <b>%{y:.2f}</b><extra></extra>",
            ))
            fig.add_trace(go.Bar(
                name="P",
                x=avg["Label"],
                y=avg["P"].round(2),
                marker_color="#e8a13a",
                marker_line_color="#0d1f18",
                marker_line_width=1.5,
                hovertemplate="<b>%{x}</b><br>Fosfor (P): <b>%{y:.2f}</b><extra></extra>",
            ))
            fig.add_trace(go.Bar(
                name="K",
                x=avg["Label"],
                y=avg["K"].round(2),
                marker_color="#5aa9d9",
                marker_line_color="#0d1f18",
                marker_line_width=1.5,
                hovertemplate="<b>%{x}</b><br>Kalium (K): <b>%{y:.2f}</b><extra></extra>",
            ))

            fig.update_layout(
                paper_bgcolor="#16201c",
                plot_bgcolor="#1e2e22",
                font=dict(color="#e8efe9"),
                barmode="group",
                bargap=0.25,
                bargroupgap=0.08,
                xaxis=dict(
                    tickfont=dict(size=12, color="#9bb0a4"),
                    gridcolor="rgba(255,255,255,0.05)",
                    linecolor="rgba(255,255,255,0.08)",
                ),
                yaxis=dict(
                    tickfont=dict(size=11, color="#9bb0a4"),
                    gridcolor="rgba(255,255,255,0.06)",
                    linecolor="rgba(255,255,255,0.08)",
                    gridwidth=0.5,
                ),
                legend=dict(
                    orientation="h",
                    x=0.5, xanchor="center",
                    y=1.08,
                    font=dict(size=12, color="#e8efe9"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                margin=dict(l=20, r=20, t=40, b=20),
                height=443,
            )

            st.plotly_chart(fig, use_container_width=True)
        st.divider()
        st.subheader(" Fitur Paling Berpengaruh terhadap Prediksi Kesuburan Tanah")

        features    = df.drop("Output", axis=1).columns.tolist()[:12]
        rf          = model.named_steps["rf"]
        importances = rf.feature_importances_


        fi_df = pd.DataFrame({
            "Fitur"      : features,
            "Importance" : importances
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        fi_df.index += 1
        fi_df.index.name = "Rank"

        fi_plot = fi_df.sort_values("Importance", ascending=True)

        q33 = fi_plot["Importance"].quantile(0.33)
        q66 = fi_plot["Importance"].quantile(0.66)



        colors = [
            GREEN  if v >= q66 else
            ORANGE if v >= q33 else
            RED
            for v in fi_plot["Importance"]
        ]

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#0f1715")
        ax.set_facecolor("#16201c")

        bars = ax.barh(fi_plot["Fitur"], fi_plot["Importance"],
                    color=colors, edgecolor="none", height=0.6)

        for bar, val in zip(bars, fi_plot["Importance"]):
            ax.text(bar.get_width() + 0.002,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9, color="#cfd8d2")

        ax.set_xlabel("Importance Score", color="#cfd8d2")
        ax.set_title("Feature Importance — Random Forest", fontsize=13,
                    fontweight="bold", color="#e8efe9", pad=14)
        ax.set_xlim(0, fi_plot["Importance"].max() + 0.06)
        ax.tick_params(colors="#cfd8d2")
        ax.spines[:].set_color("#2a3a33")

        legend = [
            Patch(color=GREEN,  label="High importance"),
            Patch(color=ORANGE, label="Mid importance"),
            Patch(color=RED,    label="Low importance"),
        ]
        ax.legend(handles=legend, loc="lower right",
                facecolor="#16201c", edgecolor="#2a3a33",
                labelcolor="#cfd8d2", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)

        st.caption(f"✅ Paling berpengaruh: **{fi_df.iloc[0]['Fitur']}** ({fi_df.iloc[0]['Importance']:.4f}) &nbsp;|&nbsp; ⚠️ Paling lemah: **{fi_df.iloc[-1]['Fitur']}** ({fi_df.iloc[-1]['Importance']:.4f})")
        

    with tab2:
        st.title("📓 Notebook Cell Viewer")
        for i, cell in enumerate(nb.cells):

                # ── Header cell ──────────────────────────────────────
                tipe  = cell.cell_type
                label = "🟦 Markdown" if tipe == "markdown" else "🟩 Code"
                st.markdown(
                    f"""<div style="background:#1e2e22; border-left:3px solid #2fa05a;
                        padding:6px 14px; border-radius:6px; margin-bottom:4px;">
                        <span style="color:#7ab8f5; font-size:12px; font-weight:600;">
                        Cell [{i+1}]</span>
                        <span style="color:rgba(255,255,255,0.4); font-size:12px;">
                        &nbsp;·&nbsp;{label}</span>
                    </div>""",
                    unsafe_allow_html=True
                )

                # ── Isi cell ─────────────────────────────────────────
                if tipe == "markdown":
                    st.markdown(cell.source)

                elif tipe == "code":
                    st.code(cell.source, language="python")

                    # ── Output cell ───────────────────────────────────
                    for output in cell.get("outputs", []):

                        # Teks / print
                        if output.output_type == "stream":
                            teks = html.escape(output.text)
                            st.markdown(
                                f"""<div style="background:#0f1715; border-radius:6px;
                                    padding:10px 14px; font-family:monospace;
                                    font-size:13px; color:#a8d5b5;
                                    white-space:pre-wrap; margin-top:4px;">{teks}</div>""",
                                unsafe_allow_html=True
                            )

                        elif output.output_type in ("display_data", "execute_result"):

                            # Gambar
                            if "image/png" in output.data:
                                import base64
                                img_data = output.data["image/png"]
                                st.markdown(
                                    f'<img src="data:image/png;base64,{img_data}" '
                                    f'style="max-width:100%; border-radius:8px; margin-top:6px;">',
                                    unsafe_allow_html=True
                                )

                            # Tabel / teks
                            elif "text/html" in output.data:
                                st.markdown(output.data["text/html"], unsafe_allow_html=True)

                            elif "text/plain" in output.data:
                                teks = html.escape(output.data["text/plain"])
                                st.markdown(
                                    f"""<div style="background:#0f1715; border-radius:6px;
                                        padding:10px 14px; font-family:monospace;
                                        font-size:13px; color:#a8d5b5;
                                        white-space:pre-wrap; margin-top:4px;">{teks}</div>""",
                                    unsafe_allow_html=True
                                )

                st.markdown("<hr style='border:0.5px solid #1e2e22; margin:12px 0'>",
                            unsafe_allow_html=True)

# ============================================================
# PAGE: TENTANG
# ============================================================
def page_tentang():
    st.title("Tentang ")
    st.write("Tentang Developer, Informasi model dan evaluasi ")
    def load_image_b64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    img_b64 = load_image_b64("Gemini_Generated_Image_md2ip6md2ip6md2i.png")

    tab1, tab2 = st.tabs(["👤 Tentang Developer", "🧠 Tentang Model"])

    with tab1:
        st.subheader("Developer Streamlit App")

        components.html(f"""
        <div style=" border: 0.5px solid #1e4030; background:#16201c; padding:28px; border-radius:16px; color:white; font-family:sans-serif;">

            <div style="display:flex; gap:28px; align-items:stretch;">

                <!-- Kiri: Foto -->
                <div style="flex-shrink:0; display:flex; flex-direction:column;
                    align-items:center; justify-content:center;
                    background:#1e2e22; border-radius:14px;border: 0.5px solid #1e4030;
                    padding:24px 20px; min-width:320px;">
                    <img src="data:image/jpeg;base64,{img_b64}"
                        style="width:200px; height:200px; border-radius:999px;
                            object-fit:cover; object-position:top;
                            border:5px solid #85A947;
                            margin-bottom:14px;">
                    <div style="font-size:22px; font-weight:700; text-align:center;
                        line-height:1.3; margin-bottom:5px;">Nurry Nurul Naomi</div>
                    <div style="display:flex; flex-direction:column; gap:6px; align-items:center; width:100%;">
                        <span style="background:#1a3a5c; color:#7ab8f5; font-size:18px;
                            padding:4px 14px; border-radius:20px; text-align:center;">Rekayasa Perangkat Lunak</span>
                        <span style="background:#162820; color:rgba(255,255,255,0.6); font-size:18px;
                            padding:4px 14px; border-radius:20px; text-align:center;
                            border:0.5px solid rgba(255,255,255,0.1);">SMKN 1 Purbalingga</span>
                    </div>
                </div>

                <!-- Kanan: Informasi -->
                <div style="flex:1; display:flex; flex-direction:column; gap:12px; justify-content:center;">

                    <!-- Tanggal Lahir -->
                    <div style="display:flex; align-items:center; gap:16px;border: 0.5px solid #1e4030;
                        background:#1e2e22; border-radius:12px; padding:16px 18px;">
                        <span style="font-size:26px;">📅</span>
                        <div>
                            <div style="font-size:16px; color:rgba(255,255,255,0.4);
                                margin-bottom:5px; ">TANGGAL LAHIR</div>
                            <div style="font-size:19px; font-weight:600;">Purwokerto, 28 Januari 2009</div>
                        </div>
                    </div>

                    <!-- Email -->
                    <div style="display:flex; align-items:center; gap:16px;border: 0.5px solid #1e4030;
                        background:#1e2e22; border-radius:12px; padding:16px 18px;">
                        <span style="font-size:26px;">📧</span>
                        <div>
                            <div style="font-size:16px; color:rgba(255,255,255,0.4);
                                margin-bottom:5px; ">EMAIL</div>
                            <a href="mailto:nurry.naomy28@gmail.com"
                                style="font-size:19px; font-weight:600; color:#7ab8f5; text-decoration:none;">
                                nurry.naomy28@gmail.com
                            </a>
                        </div>
                    </div>

                    <!-- GitHub -->
                    <a href="https://github.com/nurryyy0" target="_blank" style="
                        display:flex; align-items:center; gap:16px;
                        background:#1e2e22; border-radius:12px;border: 0.5px solid #1e4030; padding:16px 18px; text-decoration:none;">
                        <span style="font-size:26px;">🐱</span>
                        <div>
                            <div style="font-size:16px; color:rgba(255,255,255,0.4);
                                margin-bottom:5px;">GITHUB</div>
                            <div style="font-size:19px; font-weight:600; color:white;">@nurryyy0</div>
                        </div>
                    </a>

                    <!-- Instagram -->
                    <a href="https://www.instagram.com/nnaoou_/" target="_blank" style="
                        display:flex; align-items:center; gap:16px;
                        background:#1e2e22; border-radius:12px;border: 0.5px solid #1e4030; padding:16px 18px; text-decoration:none;">
                        <span style="font-size:26px;">📸</span>
                        <div>
                            <div style="font-size:16px; color:rgba(255,255,255,0.4);
                                margin-bottom:5px; ">INSTAGRAM</div>
                            <div style="font-size:19px; font-weight:600; color:white;">@nnaoou_</div>
                        </div>
                    </a>

                </div>
            </div>

        </div>
        """, height=450)
            

    with tab2:
        st.subheader("Tentang Model")
        st.write("informasi model")
    
  
        ev = evaluate_model(model)


        c1, c2 = st.columns(2)
        c1.markdown(
            "<div class='metric-card'><div class='label'>Algoritma</div>"
            "<div style='font-size:1.2rem;font-weight:600'>Random Forest Classifier</div></div>",
            unsafe_allow_html=True,
        )
        c2.markdown(
            "<div class='metric-card'><div class='label'>Preprocessing</div>"
            "<div style='font-size:1.2rem;font-weight:600'>StandardScaler + SMOTE (oversampling)</div></div>",
            unsafe_allow_html=True,
        )

        st.subheader("Best Parameters")
        params = [
            "ccp_alpha: 0.001", "criterion: entropy", "max_depth: 12",
            "max_features: sqrt", "max_samples: 0.8", "min_samples_leaf: 5",
            "min_samples_split: 13", "n_estimators: 239",
        ]
        st.markdown(
            "<div style='display:flex; flex-wrap:wrap; gap:10px;'>"
            + "".join([f"<span class='pill pill-green'>{p}</span>" for p in params])
            + "</div>",
            unsafe_allow_html=True,
        )

        st.subheader("Evaluasi Model")
        m1, m2, m3 = st.columns(3)
        for col, (l, v) in zip([m1, m2, m3], [
            ("Test Accuracy",      ev["test_acc"]),
            ("F1 Macro",           ev["f1"]),
            ("Balanced Accuracy",  ev["bal_acc"]),
        ]):
            col.markdown(
                f"<div class='metric-card'><div class='value'>{v*100:.1f}%</div>"
                f"<div class='label'>{l}</div></div>",
                unsafe_allow_html=True,
            )

        st.subheader("Classification Report")
        report = ev["report"]
        rows = []
        for cls in [0, 1, 2]:
            r = report.get(str(cls), {})
            rows.append([
                LABEL_NAME[cls],
                round(r.get("precision", 0), 3),
                round(r.get("recall", 0), 3),
                round(r.get("f1-score", 0), 3),
                int(r.get("support", 0)),
            ])
        st.dataframe(
            pd.DataFrame(rows, columns=["Kelas", "Precision", "Recall", "F1-Score", "Support"]),
            use_container_width=True,
        )

        st.subheader("Confusion Matrix")
        cm     = ev["cm"]
        labels = ["Kurang Subur", "Subur", "Sangat Subur"]
        fig, ax = plt.subplots(figsize=(3, 2.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False,
                    linewidths=0.5, linecolor="white", square=True,
                    annot_kws={"size": 8, "weight": "bold"}, ax=ax)
        ax.set_xlabel("Predicted", fontsize=7)
        ax.set_ylabel("Actual", fontsize=7)
        ax.set_xticklabels(labels, fontsize=6)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title("Confusion Matrix Model", fontsize=8)
        plt.xticks(rotation=0); plt.yticks(rotation=0)

        col, _ = st.columns([1, 1])
        with col:
            st.pyplot(fig)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Proses Model")
            st.markdown("""
            <div style="background:#16201c; padding:15px; border-radius:10px; color:white;">
            Data tanah diproses melalui tahap preprocessing, normalisasi, dan penyeimbangan data menggunakan SMOTE.
            Selanjutnya dilakukan training dan hyperparameter tuning pada model Random Forest untuk memperoleh 
            performa prediksi terbaik. Model kemudian dievaluasi menggunakan beberapa metrik pengujian sebelum 
            disimpan dan digunakan pada sistem prediksi.
            </div>
            """, unsafe_allow_html=True)


        with c2:
            tools = ["Python", "Streamlit", "Pandas", "NumPy",
                    "Scikit-learn", "Random Forest", "Label Encoder",
                    "Matplotlib", "Seaborn", "Plotly",
                    "Joblib", "FPDF", "Jupyter Notebook"]
            tools_html = "".join([f"<span class='pill pill-green'>{t}</span>" for t in tools])
            st.subheader("Tools")
            st.markdown(f"""
            <div style="background:#16201c; padding:27px; border-radius:10px; color:white; margin-bottom:10px;">
                <div style='display:flex; flex-wrap:wrap; gap8px;'>{tools_html}</div>
            </div>
            """, unsafe_allow_html=True)
        st.write("")
        st.divider()

#===========================
# JUPYTER KODE
#==========================

# ============================================================
# ROUTER — menggunakan tabs
# ============================================================
with tab1:
    page_klasifikasi()
with tab2:
    page_beranda()

with tab3:
    page_eksplorasi()

with tab4:
    page_tentang()

