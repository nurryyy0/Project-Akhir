"""
Redesign Hasil Prediksi – Soil Fertility Classifier
====================================================
Drop-in replacement untuk bagian if do_pred: pada page_klasifikasi().
Tambahkan fungsi-fungsi helper ini di atas page_klasifikasi(),
lalu ganti blok if do_pred: dengan versi baru di bawah.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ─────────────────────────────────────────────────────────────
# 0.  GLOBAL CSS  (panggil sekali di awal app / sebelum render)
# ─────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Sora:wght@300;400;500;600;700&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

    /* ── Hero card ── */
    .hero-card {
        background: linear-gradient(135deg, #071a12 0%, #0c2318 60%, #091f17 100%);
        border: 1px solid #1e4030;
        border-radius: 20px;
        padding: 2rem 2.2rem 1.8rem;
        position: relative;
        overflow: hidden;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .hero-card::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(14,240,184,0.07) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        letter-spacing: .18em;
        color: #3a7a60;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        line-height: 1.05;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        font-size: 0.88rem;
        color: #507a65;
        margin-bottom: 1.4rem;
    }
    .hero-conf {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: rgba(14,240,184,0.07);
        border: 0.5px solid rgba(14,240,184,0.2);
        border-radius: 10px;
        padding: 8px 16px;
    }
    .hero-conf-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 500;
    }
    .hero-conf-lbl {
        font-size: 0.78rem;
        color: #507a65;
        line-height: 1.3;
    }
    .conf-bar-bg {
        background: #0d2419;
        border-radius: 4px;
        height: 4px;
        margin-top: 4px;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.8s cubic-bezier(.4,0,.2,1);
    }

    /* ── Desc bullet list ── */
    .desc-list { list-style: none; padding: 0; margin: 0; }
    .desc-list li {
        display: flex;
        gap: 10px;
        align-items: flex-start;
        padding: 7px 0;
        border-bottom: 0.5px solid #122a1e;
        font-size: 0.88rem;
        color: #7aada0;
        line-height: 1.55;
    }
    .desc-list li:last-child { border-bottom: none; }
    .desc-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        margin-top: 6px;
        flex-shrink: 0;
    }

    /* ── Metric mini-cards ── */
    .mini-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-bottom: 1.5rem;
    }
    .mini-card {
        background: #071a12;
        border: 0.5px solid #1a3a2a;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        position: relative;
        overflow: hidden;
    }
    .mini-card-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 9px;
        letter-spacing: .12em;
        color: #3a7060;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .mini-card-val {
        font-size: 1.25rem;
        font-weight: 600;
        line-height: 1.1;
    }
    .mini-card-unit {
        font-size: 0.7rem;
        color: #507a65;
        margin-top: 2px;
    }

    /* ── Section header ── */
    .section-head {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 1.8rem 0 1rem;
    }
    .section-head-line {
        flex: 1;
        height: 0.5px;
        background: #1a3a2a;
    }
    .section-head-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        letter-spacing: .15em;
        color: #3a7060;
        text-transform: uppercase;
        white-space: nowrap;
    }

    /* ── Insight pills ── */
    .insight-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 1.5rem;
    }
    .insight-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        font-size: 0.8rem;
        padding: 6px 13px;
        border-radius: 20px;
        line-height: 1.4;
    }
    .insight-ok   { background: rgba(14,240,184,0.08); border: 0.5px solid rgba(14,240,184,0.25); color: #0ef0b8; }
    .insight-warn { background: rgba(234,163,58,0.08); border: 0.5px solid rgba(234,163,58,0.25); color: #e8a13a; }
    .insight-bad  { background: rgba(229,57,53,0.1);   border: 0.5px solid rgba(229,57,53,0.25);  color: #f07060; }

    /* ── Rekomendasi cards ── */
    .reko-card {
        background: #071a12;
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 10px;
        border: 0.5px solid #1a3a2a;
        position: relative;
        overflow: hidden;
    }
    .reko-card.kurang  { border-left: 3px solid #E53935; }
    .reko-card.berlebih{ border-left: 3px solid #FFA726; }
    .reko-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .reko-card-body {
        font-size: 0.8rem;
        color: #7aada0;
        line-height: 1.55;
    }

    /* ── Comparison table override ── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* ── st.metric override ── */
    [data-testid="metric-container"] {
        background: #071a12;
        border: 0.5px solid #1a3a2a;
        border-radius: 12px;
        padding: 1rem 1.1rem;
    }
    [data-testid="stMetricLabel"] > div {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px !important;
        letter-spacing: .1em;
        color: #3a7060 !important;
        text-transform: uppercase;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Sora', sans-serif;
        font-size: 1.4rem !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
_KLASS_CFG = {
    "Kurang Subur": {
        "color":      "#f07060",
        "bar_color":  "#E53935",
        "icon":       "⚠️",
        "score":      25,
        "quality":    "Rendah",
        "status":     "Perlu Perbaikan",
        "gauge_color": ["#1a0a0a", "#6b1212", "#E53935"],
    },
    "Cukup Subur": {
        "color":      "#e8a13a",
        "bar_color":  "#FFA726",
        "icon":       "🌾",
        "score":      62,
        "quality":    "Menengah",
        "status":     "Layak Tanam",
        "gauge_color": ["#1a1200", "#6b4a00", "#FFA726"],
    },
    "Sangat Subur": {
        "color":      "#0ef0b8",
        "bar_color":  "#43A047",
        "icon":       "🏆",
        "score":      95,
        "quality":    "Tinggi",
        "status":     "Siap Produksi",
        "gauge_color": ["#001a0e", "#006630", "#43A047"],
    },
}

_WARNA_BAR = {
    "Kurang Subur": "#E53935",
    "Cukup Subur":  "#FFA726",
    "Sangat Subur": "#43A047",
}


# ─────────────────────────────────────────────────────────────
# 1. HERO PREDICTION CARD
# ─────────────────────────────────────────────────────────────
def render_hero_card(predicted_label, conf, DESKRIPSI):
    cfg   = _KLASS_CFG[predicted_label]
    color = cfg["color"]
    descs = [d.strip().lstrip("- ") for d in DESKRIPSI[predicted_label].split("<br>") if d.strip()]
    dots  = "".join(
        f"<li><span class='desc-dot' style='background:{color}'></span>{d}</li>"
        for d in descs
    )
    bar_w = f"{conf:.1f}%"
    st.markdown(f"""
    <div class='hero-card'>
      <div class='hero-label'>// HASIL PREDIKSI KESUBURAN LAHAN</div>
      <div class='hero-title' style='color:{color}'>{cfg["icon"]} {predicted_label}</div>
      <div class='hero-sub'>Model memprediksi dengan keyakinan tinggi berdasarkan 12 parameter tanah.</div>
      <div class='hero-conf'>
        <div>
          <div class='hero-conf-val' style='color:{color}'>{conf:.1f}%</div>
          <div class='conf-bar-bg' style='width:90px'>
            <div class='conf-bar-fill' style='width:{bar_w}; background:{color}'></div>
          </div>
        </div>
        <div class='hero-conf-lbl'>Keyakinan<br>Model</div>
      </div>
      <div style='margin-top:1.4rem'>
        <ul class='desc-list'>{dots}</ul>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 2. METRIC MINI CARDS (4 kolom)
# ─────────────────────────────────────────────────────────────
def render_metric_cards(predicted_label, conf):
    cfg = _KLASS_CFG[predicted_label]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🏅 Kelas Lahan",   predicted_label)
    with c2:
        st.metric("📊 Confidence",    f"{conf:.1f}%")
    with c3:
        st.metric("🌿 Kualitas",       cfg["quality"])
    with c4:
        st.metric("🔖 Status",         cfg["status"])


# ─────────────────────────────────────────────────────────────
# 3. GAUGE CHART – SKOR KESUBURAN
# ─────────────────────────────────────────────────────────────
def render_gauge(predicted_label, conf):
    cfg   = _KLASS_CFG[predicted_label]
    score = cfg["score"]

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = score,
        delta = {"reference": 50, "valueformat": ".0f",
                 "increasing": {"color": "#0ef0b8"},
                 "decreasing": {"color": "#f07060"}},
        number= {"suffix": " pts", "font": {"size": 42, "family": "Sora", "color": cfg["color"]}},
        gauge = {
            "axis"      : {"range": [0, 100], "tickwidth": 0.5,
                           "tickcolor": "#1e3d30", "tickfont": {"size": 10, "color": "#3a7060"}},
            "bar"       : {"color": cfg["bar_color"], "thickness": 0.28},
            "bgcolor"   : "#071a12",
            "borderwidth": 0,
            "steps"     : [
                {"range": [0,  33], "color": "#0d1f17"},
                {"range": [33, 66], "color": "#0f2318"},
                {"range": [66,100], "color": "#112818"},
            ],
            "threshold" : {
                "line" : {"color": cfg["color"], "width": 3},
                "thickness": 0.8,
                "value"    : score,
            },
        },
        title = {"text": "Skor Kesuburan",
                 "font": {"size": 13, "family": "JetBrains Mono", "color": "#3a7060"}},
    ))
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        margin        = dict(t=40, b=10, l=20, r=20),
        height        = 230,
        font          = dict(family="Sora"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────
# 4. HORIZONTAL BAR – PROBABILITAS KELAS
# ─────────────────────────────────────────────────────────────
def render_proba_chart(model, proba, LABEL_NAME):
    labels = [LABEL_NAME[c] for c in model.classes_]
    pcts   = [p * 100 for p in proba]
    colors = [_WARNA_BAR.get(l, "#888") for l in labels]

    fig = go.Figure()
    for lbl, pct, clr in zip(labels, pcts, colors):
        fig.add_trace(go.Bar(
            x           = [pct],
            y           = [lbl],
            orientation = "h",
            marker_color= clr,
            marker_line_width = 0,
            text        = [f"  {pct:.1f}%"],
            textposition= "outside",
            textfont    = dict(size=13, family="JetBrains Mono", color=clr),
            hovertemplate = f"<b>{lbl}</b><br>Probabilitas: {pct:.2f}%<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(7,26,18,0.6)",
        showlegend    = False,
        barmode       = "overlay",
        xaxis = dict(
            range       = [0, 115],
            showgrid    = False,
            zeroline    = False,
            showticklabels = False,
        ),
        yaxis = dict(
            showgrid    = False,
            tickfont    = dict(family="Sora", size=13, color="#7aada0"),
            categoryorder = "array",
            categoryarray = labels,
        ),
        margin = dict(t=10, b=10, l=10, r=60),
        height = 180,
        bargap = 0.45,
    )
    # gridline manual
    fig.add_shape(type="line", x0=0, x1=0, y0=-0.5, y1=len(labels)-0.5,
                  line=dict(color="#1e3d30", width=1))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────
# 5. RADAR CHART – PARAMETER vs IDEAL
# ─────────────────────────────────────────────────────────────
def render_radar(inputs, DATA_IDEAL, NAMA_FITUR, predicted_label):
    cfg    = _KLASS_CFG[predicted_label]
    fitur  = list(DATA_IDEAL.keys())
    labels = [NAMA_FITUR[f].split(" (")[0] for f in fitur]

    # Normalisasi 0-100 berdasarkan ideal
    def norm(val, ideal):
        r = min(val / (ideal * 2 + 1e-9), 1.0) * 100
        return round(r, 1)

    vals_in    = [norm(float(inputs[f]), DATA_IDEAL[f]) for f in fitur]
    vals_ideal = [50.0] * len(fitur)   # selalu 50 (half of ideal*2)

    cats = labels + [labels[0]]
    vi   = vals_in    + [vals_in[0]]
    vid  = vals_ideal + [vals_ideal[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vid, theta=cats, fill="toself",
        name="Nilai Ideal",
        line=dict(color="#1e4030", width=1.5, dash="dot"),
        fillcolor="rgba(30,64,48,0.15)",
        hovertemplate="<b>Ideal</b><extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=vi, theta=cats, fill="toself",
        name="Input Kamu",
        line=dict(color=cfg["bar_color"], width=2),
        fillcolor=f"rgba({_hex_to_rgb(cfg['bar_color'])},0.18)",
        hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
    ))
    fig.update_layout(
        polar = dict(
            bgcolor   = "rgba(7,26,18,0.5)",
            radialaxis= dict(visible=True, range=[0, 100],
                             showticklabels=False, gridcolor="#1a3a2a", gridwidth=0.5),
            angularaxis=dict(tickfont=dict(size=10, color="#507a65", family="Sora"),
                             gridcolor="#1a3a2a", gridwidth=0.5,
                             linecolor="#1a3a2a"),
        ),
        paper_bgcolor = "rgba(0,0,0,0)",
        showlegend    = True,
        legend        = dict(font=dict(family="Sora", size=11, color="#7aada0"),
                             bgcolor="rgba(0,0,0,0)", borderwidth=0,
                             orientation="h", x=0.5, xanchor="center", y=-0.08),
        margin = dict(t=10, b=30, l=30, r=30),
        height = 310,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


# ─────────────────────────────────────────────────────────────
# 6. INSIGHT OTOMATIS
# ─────────────────────────────────────────────────────────────
def render_insights(inputs, DATA_IDEAL, NAMA_FITUR, predicted_label, conf,
                    TOLERANSI_ABSOLUT):
    cfg     = _KLASS_CFG[predicted_label]
    pills   = []

    # Insight confidence
    if conf >= 85:
        pills.append(("ok",   f"Model sangat yakin: confidence {conf:.1f}%"))
    elif conf >= 65:
        pills.append(("warn", f"Confidence moderat ({conf:.1f}%), pertimbangkan uji ulang"))
    else:
        pills.append(("bad",  f"Confidence rendah ({conf:.1f}%), hasil kurang pasti"))

    # Insight per fitur
    for f, ideal in DATA_IDEAL.items():
        val = float(inputs[f])
        tol = TOLERANSI_ABSOLUT.get(f, ideal * 0.1)
        sel = val - ideal
        nama = NAMA_FITUR[f].split(" (")[0]
        if abs(sel) <= tol:
            pills.append(("ok",   f"{nama} optimal (±{tol})"))
        elif sel < 0:
            pills.append(("warn", f"{nama} kurang dari ideal (−{abs(sel):.2f})"))
        else:
            pills.append(("bad",  f"{nama} berlebih (+{sel:.2f})"))

    # Insight kelas
    pills.append(("ok" if predicted_label == "Sangat Subur"
                  else "warn" if predicted_label == "Cukup Subur"
                  else "bad",
                  f"Lahan diklasifikasikan {predicted_label} — {cfg['status']}"))

    cls_map = {"ok": "insight-ok", "warn": "insight-warn", "bad": "insight-bad"}
    icon_map = {"ok": "✓", "warn": "⚡", "bad": "✕"}

    html = "<div class='insight-wrap'>"
    for kind, text in pills:
        css = cls_map[kind]
        ico = icon_map[kind]
        html += f"<span class='insight-pill {css}'><b>{ico}</b> {text}</span>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 7. TABEL PERBANDINGAN – STYLED
# ─────────────────────────────────────────────────────────────
def render_comparison_table(inputs, DATA_IDEAL, NAMA_FITUR, SATUAN,
                             TOLERANSI_ABSOLUT, SARAN_FITUR):
    rows     = []
    kurang   = []
    berlebih = []

    for f, ideal in DATA_IDEAL.items():
        val = float(inputs[f])
        sel = val - ideal
        tol = TOLERANSI_ABSOLUT.get(f, ideal * 0.10)
        if abs(sel) <= tol:
            status = "✅ Optimal"
        elif sel < 0:
            status = "⬇️ Kurang"
            kurang.append({"fitur": NAMA_FITUR[f], "saran": SARAN_FITUR[f][0]})
        else:
            status = "⬆️ Berlebih"
            berlebih.append({"fitur": NAMA_FITUR[f], "saran": SARAN_FITUR[f][1]})

        sat = SATUAN.get(f, "")
        rows.append({
            "Parameter"   : NAMA_FITUR[f],
            "Input"       : f"{val:.2f} {sat}".strip(),
            "Ideal"       : f"{ideal:.2f} {sat}".strip(),
            "Selisih"     : f"{sel:+.2f}",
            "Status"      : status,
        })

    df_cmp = pd.DataFrame(rows)

    def highlight_status(val):
        if "Optimal"  in str(val): return "color:#0ef0b8; font-weight:600"
        if "Kurang"   in str(val): return "color:#f07060; font-weight:600"
        if "Berlebih" in str(val): return "color:#FFA726; font-weight:600"
        return ""

    def highlight_selisih(val):
        try:
            v = float(str(val).replace(",", "."))
            if v > 0:  return "color:#FFA726"
            if v < 0:  return "color:#f07060"
            return "color:#0ef0b8"
        except:
            return ""

    styled = (
        df_cmp.style
        .applymap(highlight_status,   subset=["Status"])
        .applymap(highlight_selisih,  subset=["Selisih"])
        .set_table_styles([
            {"selector": "thead th", "props": [
                ("background", "#071a12"),
                ("color",      "#3a7060"),
                ("font-family","JetBrains Mono, monospace"),
                ("font-size",  "11px"),
                ("letter-spacing", "0.08em"),
                ("text-transform","uppercase"),
                ("border-bottom","1px solid #1a3a2a"),
                ("padding",    "10px 14px"),
            ]},
            {"selector": "tbody td", "props": [
                ("background", "#07150f"),
                ("color",      "#9abfb0"),
                ("font-size",  "13px"),
                ("padding",    "9px 14px"),
                ("border-bottom","0.5px solid #112218"),
            ]},
            {"selector": "tbody tr:hover td", "props": [
                ("background", "#0c2218"),
            ]},
            {"selector": "", "props": [
                ("border-radius","12px"),
                ("overflow","hidden"),
            ]},
        ])
        .hide(axis="index")
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    return kurang, berlebih


# ─────────────────────────────────────────────────────────────
# 8. REKOMENDASI CARDS
# ─────────────────────────────────────────────────────────────
def render_rekomendasi(kurang, berlebih):
    if not kurang and not berlebih:
        st.success("🎉 Semua parameter sudah optimal! Lahan dalam kondisi terbaik.")
        return

    tabs = []
    if kurang:   tabs.append("⬇️ Perlu Ditingkatkan")
    if berlebih: tabs.append("⬆️ Perlu Dikurangi")

    tab_objs = st.tabs(tabs)
    idx = 0

    if kurang:
        with tab_objs[idx]:
            for r in kurang:
                st.markdown(f"""
                <div class='reko-card kurang'>
                  <div class='reko-card-title' style='color:#f07060'>⬇ {r['fitur']}</div>
                  <div class='reko-card-body'>{r['saran']}</div>
                </div>
                """, unsafe_allow_html=True)
        idx += 1

    if berlebih:
        with tab_objs[idx]:
            for r in berlebih:
                st.markdown(f"""
                <div class='reko-card berlebih'>
                  <div class='reko-card-title' style='color:#FFA726'>⬆ {r['fitur']}</div>
                  <div class='reko-card-body'>{r['saran']}</div>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 9.  SECTION DIVIDER HELPER
# ─────────────────────────────────────────────────────────────
def section(title):
    st.markdown(f"""
    <div class='section-head'>
      <div class='section-head-line'></div>
      <div class='section-head-text'>{title}</div>
      <div class='section-head-line'></div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 10.  MASTER RENDER – ganti blok `if do_pred:` dengan ini
# ─────────────────────────────────────────────────────────────
def render_hasil_prediksi(
    model, pred, proba, conf,
    predicted_label,
    inputs, features,
    LABEL_NAME, NAMA_FITUR, SATUAN,
    DATA_IDEAL, TOLERANSI_ABSOLUT,
    SARAN_FITUR, DESKRIPSI, REKOMENDASI,
):
    """
    Panggil semua helper di atas secara berurutan.
    Tidak mengubah logika ML, hanya render UI.
    """
    inject_css()

    # ── A. Hero card ──────────────────────────────────────────
    render_hero_card(predicted_label, conf, DESKRIPSI)

    # ── B. Metric mini cards ──────────────────────────────────
    render_metric_cards(predicted_label, conf)

    # ── C. Gauge + Probabilitas (side by side) ────────────────
    section("VISUALISASI MODEL")
    col_g, col_p = st.columns([1, 1.6])
    with col_g:
        st.markdown("<div style='font-family:JetBrains Mono;font-size:10px;"
                    "letter-spacing:.12em;color:#3a7060;margin-bottom:4px'>"
                    "SKOR KESUBURAN</div>", unsafe_allow_html=True)
        render_gauge(predicted_label, conf)
    with col_p:
        st.markdown("<div style='font-family:JetBrains Mono;font-size:10px;"
                    "letter-spacing:.12em;color:#3a7060;margin-bottom:4px'>"
                    "PROBABILITAS PER KELAS</div>", unsafe_allow_html=True)
        render_proba_chart(model, proba, LABEL_NAME)

    # ── D. Radar chart ────────────────────────────────────────
    section("PARAMETER TANAH vs IDEAL")
    render_radar(inputs, DATA_IDEAL, NAMA_FITUR, predicted_label)

    # ── E. Insight otomatis ───────────────────────────────────
    section("INSIGHT OTOMATIS")
    render_insights(inputs, DATA_IDEAL, NAMA_FITUR, predicted_label,
                    conf, TOLERANSI_ABSOLUT)

    # ── F. Tabel perbandingan ─────────────────────────────────
    section("PERBANDINGAN INPUT vs IDEAL")
    kurang, berlebih = render_comparison_table(
        inputs, DATA_IDEAL, NAMA_FITUR, SATUAN, TOLERANSI_ABSOLUT, SARAN_FITUR
    )

    # ── G. Rekomendasi ────────────────────────────────────────
    section("REKOMENDASI PERBAIKAN LAHAN")
    render_rekomendasi(kurang, berlebih)

    # ── H. Rekomendasi umum per kelas ────────────────────────
    with st.expander("📋 Lihat Rekomendasi Umum Lengkap"):
        st.markdown(REKOMENDASI[predicted_label])
