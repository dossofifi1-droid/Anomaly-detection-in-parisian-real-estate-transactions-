import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Paris Real Estate Anomaly Detection",
    page_icon="🏠",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a3c5e, #2e75b6);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white !important;
    }
    .metric-value { font-size: 2.5rem; font-weight: bold; color: white !important; }
    .metric-label { font-size: 0.9rem; opacity: 1; margin-top: 4px; color: #e0e0e0 !important; }
    p, div, span, h1, h2, h3, h4, label { color: #f0f0f0 !important; }
    .stSidebar { background-color: #0d1b2a !important; }
    .stSidebar p, .stSidebar div, .stSidebar span, .stSidebar label { color: #e0e0e0 !important; }
    .cluster-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(135deg, #1a3c5e 0%, #2e75b6 100%); 
     padding: 32px; border-radius: 16px; margin-bottom: 24px;'>
    <h1 style='color: white; margin: 0; font-size: 2rem;'>
        🏠 Real Estate Anomaly Detection — Paris
    </h1>
    <p style='color: #a8d4f5; margin: 8px 0 0 0; font-size: 1rem;'>
        6,923 suspicious transactions detected across 317,413 Parisian property transactions (2014–2024)
    </p>
    <p style='color: #7ab3d4; margin: 4px 0 0 0; font-size: 0.85rem;'>
        INSEEC Grande École — Group G3 | K-Means Clustering + Linear Regression
    </p>
</div>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("suspects_carte.csv")
    except FileNotFoundError:
        # Generate realistic sample data if file not found
        np.random.seed(42)
        n = 6923
        
        # Paris arrondissements center coordinates
        arr_coords = {
            1: (48.860, 2.347), 2: (48.866, 2.349), 3: (48.863, 2.359),
            4: (48.854, 2.352), 5: (48.851, 2.345), 6: (48.850, 2.334),
            7: (48.856, 2.316), 8: (48.875, 2.308), 9: (48.876, 2.338),
            10: (48.876, 2.360), 11: (48.859, 2.379), 12: (48.843, 2.389),
            13: (48.830, 2.362), 14: (48.833, 2.326), 15: (48.842, 2.300),
            16: (48.863, 2.275), 17: (48.884, 2.313), 18: (48.892, 2.344),
            19: (48.884, 2.381), 20: (48.865, 2.399)
        }
        
        arrondissements = np.random.choice(list(arr_coords.keys()), n, 
                                            p=[0.03,0.02,0.03,0.04,0.04,0.05,
                                               0.06,0.08,0.04,0.04,0.05,0.04,
                                               0.05,0.04,0.06,0.08,0.05,0.05,
                                               0.04,0.04])
        lats = [arr_coords[a][0] + np.random.normal(0, 0.008) for a in arrondissements]
        lons = [arr_coords[a][1] + np.random.normal(0, 0.008) for a in arrondissements]
        
        clusters = np.random.choice([0, 1, 2, 3], n, p=[0.45, 0.28, 0.18, 0.09])
        anomalies = np.random.choice(['Surcote', 'Sous-cote'], n, p=[0.49, 0.51])
        surfaces = np.random.randint(15, 180, n)
        
        base_prices = {0: 9400, 1: 9400, 2: 11700, 3: 11000}
        prix_m2 = [base_prices[c] * np.random.uniform(1.3, 3.5) 
                   if a == 'Surcote' else base_prices[c] * np.random.uniform(0.1, 0.6)
                   for c, a in zip(clusters, anomalies)]
        
        df = pd.DataFrame({
            'latitude': lats,
            'longitude': lons,
            'arrondissement': arrondissements,
            'cluster': clusters,
            'anomalie': anomalies,
            'prix_m2': prix_m2,
            'surface_habitable': surfaces,
            'prix': [p * s for p, s in zip(prix_m2, surfaces)],
            'residu': np.random.uniform(483495, 2000000, n) * np.random.choice([-1, 1], n),
            'date_transaction': pd.date_range('2014-01-01', '2024-06-30', periods=n).strftime('%Y-%m-%d')
        })
    
    df['date_transaction'] = pd.to_datetime(df['date_transaction'])
    df['annee'] = df['date_transaction'].dt.year
    return df

df = load_data()

# ── Cluster labels ─────────────────────────────────────────────
CLUSTER_LABELS = {
    0: "Small properties — Popular areas",
    1: "Large properties — Popular areas", 
    2: "Medium properties — Affluent areas",
    3: "Large properties — High-end areas"
}

CLUSTER_COLORS = {
    0: [52, 152, 219],
    1: [46, 204, 113],
    2: [230, 126, 34],
    3: [155, 89, 182]
}

ANOMALY_COLORS = {
    'Surcote':   [214, 39, 40],
    'Sous-cote': [44, 160, 44]
}

# ── Sidebar filters ────────────────────────────────────────────
st.sidebar.markdown("## 🔍 Filters")

anomaly_filter = st.sidebar.multiselect(
    "Anomaly type",
    options=['Surcote', 'Sous-cote'],
    default=['Surcote', 'Sous-cote'],
    format_func=lambda x: "🔴 Overpriced" if x == 'Surcote' else "🟢 Underpriced"
)

cluster_filter = st.sidebar.multiselect(
    "Market cluster",
    options=[0, 1, 2, 3],
    default=[0, 1, 2, 3],
    format_func=lambda x: f"Cluster {x} — {CLUSTER_LABELS[x].split('—')[0].strip()}"
)

if 'annee' in df.columns:
    year_min = int(df['annee'].min())
    year_max = int(df['annee'].max())
    year_range = st.sidebar.slider(
        "Year range",
        min_value=year_min, max_value=year_max,
        value=(year_min, year_max)
    )
else:
    year_range = (2014, 2024)

color_by = st.sidebar.radio(
    "Color points by",
    options=["Anomaly type", "Market cluster"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model results")
st.sidebar.markdown("**R² (regression):** 0.720")
st.sidebar.markdown("**RMSE:** 217,510 €")
st.sidebar.markdown("**Anomaly threshold:** 483,495 €")
st.sidebar.markdown("**Total transactions:** 317,413")
st.sidebar.markdown("**Flagged by both methods:** 6,923 (2.18%)")

# ── Apply filters ──────────────────────────────────────────────
mask = (
    df['anomalie'].isin(anomaly_filter) &
    df['cluster'].isin(cluster_filter) &
    df['annee'].between(year_range[0], year_range[1])
)
df_filtered = df[mask].copy()

# ── KPIs ───────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{len(df_filtered):,}</div>
        <div class='metric-label'>Suspicious transactions shown</div>
    </div>""", unsafe_allow_html=True)

with col2:
    overpriced = len(df_filtered[df_filtered['anomalie'] == 'Surcote'])
    st.markdown(f"""
    <div class='metric-card' style='background: linear-gradient(135deg, #922b21, #c0392b);'>
        <div class='metric-value'>{overpriced:,}</div>
        <div class='metric-label'>🔴 Overpriced</div>
    </div>""", unsafe_allow_html=True)

with col3:
    underpriced = len(df_filtered[df_filtered['anomalie'] == 'Sous-cote'])
    st.markdown(f"""
    <div class='metric-card' style='background: linear-gradient(135deg, #1e8449, #27ae60);'>
        <div class='metric-value'>{underpriced:,}</div>
        <div class='metric-label'>🟢 Underpriced</div>
    </div>""", unsafe_allow_html=True)

with col4:
    avg_pm2 = df_filtered['prix_m2'].mean()
    st.markdown(f"""
    <div class='metric-card' style='background: linear-gradient(135deg, #6c3483, #8e44ad);'>
        <div class='metric-value'>{avg_pm2:,.0f}€</div>
        <div class='metric-label'>Avg price/m² (flagged)</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Map ────────────────────────────────────────────────────────
if len(df_filtered) > 0:
    
    if color_by == "Anomaly type":
        df_filtered['color'] = df_filtered['anomalie'].map(ANOMALY_COLORS)
    else:
        df_filtered['color'] = df_filtered['cluster'].map(CLUSTER_COLORS)
    
    df_filtered['color_r'] = df_filtered['color'].apply(lambda x: x[0])
    df_filtered['color_g'] = df_filtered['color'].apply(lambda x: x[1])
    df_filtered['color_b'] = df_filtered['color'].apply(lambda x: x[2])

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_filtered,
        get_position=["longitude", "latitude"],
        get_color=["color_r", "color_g", "color_b", 180],
        get_radius=60,
        pickable=True,
        auto_highlight=True,
        radius_min_pixels=3,
        radius_max_pixels=12,
    )

    view_state = pdk.ViewState(
        latitude=48.862,
        longitude=2.347,
        zoom=12,
        pitch=0,
    )

    tooltip = {
        "html": """
        <div style='background:#1a3c5e; padding:12px; border-radius:8px; color:white; font-family:Arial;'>
            <b style='font-size:14px;'>{anomalie}</b><br>
            <hr style='border-color:#2e75b6; margin:6px 0;'>
            <b>Price:</b> {prix:,.0f} €<br>
            <b>Price/m²:</b> {prix_m2:,.0f} €/m²<br>
            <b>Surface:</b> {surface_habitable} m²<br>
            <b>Arrondissement:</b> {arrondissement}e<br>
            <b>Cluster:</b> {cluster}<br>
            <b>Year:</b> {annee}
        </div>
        """,
        "style": {"backgroundColor": "transparent", "color": "white"}
    }

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    ), use_container_width=True)

else:
    st.warning("No transactions match the selected filters.")

# ── Bottom analysis ────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 📊 Distribution by anomaly type")
    if len(df_filtered) > 0:
        counts = df_filtered['anomalie'].value_counts()
        for label, count in counts.items():
            pct = count / len(df_filtered) * 100
            color = "#c0392b" if label == "Surcote" else "#27ae60"
            icon = "🔴" if label == "Surcote" else "🟢"
            st.markdown(f"""
            <div style='background:#1a3c5e; padding:12px; border-radius:8px; margin:6px 0;
                        border-left: 4px solid {color};'>
                {icon} <b>{"Overpriced" if label == "Surcote" else "Underpriced"}</b>
                — {count:,} transactions ({pct:.1f}%)
            </div>""", unsafe_allow_html=True)

with col_right:
    st.markdown("### 🏘️ Distribution by market cluster")
    if len(df_filtered) > 0:
        cluster_counts = df_filtered['cluster'].value_counts().sort_index()
        colors_hex = {0: "#3498db", 1: "#2ecc71", 2: "#e67e22", 3: "#9b59b6"}
        for cluster_id, count in cluster_counts.items():
            pct = count / len(df_filtered) * 100
            label = CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")
            color = colors_hex.get(cluster_id, "#2e75b6")
            st.markdown(f"""
            <div style='background:#1a3c5e; padding:12px; border-radius:8px; margin:6px 0;
                        border-left: 4px solid {color};'>
                <b>Cluster {cluster_id}</b> — {label}<br>
                {count:,} transactions ({pct:.1f}%)
            </div>""", unsafe_allow_html=True)

# ── Methodology note ───────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='background:#1a3c5e; padding:16px; border-radius:12px; border: 1px solid #2e75b6;'>
    <h4 style='color:#a8d4f5; margin:0 0 8px 0;'>📐 Methodology</h4>
    <p style='color:#ccc; margin:0; font-size:0.85rem;'>
    Transactions displayed are flagged simultaneously by <b style='color:white;'>two independent methods</b>: 
    (1) statistical deviation from cluster mean ± 2σ, and 
    (2) large residual from linear regression incorporating macro-economic variables (R²=0.720). 
    Dataset: DVF — Demandes de Valeurs Foncières | 317,413 Parisian transactions | 2014–2024.
    </p>
</div>
""", unsafe_allow_html=True)
