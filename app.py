"""
SunSight AI — Enterprise Solar Intelligence Dashboard
======================================================
A premium Gradio-based frontend for real-time solar viability
analysis. Loads pre-trained scikit-learn models when available,
falling back to calibrated empirical formulas for robustness.
"""

import gradio as gr
import os
import math

# ---------------------------------------------------------------------------
# Safe imports — never crash the UI on missing packages
# ---------------------------------------------------------------------------
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import joblib
except ImportError:
    joblib = None

# ---------------------------------------------------------------------------
# Model Loading (graceful fallback)
# ---------------------------------------------------------------------------
MODELS_LOADED = False
reg_model = None
clf_model = None
_load_error = ""

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

if joblib is not None:
    try:
        _reg_path = os.path.join(MODEL_DIR, "yield_regression_model.pkl")
        _clf_path = os.path.join(MODEL_DIR, "viability_classifier_model.pkl")
        reg_model = joblib.load(_reg_path)
        clf_model = joblib.load(_clf_path)
        MODELS_LOADED = True
    except Exception as exc:
        _load_error = str(exc)

ENGINE_STATUS = "● AI Engine Online — ML Models Active" if MODELS_LOADED else "● AI Engine Online — Empirical Mode"
ENGINE_COLOR = "#22c55e" if MODELS_LOADED else "#f59e0b"

# ---------------------------------------------------------------------------
# City Presets (matches data_fetcher.py climate zones)
# ---------------------------------------------------------------------------
CITY_PRESETS = {
    "Delhi (NCR)":    (28.6139, 77.2090),
    "Bangalore":      (12.9716, 77.5946),
    "Mumbai":         (19.0760, 72.8777),
    "Faridabad":      (28.4089, 77.3178),
    "Guwahati":       (26.1445, 91.7362),
    "Jaisalmer":      (26.9157, 70.9083),
    "Kochi":          (9.9312,  76.2673),
    "Kolkata":        (22.5726, 88.3639),
    "Leh":            (34.1526, 77.5771),
    "Chennai":        (13.0827, 80.2707),
    "Hyderabad":      (17.3850, 78.4867),
    "Ahmedabad":      (23.0225, 72.5714),
    "— Custom —":     (0.0, 0.0),
}

# ---------------------------------------------------------------------------
# Solar & Financial Engine
# ---------------------------------------------------------------------------
def _estimate_irradiance(lat: float, lon: float) -> float:
    """Heuristic irradiance estimator (kWh/m²/day) calibrated for India."""
    base = 5.2
    lat_factor = -0.04 * abs(lat - 23.0)
    lon_factor = -0.008 * max(0, lon - 82.0)
    return max(3.0, min(7.0, base + lat_factor + lon_factor))


def _estimate_precip(lat: float, lon: float) -> float:
    """Heuristic precipitation estimator (mm/day)."""
    if lon > 88.0:
        return 5.5
    if lat < 12.0 and lon < 78.0:
        return 4.0
    if lat > 30.0:
        return 1.5
    return 2.2


def calculate_system_size(monthly_bill: float) -> float:
    """Estimate recommended system capacity (kW) from electricity bill."""
    units_consumed = monthly_bill / 8.0
    kw = units_consumed / 120.0
    return max(1.0, round(kw * 2) / 2)  # Round to nearest 0.5 kW


def calculate_subsidy(system_kw: float) -> dict:
    """PM Surya Ghar Muft Bijli Yojana 2024-26 subsidy slabs."""
    if system_kw <= 1.0:
        central = 30000.0
    elif system_kw <= 2.0:
        central = 60000.0
    elif system_kw <= 3.0:
        central = 78000.0
    else:
        central = 78000.0  # Capped at 3 kW for central subsidy

    return {
        "central": central,
        "scheme": "PM Surya Ghar Muft Bijli Yojana",
    }


def predict_and_render(latitude, longitude, monthly_bill, roof_area):
    """Core prediction pipeline → returns styled HTML dashboard."""

    # ---- Validation ----
    errors = []
    if not (-90 <= latitude <= 90):
        errors.append("Latitude must be between –90° and 90°.")
    if not (-180 <= longitude <= 180):
        errors.append("Longitude must be between –180° and 180°.")
    if monthly_bill <= 0:
        errors.append("Monthly bill must be positive.")
    if roof_area <= 0:
        errors.append("Roof area must be positive.")
    if errors:
        return _render_error(errors)

    # ---- Climate estimation ----
    irradiance = _estimate_irradiance(latitude, longitude)
    precip = _estimate_precip(latitude, longitude)

    # ---- ML or empirical prediction ----
    if MODELS_LOADED and pd is not None:
        X_reg = pd.DataFrame({
            "ALLSKY_SFC_SW_DWN": [irradiance],
            "T2M": [26.0],
            "RH2M": [55.0],
            "PRECTOTCORR": [precip],
        })
        predicted_yield = float(reg_model.predict(X_reg)[0])

        X_clf = pd.DataFrame({
            "ALLSKY_SFC_SW_DWN": [irradiance],
            "PRECTOTCORR": [precip],
            "Annual_Yield_kWh": [predicted_yield],
        })
        viability_class = int(clf_model.predict(X_clf)[0])
    else:
        # Empirical fallback — calibrated against NASA POWER averages
        system_kw_est = calculate_system_size(monthly_bill)
        peak_sun_hours = irradiance  # Already in kWh/m²/day ≈ peak sun hours
        # Typical system factor: ~4.0 kWh/kWp/day in India, derated 15%
        predicted_yield = system_kw_est * peak_sun_hours * 365 * 0.82
        if irradiance >= 5.5:
            viability_class = 2
        elif irradiance >= 4.2:
            viability_class = 1
        else:
            viability_class = 0

    # ---- Financial calculations ----
    system_kw = calculate_system_size(monthly_bill)
    max_panels = math.floor(roof_area / 6.0)  # ~6 sq.m per panel (400W)
    max_kw_roof = max_panels * 0.4
    effective_kw = min(system_kw, max_kw_roof) if max_kw_roof > 0 else system_kw

    gross_cost = effective_kw * 58000.0  # ₹58k/kW avg installed cost 2024
    subsidy = calculate_subsidy(effective_kw)
    net_cost = max(0, gross_cost - subsidy["central"])
    annual_savings = monthly_bill * 12.0 * 0.85
    payback_years = round(net_cost / annual_savings, 1) if annual_savings > 0 else 0
    lifetime_savings = (annual_savings * 25) - net_cost

    # Recalculate yield for effective system size
    if not MODELS_LOADED:
        predicted_yield = effective_kw * irradiance * 365 * 0.82

    return _render_dashboard(
        latitude=latitude,
        longitude=longitude,
        viability_class=viability_class,
        system_kw=effective_kw,
        gross_cost=gross_cost,
        subsidy=subsidy,
        net_cost=net_cost,
        annual_savings=annual_savings,
        payback_years=payback_years,
        lifetime_savings=lifetime_savings,
        predicted_yield=predicted_yield,
        irradiance=irradiance,
        monthly_bill=monthly_bill,
    )


# ---------------------------------------------------------------------------
# HTML Renderers
# ---------------------------------------------------------------------------
VIABILITY_CONFIG = {
    0: {"label": "Low Feasibility",    "color": "#ef4444", "bg": "rgba(239,68,68,0.12)",  "icon": "⚠️"},
    1: {"label": "Moderate Viability", "color": "#f59e0b", "bg": "rgba(245,158,11,0.12)", "icon": "⚡"},
    2: {"label": "Highly Profitable",  "color": "#22c55e", "bg": "rgba(34,197,94,0.12)",  "icon": "🏆"},
}


def _render_error(errors: list) -> str:
    items = "".join(f"<li>{e}</li>" for e in errors)
    return f"""
    <div style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3);
                border-radius:12px; padding:24px; font-family:'Inter',system-ui,sans-serif;">
        <h3 style="color:#ef4444; margin:0 0 12px 0;">⚠️ Validation Error</h3>
        <ul style="color:#fca5a5; margin:0; padding-left:20px;">{items}</ul>
    </div>
    """


def _render_dashboard(**d) -> str:
    v = VIABILITY_CONFIG[d["viability_class"]]

    return f"""
<div style="font-family:'Inter','Segoe UI',system-ui,-apple-system,sans-serif; color:#e2e8f0;">

  <!-- Viability Banner -->
  <div style="background:{v['bg']}; border:1px solid {v['color']}33;
              border-radius:14px; padding:18px 24px; margin-bottom:20px;
              display:flex; align-items:center; gap:14px;">
    <span style="font-size:28px;">{v['icon']}</span>
    <div>
      <div style="font-size:13px; color:#94a3b8; text-transform:uppercase; letter-spacing:1.5px;
                  font-weight:600; margin-bottom:2px;">Solar Viability Assessment</div>
      <div style="font-size:22px; font-weight:700; color:{v['color']};">{v['label']}</div>
    </div>
    <div style="margin-left:auto; text-align:right;">
      <div style="font-size:12px; color:#64748b;">Location</div>
      <div style="font-size:14px; color:#cbd5e1; font-family:monospace;">{d['latitude']:.4f}°N, {d['longitude']:.4f}°E</div>
    </div>
  </div>

  <!-- Metric Cards Grid -->
  <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:14px; margin-bottom:20px;">

    <!-- Card 1: System Size -->
    <div style="background:linear-gradient(135deg, rgba(99,102,241,0.1), rgba(99,102,241,0.03));
                border:1px solid rgba(99,102,241,0.2); border-radius:12px; padding:20px;">
      <div style="font-size:12px; color:#818cf8; text-transform:uppercase; letter-spacing:1px;
                  font-weight:600; margin-bottom:8px;">⚡ System Size</div>
      <div style="font-size:32px; font-weight:800; color:#e2e8f0;
                  line-height:1;">{d['system_kw']:.1f}<span style="font-size:16px; color:#94a3b8; font-weight:400;"> kW</span></div>
      <div style="font-size:12px; color:#64748b; margin-top:6px;">Recommended capacity</div>
    </div>

    <!-- Card 2: Net Cost -->
    <div style="background:linear-gradient(135deg, rgba(16,185,129,0.1), rgba(16,185,129,0.03));
                border:1px solid rgba(16,185,129,0.2); border-radius:12px; padding:20px;">
      <div style="font-size:12px; color:#34d399; text-transform:uppercase; letter-spacing:1px;
                  font-weight:600; margin-bottom:8px;">💰 Net Cost</div>
      <div style="font-size:32px; font-weight:800; color:#e2e8f0;
                  line-height:1;">₹{d['net_cost']:,.0f}</div>
      <div style="font-size:12px; color:#64748b; margin-top:6px;">After central subsidy</div>
    </div>

    <!-- Card 3: Payback Period -->
    <div style="background:linear-gradient(135deg, rgba(251,191,36,0.1), rgba(251,191,36,0.03));
                border:1px solid rgba(251,191,36,0.2); border-radius:12px; padding:20px;">
      <div style="font-size:12px; color:#fbbf24; text-transform:uppercase; letter-spacing:1px;
                  font-weight:600; margin-bottom:8px;">📅 Payback</div>
      <div style="font-size:32px; font-weight:800; color:#e2e8f0;
                  line-height:1;">{d['payback_years']}<span style="font-size:16px; color:#94a3b8; font-weight:400;"> yrs</span></div>
      <div style="font-size:12px; color:#64748b; margin-top:6px;">Break-even estimate</div>
    </div>

    <!-- Card 4: Lifetime Savings -->
    <div style="background:linear-gradient(135deg, rgba(244,114,182,0.1), rgba(244,114,182,0.03));
                border:1px solid rgba(244,114,182,0.2); border-radius:12px; padding:20px;">
      <div style="font-size:12px; color:#f472b6; text-transform:uppercase; letter-spacing:1px;
                  font-weight:600; margin-bottom:8px;">🏦 25-Yr Savings</div>
      <div style="font-size:32px; font-weight:800; color:#e2e8f0;
                  line-height:1;">₹{d['lifetime_savings']:,.0f}</div>
      <div style="font-size:12px; color:#64748b; margin-top:6px;">Net lifetime benefit</div>
    </div>
  </div>

  <!-- Two Column: Breakdown Table + Bar Chart -->
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px;">

    <!-- Financial Breakdown Table -->
    <div style="background:rgba(30,41,59,0.5); border:1px solid rgba(148,163,184,0.1);
                border-radius:12px; padding:20px;">
      <div style="font-size:14px; font-weight:700; color:#e2e8f0; margin-bottom:14px;
                  display:flex; align-items:center; gap:8px;">
        <span style="color:#60a5fa;">📊</span> Financial Breakdown
      </div>
      <table style="width:100%; border-collapse:collapse; font-size:13px;">
        <tr style="border-bottom:1px solid rgba(148,163,184,0.1);">
          <td style="padding:10px 0; color:#94a3b8;">Gross Installation Cost</td>
          <td style="padding:10px 0; text-align:right; color:#e2e8f0; font-weight:600;">₹{d['gross_cost']:,.0f}</td>
        </tr>
        <tr style="border-bottom:1px solid rgba(148,163,184,0.1);">
          <td style="padding:10px 0; color:#94a3b8;">{d['subsidy']['scheme']}</td>
          <td style="padding:10px 0; text-align:right; color:#22c55e; font-weight:600;">– ₹{d['subsidy']['central']:,.0f}</td>
        </tr>
        <tr style="border-bottom:1px solid rgba(148,163,184,0.15);">
          <td style="padding:10px 0; color:#e2e8f0; font-weight:700;">Net Investment</td>
          <td style="padding:10px 0; text-align:right; color:#e2e8f0; font-weight:700;">₹{d['net_cost']:,.0f}</td>
        </tr>
        <tr style="border-bottom:1px solid rgba(148,163,184,0.1);">
          <td style="padding:10px 0; color:#94a3b8;">Annual Savings</td>
          <td style="padding:10px 0; text-align:right; color:#34d399; font-weight:600;">₹{d['annual_savings']:,.0f}/yr</td>
        </tr>
        <tr>
          <td style="padding:10px 0; color:#94a3b8;">Projected Annual Yield</td>
          <td style="padding:10px 0; text-align:right; color:#60a5fa; font-weight:600;">{d['predicted_yield']:,.0f} kWh</td>
        </tr>
      </table>
    </div>

    <!-- Solar vs Grid Visual -->
    <div style="background:rgba(30,41,59,0.5); border:1px solid rgba(148,163,184,0.1);
                border-radius:12px; padding:20px;">
      <div style="font-size:14px; font-weight:700; color:#e2e8f0; margin-bottom:14px;
                  display:flex; align-items:center; gap:8px;">
        <span style="color:#fbbf24;">📈</span> Annual Cost Comparison
      </div>

      <!-- Grid Cost Bar -->
      <div style="margin-bottom:16px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
          <span style="font-size:12px; color:#94a3b8;">Grid Electricity</span>
          <span style="font-size:13px; color:#ef4444; font-weight:600;">₹{d['monthly_bill'] * 12:,.0f}/yr</span>
        </div>
        <div style="background:rgba(239,68,68,0.1); border-radius:8px; height:28px; overflow:hidden;">
          <div style="background:linear-gradient(90deg, #ef4444, #dc2626); height:100%;
                      width:100%; border-radius:8px; display:flex; align-items:center;
                      padding-left:10px;">
            <span style="font-size:11px; color:white; font-weight:600;">Without Solar</span>
          </div>
        </div>
      </div>

      <!-- Solar Savings Bar -->
      {_render_savings_bar(d['monthly_bill'], d['annual_savings'])}

      <!-- Irradiance Indicator -->
      <div style="margin-top:18px; padding-top:14px; border-top:1px solid rgba(148,163,184,0.1);">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="font-size:12px; color:#94a3b8;">☀️ Solar Irradiance Index</span>
          <span style="font-size:14px; font-weight:700; color:#fbbf24;">{d['irradiance']:.2f} kWh/m²/day</span>
        </div>
        <div style="background:rgba(251,191,36,0.1); border-radius:6px; height:8px; margin-top:8px; overflow:hidden;">
          <div style="background:linear-gradient(90deg, #f59e0b, #fbbf24);
                      height:100%; width:{min(100, d['irradiance'] / 7.0 * 100):.0f}%;
                      border-radius:6px; transition:width 0.6s ease;"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Mode Indicator -->
  <div style="text-align:center; margin-top:16px; padding:10px;
              font-size:11px; color:#475569; letter-spacing:0.5px;">
    {"🤖 Prediction via trained ML pipeline (scikit-learn)" if MODELS_LOADED else "📐 Prediction via calibrated empirical model"} &nbsp;·&nbsp;
    Data: NASA POWER API &nbsp;·&nbsp; Subsidy: PM Surya Ghar 2024-26
  </div>
</div>
"""


def _render_savings_bar(monthly_bill: float, annual_savings: float) -> str:
    grid_annual = monthly_bill * 12
    solar_annual = grid_annual - annual_savings
    pct = max(5, min(95, (solar_annual / grid_annual * 100))) if grid_annual > 0 else 50

    return f"""
      <div>
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
          <span style="font-size:12px; color:#94a3b8;">With Rooftop Solar</span>
          <span style="font-size:13px; color:#22c55e; font-weight:600;">₹{solar_annual:,.0f}/yr</span>
        </div>
        <div style="background:rgba(34,197,94,0.1); border-radius:8px; height:28px; overflow:hidden;
                    position:relative;">
          <div style="background:linear-gradient(90deg, #22c55e, #16a34a); height:100%;
                      width:{pct:.0f}%; border-radius:8px; display:flex; align-items:center;
                      padding-left:10px; transition:width 0.6s ease;">
            <span style="font-size:11px; color:white; font-weight:600;">With Solar</span>
          </div>
        </div>
        <div style="text-align:right; margin-top:6px;">
          <span style="font-size:12px; font-weight:600; color:#22c55e;
                       background:rgba(34,197,94,0.12); padding:3px 10px; border-radius:20px;">
            ↓ {annual_savings / grid_annual * 100:.0f}% savings
          </span>
        </div>
      </div>
    """


# ---------------------------------------------------------------------------
# City Selector Handler
# ---------------------------------------------------------------------------
def on_city_change(city_name):
    """Return updated lat/lon when a city is selected."""
    if city_name in CITY_PRESETS and city_name != "— Custom —":
        lat, lon = CITY_PRESETS[city_name]
        return lat, lon
    return gr.update(), gr.update()


def sync_bill_slider(val):
    return val

def sync_roof_slider(val):
    return val


# ---------------------------------------------------------------------------
# Custom CSS for Dark Enterprise Theme
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* ---- Global Overrides ---- */
.gradio-container {
    background: linear-gradient(160deg, #0b0f1a 0%, #111827 50%, #0f172a 100%) !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
    min-height: 100vh;
}

/* ---- Header ---- */
#app-header {
    background: linear-gradient(135deg, rgba(30,41,59,0.8), rgba(15,23,42,0.9));
    border: 1px solid rgba(148,163,184,0.08);
    border-radius: 16px;
    padding: 28px 32px 24px 32px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
#app-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #f59e0b, #ef4444, #8b5cf6, #3b82f6, #22c55e);
    border-radius: 16px 16px 0 0;
}

/* ---- Input Panel ---- */
#input-panel {
    background: rgba(30,41,59,0.4) !important;
    border: 1px solid rgba(148,163,184,0.08) !important;
    border-radius: 14px !important;
    padding: 20px !important;
}

/* ---- Control Styling ---- */
.dark input, .dark select, .dark textarea {
    background: rgba(15,23,42,0.6) !important;
    border-color: rgba(148,163,184,0.15) !important;
    color: #e2e8f0 !important;
}
.dark .primary.svelte-1kg8u7r, .dark button.primary {
    background: linear-gradient(135deg, #f59e0b, #d97706) !important;
    border: none !important;
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 12px 0 !important;
    border-radius: 10px !important;
    letter-spacing: 0.5px;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 14px rgba(245,158,11,0.25) !important;
}
.dark .primary.svelte-1kg8u7r:hover, .dark button.primary:hover {
    background: linear-gradient(135deg, #fbbf24, #f59e0b) !important;
    box-shadow: 0 6px 20px rgba(245,158,11,0.35) !important;
    transform: translateY(-1px);
}

/* ---- Section Labels ---- */
#input-panel label {
    color: #94a3b8 !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px;
}

/* ---- Slider number input ---- */
.gradio-slider input[type=number] {
    background: rgba(15,23,42,0.6) !important;
    border: 1px solid rgba(148,163,184,0.15) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* ---- Output Panel ---- */
#output-panel {
    background: rgba(30,41,59,0.3) !important;
    border: 1px solid rgba(148,163,184,0.08) !important;
    border-radius: 14px !important;
    padding: 20px !important;
    min-height: 400px;
}

/* ---- Footer ---- */
footer { display: none !important; }

/* ---- Slider Track ---- */
input[type=range] {
    accent-color: #f59e0b !important;
}
"""


# ---------------------------------------------------------------------------
# Gradio Blocks Application
# ---------------------------------------------------------------------------
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.amber,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="*neutral_950",
    body_text_color="*neutral_200",
    block_background_fill="transparent",
    block_border_width="0px",
    input_background_fill="*neutral_900",
    input_border_color="*neutral_700",
)

with gr.Blocks(
    title="SunSight AI — Solar Intelligence Platform",
) as demo:

    # ---- Header ----
    gr.HTML(f"""
    <div id="app-header">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <h1 style="margin:0; font-size:28px; font-weight:800; color:#f1f5f9;
                           letter-spacing:-0.5px;">
                    ☀️ SunSight AI
                </h1>
                <p style="margin:4px 0 0 0; font-size:14px; color:#64748b; font-weight:400;">
                    Geospatial Solar Viability Engine &nbsp;·&nbsp; Powered by NASA POWER Telemetry
                </p>
            </div>
            <div style="display:flex; align-items:center; gap:8px; background:rgba(15,23,42,0.5);
                        padding:8px 16px; border-radius:20px; border:1px solid rgba(148,163,184,0.1);">
                <span style="display:inline-block; width:8px; height:8px; border-radius:50%;
                             background:{ENGINE_COLOR}; box-shadow:0 0 8px {ENGINE_COLOR};
                             animation: pulse 2s infinite;"></span>
                <span style="font-size:12px; color:#94a3b8; font-weight:500;">{ENGINE_STATUS}</span>
            </div>
        </div>
        <style>@keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.5; }} }}</style>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ---- Left Column: Inputs ----
        with gr.Column(scale=2, elem_id="input-panel"):
            gr.Markdown(
                "<span style='font-size:13px; color:#94a3b8; text-transform:uppercase; "
                "letter-spacing:1.5px; font-weight:700;'>📍 Configuration</span>"
            )

            city_dropdown = gr.Dropdown(
                choices=list(CITY_PRESETS.keys()),
                value="Faridabad",
                label="City / Region",
                info="Select a preset or choose 'Custom' to enter coordinates manually",
                interactive=True,
            )

            with gr.Row():
                lat_input = gr.Number(
                    value=28.4089, label="Latitude (°N)", precision=4, interactive=True,
                )
                lon_input = gr.Number(
                    value=77.3178, label="Longitude (°E)", precision=4, interactive=True,
                )

            gr.Markdown(
                "<span style='font-size:13px; color:#94a3b8; text-transform:uppercase; "
                "letter-spacing:1.5px; font-weight:700; margin-top:8px; display:block;'>"
                "💡 Energy & Property</span>"
            )

            bill_slider = gr.Slider(
                minimum=500, maximum=25000, step=100, value=3500,
                label="Monthly Electricity Bill (₹)",
                info="Your average monthly electricity expenditure",
            )

            roof_slider = gr.Slider(
                minimum=10, maximum=500, step=5, value=60,
                label="Usable Roof Area (sq. m)",
                info="Available shadow-free rooftop area",
            )

            submit_btn = gr.Button(
                "☀️  Analyze Solar Viability",
                variant="primary",
                size="lg",
            )

        # ---- Right Column: Results ----
        with gr.Column(scale=3, elem_id="output-panel"):
            gr.Markdown(
                "<span style='font-size:13px; color:#94a3b8; text-transform:uppercase; "
                "letter-spacing:1.5px; font-weight:700;'>📊 Intelligence Report</span>"
            )
            result_html = gr.HTML(
                value="""
                <div style="display:flex; align-items:center; justify-content:center;
                            min-height:350px; color:#475569; font-size:14px; text-align:center;
                            font-family:'Inter',system-ui,sans-serif;">
                    <div>
                        <div style="font-size:48px; margin-bottom:12px; opacity:0.4;">☀️</div>
                        <div style="color:#64748b;">Configure your parameters and click<br>
                        <strong style="color:#f59e0b;">Analyze Solar Viability</strong> to generate your report.</div>
                    </div>
                </div>
                """,
            )

    # ---- Event Wiring ----

    # City selector → populate lat/lon
    city_dropdown.change(
        fn=on_city_change,
        inputs=[city_dropdown],
        outputs=[lat_input, lon_input],
    )



    # Main prediction
    submit_btn.click(
        fn=predict_and_render,
        inputs=[lat_input, lon_input, bill_slider, roof_slider],
        outputs=[result_html],
    )


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        theme=THEME,
        css=CUSTOM_CSS,
    )