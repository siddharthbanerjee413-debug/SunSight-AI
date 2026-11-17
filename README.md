# ☀️ SunSight AI — Enterprise Geospatial Solar Viability & Intelligence Engine

> An enterprise-grade AI platform that predicts rooftop solar energy yield, financial ROI, and environmental ESG impact across the Indian subcontinent using localized NASA POWER climate telemetry, state-level tariffs, and machine learning models.

---

## 🌟 Overview

**SunSight AI** eliminates the guesswork in residential and commercial solar adoption. By synthesizing 10+ years of satellite solar irradiance data, ambient temperature trends, and regional meteorological parameters from the **NASA POWER API**, SunSight AI provides real-time solar viability scoring, expected annual generation (kWh), capital expenditure estimation, government subsidy calculations (PM Surya Ghar Muft Bijli Yojana), and payback timelines.

---

## 🏗️ Architecture Overview

```
                                  ┌───────────────────────────────┐
                                  │       Client Interfaces       │
                                  └───────────────┬───────────────┘
                                                  │
                   ┌──────────────────────────────┴──────────────────────────────┐
                   ▼                                                             ▼
     ┌────────────────────────────┐                                ┌────────────────────────────┐
     │  Enterprise SaaS Web App   │                                │      Gradio Dashboard      │
     │     (static/index.html)    │                                │          (app.py)          │
     │  Leaflet · Turf.js · Tailwind│                              │   Rapid Viability Explorer │
     └─────────────┬──────────────┘                                └─────────────┬──────────────┘
                   │ HTTP / JSON                                                 │
                   ▼                                                             ▼
     ┌──────────────────────────────────────────────────────────────────────────────────────────┐
     │                               FastAPI Gateway (server.py)                                │
     │                           REST API · CORS · Static Hosting                               │
     └─────────────────────────────────────────────┬────────────────────────────────────────────┘
                                                   │
                   ┌───────────────────────────────┴───────────────────────────────┐
                   ▼                                                               ▼
     ┌────────────────────────────┐                                  ┌────────────────────────────┐
     │    Climate Data Engine     │                                  │   ML Inference Pipeline    │
     │ ┌────────────────────────┐ │                                  │ ┌────────────────────────┐ │
     │ │ NASA POWER API (Live)  │ │                                  │ │ Yield Regression Model │ │
     │ └───────────┬────────────┘ │                                  │ │(yield_regression_model)│ │
     │             ▼ (fallback)   │                                  │ └───────────┬────────────┘ │
     │ ┌────────────────────────┐ │                                  │             ▼              │
     │ │ Geospatial Heuristics  │ │                                  │ ┌────────────────────────┐ │
     │ └────────────────────────┘ │                                  │ │ Viability Classifier   │ │
     └────────────────────────────┘                                  │ │(viability_classifier)  │ │
                                                                     │ └────────────────────────┘ │
                                                                     └─────────────┬──────────────┘
                                                                                   │
                   ┌───────────────────────────────────────────────────────────────┴──────────────┐
                   ▼                                                                              ▼
     ┌────────────────────────────┐                                                 ┌────────────────────────────┐
     │  Financial & Subsidy Rules │                                                 │      ESG Impact Engine     │
     │  PM Surya Ghar Scheme      │                                                 │  CO₂ Offset · Trees Planted│
     │  Capex · Net Cost · Payback│                                                 │  Car Emission Equivalence  │
     └────────────────────────────┘                                                 └────────────────────────────┘
```

---

## 🚀 Key Features

- **Interactive Geospatial Dashboard**: Leaflet-powered satellite map with polygon drawing tools (`Leaflet.draw` + `Turf.js`) to measure exact usable rooftop area.
- **Dual-Model ML Pipeline**:
  - **Yield Regression Model** (`yield_regression_model.pkl`): Predicts annual solar output (kWh) based on multi-variate climate indicators.
  - **Viability Classifier Model** (`viability_classifier_model.pkl`): Categorizes installation viability into distinct tiers (Optimal, Moderate, Low).
- **Deep Climate Telemetry**: Ingests 11 critical NASA POWER parameters including:
  - All-Sky / Clear-Sky Surface Downwelling Irradiance (`ALLSKY_SFC_SW_DWN`, `CLRSKY_SFC_SW_DWN`)
  - Insolation Clearness Index (`ALLSKY_KT`)
  - Maximum, Minimum, and Dew Point Temperatures (`T2M_MAX`, `T2M_MIN`, `T2MDEW`)
  - Relative Humidity (`RH2M`), Precipitation (`PRECTOTCORR`), Wind Speed (`WS10M`), and Surface Pressure (`PS`)
- **Fault-Tolerant Telemetry Fetching**: Live queries to the NASA POWER API with automatic fallback to geospatial climate heuristics if the upstream API times out.
- **Financial & Subsidy Simulation**: Real-world modeling of system sizing (kW), gross investment, central subsidies (e.g. PM Surya Ghar Muft Bijli Yojana), annual bill savings, and payback period.
- **ESG & Sustainability Impact**: Real-time computation of 25-year lifetime CO₂ displacement, equivalent trees planted, and vehicle emission offsets.
- **Exportable PDF Reports**: One-click generation of client-facing feasibility documents via `html2pdf`.

---

## 📂 Project Structure

```
SunSight-AI/
├── .gitignore                      # Git exclusion rules
├── README.md                       # Comprehensive project documentation
├── requirements.txt                # Python package dependencies
├── server.py                       # FastAPI application & REST endpoint (/api/predict)
├── app.py                          # Gradio dashboard interface
├── data_fetcher.py                 # Massive NASA POWER grid batch ingestion script
├── schema.sql                      # MySQL schema for state tariffs & audit queries
├── raw_telemetry_massive.csv       # Multi-year climate dataset across 1,000+ Indian coordinates
├── yield_regression_model.pkl      # Serialized scikit-learn regression model (Annual kWh)
├── viability_classifier_model.pkl  # Serialized scikit-learn viability classifier model
├── static/
│   └── index.html                  # Enterprise dashboard (Tailwind CSS, Leaflet, Turf.js)
└── document/
    ├── SunSight_AI_Project_Report.pdf   # Complete project documentation & analysis
    ├── SunSight AI - Project Report.pdf # Research report
    ├── SunSight AI Project Report.pptx  # Presentation deck
    └── SunSight_AI_Project_Report.docx  # Editable manuscript
```

---

## 🛠️ Setup & Installation

### Prerequisites

- **Python**: 3.9, 3.10, or 3.11 recommended
- **Git**
- Optional: **MySQL 8.0+** (if persisting query logs and tariffs via `schema.sql`)

### 1. Clone the Repository

```bash
git clone https://github.com/siddharthbanerjee413-debug/SunSight-AI.git
cd SunSight-AI
```

### 2. Create and Activate a Virtual Environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🖥️ Running the Application

### Option A: Enterprise SaaS Web Application (FastAPI + Modern Web Dashboard)

Launch the FastAPI backend server:

```bash
python server.py
# or using uvicorn directly:
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

- Interactive rooftop area tracing on map
- Real-time viability classification & financial metrics
- Instant PDF executive report download

### Option B: Gradio Analytics Interface

SunSight AI also includes an interactive Gradio dashboard:

```bash
python app.py
```

Open your browser and navigate to:
**[http://127.0.0.1:7860](http://127.0.0.1:7860)**

---

## 📡 REST API Documentation

### `POST /api/predict`

Calculates solar viability, annual generation, cost breakdown, and environmental impact for a given coordinate and consumption profile.

#### Request Body (`application/json`)

```json
{
  "latitude": 12.9716,
  "longitude": 77.5946,
  "monthly_bill": 4500.0,
  "roof_area": 850.0
}
```

#### Response (`application/json`)

```json
{
  "prediction": {
    "annual_yield_kwh": 6542.80,
    "viability_tier": 2,
    "system_kw": 4.7
  },
  "financials": {
    "gross_cost": 282000.0,
    "subsidy": 78000.0,
    "net_cost": 204000.0,
    "annual_savings": 52342.40,
    "payback_years": 3.90,
    "lifetime_savings": 1308560.0
  },
  "esg": {
    "co2_offset_tons": 134.1,
    "trees_planted": 3438,
    "cars_removed": 28.4
  },
  "explanation": {
    "primary_driver": "Surface irradiance is 5.34 kWh/m²/day.",
    "atmospheric_loss": "Clearness Index of 0.58 indicates atmospheric scattering/cloud cover.",
    "thermal_impact": "Peak temperatures around 32.1°C will cause moderate thermal derating of panels.",
    "recommendation": "Highly optimal for solar. Expect rapid ROI and maximum yield."
  },
  "climate": {
    "_source": "NASA POWER API (Live)",
    "ALLSKY_SFC_SW_DWN": 5.34,
    "CLRSKY_SFC_SW_DWN": 6.12,
    "ALLSKY_KT": 0.58,
    "T2M_MAX": 32.1,
    "T2M_MIN": 21.4,
    "T2M": 26.5,
    "RH2M": 58.2,
    "PRECTOTCORR": 2.1,
    "WS10M": 3.4,
    "PS": 92.5,
    "T2MDEW": 17.1
  }
}
```

---

## 🗄️ Optional Database Setup

To deploy the MySQL schema for state-wise tariffs and historical audit logs:

```bash
mysql -u root -p < schema.sql
```

Tables initialized:
- `state_tariffs`: Benchmark commercial and residential electricity tariffs per Indian state.
- `user_queries`: Historical search query audits with user parameters and predicted ROI.

---

## 👥 Team Members

- **Siddharth Banerjee**
- **Arsiya Panwar**
- **Katragadda Bhargav**

---

## 📜 License & Acknowledgments

- **Data Attribution**: Climate telemetry provided courtesy of the [NASA Prediction of Worldwide Energy Resources (POWER)](https://power.larc.nasa.gov/) project.
- **License**: This project is developed for academic, research, and non-commercial educational purposes.
