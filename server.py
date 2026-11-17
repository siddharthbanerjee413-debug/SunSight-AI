import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="SunSight AI - Enterprise Solar Viability API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    reg_model = joblib.load(os.path.join(MODEL_DIR, "yield_regression_model.pkl"))
    clf_model = joblib.load(os.path.join(MODEL_DIR, "viability_classifier_model.pkl"))
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    reg_model, clf_model = None, None

class PredictRequest(BaseModel):
    latitude: float
    longitude: float
    monthly_bill: float
    roof_area: float

def get_nasa_power_data(lat, lon):
    url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,ALLSKY_KT,T2M_MAX,T2M_MIN,T2M,RH2M,PRECTOTCORR,WS10M,PS,T2MDEW",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "format": "JSON",
        "start": 2022,
        "end": 2022
    }
    response = requests.get(url, params=params, timeout=5)
    response.raise_for_status()
    data = response.json()
    
    features = {}
    parameters = data.get("properties", {}).get("parameter", {})
    for param, values in parameters.items():
        valid_months = [f"2022{str(m).zfill(2)}" for m in range(1, 13)]
        monthly_vals = [val for key, val in values.items() if key in valid_months and val != -999.0]
        if monthly_vals:
            features[param] = sum(monthly_vals) / len(monthly_vals)
        else:
            features[param] = 0.0
            
    return features

def get_climate_heuristics(lat, lon):
    irradiance = 5.2
    precip = 2.5
    
    lat_factor = -0.04 * abs(lat - 23.0)
    lon_factor = -0.008 * max(0, lon - 82.0)
    
    irradiance = max(3.0, min(7.0, irradiance + lat_factor + lon_factor))
    
    if lon > 88.0: precip = 5.5
    elif lat < 12.0 and lon < 78.0: precip = 4.0
    elif lat > 30.0: precip = 1.5

    return {
        'ALLSKY_SFC_SW_DWN': irradiance,
        'CLRSKY_SFC_SW_DWN': irradiance * 1.15,
        'ALLSKY_KT': irradiance / (irradiance * 1.15),
        'T2M_MAX': 31.0,
        'T2M_MIN': 21.0,
        'T2M': 26.0,
        'RH2M': 55.0,
        'PRECTOTCORR': precip,
        'WS10M': 3.8,
        'PS': 95.0,
        'T2MDEW': 16.0
    }

@app.post("/api/predict")
async def predict_viability(req: PredictRequest):
    if not reg_model or not clf_model:
        raise HTTPException(status_code=500, detail="Models not loaded")

    # 1. Feature Vector
    try:
        climate = get_nasa_power_data(req.latitude, req.longitude)
        climate['_source'] = 'NASA POWER API (Live)'
    except Exception as e:
        print(f"NASA API failed: {e}. Falling back to heuristics.")
        climate = get_climate_heuristics(req.latitude, req.longitude)
        climate['_source'] = 'Heuristic Engine (Fallback)'

    features = [
        'ALLSKY_SFC_SW_DWN', 'CLRSKY_SFC_SW_DWN', 'ALLSKY_KT', 
        'T2M_MAX', 'T2M_MIN', 'T2M', 'RH2M', 'PRECTOTCORR', 
        'WS10M', 'PS', 'T2MDEW'
    ]
    x_reg = pd.DataFrame([{f: climate[f] for f in features}])

    # 2. Inference Pipeline
    try:
        base_yield = float(reg_model.predict(x_reg)[0])
        
        x_clf = x_reg.copy()
        x_clf['Annual_Yield_kWh'] = base_yield
        viability_class = int(clf_model.predict(x_clf)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # 3. Dynamic Scaling & Physical Financial Engine
    system_kw = max(1.0, round((req.monthly_bill / 8.0) / 120.0, 1))
    actual_yield_kwh = base_yield * (system_kw / 4.0)

    gross_cost = system_kw * 60000.0
    if system_kw <= 2.0:
        subsidy = system_kw * 30000.0
    elif system_kw < 3.0:
        subsidy = 60000.0 + ((system_kw - 2.0) * 18000.0)
    else:
        subsidy = 78000.0

    net_cost = max(0, gross_cost - subsidy)
    
    true_annual_savings = actual_yield_kwh * 8.0
    payback_years = net_cost / true_annual_savings if true_annual_savings > 0 else 0
    lifetime_savings = true_annual_savings * 25
    
    # 4. ESG Impact Engine
    co2_offset = round(actual_yield_kwh * 25 * 0.00082, 1)
    trees_planted = round(co2_offset / 0.039)
    cars_removed = round(co2_offset / 4.73, 1)

    # 5. Structured Explanations
    irradiance = climate['ALLSKY_SFC_SW_DWN']
    kt = climate['ALLSKY_KT']
    tmax = climate['T2M_MAX']

    if viability_class == 2:
        recommendation = "Highly optimal for solar. Expect rapid ROI and maximum yield."
    elif viability_class == 1:
        recommendation = "Moderately viable. Solar is still beneficial but with longer payback times."
    else:
        recommendation = "Low financial viability due to suboptimal climate parameters."

    explanation = {
        "primary_driver": f"Surface irradiance is {irradiance:.2f} kWh/m²/day.",
        "atmospheric_loss": f"Clearness Index of {kt:.2f} indicates atmospheric scattering/cloud cover.",
        "thermal_impact": f"Peak temperatures around {tmax:.1f}°C will cause moderate thermal derating of panels.",
        "recommendation": recommendation
    }

    return {
        "prediction": {
            "annual_yield_kwh": round(actual_yield_kwh, 2),
            "viability_tier": viability_class,
            "system_kw": system_kw
        },
        "financials": {
            "gross_cost": round(gross_cost, 2),
            "subsidy": round(subsidy, 2),
            "net_cost": round(net_cost, 2),
            "annual_savings": round(true_annual_savings, 2),
            "payback_years": round(payback_years, 2),
            "lifetime_savings": round(lifetime_savings, 2)
        },
        "esg": {
            "co2_offset_tons": co2_offset,
            "trees_planted": trees_planted,
            "cars_removed": cars_removed
        },
        "explanation": explanation,
        "climate": climate
    }

os.makedirs(os.path.join(MODEL_DIR, "static"), exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(MODEL_DIR, "static")), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(MODEL_DIR, "static", "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
