"""
SunSight AI — Massive Geospatial Grid Ingestion (1000+ Points)
==============================================================
Generates a massive geospatial coordinate grid across the Indian 
subcontinent and fetches 10 years of deep climate telemetry from 
the NASA POWER API with auto-retry and rate-limit handling.
"""

import os
import sys
import requests
import pandas as pd
import time
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"

# 11 Deep Climate Parameters (Max allowed by NASA without chunking)
PARAMETERS = [
    "ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "ALLSKY_KT", 
    "T2M_MAX", "T2M_MIN", "T2M", 
    "RH2M", "PRECTOTCORR", "WS10M",
    "PS", "T2MDEW" 
]

START_YEAR = 2014
END_YEAR = 2023
OUTPUT_CSV = "raw_telemetry_massive.csv"

# Geospatial Bounding Box for India
# Latitudes: 8.0 to 37.0 | Longitudes: 68.0 to 97.0
# A 30x35 grid yields exactly 1,050 coordinates
LATS = np.linspace(8.0, 37.0, 30)
LONS = np.linspace(68.0, 97.0, 35)

def generate_grid():
    grid = []
    for lat in LATS:
        for lon in LONS:
            grid.append((round(lat, 4), round(lon, 4)))
    return grid

# ---------------------------------------------------------------------------
# Core Fetcher with Exponential Backoff
# ---------------------------------------------------------------------------
def fetch_solar_data(lat: float, lon: float, retries: int = 3):
    params = {
        "parameters":  ",".join(PARAMETERS),
        "community":   "RE",                       
        "longitude":   lon,
        "latitude":    lat,
        "start":       START_YEAR,
        "end":         END_YEAR,
        "format":      "json",
    }
    
    for attempt in range(retries):
        try:
            # We use a 10-second timeout. If NASA hangs, we catch it.
            response = requests.get(NASA_POWER_BASE_URL, params=params, timeout=10)
            
            if response.status_code == 429: # Too Many Requests
                time.sleep(5 * (attempt + 1)) # Back off for 5, then 10, then 15 secs
                continue
                
            response.raise_for_status()
            data = response.json()
            
            parameter_data = data["properties"]["parameter"]
            records = []
            sample_param = PARAMETERS[0]
            
            for date_key in parameter_data[sample_param]:
                month = int(date_key[4:])
                if month == 13: # Skip annual averages
                    continue

                row = {
                    "year": int(date_key[:4]),
                    "month": month,
                    "latitude": lat,
                    "longitude": lon,
                }
                for param in PARAMETERS:
                    val = parameter_data[param].get(date_key)
                    row[param] = val if val != -999.0 else None

                records.append(row)
                
            return pd.DataFrame(records)

        except (requests.exceptions.RequestException, KeyError):
            time.sleep(2) # Brief pause on standard errors before retry
            
    return pd.DataFrame() # Return empty if all retries fail

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def main():
    coordinates = generate_grid()
    total_points = len(coordinates)
    
    print(f"[SunSight] Initializing MASSIVE dataset ingestion.")
    print(f"[SunSight] Target Grid: {total_points} locations across India.")
    print(f"[SunSight] Estimated Run Time: ~30 to 45 minutes.\n")
    
    all_data = []

    # tqdm creates a beautiful progress bar in your terminal
    for lat, lon in tqdm(coordinates, desc="Downloading Telemetry", unit="loc"):
        df = fetch_solar_data(lat, lon)
        if not df.empty:
            all_data.append(df)
            
        # Hard sleep of 1 second per request to respect NASA limits
        time.sleep(1)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n[SunSight] ✓ Assembly complete! Dataset saved to {OUTPUT_CSV}")
        print(f"[SunSight] Total Dataset Size: {len(final_df):,} rows.")
    else:
        print("\n[SunSight] ✗ Pipeline failed. No data retrieved.")

if __name__ == "__main__":
    main()