# --- Imports ---
import os
from folium.plugins import TimestampedGeoJson
import json
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import branca.colormap as cm
from sklearn.preprocessing import MinMaxScaler
from functions import *
import constants
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# --- Disable SSL warnings globally ---
urllib3.disable_warnings(category=InsecureRequestWarning)

# --- Utility: Unsafe requests for APIs ---
import requests
orig_get = requests.get
def unsafe_get(*args, **kwargs):
    kwargs['verify'] = False
    return orig_get(*args, **kwargs)
requests.get = unsafe_get

# --- Load external data ---
usda = pd.read_csv("data/FoodAccessResearchAtlas.csv", dtype={"CensusTract": str})

# --- Load NC shapefiles ---
bg = gpd.read_file("data/tl_2024_37_bg.shp")
bg = bg[bg['COUNTYFP'].isin(['077'])]
bg["tract"] = bg["GEOID"].str[:11]

# Merge USDA food desert info
bg_usda = bg.merge(usda, left_on="tract", right_on="CensusTract", how="left")
bg_usda["food_desert"] = bg_usda["LILATracts_1And10"] == 1

# --- Create cache directory ---
os.makedirs("cache", exist_ok=True)

# --- Download ACS data ---
dfs = []

for vintage in constants.years:
    print(f"\nPulling data for {vintage}")
    data = safe_download_acs_tract(vintage)
    if data is None:
        print(f"Skipping {vintage}")
        continue

    data.to_parquet(f"cache/bg_{vintage}.parquet")

    # --- GEOID and timestamps ---
    data["GEOID"] = data["STATE"] + data["COUNTY"] + data["TRACT"]
    data["year"] = vintage
    data["timestamp"] = pd.to_datetime(f"{vintage}-01-01")

    # --- Basic demographic indicators ---
    data["senior_pop"] = data[constants.senior_vars].sum(axis=1)
    data["senior_share"] = data["senior_pop"] / data["B01001_001E"]
    data["poverty_rate"] = data["B17021_002E"] / data["B17021_001E"]
    data = data.dropna(subset=["poverty_rate"])
    data["unemployment_rate"] = data["B23025_005E"] / data["B23025_003E"]
    data["hs_or_more"] = (data["B15003_017E"] + data["B15003_022E"]) / data["B15003_001E"]
    less_than_30 = data[["B25070_003E","B25070_004E","B25070_005E","B25070_006E"]].sum(axis=1)
    cost_burdened = data[["B25070_007E","B25070_008E","B25070_009E","B25070_010E"]].sum(axis=1)
    data["median_rent"] = data["B25064_001E"]
    data["median_rent_str"] = data["median_rent"].fillna(0).apply(lambda x: "${:,.0f}".format(x))
    data["percent_less_than_30"] = less_than_30 / data["B25070_001E"]
    data["percent_cost_burdened"] = cost_burdened / data["B25070_001E"]
    data["rental_vacancy_rate"] = data["B25004_002E"] / data["B25002_001E"]
    data["for_sale_vacancy_rate"] = data["B25004_004E"] / data["B25002_001E"]
    data["rent_share"] = data["B25003_003E"] / data["B25003_001E"]
    data["snap_share"] = data["B22010_002E"] / data["B22010_001E"]
    data["no_car_share"] = data["B08201_002E"] / data["B08201_001E"]
    data["black_share"] = data["B03002_004E"] / data["B03002_001E"]
    data["latino_share"] = data["B03002_012E"] / data["B03002_001E"]
    data["white_share"] = data["B03002_003E"] / data["B03002_001E"]

    # --- Flags ---
    data["high_poverty"] = data["poverty_rate"] > 0.2
    data["high_rent_share"] = data["rent_share"] > 0.6
    data["high_cost_burden"] = data["percent_cost_burdened"] > 0.3
    data["high_snap"] = data["snap_share"] > 0.2
    data["low_income"] = data["B19013_001E"] < 40000
    data["high_unemployment"] = data["unemployment_rate"] > 0.1
    data["senior_heavy"] = data["senior_share"] > 0.2
    data["food_desert_flag"] = (
        bg_usda.set_index("GEOID").reindex(data["GEOID"])["food_desert"].fillna(False).values
    )

    # --- Worker data ---
    data["pct_work_from_home"] = data["B08301_010E"] / data["B08301_001E"]
    data["median_income"] = data["B19013_001E"]
    data["median_income_str"] = data["B19013_001E"].fillna(0).apply(lambda x: "${:,.0f}".format(x))
    data["population_str"] = data["B01001_001E"].fillna(0).astype(int).apply(lambda x: f"{x:,}")

    # --- Econ development score ---
    data["econ_dev_need_score"] = (
        data[["high_poverty","high_rent_share","high_cost_burden",
              "high_snap","low_income","high_unemployment","food_desert_flag"]]
        .astype(int).sum(axis=1)
    )
    data["high_econ_dev_need"] = data["econ_dev_need_score"] >= 5

    dfs.append(data)

# --- Combine all years ---
if not dfs:
    print("❌ No ACS data downloaded.")
    sys.exit(1)

long_data_geo = pd.concat(dfs, ignore_index=True)
long_data_geo.sort_values(["GEOID","year"], inplace=True)

# --- Demographic change flags ---
long_data_geo["black_share_change"] = long_data_geo.groupby("GEOID")["black_share"].diff()
long_data_geo["latino_share_change"] = long_data_geo.groupby("GEOID")["latino_share"].diff()
long_data_geo["black_decline"] = (long_data_geo["black_share_change"] < -0.02).astype(int)
long_data_geo["latino_decline"] = (long_data_geo["latino_share_change"] < -0.02).astype(int)

# --- Rent change ---
long_data_geo["median_rent_pct_change"] = long_data_geo.groupby("GEOID")["median_rent"].pct_change()
county_median = long_data_geo.groupby(["COUNTY","year"])["median_rent_pct_change"].transform("median")
long_data_geo["rapid_rent_increase"] = (long_data_geo["median_rent_pct_change"] >= county_median*1.25).astype(int)

# --- Smoothed SNAP and vacancy ---
long_data_geo["snap_share_smoothed"] = (
    long_data_geo.groupby("GEOID")["snap_share"].rolling(3,min_periods=2).mean().reset_index(level=0,drop=True)
)
long_data_geo["inv_vacancy"] = 1 - long_data_geo["rental_vacancy_rate"]

# --- Ensure all risk fields exist ---
for f in constants.risk_fields:
    if f not in long_data_geo.columns:
        long_data_geo[f] = 0

# =========================================================
# Logistic Displacement Risk (Probability-Based)
# =========================================================

from scipy.stats import zscore

# --- 1. Ensure all risk fields exist ---
for f in constants.risk_fields:
    if f not in long_data_geo.columns:
        long_data_geo[f] = 0

# --- 2. Z-score risk fields across ALL tracts & years ---
z_fields = constants.risk_fields
z_df = long_data_geo[z_fields].copy()

z_df = z_df.apply(lambda x: zscore(x, nan_policy="omit"))
z_df = z_df.clip(-3, 3)  # cap extremes

for col in z_fields:
    long_data_geo[f"z_{col}"] = z_df[col]

# --- 3. Build log-odds model ---
INTERCEPT = -0.75  # baseline displacement pressure (~32%)

long_data_geo["log_odds"] = INTERCEPT

for f in z_fields:
    long_data_geo[f"z_{f}"] = (
        long_data_geo
        .groupby("year")[f]
        .transform(lambda x: zscore(x, nan_policy="omit"))
        .clip(-3, 3)
    )

# --- 4. Add bounded structural pressure flags (log-odds nudges) ---
long_data_geo["log_odds"] += (
    (long_data_geo["black_share_change"] < -0.02).astype(int) * 0.35
)

long_data_geo["log_odds"] += (
    (long_data_geo["latino_share_change"] < -0.02).astype(int) * 0.35
)

long_data_geo["log_odds"] += (
    long_data_geo["rapid_rent_increase"] * 0.4
)

# --- 5. Convert to probability ---
long_data_geo["displacement_probability"] = (
    1 / (1 + np.exp(-long_data_geo["log_odds"]))
)

# --- 6. Risk bands (policy-friendly) ---
def risk_band(p):
    if p < 0.25:
        return "Low"
    elif p < 0.45:
        return "Moderate"
    elif p < 0.65:
        return "High"
    else:
        return "Severe"

long_data_geo["risk_band"] = long_data_geo["displacement_probability"].apply(risk_band)

# --- 7. Backward compatibility (optional) ---
# If other parts of the script expect `displacement_risk`
long_data_geo["displacement_risk"] = long_data_geo["displacement_probability"]

# ---------------------------------------------
# HUD-Aligned Vulnerability & Market Pressure
# ---------------------------------------------

# Vulnerability: structurally at-risk populations
long_data_geo["hud_vulnerable"] = (
    (long_data_geo["poverty_rate"] >= 0.20) |
    (long_data_geo["percent_cost_burdened"] >= 0.30) |
    (long_data_geo["rent_share"] >= 0.50) |
    (long_data_geo["snap_share"] >= 0.20)
)

# Market pressure: signs of displacement dynamics
long_data_geo["hud_pressure"] = (
    (long_data_geo["rapid_rent_increase"] == 1) |
    (long_data_geo["black_decline"] == 1) |
    (long_data_geo["latino_decline"] == 1)
)

def hud_typology(row):
    if not row["hud_vulnerable"] and not row["hud_pressure"]:
        return "Stable / Not Vulnerable"

    if row["hud_vulnerable"] and not row["hud_pressure"]:
        return "At Risk of Displacement"

    if row["hud_vulnerable"] and row["hud_pressure"]:
        return "Ongoing Displacement"

    if not row["hud_vulnerable"] and row["hud_pressure"]:
        return "Advanced Gentrification"

    return "Unclassified"

long_data_geo["hud_typology"] = long_data_geo.apply(hud_typology, axis=1)




# --- Convert to GeoDataFrame ---
gdf = gpd.GeoDataFrame(long_data_geo, geometry="geometry", crs="EPSG:4326")





# --- Copy gdf and ensure timestamp ---
long_data_geo = gdf.copy()
print(long_data_geo["displacement_probability"].describe())

from folium.plugins import TimestampedGeoJson
from branca.element import Element

indicator = "displacement_probability"
print(f"📍 Building tract TimeSlider for: {indicator}")

long_data_geo[indicator] = long_data_geo[indicator].fillna(0)

# Normalize for color scaling
scaler = MinMaxScaler()
long_data_geo["scaled"] = scaler.fit_transform(long_data_geo[[indicator]])

# Change calculations
long_data_geo[f"{indicator}_yoy_change"] = long_data_geo.groupby("GEOID")[indicator].diff()

# Color map
colormap = cm.linear.Reds_09.scale(0.15, 0.75)
colormap.caption = "Displacement Risk (Estimated Probability)"


# Build GeoJSON features
features = []
for _, row in long_data_geo.iterrows():
    val = row[indicator]
    color = colormap(row["scaled"])

    popup_html = f"""
    <div style='max-width: 260px; font-size: 13px'>
    <strong>Year:</strong> {row['year']}<br>
    <strong>Tract:</strong> {row['GEOID']}<br>
    <strong>HUD Typology:</strong> {row['hud_typology']}<br>
    <strong>Displacement Risk:</strong> {val*100:.1f}% ({row['risk_band']})<br>
    <strong>YoY Change:</strong> {(row[f"{indicator}_yoy_change"] or 0)*100:+.1f}%<br>
    <strong>Median Rent:</strong> ${row['median_rent']:,.0f}<br>
    <strong>Median Income:</strong> ${row['median_income']:,.0f}<br>
    <strong>Renter Share:</strong> {row['rent_share']*100:.1f}%<br>
    <strong>Rent Burden (30%+):</strong> {row['percent_cost_burdened']*100:.1f}%<br>
    <strong>Poverty Rate:</strong> {row['poverty_rate']*100:.1f}%<br><br>
    <strong>Race / Ethnicity:</strong><br>
    Black: {row['black_share']*100:.1f}%<br>
    Latino: {row['latino_share']*100:.1f}%<br>
    White: {row['white_share']*100:.1f}%
    </div>
    """

    features.append({
        "type": "Feature",
        "geometry": row["geometry"].__geo_interface__,
        "properties": {
            "time": row["timestamp"].strftime("%Y-%m-%d"),
            "style": {
                "color": "#333",
                "weight": 0.6,
                "fillColor": color,
                "fillOpacity": 0.5,
            },
            "popup": popup_html,
        }
    })

# Build map
m_ts = folium.Map(location=[36.31, -78.59], zoom_start=12, tiles="cartodbpositron")

TimestampedGeoJson(
    {
        "type": "FeatureCollection",
        "features": features,
    },
    transition_time=2000,
    loop=False,
    auto_play=False,
    period="P1Y",
    duration="P1Y",
    add_last_point=False,
).add_to(m_ts)

colormap.add_to(m_ts)

# Optional: hide play/loop buttons
m_ts.get_root().html.add_child(Element("""
<style>
.leaflet-control-timecontrol .leaflet-control-timecontrol-play,
.leaflet-control-timecontrol .leaflet-control-timecontrol-loop {
    display: none !important;
}
</style>
"""))

m_ts.save("html/displacement_risk_timeslider_tract.html")
print("✅ Tract-level TimeSlider saved.")
