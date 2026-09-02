# --- Imports ---

import pandas as pd
import geopandas as gpd
import requests
import folium
import branca.colormap as cm
from folium import Element
from folium.plugins import TimestampedGeoJson
from sklearn.preprocessing import MinMaxScaler
from urllib3.exceptions import InsecureRequestWarning
import urllib3
from functions import *
import constants
import sys
import os



# --- Load External Data ---
# --- Load USDA Data ---

usda = pd.read_csv("data/FoodAccessResearchAtlas.csv", dtype={"CensusTract": str})

# --- Load NC shapefiles ---

bg = gpd.read_file("data/tl_2024_37_bg.shp")

# --- Filter by County ---

bg = bg[bg['COUNTYFP'].isin(['077'])]  

# --- Formatting the GEOID to make sure that the geometries match to the tract values in the USDA data ---

bg["tract"] = bg["GEOID"].str[:11]  

# --- Merge geometries into the USDA Data ---

bg_usda = bg.merge(usda, left_on="tract", right_on="CensusTract", how="left")
# --- Create boolean marker for the Census Tracts --- 

bg_usda["food_desert"] = bg_usda["LILATracts_1And10"] == 1

# --- create cache for API vintages ---

os.makedirs("cache", exist_ok=True)

# --- create empty dataframe for returned API data

dfs = []

# --- Loop Over Vintages from Census data API ---

for vintage in constants.years:
    print(f"\nPulling data for {vintage}")
    data = safe_download_acs(vintage) #safe_download_acs gracefully downloads the data from the Census API

    if data is None:
        print(f"Skipping {vintage} due to download failure.")
        continue

    data.to_parquet(f"cache/bg_{vintage}.parquet")

    # --- create new useful values: GEOID + Timestamp
    data["GEOID"] = data["STATE"] + data["COUNTY"] + data["TRACT"] + data["BLOCK_GROUP"]
    data["year"] = vintage 
    data["timestamp"] = pd.to_datetime(f"{vintage}-01-01") #fixed dates allow for YOY comparisons
    data["senior_pop"] = data[constants.senior_vars].sum(axis=1) # How many people over 65 live in this census block?
    data["senior_share"] = data["senior_pop"] / data["B01001_001E"] # How many seniors are there as a percentage of population?

    # Indicators
    data["poverty_rate"] = data["B17021_002E"] / data["B17021_001E"] #percentage of those below poverty rate in the last 12 months
    data = data.dropna(subset=["poverty_rate"]) 
    data["unemployment_rate"] = data["B23025_005E"] / data["B23025_003E"] # number of unemployed
    data["hs_or_more"] = (data["B15003_017E"] + data["B15003_022E"]) / data["B15003_001E"] # population with at least high school education level
    less_than_30 = data[["B25070_003E", "B25070_004E", "B25070_005E", "B25070_006E"]].sum(axis=1) #number of housholds below the 30% rent burden mark
    cost_burdened = data[["B25070_007E", "B25070_008E", "B25070_009E", "B25070_010E"]].sum(axis=1) #number of housholds above the 30% rent burden mark
    data["median_rent"] = data["B25064_001E"] # median gross rent
    data["median_rent_str"] = data["median_rent"].fillna(0).apply(lambda x: "${:,.0f}".format(x)) #formatted for map display
    data["percent_less_than_30"] = less_than_30 / data["B25070_001E"] # percentage of those not rent burdened 
    data["percent_cost_burdened"] = cost_burdened / data["B25070_001E"] # percentage of those rent burdened 
    data["rental_vacancy_rate"] = data["B25004_002E"] / data["B25002_001E"] #rental vacancy rate for census block
    data["for_sale_vacancy_rate"] = data["B25004_004E"] / data["B25002_001E"] #units for sale
    data["rent_share"] = data["B25003_003E"] / data["B25003_001E"] #total occupied houses
    data["snap_share"] = data["B22010_002E"] / data["B22010_001E"] #percent receiving federal food assistance
    data["no_car_share"] = data["B08201_002E"] / data["B08201_001E"] #percent without transportation
    data["black_share"] = data["B03002_004E"] / data["B03002_001E"] #AFAM share
    data["latino_share"] = data["B03002_012E"] / data["B03002_001E"] #Latino share
    data["white_share"] = data["B03002_003E"] / data["B03002_001E"] #White share

    # Binary Flags
    data["high_poverty"] = data["poverty_rate"] > 0.20
    data["high_rent_share"] = data["rent_share"] > 0.60
    data["high_cost_burden"] = data["percent_cost_burdened"] > 0.30
    data["high_snap"] = data["snap_share"] > 0.20
    data["low_income"] = data["B19013_001E"] < 40000
    data["high_unemployment"] = data["unemployment_rate"] > 0.10
    data["senior_heavy"] = data["senior_share"] > 0.20
    data["food_desert_flag"] = (
        bg_usda.set_index("GEOID").reindex(data["GEOID"])["food_desert"]
        .fillna(False).values
    )

    # Worker data
    data["pct_work_from_home"] = data["B08301_010E"] / data["B08301_001E"]
    data["median_income"] = data["B19013_001E"]
    data["median_income_str"] = data["B19013_001E"].fillna(0).apply(lambda x: "${:,.0f}".format(x))
    data["population_str"] = data["B01001_001E"].fillna(0).astype(int).apply(lambda x: f"{x:,}")

    # Composite Scores for Displacement Risk
    data["econ_dev_need_score"] = (
        data["high_poverty"].astype(int)
        + data["high_rent_share"].astype(int)
        + data["high_cost_burden"].astype(int)
        + data["high_snap"].astype(int)
        + data["low_income"].astype(int)
        + data["high_unemployment"].astype(int)
        + data["food_desert_flag"].astype(int)
        # + data["senior_heavy"].astype(int)
    )
    data["high_econ_dev_need"] = data["econ_dev_need_score"] >= 5
    dfs.append(data)

# --- Handle a graceful exit if there's no data ---

if not dfs:
    print("❌ No ACS data was downloaded for any year. Exiting gracefully.")
    sys.exit(1)

# --- Clean and sort ---
long_data_geo = pd.concat(dfs, ignore_index=True)
long_data_geo = long_data_geo.sort_values(["GEOID", "year"])

# --- Demographic displacement proxies ---
long_data_geo["black_share_change"] = long_data_geo.groupby("GEOID")["black_share"].diff()
long_data_geo["latino_share_change"] = long_data_geo.groupby("GEOID")["latino_share"].diff()
long_data_geo["black_decline"] = (long_data_geo["black_share_change"] < -0.02).astype(int)
long_data_geo["latino_decline"] = (long_data_geo["latino_share_change"] < -0.02).astype(int)

# --- Rent change tracking ---
long_data_geo["median_rent_pct_change"] = long_data_geo.groupby("GEOID")["median_rent"].pct_change()
long_data_geo["rapid_rent_increase"] = (long_data_geo["median_rent_pct_change"] > 0.10).astype(int)

# --- Normalize displacement inputs ---
long_data_geo["inv_vacancy"] = 1 - long_data_geo["rental_vacancy_rate"]

scaler = MinMaxScaler()

normalized = pd.DataFrame(
    scaler.fit_transform(long_data_geo[constants.risk_fields].fillna(0)),
    columns=constants.risk_fields,
    index=long_data_geo.index,
)

long_data_geo["base_displacement_index"] = (
    sum(normalized[field] * constants.weights[field] for field in constants.risk_fields) / constants.total_weight
)

# --- Final displacement risk score (with binary flags) ---
long_data_geo["displacement_risk"] = (
    long_data_geo["base_displacement_index"]
    + long_data_geo["black_decline"] * 0.5
    + long_data_geo["latino_decline"] * 0.5
    + long_data_geo["rapid_rent_increase"] * 0.5
)
long_data_geo["displacement_risk"] = MinMaxScaler().fit_transform(
    long_data_geo[["displacement_risk"]]
)

# --- Generate TimeSlider maps ---
for indicator in constants.indicators:
    print(f"📍 Building map for: {indicator}")
    long_data_geo[indicator] = long_data_geo[indicator].fillna(0)

    # Normalize for color scaling
    if constants.indicator_ranges.get(indicator):
        vmin, vmax = constants.indicator_ranges[indicator]
        long_data_geo["scaled"] = long_data_geo[indicator].clip(vmin, vmax)
        long_data_geo["scaled"] = (long_data_geo["scaled"] - vmin) / (vmax - vmin)
    else:
        scaler = MinMaxScaler()
        long_data_geo["scaled"] = scaler.fit_transform(long_data_geo[[indicator]])

    # Change calculations
    long_data_geo[f"{indicator}_yoy_change"] = long_data_geo.groupby("GEOID")[indicator].diff()
    long_data_geo[f"{indicator}_pct_change_from_start"] = (
        long_data_geo[indicator] / long_data_geo.groupby("GEOID")[indicator].transform("first") - 1
    )

    # Create color map
    vmin = constants.indicator_ranges.get(indicator, (long_data_geo[indicator].min(), long_data_geo[indicator].max()))[0]
    vmax = constants.indicator_ranges.get(indicator, (long_data_geo[indicator].min(), long_data_geo[indicator].max()))[1]

    colormap = cm.linear.Blues_09.scale(vmin, vmax)
    colormap.caption = {
        "median_income": "Median Income ($)",
        "displacement_risk": "Displacement Risk (%)",
        "median_rent": "Median Rent ($)"
    }.get(indicator, indicator.replace("_", " ").title() + (" (%)" if "rate" in indicator or "share" in indicator or "percent" in indicator else ""))

    # Build GeoJSON features
    features = []
    for _, row in long_data_geo.iterrows():
        val = row[indicator]
        color = colormap(row["scaled"])
        popup_val = (
            f"{val * 100:.1f}%" if "rate" in indicator or "share" in indicator or "percent" in indicator or indicator == "displacement_risk"
            else f"${val:,.0f}" if "median" in indicator
            else f"{val}"
        )
        yoy = row.get(f"{indicator}_yoy_change")
        pct_total = row.get(f"{indicator}_pct_change_from_start")

        yoy_str = (
            f"{yoy * 100:+.1f}%" if "rate" in indicator or "share" in indicator or "percent" in indicator or indicator == "displacement_risk"
            else f"{yoy:+,.0f}" if "median" in indicator
            else f"{yoy:+.2f}" if pd.notnull(yoy) else "N/A"
        )
        pct_total_str = (
            f"{pct_total * 100:.1f}%" if pd.notnull(pct_total) else "N/A"
        )

# --- Generate Popup HTML formatting ---

        popup_html = f"""
        <div style='max-width: 250px; font-size: 13px'>
        <strong>Year:</strong> {row['year']}<br>
        <strong>GEOID:</strong> {row['GEOID']}<br>
        <strong>{indicator.replace('_', ' ').title()}:</strong> {popup_val}<br>
        <strong>Year-over-Year Change:</strong> {yoy_str}<br>
        <strong>% Change from 2021:</strong> {pct_total_str}<br>
        <strong>Median Income:</strong> ${row['median_income']:,.0f}<br>
        <strong>Median Rent:</strong> ${row['median_rent']:,.0f}<br>
        <strong>Renter Share:</strong> {row['rent_share']*100:.1f}%<br>
        <strong>Rent Burden (30%+):</strong> {row['percent_cost_burdened']*100:.1f}%<br>
        <strong>Poverty Rate:</strong> {row['poverty_rate']*100:.1f}%<br>
        <strong>SNAP Share:</strong> {row['snap_share']*100:.1f}%<br><br>
        <strong>Race / Ethnicity:</strong><br>
        &nbsp;&nbsp;Black: {row['black_share']*100:.1f}%<br>
        &nbsp;&nbsp;Latino: {row['latino_share']*100:.1f}%<br>
        &nbsp;&nbsp;White: {row['white_share']*100:.1f}%<br>
        </div>
        """
        features.append({
            "type": "Feature",
            "geometry": row["geometry"].__geo_interface__,
            "properties": {
                "time": row["timestamp"].strftime("%Y-%m-%d"),
                "style": {"color": "black", "weight": 0.7, "fillColor": color, "fillOpacity": 0.25},
                "popup": popup_html,
            }
        })

    # --- Build and save each map ---

    m = folium.Map(location=[36.3, -78.6], zoom_start=14, tiles="OpenStreetMap")

    TimestampedGeoJson({
        "type": "FeatureCollection",
        "features": features,
    }, transition_time=2000, loop=False, auto_play=False, period="P1Y", duration="P1Y", add_last_point=False).add_to(m)

    colormap.add_to(m)
    m.get_root().html.add_child(Element("""
    <style>
    .leaflet-control-timecontrol .leaflet-control-timecontrol-play,
    .leaflet-control-timecontrol .leaflet-control-timecontrol-loop {
        display: none !important;
    }
    </style>
    """))

    output_path = f"html/demo/{indicator}_timeslider.html"
    m.save(output_path)
    print(f"✅ Saved: {output_path}")

latest_displacement = long_data_geo[long_data_geo["year"] == 2023][["GEOID", "displacement_risk"]].copy()
latest_displacement["displacement_risk_pct"] = (latest_displacement["displacement_risk"] * 100).map("{:.1f}%".format)
# Save to CSV


