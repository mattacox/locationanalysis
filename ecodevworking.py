# --- Imports ---
import pandas as pd
import geopandas as gpd
import requests
import folium
import branca.colormap as cm
from folium import Element
from folium.plugins import TimestampedGeoJson
from sklearn.preprocessing import MinMaxScaler
from censusdis.data import download
from censusdis.datasets import ACS5
from censusdis import states
import censusdis.data as ced
import censusdis.maps as dem
from urllib3.exceptions import InsecureRequestWarning
import urllib3
import numpy as np


# --- Disable SSL warnings globally ---
urllib3.disable_warnings(category=InsecureRequestWarning)
orig_get = requests.get
def unsafe_get(*args, **kwargs):
    kwargs['verify'] = False
    return orig_get(*args, **kwargs)
requests.get = unsafe_get

# --- Config ---
years = [2017, 2018, 2019, 2021, 2022, 2023, 
        #  2024,
         ]
all_years = []

# --- Load External Data ---
usda = pd.read_csv("data/FoodAccessResearchAtlas.csv", dtype={"CensusTract": str})
bg = gpd.read_file("data/tl_2024_37_bg.shp")
bg = bg[bg['COUNTYFP'].isin(['077'])]  # Filter to Granville County
bg["tract"] = bg["GEOID"].str[:11]
bg_usda = bg.merge(usda, left_on="tract", right_on="CensusTract", how="left")
bg_usda["food_desert"] = bg_usda["LILATracts_1And10"] == 1




# --- ACS Variable List ---
bg_vars = [
    "B19013_001E", "B17021_002E", "B17021_001E", "B23025_005E", "B23025_003E",
    "B15003_001E", "B15003_017E", "B15003_022E", "B25064_001E",
    "B25070_003E", "B25070_004E", "B25070_005E", "B25070_006E",
    "B25070_007E", "B25070_008E", "B25070_009E", "B25070_010E", "B25070_001E",
    "B25002_003E", "B25002_001E", "B25003_003E", "B25003_001E",
    "B01001_001E", "B01001_020E", "B01001_021E", "B01001_022E",
    "B01001_023E", "B01001_024E", "B01001_025E", "B01001_044E",
    "B01001_045E", "B01001_046E", "B01001_047E", "B01001_048E", "B01001_049E",
    "B22010_001E", "B22010_002E",
    "B08201_001E", "B08201_002E",
    "B03002_001E", "B03002_003E", "B03002_004E", "B03002_012E",
    "B25004_002E", # For rent

"B25004_003E", #Rented, not occupied

"B25004_004E", #For sale only

"B08301_001E", #total workers
"B08301_010E", #workers from home
]



# --- Loop Over Vintages ---
for vintage in years:
    print(f"Pulling data for {vintage}")
    data = ced.download(
        dataset=ACS5,
        vintage=vintage,
        download_variables=bg_vars,
        state=states.NC,
        county=['077'],
        block_group='*',
        with_geometry=True,
    )

    # GEOID + Timestamp
    data["GEOID"] = data["STATE"] + data["COUNTY"] + data["TRACT"] + data["BLOCK_GROUP"]
    data["year"] = vintage
    data["timestamp"] = pd.to_datetime(f"{vintage}-01-01")

    # Senior Share
    senior_vars = [
        "B01001_020E", "B01001_021E", "B01001_022E", "B01001_023E", "B01001_024E", "B01001_025E",
        "B01001_044E", "B01001_045E", "B01001_046E", "B01001_047E", "B01001_048E", "B01001_049E",
    ]
    data["senior_pop"] = data[senior_vars].sum(axis=1)
    data["senior_share"] = data["senior_pop"] / data["B01001_001E"]

    # Indicators
    data["poverty_rate"] = data["B17021_002E"] / data["B17021_001E"]
    data = data.dropna(subset=["poverty_rate"])
    data["unemployment_rate"] = data["B23025_005E"] / data["B23025_003E"]
    data["hs_or_more"] = (data["B15003_017E"] + data["B15003_022E"]) / data["B15003_001E"]

    less_than_30 = data[["B25070_003E", "B25070_004E", "B25070_005E", "B25070_006E"]].sum(axis=1)
    cost_burdened = data[["B25070_007E", "B25070_008E", "B25070_009E", "B25070_010E"]].sum(axis=1)

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

    # Composite Score
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

    all_years.append(data)

# --- Clean and sort ---
long_data_geo = pd.concat(all_years, ignore_index=True)
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
risk_fields = [
    "rent_share", "percent_cost_burdened", "poverty_rate", "snap_share",
    "unemployment_rate", "senior_share", "inv_vacancy"
]
scaler = MinMaxScaler()
normalized = pd.DataFrame(
    scaler.fit_transform(long_data_geo[risk_fields].fillna(0)),
    columns=risk_fields,
    index=long_data_geo.index,
)

# --- Weighted composite index ---
weights = {
    "rent_share": 1,
    "percent_cost_burdened": 2,
    "poverty_rate": 1,
    "snap_share": 1,
    "unemployment_rate": 1,
    "senior_share": 1,
    "inv_vacancy": 2,
}
total_weight = sum(weights.values())

long_data_geo["base_displacement_index"] = (
    sum(normalized[field] * weights[field] for field in risk_fields) / total_weight
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

# Select Oxford block groups
oxford_bgs = long_data_geo[long_data_geo["year"] == 2023].copy()

# Simulate infill + IZ effect
scenario_infill = simulate_iz_effect(
    df=oxford_bgs,
    new_units=400 * len(oxford_bgs),  # 200 units per block group
    iz_rate=0.15,                     # 10% of new units are IZ
    timeline=10,
    share_from_renters=0.3,           # 30% of IZ units come from current renters
    pass_through=0.5                   # fraction of rent decrease that passes through
)

# --- Recompute displacement risk using IZ-adjusted rent burden ---
risk_fields = [
    "rent_share", "poverty_rate", "snap_share", "unemployment_rate",
    "senior_share", "inv_vacancy"
]

# Use IZ-adjusted rent burden instead of baseline
scenario_infill["percent_cost_burdened_for_risk"] = scenario_infill["percent_cost_burdened_iz"]

risk_fields_with_iz = risk_fields + ["percent_cost_burdened_for_risk"]

normalized_iz = pd.DataFrame(
    scaler.fit_transform(scenario_infill[risk_fields_with_iz].fillna(0)),
    columns=risk_fields_with_iz,
    index=scenario_infill.index,
)

scenario_infill["base_displacement_index_iz"] = (
    sum(normalized_iz[field] * weights.get(field, 1) for field in risk_fields_with_iz) / total_weight
)

# Add the same binary penalties (demographic/rent change)
scenario_infill["displacement_risk_iz"] = (
    scenario_infill["base_displacement_index_iz"]
    + scenario_infill["black_decline"] * 0.5
    + scenario_infill["latino_decline"] * 0.5
    + scenario_infill["rapid_rent_increase"] * 0.5
)

scenario_infill["displacement_risk_iz"] = MinMaxScaler().fit_transform(
    scenario_infill[["displacement_risk_iz"]]
)

scenario_infill["delta_displacement_risk"] = (
    scenario_infill["displacement_risk_iz"] - scenario_infill["displacement_risk"]
)


summary = scenario_infill[[
    "GEOID",
    "percent_cost_burdened",
    "percent_cost_burdened_iz",
    "renters_exiting",
    "displacement_risk",
    "displacement_risk_iz",
    "delta_displacement_risk"
]].copy()


# Add delta columns
summary["delta_rent_burden"] = summary["percent_cost_burdened_iz"] - summary["percent_cost_burdened"]

# Format as percentages
summary["delta_rent_burden_numeric"] = summary["percent_cost_burdened_iz"] - summary["percent_cost_burdened"]
summary["percent_cost_burdened"] = summary["percent_cost_burdened"].map("{:.1%}".format)
summary["percent_cost_burdened_iz"] = summary["percent_cost_burdened_iz"].map("{:.1%}".format)
summary["delta_rent_burden"] = summary["delta_rent_burden_numeric"].map("{:+.1%}".format)




print(summary.head(10))  # first 10 block groups

avg_delta = scenario_infill["percent_cost_burdened_iz"].mean() - scenario_infill["percent_cost_burdened"].mean()
print(f"Average reduction in rent burden: {avg_delta:.2%}")

total_renters_exiting = scenario_infill["renters_exiting"].sum()
print(f"Total renters able to move into new IZ units: {int(total_renters_exiting)}")

# Use numeric column for sorting
top_blocks = summary.sort_values("delta_rent_burden_numeric", ascending=True).head(5)
print(top_blocks[["GEOID", "delta_rent_burden", "percent_cost_burdened", "percent_cost_burdened_iz"]])


# --- Clean geometries ---
if data.crs is None:
    data.set_crs(epsg=4269, inplace=True)

data = data.to_crs(epsg=4326)
data = data[data.is_valid & ~data.geometry.is_empty]

# --- Map setup ---
indicators = [
    "poverty_rate", "percent_cost_burdened", "unemployment_rate", "snap_share",
    "rent_share", "senior_share", "displacement_risk", "rental_vacancy_rate",
    "median_income", "median_rent", "black_share", "white_share", "latino_share", "pct_work_from_home"

]

indicator_ranges = {
    "poverty_rate": (0, 0.5),
    "percent_cost_burdened": (0, 0.6),
    "unemployment_rate": (0, 0.25),
    "snap_share": (0, 0.5),
    "rent_share": (0, 0.8),
    "senior_share": (0, 0.5),
    "rental_vacancy_rate": (0, 0.4),
    "displacement_risk": (0, 1.0),
    "median_rent": (300, 1500),
    "black_share": (0, 1),
    "white_share": (0, 1),
    "latino_share": (0, 1),
    "median_income": (30000, 200000),  # adjust if needed
}

# --- Generate TimeSlider maps ---
for indicator in indicators:
    print(f"📍 Building map for: {indicator}")
    long_data_geo[indicator] = long_data_geo[indicator].fillna(0)

    # Normalize for color scaling
    if indicator_ranges.get(indicator):
        vmin, vmax = indicator_ranges[indicator]
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
    vmin = indicator_ranges.get(indicator, (long_data_geo[indicator].min(), long_data_geo[indicator].max()))[0]
    vmax = indicator_ranges.get(indicator, (long_data_geo[indicator].min(), long_data_geo[indicator].max()))[1]

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

        popup_html = f"""
        <div style='max-width: 250px; font-size: 13px'>
        <strong>Year:</strong> {row['year']}<br>
        <strong>GEOID:</strong> {row['GEOID']}<br>
        <strong>{indicator.replace('_', ' ').title()}:</strong> {popup_val}<br>
        <strong>Year-over-Year Change:</strong> {yoy_str}<br>
        <strong>% Change from 2017:</strong> {pct_total_str}<br>
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

    # Build and save map
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

    output_path = f"html/{indicator}_timeslider.html"
    m.save(output_path)
    print(f"✅ Saved: {output_path}")

latest_displacement = long_data_geo[long_data_geo["year"] == 2023][["GEOID", "displacement_risk"]].copy()
latest_displacement["displacement_risk_pct"] = (latest_displacement["displacement_risk"] * 100).map("{:.1f}%".format)
# Save to CSV
latest_displacement.to_csv("data/displacement_risk_2023.csv", index=False)
print("✅ Saved 2023 displacement risk scores to data/displacement_risk_2023.csv")

# --- Turn summary into a GeoDataFrame ---
summary_gdf = scenario_infill.merge(
    summary,
    on="GEOID",
    how="left",
    suffixes=("", "_summary")
)
summary_gdf = gpd.GeoDataFrame(summary_gdf, geometry="geometry", crs="EPSG:4326")

# --- Set up color map based on delta_rent_burden_numeric ---
vmin, vmax = summary_gdf["delta_rent_burden_numeric"].min(), summary_gdf["delta_rent_burden_numeric"].max()
colormap = cm.linear.RdYlGn_11.scale(vmin, vmax)
colormap.caption = "Change in Rent Burden (After IZ Simulation)"

# --- Build Folium Map ---
m = folium.Map(location=[36.3, -78.6], zoom_start=14, tiles="OpenStreetMap")

folium.TileLayer(
    tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    name='OSM'
).add_to(m)

# IZ-adjusted polygons
for _, row in summary_gdf.iterrows():
    col = colormap(row["delta_rent_burden_numeric"])
    popup_html = f"""
    <div style='font-size:13px; max-width:220px;'>
        <strong>GEOID:</strong> {row['GEOID']}<br>
        <strong>Original Rent Burden:</strong> {row['percent_cost_burdened']*100:.1f}%<br>
        <strong>IZ Adjusted Rent Burden:</strong> {row['percent_cost_burdened_iz']*100:.1f}%<br>
        <strong>Delta Rent Burden:</strong> {row['delta_rent_burden_numeric']*100:+.1f}%<br>
        <strong>Renters Exiting:</strong> {int(row['renters_exiting'])}<br><br>
        <strong>Original Displacement Risk:</strong> {row['displacement_risk']*100:.1f}%<br>
        <strong>IZ Displacement Risk:</strong> {row['displacement_risk_iz']*100:.1f}%<br>
        <strong>Delta Displacement Risk:</strong> {row['delta_displacement_risk']*100:+.1f}%
    </div>
    """
    folium.GeoJson(
        row["geometry"],
        style_function=lambda feature, col=col: {
            "fillColor": col,
            "color": "black",
            "weight": 0.7,
            "fillOpacity": 0.2,
        },
        tooltip=popup_html
    ).add_to(m)


colormap.add_to(m)
m.save("html/iz_displacement_rent_map.html")
print("✅ IZ + Displacement Risk map saved!")



def recompute_displacement_with_adjusted_burden(df_all, adjusted_bg_gdf):
    """
    Given the entire county-level df_all (GeoDataFrame) and a version that has
    percent_cost_burdened adjusted for one block group (in adjusted_bg_gdf),
    recompute the displacement_risk for all BGs using the same method you use
    in your script (normalize, weighted sum, add binary flags).
    Returns a copy of adjusted_bg_gdf with new displacement_risk column.
    """
    tmp = df_all.copy().reset_index(drop=True)
    # replace row(s) for the BG(s) being adjusted
    tmp = tmp.set_index("GEOID")
    adj = adjusted_bg_gdf.set_index("GEOID")
    for col in adj.columns:
        tmp.loc[adj.index, col] = adj[col]
    tmp = tmp.reset_index()

    # Recompute normalized inputs using your risk_fields (but with adjusted percent_cost_burdened)
    fields_for_norm = [
        "rent_share", "percent_cost_burdened", "poverty_rate", "snap_share",
        "unemployment_rate", "senior_share", "inv_vacancy"
    ]
    # If percent_cost_burdened in tmp may be NaN, fill with 0
    scaler_local = MinMaxScaler()
    normalized_local = pd.DataFrame(
        scaler_local.fit_transform(tmp[fields_for_norm].fillna(0)),
        columns=fields_for_norm,
        index=tmp.index,
    )

    tmp["base_displacement_index"] = (
        sum(normalized_local[f] * weights[f] for f in fields_for_norm) / total_weight
    )

    # add back your binary flags (these are already in tmp if you computed earlier)
    tmp["displacement_risk"] = (
        tmp["base_displacement_index"]
        + tmp.get("black_decline", 0) * 0.5
        + tmp.get("latino_decline", 0) * 0.5
        + tmp.get("rapid_rent_increase", 0) * 0.5
    )

    tmp["displacement_risk"] = MinMaxScaler().fit_transform(tmp[["displacement_risk"]])

    # return updated geometry rows (GeoDataFrame)
    return gpd.GeoDataFrame(tmp, geometry="geometry", crs=tmp.crs)

def simulate_local_infill_effect_for_bg(
    base_df, bg_geoid, total_new_units, iz_rate=0.15, share_from_renters=0.30, pass_through=0.5,
    renter_col="B25003_003E", rent_burden_col="percent_cost_burdened"
):
    """
    Apply an infill scenario where 'total_new_units' are all built in bg_geoid.
    Returns an adjusted GeoDataFrame (copy of base_df) with updated percent_cost_burdened
    for that BG and recomputed displacement_risk across all BGs.
    """
    df = base_df.copy().set_index("GEOID")
    if bg_geoid not in df.index:
        raise KeyError(f"{bg_geoid} not found in base_df GEOID index")

    iz_units = total_new_units * iz_rate
    renters_exiting = iz_units * share_from_renters

    # allocate renters_exiting entirely to the tested BG (local infill)
    renters_in_bg = df.at[bg_geoid, renter_col]
    if renters_in_bg <= 0:
        pct_reduction = 0.0
    else:
        pct_reduction = renters_exiting / renters_in_bg

    rent_change_pct = pct_reduction * pass_through

    # Compute new percent_cost_burdened for that BG (avoid negative)
    baseline_burden = df.at[bg_geoid, rent_burden_col]
    new_burden = baseline_burden * (1 - rent_change_pct)
    new_burden = max(new_burden, 0.0)

    adjusted = df.reset_index()
    adjusted.loc[adjusted["GEOID"] == bg_geoid, "percent_cost_burdened"] = new_burden
    adjusted_gdf = gpd.GeoDataFrame(adjusted, geometry="geometry", crs=adjusted.crs)

    # Recompute full displacement index across the county using the helper
    recomputed = recompute_displacement_with_adjusted_burden(base_df, adjusted_gdf)

    return recomputed

def find_min_units_for_target(
    base_df,
    bg_geoid,
    target_risk=0.4,
    iz_rate=0.10,
    share_from_renters=0.30,
    pass_through=0.5,
    max_units=20000,
    tol=1e-3,
    renter_col="B25003_003E"
):
    """
    Binary search on total_new_units built in bg_geoid (infill) to find the minimum total_new_units
    such that the bg's displacement risk falls to <= target_risk.
    Returns the number of units (int), and the final recomputed row for the BG (Series).
    If not achievable within max_units returns None and a message.
    """
    # baseline risk for the BG
    baseline_row = base_df.set_index("GEOID").loc[bg_geoid]
    baseline_risk = baseline_row["displacement_risk"]

    # quick check: if baseline already below target
    if baseline_risk <= target_risk:
        return 0, baseline_row

    lo, hi = 0, max_units
    found = None
    while lo <= hi:
        mid = (lo + hi) // 2
        recomputed = simulate_local_infill_effect_for_bg(
            base_df, bg_geoid, total_new_units=mid,
            iz_rate=iz_rate, share_from_renters=share_from_renters,
            pass_through=pass_through, renter_col=renter_col
        )
        risk_mid = recomputed.set_index("GEOID").at[bg_geoid, "displacement_risk"]
        # print(bg_geoid, "units", mid, "risk", risk_mid)  # debug if you want
        if risk_mid <= target_risk + tol:
            found = mid
            hi = mid - 1
        else:
            lo = mid + 1

    if found is None:
        return None, None
    recomputed_final = simulate_local_infill_effect_for_bg(
        base_df, bg_geoid, total_new_units=found,
        iz_rate=iz_rate, share_from_renters=share_from_renters,
        pass_through=pass_through, renter_col=renter_col
    )
    return found, recomputed_final.set_index("GEOID").loc[bg_geoid]

# Example batch run for all Oxford BGs
def compute_required_units_for_all_bgs(
    base_df,
    bg_list=None,
    target_risk=0.4,
    iz_rate=0.15,
    share_from_renters=0.30,
    pass_through=0.5,
    max_units=20000
):
    """
    For each BG in bg_list (or all in base_df if None), run binary search and return results DF.
    """
    results = []
    if bg_list is None:
        bg_list = base_df["GEOID"].unique().tolist()

    for geoid in bg_list:
        print("Testing", geoid)
        units_needed, final_row = find_min_units_for_target(
            base_df, geoid, target_risk=target_risk,
            iz_rate=iz_rate, share_from_renters=share_from_renters,
            pass_through=pass_through, max_units=max_units
        )
        if units_needed is None:
            results.append({
                "GEOID": geoid,
                "units_needed": None,
                "message": f"> {max_units} units required (not achieved)"
            })
        else:
            results.append({
                "GEOID": geoid,
                "units_needed": int(units_needed),
                "displacement_risk_final": final_row["displacement_risk"],
                "percent_cost_burdened_final": final_row["percent_cost_burdened"],
                "message": "achieved"
            })
    return pd.DataFrame(results)


# base_df should be the 2023 county-level geo-dataframe with columns like percent_cost_burdened, displacement_risk, etc.
base_df = long_data_geo[long_data_geo["year"] == 2023].copy()

# Optionally restrict to Oxford block groups only if you have a list
oxford_geoids = base_df["GEOID"].unique().tolist()  # or a subset if you know which are Oxford

results_df = compute_required_units_for_all_bgs(
    base_df,
    bg_list=oxford_geoids,
    target_risk=0.4,            # set acceptable threshold
    iz_rate=0.15,               # 10% IZ
    share_from_renters=0.30,    # 30% of IZ units from existing renters
    pass_through=0.5,
    max_units=20000           # cap search at e.g. 2000 units
)

results_df.to_csv("data/units_needed_per_bg.csv", index=False)
print(results_df)

def sensitivity_curve_for_bg(
    base_df,
    bg_geoid,
    unit_steps=None,
    iz_rate=0.15,
    share_from_renters=0.30,
    pass_through=0.5,
    renter_col="B25003_003E",
    rent_burden_col="percent_cost_burdened"
):
    """
    Run a sensitivity test: incrementally add housing units within a BG
    and record displacement risk at each step.
    
    Returns a DataFrame with GEOID, units_added, displacement_risk, and percent_cost_burdened.
    """
    if unit_steps is None:
        unit_steps = list(range(0, 2001, 100))  # default: 0 to 2000 in steps of 100

    records = []
    for u in unit_steps:
        recomputed = simulate_local_infill_effect_for_bg(
            base_df, bg_geoid, total_new_units=u,
            iz_rate=iz_rate,
            share_from_renters=share_from_renters,
            pass_through=pass_through,
            renter_col=renter_col,
            rent_burden_col=rent_burden_col
        )
        row = recomputed.set_index("GEOID").loc[bg_geoid]
        records.append({
            "GEOID": bg_geoid,
            "units_added": u,
            "displacement_risk": row["displacement_risk"],
            "percent_cost_burdened": row["percent_cost_burdened"]
        })

    return pd.DataFrame(records)



def batch_sensitivity_curves(
    base_df,
    bg_list=None,
    unit_steps=None,
    iz_rate=0.15,
    share_from_renters=0.30,
    pass_through=0.5,
    renter_col="B25003_003E",
    rent_burden_col="percent_cost_burdened",
    out_csv="data/sensitivity_curves.csv",
    plot_dir="plots/sensitivity_curves"
):
    """
    Run sensitivity curves for all BGs in bg_list.
    Saves combined results to CSV and optionally plots each BG's curve.
    """
    if bg_list is None:
        bg_list = base_df["GEOID"].unique().tolist()
    if unit_steps is None:
        unit_steps = list(range(0, 2001, 100))  # default sweep

    all_records = []

    # Create plot directory
    import os
    os.makedirs(plot_dir, exist_ok=True)

    for geoid in bg_list:
        print(f"Running sensitivity test for {geoid}...")
        curve_df = sensitivity_curve_for_bg(
            base_df, geoid, unit_steps=unit_steps,
            iz_rate=iz_rate, share_from_renters=share_from_renters,
            pass_through=pass_through,
            renter_col=renter_col, rent_burden_col=rent_burden_col
        )
        curve_df.to_csv(f"{plot_dir}/{geoid}_curve.csv", index=False)  # individual CSV
        all_records.append(curve_df)

        # Plot curve
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(curve_df["units_added"], curve_df["displacement_risk"], marker="o")
        plt.axhline(0.4, color="red", linestyle="--", label="Target Risk 0.4")
        plt.title(f"Sensitivity Curve for BG {geoid}")
        plt.xlabel("Units Added (infill)")
        plt.ylabel("Displacement Risk")
        plt.legend()
        plt.savefig(f"{plot_dir}/{geoid}_curve.png")
        plt.close()

    # Concatenate all into one CSV
    results_df = pd.concat(all_records, ignore_index=True)
    results_df.to_csv(out_csv, index=False)
    print(f"✅ Saved combined sensitivity results to {out_csv}")

    return results_df



# Merge the results back onto the GeoDataFrame
map_gdf = base_df.merge(results_df, on="GEOID", how="left")
for col in map_gdf.select_dtypes(include=["datetime", "datetimetz"]).columns:
    map_gdf[col] = map_gdf[col].astype(str)

# --- Map 1: Units Needed to reach target risk ---
vmin, vmax = 0, map_gdf["units_needed"].dropna().max()
colormap = cm.linear.YlGnBu_09.scale(vmin, vmax)
colormap.caption = "Units Needed (to reach target risk)"

m = folium.Map(location=[36.31, -78.59], zoom_start=13, tiles="OpenStreetMap")

def style_fn(feature):
    value = feature["properties"]["units_needed"]
    if value is None:
        return {"fillColor": "lightgray", "color": "black", "weight": 0.5, "fillOpacity": 0.3}
    return {
        "fillColor": colormap(value),
        "color": "black",
        "weight": 0.5,
        "fillOpacity": 0.7,
    }

folium.GeoJson(
    map_gdf.to_json(),
    style_function=style_fn,
    tooltip=folium.features.GeoJsonTooltip(
        fields=["GEOID", "units_needed", "displacement_risk_final", "message"],
        aliases=["BG", "Units Needed", "Final Risk", "Status"],
        localize=True
    )
).add_to(m)

colormap.add_to(m)
m.save("html/oxford_units_needed_map.html")


import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler

base_bg_gdf = base_df.copy()

def simulate_equilibrium_projection(
    base_bg_gdf,
    years=20,
    annual_infill_units_per_bg=0,        # if scalar -> applied to all BGs; if dict/series -> per-BG
    annual_external_units=0,             # units added outside (annexation) - reduces county pressure, not local
    iz_rate=0.15,                        # share of new units that are inclusionary (owner-occupied here)
    share_renter_to_owner_iz=0.30,       # share of IZ units that come from existing renters
    pass_through=0.5,                    # fraction of demand reduction that passes into rent decline
    rent_elasticity=0.8,                 # sensitivity of rents to demand-supply imbalance
    vacancy_dampener=0.5,                # reduces rent growth when vacancy increases
    exogenous_renter_growth_rate=0.00,   # annual % change in rental demand from outside (e.g., population growth)
    displacement_sensitivity=1.0,        # scale factor for how rent burden translates to displacement prob
    displacement_burden_threshold=0.40,  # rent burden (share) above which displacement accelerates (0-1)
    income_growth_rate=0.04,             # annual household income growth (affects cost-burden)
    normalize_index_each_year=True,      # whether to recompute MinMax normalization for displacement index
    stop_tol=1e-4,                       # equilibrium tolerance (mean abs change in displacement risk)
    verbose=False
):
    """
    Run a multi-year projection at block-group level and return a time-series DataFrame.
    base_bg_gdf: GeoDataFrame for baseline year containing necessary fields (see below).
    Returns: DataFrame with index = year and columns per BG (or long format time-series).
    IMPORTANT: This is a simplified structural model for scenario comparison, not a forecast.
    """

    # Make working copy and ensure necessary columns exist
    cols_needed = [
        "GEOID", "B25003_003E", "B25003_001E", "median_rent",
        "percent_cost_burdened", "rental_vacancy_rate",
        "rent_share", "poverty_rate", "snap_share", "unemployment_rate",
        "senior_share", "black_share", "latino_share", "white_share"
    ]
    for c in cols_needed:
        if c not in base_bg_gdf.columns:
            raise KeyError(f"Missing required column in base_bg_gdf: {c}")

    # Prepare per-BG arrays
    bg = base_bg_gdf.copy().set_index("GEOID")
    geoids = bg.index.tolist()

    # Baseline arrays
    renters = bg["B25003_003E"].astype(float).fillna(0)
    total_units = bg["B25003_001E"].astype(float).replace(0, np.nan).fillna(1)
    median_rent = bg["median_rent"].astype(float).fillna(0)
    rent_burden = bg["percent_cost_burdened"].astype(float).fillna(0)
    vacancy = bg["rental_vacancy_rate"].astype(float).fillna(0)
    rent_share = bg["rent_share"].astype(float).fillna(0)

    # Keep other risk fields for recomputing index each year
    static_fields = ["poverty_rate", "snap_share", "unemployment_rate", "senior_share", "black_share", "latino_share", "white_share"]
    static_df = bg[static_fields].fillna(0)

    # Pre-allocate time series store
    records = []

    # Normalize helper (we will reuse your weights and total_weight from the script environment)
    # Ensure 'weights' and 'total_weight' are available in caller namespace; else define defaults:
    try:
        weights_local = weights
        total_weight_local = total_weight
    except NameError:
        # fallback defaults in case not defined in caller namespace
        weights_local = {
            "rent_share": 1,
            "percent_cost_burdened": 2,
            "poverty_rate": 1,
            "snap_share": 1,
            "unemployment_rate": 1,
            "senior_share": 1,
            "inv_vacancy": 2,
        }
        total_weight_local = sum(weights_local.values())

    # For convenience allow annual_infill_units_per_bg to be scalar or Series/dict
    if np.isscalar(annual_infill_units_per_bg):
        annual_infill_array = pd.Series(annual_infill_units_per_bg, index=geoids, dtype=float)
    else:
        # try convertable
        annual_infill_array = pd.Series(annual_infill_units_per_bg).reindex(geoids).fillna(0).astype(float)

    # county-wide totals for normalization if needed
    base_for_normalization = bg.copy()

    # Iterative simulation
    prev_disp_risk = bg.get("displacement_risk", pd.Series(0, index=geoids)).astype(float).fillna(0)

    for year in range(1, years + 1):
        # 1) Supply additions this year (infill in-block + external)
        new_units_bg = annual_infill_array.copy()  # units built inside each BG this year
        # Add external units to county (affects county vacancy/rent dynamics but not local supply)
        county_new_units_external = float(annual_external_units)

        # 2) IZ conversions (owner-occupied affordable units) — remove demand of renters if share_from_renters
        iz_units_bg = new_units_bg * iz_rate
        renters_exiting_to_iz = iz_units_bg * share_renter_to_owner_iz

        # Clamp so renters_exiting cannot exceed renters in BG
        renters_exiting_to_iz = renters_exiting_to_iz.clip(upper=renters)

        # 3) Update renter demand with exogenous growth
        renters = renters * (1 + exogenous_renter_growth_rate) - renters_exiting_to_iz

        # 4) Displacement driven by high rent burden
        # displacement_prob ~ logistic or linear above threshold. Simple linear:
        # when rent_burden > threshold, displacement probability increases proportionally
        over_thresh = (rent_burden - displacement_burden_threshold).clip(lower=0)
        # displacement fraction this year
        disp_frac = (over_thresh / (1.0 - displacement_burden_threshold)) * displacement_sensitivity
        disp_frac = disp_frac.clip(0, 0.5)  # cap annual displacement fraction to 50% to keep stable
        renters_displaced = (renters * disp_frac).fillna(0)

        # Remove displaced renters from renter pool
        renters = (renters - renters_displaced).clip(lower=0)

        # 5) Update housing stock (approx): total_units increases by infill. For IZ owner units, they are still units (but reduce renter demand)
        total_units = total_units + new_units_bg

        # county-level demand/supply changes to estimate rent response
        # demand metric: total renters across county (after adjustments)
        county_renter_now = renters.sum()
        county_units_now = total_units.sum() + county_new_units_external

        # compute percent changes relative to previous step for demand and supply
        # For first iteration, assume previous values from baseline bg
        if year == 1:
            prev_county_renter = (base_bg_gdf.set_index("GEOID")["B25003_003E"].astype(float).fillna(0)).sum()
            prev_county_units = (base_bg_gdf.set_index("GEOID")["B25003_001E"].astype(float).replace(0, np.nan).fillna(1)).sum()
        else:
            prev_county_renter = county_renter_prev
            prev_county_units = county_units_prev

        demand_change_pct = (county_renter_now - prev_county_renter) / max(prev_county_renter, 1)
        supply_change_pct = (county_units_now - prev_county_units) / max(prev_county_units, 1)

        # rent growth rule (simple)
        rent_growth_pct_county = rent_elasticity * (demand_change_pct - supply_change_pct) - vacancy_dampener * ( (vacancy.mean() - vacancy.mean()) if False else 0.0 )
        # note: vacancy_dampener placeholder; we don't compute vacancy mean change here - you can extend

        # distribute county rent growth back to BG-level proportionally to exposure (you could weight by rent_share)
        bg_rent_growth = pd.Series(rent_growth_pct_county, index=geoids)

        # 6) Update median_rent and percent_cost_burdened (approx) per BG
        median_rent = median_rent * (1 + bg_rent_growth)
        # income grows slowly -> reduces burden slightly if positive
        # approximate: percent_cost_burdened scales with rent/income. If income grows by g and rent by r, burden scales by (1+r)/(1+g)
        rent_to_income_scale = (1 + bg_rent_growth) / (1 + income_growth_rate)
        rent_burden = (rent_burden * rent_to_income_scale).clip(0, 1)

        # 7) Recompute rent_share and vacancy approximately
        # rent_share = renters / (total_occupied_units). Here approximate total_occupied = total_units * (1 - vacancy)
        occupied_units = total_units * (1 - vacancy)
        # avoid divide by zero
        rent_share = (renters / occupied_units).fillna(0).clip(0, 1)

        # 8) Recompute inv_vacancy for displacement index and normalized fields
        inv_vac = 1 - vacancy
        # Build tmp DF for normalization
        tmp_df = pd.DataFrame({
            "rent_share": rent_share,
            "percent_cost_burdened": rent_burden,
            "poverty_rate": static_df["poverty_rate"],
            "snap_share": static_df["snap_share"],
            "unemployment_rate": static_df["unemployment_rate"],
            "senior_share": static_df["senior_share"],
            "inv_vacancy": inv_vac
        }, index=geoids).fillna(0)

        # Normalize (MinMax) across county for the year if requested
        if normalize_index_each_year:
            scaler_local = MinMaxScaler()
            normalized_local = pd.DataFrame(
                scaler_local.fit_transform(tmp_df),
                columns=tmp_df.columns,
                index=tmp_df.index
            )
        else:
            # if not normalizing each year, you could reuse a baseline scaler (not implemented here)
            scaler_local = MinMaxScaler()
            normalized_local = pd.DataFrame(
                scaler_local.fit_transform(tmp_df),
                columns=tmp_df.columns,
                index=tmp_df.index
            )

        # compute base_displacement_index per your weights
        base_idx = sum(normalized_local[f] * weights_local.get(f, 1) for f in normalized_local.columns) / total_weight_local

        # add binary flags — we reuse the ones from base_bg_gdf where available
        black_decl = bg.get("black_decline", pd.Series(0, index=geoids)).astype(float).fillna(0)
        lat_decl = bg.get("latino_decline", pd.Series(0, index=geoids)).astype(float).fillna(0)
        rapid_rent_inc = bg.get("rapid_rent_increase", pd.Series(0, index=geoids)).astype(float).fillna(0)

        displacement_risk = base_idx + 0.5 * black_decl + 0.5 * lat_decl + 0.5 * rapid_rent_inc
        # re-normalize displacement_risk across BGs to 0-1 for this year
        displacement_risk = MinMaxScaler().fit_transform(displacement_risk.values.reshape(-1, 1)).flatten()
        displacement_risk = pd.Series(displacement_risk, index=geoids)

        # store year snapshot
        rec = pd.DataFrame({
            "GEOID": geoids,
            "year": year,
            "renters": renters,
            "total_units": total_units,
            "median_rent": median_rent,
            "percent_cost_burdened": rent_burden,
            "rental_vacancy_rate": vacancy,
            "rent_share": rent_share,
            "displacement_risk": displacement_risk
        }).reset_index(drop=True)
        records.append(rec)

        # update prev county totals
        county_renter_prev = county_renter_now
        county_units_prev = county_units_now

        # check equilibrium: mean absolute change in displacement risk vs previous year
        mean_abs_change = np.mean(np.abs(displacement_risk.values - prev_disp_risk.values))
        prev_disp_risk = displacement_risk.copy()

        if verbose:
            print(f"Year {year}: mean abs displacement risk change = {mean_abs_change:.6f}")

        if mean_abs_change < stop_tol:
            if verbose:
                print(f"Equilibrium reached at year {year} (mean abs change {mean_abs_change:.6f} < tol {stop_tol})")
            break

    # concatenate records into a long dataframe
    timeseries_df = pd.concat(records, ignore_index=True)
    # join geometry from base_bg_gdf for mapping convenience
    geom = base_bg_gdf.set_index("GEOID")[["geometry"]]
    timeseries_df = timeseries_df.merge(geom, left_on="GEOID", right_index=True, how="left")
    return timeseries_df

# Example usage:
# base = long_data_geo[long_data_geo["year"] == 2023].copy()
# ts = simulate_equilibrium_projection(base, years=20, annual_infill_units_per_bg=10,
#                                      annual_external_units=200, iz_rate=0.1,
#                                      share_renter_to_owner_iz=0.3, verbose=True)
ts_df = simulate_equilibrium_projection(
    base_df,
    years=10,
    annual_infill_units_per_bg=100,
    annual_external_units=200,
    iz_rate=0.15,
    share_renter_to_owner_iz=0.3,
 
    verbose=True
)

print(ts_df.groupby("year")["displacement_risk"].mean())

import matplotlib.pyplot as plt

# Identify high-risk BGs at baseline
high_risk_geoids = base_df.loc[
    base_df["displacement_risk"] > 0.40, "GEOID"
]

# Filter time series to those BGs
high_risk_ts = ts_df[ts_df["GEOID"].isin(high_risk_geoids)]

# Pivot to wide format for line plotting
pivot = high_risk_ts.pivot(index="year", columns="GEOID",
                           values="displacement_risk")

plt.figure(figsize=(10,6))
for col in pivot.columns:
    plt.plot(pivot.index, pivot[col], alpha=0.5)
plt.title("Displacement Risk Trajectories for High-Risk BGs")
plt.xlabel("Year")
plt.ylabel("Displacement Risk Index")
plt.grid(True)
# plt.show()

