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
       
        "B25038_001E",  # Total occupied housing units
        "B25038_002E",  # Owner occupied - moved in 2005 or earlier
        "B25038_003E",  # Owner occupied - 2006-2010
        "B25038_004E",  # Owner occupied - 2011-2015
        "B25038_005E",  # Owner occupied - 2016-2020
        "B25038_006E",  # Owner occupied - 2021 or later
        "B25038_007E",  # Renter occupied - moved in 2005 or earlier
        "B25038_008E",  # Renter occupied - 2006-2010
        "B25038_009E",  # Renter occupied - 2011-2015
        "B25038_010E",  # Renter occupied - 2016-2020
        "B25038_011E",  # Renter occupied - 2021 or later
             "B25036_001E",  # Total units
    "B25036_002E",  # Owner occupied - built 2014 or later
    "B25036_003E",  # Owner occupied - built 2010-2013
    "B25036_004E",  # Owner occupied - built 2000-2009
    "B25036_005E",  # Owner occupied - built 1980-1999
    "B25036_006E",  # Owner occupied - built 1960-1979
    "B25036_007E",  # Owner occupied - built 1940-1959
    "B25036_008E",  # Owner occupied - built 1939 or earlier
    "B25036_009E",  # Renter occupied - built 2014 or later
    "B25036_010E",  # Renter occupied - built 2010-2013
    "B25036_011E",  # Renter occupied - built 2000-2009
    "B25036_012E",  # Renter occupied - built 1980-1999
    "B25036_013E",  # Renter occupied - built 1960-1979
    "B25036_014E",  # Renter occupied - built 1940-1959
    "B25036_015E",  # Renter occupied - built 1939 or earlier
    "B25003_002E",
    "B25003_003E",
]

def simulate_iz_effect(
    df, 
    new_units=20000, 
    iz_rate=0.15, 
    timeline=10, 
    share_from_renters=0.30,
    pass_through=0.5,
    renter_col="B25003_003E",  # renter households
    rent_burden_col="percent_cost_burdened",
    allocation="proportional"
):
    """
    Adjusts rent burden based on an Inclusionary Zoning (IZ) scenario.
    """
    df = df.copy()
    
    # Total IZ units
    iz_units = new_units * iz_rate
    
    # Renters leaving rental pool
    renters_exiting_total = iz_units * share_from_renters
    
    # Allocate renters exiting
    if allocation == "proportional":
        weights = df[renter_col] / df[renter_col].sum()
    else:  # equal allocation
        weights = 1 / len(df)
    
    df["renters_exiting"] = renters_exiting_total * weights
    df["pct_reduction_renters"] = df["renters_exiting"] / df[renter_col]
    
    # Estimate rent change from reduced demand
    df["rent_change_pct"] = df["pct_reduction_renters"] * pass_through
    
    # Adjust rent burden
    df["iz_rent_burden_change"] = df[rent_burden_col] * df["rent_change_pct"]
    df["percent_cost_burdened_iz"] = df[rent_burden_col] - df["iz_rent_burden_change"]
    
    return df


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

        # --- Recent Mover Indicators (B07013) ---
        # --- Mobility / Recent Movers ---
    # Recent movers in past year by tenure (from B07013)
  
    data["recent_owner_movers"] = data["B25038_005E"] + data["B25038_006E"]  # moved 2016-2020 & 2021+
    data["recent_renter_movers"] = data["B25038_010E"] + data["B25038_011E"]
    data["percent_recent_owner_movers"] = data["recent_owner_movers"] / data["B25038_001E"]
    data["percent_recent_renter_movers"] = data["recent_renter_movers"] / data["B25038_001E"]



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

long_data_geo.to_csv("data/allyearshousing.csv", index=False)

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
    "median_income", "median_rent", "black_share", "white_share", "latino_share", "pct_work_from_home","percent_recent_owner_movers","percent_recent_owner_movers"

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
    "percent_recent_owner_movers": (0,.5),
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
