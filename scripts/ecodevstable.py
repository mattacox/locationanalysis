from censusdis.data import download
from censusdis.datasets import ACS5
from censusdis import states
import censusdis.maps as dem
import censusdis.data as ced
import folium
import branca.colormap as cm
import pandas as pd
import geopandas as gpd
from folium import Element
from sklearn.preprocessing import MinMaxScaler
import requests
from urllib3.exceptions import InsecureRequestWarning
import urllib3
urllib3.disable_warnings(category=InsecureRequestWarning)

orig_get = requests.get
def unsafe_get(*args, **kwargs):
    kwargs['verify'] = False
    return orig_get(*args, **kwargs)
requests.get = unsafe_get

years = [
    2017,
    2018,
    2019, 
    2020, 
    2021, 
    2022, 
    2023, 
    ]
all_years = []

# Load USDA data
usda = pd.read_csv("data/FoodAccessResearchAtlas.csv", dtype={"CensusTract": str})

# Load block group geometries
bg = gpd.read_file("data/tl_2024_37_bg.shp")
bg = bg[bg['COUNTYFP'].isin([
    '077', 
    # '181', 
    # '063',
    ])]  

# Truncate GEOID to 11-digit tract for merging
bg["tract"] = bg["GEOID"].str[:11]

# Merge tract-level USDA data onto block groups
bg_usda = bg.merge(usda, left_on="tract", right_on="CensusTract", how="left")

# Create boolean food desert flag
bg_usda["food_desert"] = bg_usda["LILATracts_1And10"] == 1



bg_vars = [
    "B19013_001E",  # Median household income
    "B17021_002E",  # Below poverty
    "B17021_001E",  # Poverty universe
    "B23025_005E",  # Unemployed
    "B23025_003E",  # In labor force
    "B15003_001E",  # Educational attainment total
    "B15003_017E",  # High school graduate
    "B15003_022E",  # Bachelor's degree
    "B25064_001E",  # Median gross rent

    "B25070_003E", # = households paying 0-14.9%

    "B25070_004E", # = households paying 15-19.9%

    "B25070_005E", # = households paying 20-24.9%

    "B25070_006E", # = households paying 25-29.9%

    "B25070_007E", # = households paying 30-34.9%

    "B25070_008E", # = households paying 35-39.9%

    'B25070_009E', # = households paying 40-49.9%

    'B25070_010E', #households paying 50% or more
    "B25070_001E",  # Renters total
    "B25002_003E",  # Vacant housing units
    "B25002_001E",  # Total housing units
    "B25003_003E",  # Renter-occupied units
    "B25003_001E",  # Total occupied units
    "B01001_001E",  # Total population
    "B01001_020E", "B01001_021E", "B01001_022E", "B01001_023E", "B01001_024E", "B01001_025E",  # Males 65+
    "B01001_044E", "B01001_045E", "B01001_046E", "B01001_047E", "B01001_048E", "B01001_049E",  # Females 65+
    "B22010_001E", "B22010_002E",  # SNAP
    "B08201_001E", "B08201_002E",  # Car access
    "B03002_001E", "B03002_003E", "B03002_004E", "B03002_012E"  # Race/ethnicity
]


for vintage in years:
    print(f"Pulling data for {vintage}")
    data = ced.download(
        dataset=ACS5,
        vintage=vintage,
        download_variables=bg_vars,
        state=states.NC,
        county=['077', #Granville
                # '181', #Vance
                # '063',#Durham
                ],  
        block_group='*',
        with_geometry=True,  # Geometry only needed once
    )

    # Construct GEOID from state + county + tract + block group
    data["GEOID"] = data["STATE"] + data["COUNTY"] + data["TRACT"] + data["BLOCK_GROUP"]
    data["year"] = vintage  # <-- Add this inside the for-loop
    data["timestamp"] = pd.to_datetime(data["year"].astype(str) + "-01-01")


    senior_vars = [
        "B01001_020E", "B01001_021E", "B01001_022E", "B01001_023E", "B01001_024E", "B01001_025E",
        "B01001_044E", "B01001_045E", "B01001_046E", "B01001_047E", "B01001_048E", "B01001_049E",
    ]

    data["senior_pop"] = data[senior_vars].sum(axis=1)
    data["senior_share"] = data["senior_pop"] / data["B01001_001E"]


    data["poverty_rate"] = data["B17021_002E"] / data["B17021_001E"]
    data = data.dropna(subset=["poverty_rate"])

    data["unemployment_rate"] = data["B23025_005E"] / data["B23025_003E"]
    data["hs_or_more"] = (data["B15003_017E"] + data["B15003_022E"]) / data["B15003_001E"]
    # Households paying less than 30%
    less_than_30 = data["B25070_003E"] + data["B25070_004E"] + data["B25070_005E"] + data["B25070_006E"]

    percent_less_than_30 = less_than_30 / data["B25070_001E"]

    # Or, for cost-burdened (30% or more)
    cost_burdened = data["B25070_007E"] + data["B25070_008E"] + data["B25070_009E"] + data["B25070_010E"]
    # Median rent value and formatted label
    data["median_rent"] = data["B25064_001E"]
    data["median_rent_str"] = data["median_rent"].fillna(0).apply(lambda x: "${:,.0f}".format(x))

    percent_cost_burdened = cost_burdened / data["B25070_001E"]
    data["percent_less_than_30"] = percent_less_than_30
    data["percent_cost_burdened"] = percent_cost_burdened
    data["vacancy_rate"] = data["B25002_003E"] / data["B25002_001E"]
    data["rent_share"] = data["B25003_003E"] / data["B25003_001E"]
    data["snap_share"] = data["B22010_002E"] / data["B22010_001E"]
    data["no_car_share"] = data["B08201_002E"] / data["B08201_001E"]
    data["black_share"] = data["B03002_004E"] / data["B03002_001E"]
    data["latino_share"] = data["B03002_012E"] / data["B03002_001E"]
    data["white_share"] = data["B03002_003E"] / data["B03002_001E"]
    data["high_poverty"] = data["poverty_rate"] > 0.20
    data["high_rent_share"] = data["rent_share"] > 0.60
    data["high_cost_burden"] = data["percent_cost_burdened"] > 0.30
    data["high_snap"] = data["snap_share"] > 0.20
    data["low_income"] = data["B19013_001E"] < 40000
    data["high_unemployment"] = data["unemployment_rate"] > 0.10
    data["senior_heavy"] = data["senior_share"] > 0.20  # Optional
    data["food_desert_flag"] = (
        bg_usda.set_index("GEOID")
        .reindex(data["GEOID"])
        ["food_desert"]
        .fillna(False)
        .values
    )
    data["median_income"] = data["B19013_001E"]
    data["median_income_str"] = data["B19013_001E"].fillna(0).apply(lambda x: "${:,.0f}".format(x))
    data["population_str"] = data["B01001_001E"].fillna(0).astype(int).apply(lambda x: f"{x:,}")



    # Additive score (can use weights later if desired)
    data["econ_dev_need_score"] = (
        data["high_poverty"].astype(int)
        + data["high_rent_share"].astype(int)
        + data["high_cost_burden"].astype(int)
        + data["high_snap"].astype(int)
        + data["low_income"].astype(int)
        + data["high_unemployment"].astype(int)
        + data["food_desert_flag"].astype(int)
        # + data["senior_heavy"].astype(int)  # Optional
    )

    # Flag top need
    data["high_econ_dev_need"] = data["econ_dev_need_score"] >= 5  # Adjust threshold as needed

    # scaler = MinMaxScaler()

    # # Define the fields to use
    # risk_fields = [
    #     "rent_share",
    #     "percent_cost_burdened",
    #     "poverty_rate",
    #     "snap_share",
    #     "unemployment_rate",
    #     # "no_car_share",
    #     "senior_share",
    #     # invert vacancy so that low vacancy = higher risk
    #     "vacancy_rate"
    # ]
    
    # # Create a copy with inverted vacancy
    # data["inv_vacancy"] = 1 - data["vacancy_rate"]

    # # Replace the original in list
    # risk_fields = [f if f != "vacancy_rate" else "inv_vacancy" for f in risk_fields]


    # # Drop rows with missing values for these fields
    # risk_data = data[risk_fields].dropna()


    # # Normalize
    # normalized = pd.DataFrame(scaler.fit_transform(risk_data), columns=risk_fields, index=risk_data.index)

    # # Optional: Add weights (equal weights by default)
    # # Weighted version: assign 2× weight to rent burden
    # weights = {
    #     "rent_share": 1,
    #     "percent_cost_burdened": 2,
    #     "poverty_rate": 1,
    #     "snap_share": 1,
    #     "unemployment_rate": 1,
    #     "senior_share": 1,
    #     "inv_vacancy": 1,
    # }
    # total_weight = sum(weights.values())
    # data.loc[normalized.index, "displacement_risk"] = (
    #     sum(normalized[field] * weight for field, weight in weights.items()) / total_weight
    # )


    all_years.append(data)

# --- Clean and sort ---
long_data_geo = pd.concat(all_years, ignore_index=True)
long_data_geo = long_data_geo.sort_values(["GEOID", "year"])

# --- Create Demographic Displacement Proxies ---
long_data_geo["black_share_change"] = long_data_geo.groupby("GEOID")["black_share"].diff()
long_data_geo["latino_share_change"] = long_data_geo.groupby("GEOID")["latino_share"].diff()

long_data_geo["black_decline"] = (long_data_geo["black_share_change"] < -0.02).astype(int)
long_data_geo["latino_decline"] = (long_data_geo["latino_share_change"] < -0.02).astype(int)

# --- Track Median Rent Change ---
long_data_geo["median_rent_pct_change"] = long_data_geo.groupby("GEOID")["median_rent"].pct_change()
long_data_geo["rapid_rent_increase"] = (long_data_geo["median_rent_pct_change"] > 0.10).astype(int)

# --- Normalize Core Displacement Risk Inputs ---
# Replace vacancy_rate with its inverse (low vacancy = higher risk)
long_data_geo["inv_vacancy"] = 1 - long_data_geo["vacancy_rate"]

# Define indicators to scale
risk_fields = [
    "rent_share",
    "percent_cost_burdened",
    "poverty_rate",
    "snap_share",
    "unemployment_rate",
    "senior_share",
    "inv_vacancy",
]

# Normalize with MinMaxScaler
scaler = MinMaxScaler()
normalized = pd.DataFrame(
    scaler.fit_transform(long_data_geo[risk_fields].fillna(0)),
    columns=risk_fields,
    index=long_data_geo.index
)

# --- Apply Custom Weights ---
weights = {
    "rent_share": 1,
    "percent_cost_burdened": 2,
    "poverty_rate": 1,
    "snap_share": 1,
    "unemployment_rate": 1,
    "senior_share": 1,
    "inv_vacancy": 1,
}
total_weight = sum(weights.values())

# Compute weighted composite index
long_data_geo["base_displacement_index"] = (
    sum(normalized[field] * weights[field] for field in risk_fields) / total_weight
)

# --- Final Displacement Risk Score ---
# Include binary flags: demographic decline + rent increase
long_data_geo["displacement_risk"] = (
    long_data_geo["base_displacement_index"]
    + long_data_geo["black_decline"] * 0.5
    + long_data_geo["latino_decline"] * 0.5
    + long_data_geo["rapid_rent_increase"] * 0.5
)

# Normalize to 0–1 scale again
long_data_geo["displacement_risk"] = MinMaxScaler().fit_transform(
    long_data_geo[["displacement_risk"]]
)



#clean geometries:

if data.crs is None:
    data.set_crs(epsg=4269, inplace=True)

data = data.to_crs(epsg=4326)
data = data[data.is_valid]
data = data[~data.geometry.is_empty]

from sklearn.preprocessing import MinMaxScaler
import folium
import branca.colormap as cm
from folium.plugins import TimestampedGeoJson

# List of indicators you want to map
indicators = [
    "poverty_rate",
    "percent_cost_burdened",
    "unemployment_rate",
    "snap_share",
    "rent_share",
    "senior_share",
    "displacement_risk",
    "vacancy_rate",
    "median_income",
    "median_rent", 
    "black_share",
    "white_share",
    "latino_share",

]

# Define optional fixed color range per indicator (None = use MinMaxScaler)
indicator_ranges = {
    "poverty_rate": (0, 0.5),
    "percent_cost_burdened": (0, 0.6),
    "unemployment_rate": (0, 0.25),
    "snap_share": (0, 0.5),
    "rent_share": (0, 0.8),
    "senior_share": (0, 0.5),
    "vacancy_rate": (0, 0.4),  # <-- Add this line
    

    "displacement_risk": (0, 1.0),  # Already scaled
    "median_rent": (300, 1500),  # Adjust range based on your data

}

for indicator in indicators:
    print(f"📍 Building map for: {indicator}")

    # Handle missing values
    long_data_geo[indicator] = long_data_geo[indicator].fillna(0)

    # Normalize
    if indicator_ranges.get(indicator):
        vmin, vmax = indicator_ranges[indicator]
        long_data_geo["scaled"] = long_data_geo[indicator].clip(lower=vmin, upper=vmax)
        long_data_geo["scaled"] = (long_data_geo["scaled"] - vmin) / (vmax - vmin)
    else:
        scaler = MinMaxScaler()
        long_data_geo["scaled"] = scaler.fit_transform(long_data_geo[[indicator]])


    # Compute year-over-year change and percent change from start year
    long_data_geo[f"{indicator}_yoy_change"] = long_data_geo.groupby("GEOID")[indicator].diff()
    long_data_geo[f"{indicator}_pct_change_from_start"] = (
        long_data_geo[indicator] / long_data_geo.groupby("GEOID")[indicator].transform("first") - 1
    )

    # Create color map
    if indicator_ranges.get(indicator):
        vmin, vmax = indicator_ranges[indicator]
    else:
        vmin = long_data_geo[indicator].min()
        vmax = long_data_geo[indicator].max()

    colormap = cm.linear.Blues_09.scale(vmin, vmax)
    if indicator == "median_income":
        colormap.caption = "Median Income ($)"
    elif indicator.endswith("_rate") or "share" in indicator or "percent" in indicator:
        colormap.caption = indicator.replace("_", " ").title() + " (%)"
    elif indicator == "displacement_risk":
        colormap.caption = "Displacement Risk (%)"
    elif indicator == "median_rent":
        colormap.caption = "Median Rent ($)"
    elif indicator in ["black_share", "white_share", "latino_share"]:
        colormap.caption = indicator.replace("_", " ").title() + " (%)"


    else:
        colormap.caption = indicator.replace("_", " ").title()



    # Create GeoJSON features
    features = []
    for _, row in long_data_geo.iterrows():
        color = colormap(row["scaled"])
        # Format values nicely for popup
        val = row[indicator]
        if indicator in ["poverty_rate", "percent_cost_burdened", "unemployment_rate", "snap_share", "rent_share", "senior_share","vacancy_rate", ]:
            popup_val = f"{val * 100:.1f}%"
        elif indicator == "median_income":
            popup_val = f"${val:,.0f}"  
        elif indicator == "displacement_risk":
            popup_val = f"{val * 100:.1f}%"
        elif indicator == "median_income" or indicator == "median_rent":
            popup_val = f"${val:,.0f}"
        elif indicator in ["black_share", "white_share", "latino_share"]:
            popup_val = f"{val * 100:.1f}%"

        else:
            popup_val = f"{val}"
            # Format YoY and total change
        yoy = row.get(f"{indicator}_yoy_change", None)
        pct_total = row.get(f"{indicator}_pct_change_from_start", None)

        if pd.notnull(yoy):
            if "rate" in indicator or "share" in indicator or "percent" in indicator or indicator == "displacement_risk":
                yoy_str = f"{yoy * 100:+.1f}%"
            elif "median" in indicator:
                yoy_str = f"{yoy:+,.0f}"
            else:
                yoy_str = f"{yoy:+.2f}"
        else:
            yoy_str = "N/A"

        if pd.notnull(pct_total):
            if "rate" in indicator or "share" in indicator or "percent" in indicator or indicator == "displacement_risk":
                pct_total_str = f"{pct_total * 100:.1f}%"
            elif "median" in indicator:
                pct_total_str = f"{pct_total * 100:.1f}%"
            else:
                pct_total_str = f"{pct_total:.2f}"
        else:
            pct_total_str = "N/A"


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

        feature = {
            "type": "Feature",
            "geometry": row["geometry"].__geo_interface__,
            "properties": {
                "time": row["timestamp"].strftime("%Y-%m-%d"),
                "style": {
                    "color": "black",
                    "weight": 0.7,
                    "fillColor": color,
                    "fillOpacity": .25,
                },
                "popup": popup_html,
            },
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    # Create map
    m = folium.Map(location=[36.3, -78.6], zoom_start=14, tiles='OpenStreetMap'   # Explicitly sets OSM base layer
)



    TimestampedGeoJson(
        geojson,
        transition_time=2000,
        loop=False,
        auto_play=False,
        period="P1Y",
        duration="P1Y",
        add_last_point=False,
    ).add_to(m)

    colormap.add_to(m)

    # Optional: Hide play/loop controls
    hide_controls = Element("""
    <style>
    .leaflet-control-timecontrol .leaflet-control-timecontrol-play,
    .leaflet-control-timecontrol .leaflet-control-timecontrol-loop {
        display: none !important;
    }
    </style>
    """)
    m.get_root().html.add_child(hide_controls)

    # Save to file
    output_path = f"html/{indicator}_timeslider.html"
    m.save(output_path)
    print(f"✅ Saved: {output_path}")

latest = long_data_geo[long_data_geo["year"] == 2023].copy()


m = folium.Map(location=[36.3, -78.6], zoom_start=14, tiles="OpenStreetMap")

for _, row in latest.iterrows():
    popup_html = f"""
<div style='max-width: 280px; font-size: 13px'>
    <strong>Year:</strong> {row['year']}<br>
    <strong>GEOID:</strong> {row['GEOID']}<br><hr style="margin: 4px 0">
    <strong>Median Income:</strong> ${row['median_income']:,.0f}<br>
    <strong>Median Rent:</strong> ${row['median_rent']:,.0f}<br>
    <strong>Rent Burden (30%+):</strong> {row['percent_cost_burdened']*100:.1f}%<br>
    <strong>Poverty Rate:</strong> {row['poverty_rate']*100:.1f}%<br>
    <strong>SNAP Share:</strong> {row['snap_share']*100:.1f}%<br><br>
    <strong>Race / Ethnicity:</strong><br>
    &nbsp;&nbsp;Black: {row['black_share']*100:.1f}%<br>
    &nbsp;&nbsp;Latino: {row['latino_share']*100:.1f}%<br>
    &nbsp;&nbsp;White: {row['white_share']*100:.1f}%<br>
</div>
"""
    folium.GeoJson(
        row["geometry"],
        style_function=lambda x: {
            "color": "black",
            "weight": 0.3,
            "fillColor": "#2171b5",  # You can scale color based on one variable if you want
            "fillOpacity": 0.4,
        },
        tooltip=folium.Tooltip(popup_html),
    ).add_to(m)

m.save(f"html/2023_income_burden_race_2023.html")

latest = long_data_geo[long_data_geo["year"] == 2017].copy()


m = folium.Map(location=[36.3, -78.6], zoom_start=14, tiles="OpenStreetMap")

for _, row in latest.iterrows():
    popup_html = f"""
    <div style='max-width: 250px; font-size: 13px'>
        <strong>GEOID:</strong> {row['GEOID']}<br>
        <strong>Median Income:</strong> ${row['median_income']:,.0f}<br>
        <strong>% Cost Burdened:</strong> {row['percent_cost_burdened'] * 100:.1f}%<br>
        <strong>Black Share:</strong> {row['black_share'] * 100:.1f}%<br>
        <strong>Latino Share:</strong> {row['latino_share'] * 100:.1f}%<br>
        <strong>White Share:</strong> {row['white_share'] * 100:.1f}%
    </div>
    """
    folium.GeoJson(
        row["geometry"],
        style_function=lambda x: {
            "color": "black",
            "weight": 0.3,
            "fillColor": "#2171b5",  # You can scale color based on one variable if you want
            "fillOpacity": 0.4,
        },
        tooltip=folium.Tooltip(popup_html),
    ).add_to(m)

m.save(f"html/2017_income_burden_race_2023.html")
