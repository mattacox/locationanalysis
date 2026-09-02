import pandas as pd
import geopandas as gpd
from censusdis.data import download
from censusdis.datasets import ACS5
from censusdis import states
import censusdis.data as ced
import censusdis.maps as dem
from shapely.geometry import LineString

import folium
from folium.features import GeoJsonTooltip

import math

def curved_line_coords(lat1, lon1, lat2, lon2, curvature=0.08):
    """
    Create a curved line (arc) between two coordinates.
    curvature: controls how far the arc bends (positive = right, negative = left)
    """
    # midpoint
    lat_mid = (lat1 + lat2) / 2
    lon_mid = (lon1 + lon2) / 2

    # offset perpendicular to the line
    dx = lon2 - lon1
    dy = lat2 - lat1
    dist = math.sqrt(dx**2 + dy**2)
    if dist == 0:
        return [(lat1, lon1), (lat2, lon2)]

    # normalized perpendicular vector
    nx = -dy / dist
    ny = dx / dist

    # apply offset for curvature
    offset_lat = lat_mid + curvature * ny
    offset_lon = lon_mid + curvature * nx

    # return as 3-point arc
    return [(lat1, lon1), (offset_lat, offset_lon), (lat2, lon2)]

years = [2017, 2018, 2019, 2021, 2022, 2023, 
        #  2024,
         ]
all_years = []

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


# --- Download tract geometries for NC using censusdis ---
print("Retrieving 2023 Census Data...")

tracts = ced.download(
        dataset=ACS5,
        vintage=2023,
        download_variables=bg_vars,
        state=states.NC,
        county=['*'],
        block_group='*',
        with_geometry=True,
)

tracts["TRACT"] = tracts["TRACT"].astype(str).str.zfill(6)


tracts["GEOID"] = tracts["STATE"] + tracts["COUNTY"] + tracts["TRACT"] + tracts["BLOCK_GROUP"]

tracts["GEOID"] = tracts["GEOID"].astype(str)

print("Loading LODES data...")

# URLs for RAC and WAC data (2021 is latest for most states)
rac_url = "https://lehd.ces.census.gov/data/lodes/LODES8/nc/rac/nc_rac_S000_JT00_2022.csv.gz"
wac_url = "https://lehd.ces.census.gov/data/lodes/LODES8/nc/wac/nc_wac_S000_JT00_2022.csv.gz"
od_url = "https://lehd.ces.census.gov/data/lodes/LODES8/nc/od/nc_od_main_JT00_2022.csv.gz"


# Load into DataFrames
rac = pd.read_csv(rac_url)
wac = pd.read_csv(wac_url)
od = pd.read_csv(od_url)


# Convert block IDs to string and trim to block group
rac["bg_geoid"] = rac["h_geocode"].astype(str).str[:12]
wac["bg_geoid"] = wac["w_geocode"].astype(str).str[:12]

# Aggregate to block group
rac_bg = rac.groupby("bg_geoid", as_index=False)["C000"].sum()
wac_bg = wac.groupby("bg_geoid", as_index=False)["C000"].sum()

# Rename for clarity
rac_bg.rename(columns={"C000": "workers_residing"}, inplace=True)
wac_bg.rename(columns={"C000": "workers_working"}, inplace=True)

# Simplify to home/work block group IDs (first 12 digits)
od["home_bg"] = od["h_geocode"].astype(str).str[:12]
od["work_bg"] = od["w_geocode"].astype(str).str[:12]

od_agg = od.groupby(["home_bg", "work_bg"], as_index=False)["S000"].sum()
od_agg.rename(columns={"S000": "workers"}, inplace=True)


# Convert to 4326 (just in case)
tracts = tracts.to_crs(4326)

# Merge by block group GEOID
rac_gdf = tracts.merge(rac_bg, left_on="GEOID", right_on="bg_geoid", how="left")
wac_gdf = tracts.merge(wac_bg, left_on="GEOID", right_on="bg_geoid", how="left")

# Fill missing with zeros for easier mapping
rac_gdf["workers_residing"] = rac_gdf["workers_residing"].fillna(0)
wac_gdf["workers_working"] = wac_gdf["workers_working"].fillna(0)
oxford_tracts = ['370779704002',
                 '370779704001',
                 '370779704002',
                '370779704003',
                '370779703002',
                '370779702002',
                '370779702003',
                '370779705001',
                '370779703001',
                '370779702001',
                '370779703003',

                 ]

oxford_bgs = tracts[
    (tracts["GEOID"].isin(oxford_tracts))
]

rac_oxford = rac_gdf[rac_gdf["GEOID"].isin(oxford_bgs["GEOID"])]
wac_oxford = wac_gdf[wac_gdf["GEOID"].isin(oxford_bgs["GEOID"])]

# Merge workers_residing from RAC data into oxford_bgs
oxford_bgs = oxford_bgs.merge(
    rac_bg[["bg_geoid", "workers_residing"]].rename(columns={"bg_geoid": "GEOID"}),
    on="GEOID",
    how="left"
)

oxford_bgs = oxford_bgs.merge(
    wac_bg[["bg_geoid", "workers_working"]].rename(columns={"bg_geoid": "GEOID"}),
    on="GEOID",
    how="left"
)



# Fill NaN with 0 just in case
oxford_bgs["workers_residing"] = oxford_bgs["workers_residing"].fillna(0).astype(int)


# Ensure GEOID is a string
rac_oxford["GEOID"] = rac_oxford["GEOID"].astype(str)
wac_oxford["GEOID"] = wac_oxford["GEOID"].astype(str)

# Convert to GeoJSON
rac_geojson = rac_oxford.to_json()
wac_geojson = wac_oxford.to_json()

#Mapp Commuting Patterns

print("Mapping commuting patterns...")

m = folium.Map(location=[36.31, -78.59], zoom_start=12, tiles="OpenStreetMap")

# RAC layer with tooltip
folium.Choropleth(
    geo_data=rac_geojson,
    data=rac_oxford,
    columns=["GEOID", "workers_residing"],
    key_on="feature.properties.GEOID",
    fill_color="YlGnBu",
    name="Residents (RAC)",
    legend_name="Workers Living in Area"
).add_to(m)

folium.GeoJson(
    rac_geojson,
    name="RAC Tooltips",
    style_function=lambda feature: {
        "fillColor": "transparent",
        "color": "transparent",
        "weight": 0
    },
    tooltip=GeoJsonTooltip(
        fields=["GEOID", "workers_residing"],
        aliases=["Block Group:", "Workers Living Here:"],
        localize=True
    )
).add_to(m)


# WAC layer with tooltip
folium.Choropleth(
    geo_data=wac_geojson,
    data=wac_oxford,
    columns=["GEOID", "workers_working"],
    key_on="feature.properties.GEOID",
    fill_color="YlOrRd",
    name="Jobs (WAC)",
    legend_name="Workers Working in Area"
).add_to(m)

folium.GeoJson(
    wac_geojson,
    name="WAC Tooltips",    
    style_function=lambda feature: {
        "fillColor": "transparent",
        "color": "transparent",
        "weight": 0
    },
    tooltip=GeoJsonTooltip(
        fields=["GEOID", "workers_working"],
        aliases=["Block Group:", "Workers Working Here:"],
        localize=True
    )
).add_to(m)

folium.LayerControl().add_to(m)
m.save("html/employment/oxford_commuting_patterns.html")

m = folium.Map(location=[36.31, -78.59], zoom_start=12, tiles="OpenStreetMap")


# --- Step 1: Get centroids for each block group safely ---
centroids = tracts.copy()
centroids = centroids[~tracts.geometry.is_empty & tracts.geometry.notnull()].copy()

# Handle invalid geometries (fix or skip)
centroids["geometry"] = centroids["geometry"].buffer(0)

# Compute centroids and drop any None results
centroids["centroid"] = centroids.geometry.centroid
centroids = centroids.dropna(subset=["centroid"])
centroids = centroids.set_geometry("centroid")[["GEOID", "centroid"]]


# --- Step 2: Merge centroids into OD data ---
od_geo = (
    od_agg
    .merge(centroids.rename(columns={"GEOID": "home_bg", "centroid": "home_geom"}), on="home_bg", how="left")
    .merge(centroids.rename(columns={"GEOID": "work_bg", "centroid": "work_geom"}), on="work_bg", how="left")
)

# Drop rows with missing geometries (e.g., work or home outside your county shapefile)
od_geo = od_geo.dropna(subset=["home_geom", "work_geom"])

# --- Step 3: Filter for flows involving Oxford ---

# --- Identify Oxford Block Groups ---
oxford_ids = oxford_bgs["GEOID"].tolist()

# --- Split OD data ---
od_outbound = od_geo[od_geo["home_bg"].isin(oxford_ids) & (~od_geo["work_bg"].isin(oxford_ids))]  # Oxford residents working elsewhere
od_inbound = od_geo[od_geo["work_bg"].isin(oxford_ids) & (~od_geo["home_bg"].isin(oxford_ids)) & (od_geo["workers"])]  # Non-residents working in Oxford
od_internal = od_geo[od_geo["home_bg"].isin(oxford_ids) & od_geo["work_bg"].isin(oxford_ids)  & (od_geo["workers"])]     # Live and work in Oxford

# For outbound (Oxford → outside)
outbound_home_ids = od_outbound["home_bg"].unique().tolist()  # all Oxford homes
outbound_work_ids = od_outbound["work_bg"].unique().tolist()  # destinations

# For inbound (outside → Oxford)
inbound_home_ids = od_inbound["home_bg"].unique().tolist()    # origins
inbound_work_ids = od_inbound["work_bg"].unique().tolist()    # all Oxford destinations

# Subset geometries for external origins/destinations
external_home_bgs = tracts[tracts["GEOID"].isin(inbound_home_ids)]
# Only include valid destination geometries
external_work_bgs = tracts[tracts["GEOID"].isin(outbound_work_ids)].copy()

# Optional: fade out all non-Oxford geometries for subtle background
background_bgs = tracts[~tracts["GEOID"].isin(oxford_ids + outbound_work_ids)].copy()

# --- Aggregate inbound, outbound, and internal counts per Oxford block group ---

# Outbound: workers living in Oxford (home_bg) but working elsewhere
outbound_counts = (
    od_outbound.groupby("home_bg")["workers"].sum().reset_index()
    .rename(columns={"home_bg": "GEOID", "workers": "outbound_workers"})
)

# Inbound: workers living elsewhere but working in Oxford
inbound_counts = (
    od_inbound.groupby("work_bg")["workers"].sum().reset_index()
    .rename(columns={"work_bg": "GEOID", "workers": "inbound_workers"})
)

# Internal: workers who live and work in same Oxford block group
internal_counts = (
    od_internal.groupby("home_bg")["workers"].sum().reset_index()
    .rename(columns={"home_bg": "GEOID", "workers": "internal_workers"})
)

oxford_bgs = (
    oxford_bgs.merge(outbound_counts, on="GEOID", how="left")
              .merge(inbound_counts, on="GEOID", how="left")
              .merge(internal_counts, on="GEOID", how="left")
)

# Replace NaN with 0
for col in ["inbound_workers", "outbound_workers", "internal_workers"]:
    oxford_bgs[col] = oxford_bgs[col].fillna(0).astype(int)


od_oxford = od_geo[
    (od_geo["home_bg"].isin(oxford_ids)) 
]

from shapely.geometry import LineString

# --- Step 4: Create LineStrings safely ---
od_oxford = od_oxford.copy()  # avoid SettingWithCopy warnings

# Function to safely create LineString
def make_line(row):
    try:
        return LineString([row["home_geom"], row["work_geom"]])
    except Exception:
        return None

for df in [od_outbound, od_inbound, od_internal]:
    df["line_geom"] = df.apply(make_line, axis=1)
    df.dropna(subset=["line_geom"], inplace=True)

od_outbound_gdf = gpd.GeoDataFrame(od_outbound, geometry="line_geom", crs=4326)
od_inbound_gdf = gpd.GeoDataFrame(od_inbound, geometry="line_geom", crs=4326)
od_internal_gdf = gpd.GeoDataFrame(od_internal, geometry="line_geom", crs=4326)

#Map Outbound Workers
print("Mapping outbound worker commutes...")
m_out = folium.Map(location=[36.31, -78.59], zoom_start=12, tiles="OpenStreetMap")

# --- Lightly fade out statewide background for context ---
folium.GeoJson(
    background_bgs.to_json(),
    name="Background (NC Block Groups)",
    style_function=lambda x: {
        "color": "#cccccc",
        "weight": 0.3,
        "fillColor": "#f2f2f2",
        "fillOpacity": 0.1
    },
    highlight_function=lambda x: {"weight": 0.5, "color": "#999999"}
).add_to(m_out)

# --- Filter destinations to only those with 10 or more Oxford-based workers ---
top_work_ids = (
    od_outbound.groupby("work_bg", as_index=False)["workers"]
    .sum()
    .query("workers > 10")["work_bg"]
    .tolist()
)

# Filter outbound flows to only include work_bg in top_work_ids
od_outbound_gdf = od_outbound_gdf[od_outbound_gdf["work_bg"].isin(top_work_ids)]

# Limit external_work_bgs to those destinations only
external_work_bgs = external_work_bgs[external_work_bgs["GEOID"].isin(top_work_ids)]

# --- Label number of Oxford-based workers in each destination block group ---

# 1. Aggregate total outbound workers by destination (work_bg)
dest_counts = (
    od_outbound.groupby("work_bg")["workers"]
    .sum()
    .reset_index()
    .rename(columns={"work_bg": "GEOID", "workers": "workers_from_oxford"})
)

# 2. Merge with destination geometries
external_work_bgs_labeled = external_work_bgs.merge(dest_counts, on="GEOID", how="left")

# 3. Add labels (CircleMarkers or Text)
for _, row in external_work_bgs_labeled.iterrows():
    if pd.notnull(row.geometry):
        centroid = row.geometry.centroid
        folium.map.Marker(
            [centroid.y, centroid.x],
            icon=folium.DivIcon(
                html=f"""
                    <div style='font-size:10pt; color:blue; text-align:center;
                                font-weight:bold; text-shadow:1px 1px 2px white;'>
                        {int(row['workers_from_oxford']):,}
                    </div>
                """
            )
        ).add_to(m_out)

# 4. GeoJson tooltip to show worker totals
folium.GeoJson(
    external_work_bgs_labeled.to_json(),
    name="Destination Block Groups (Work)",
    style_function=lambda x: {
        "color": "blue",
        "weight": 1.5,
        "fillColor": "blue",
        "fillOpacity": 0.15
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["GEOID", "workers_from_oxford"],
        aliases=["Destination Block Group:", "Oxford-Based Workers Working Here:"],
        localize=True
    )
).add_to(m_out)


for i, (_, row) in enumerate(od_outbound_gdf[od_outbound_gdf["workers"] > 1].iterrows()):
    coords = list(row.line_geom.coords)
    start = [coords[0][1], coords[0][0]]
    end = [coords[1][1], coords[1][0]]

    # alternate curvature direction to reduce overlap
    curvature = 0.05 if i % 2 == 0 else -0.05
    arc_coords = curved_line_coords(start[0], start[1], end[0], end[1], curvature=curvature)

    folium.PolyLine(
        locations=arc_coords,
        color="blue",
        weight=1 + (row["workers"] / 50),
        opacity=0.1,
        tooltip=f"{row['workers']} workers<br>Home: {row['home_bg']}<br>Work: {row['work_bg']}"
    ).add_to(m_out)



# --- Add Oxford block group boundaries ---
folium.GeoJson(
    oxford_bgs.to_json(),
    name="Oxford Block Groups",
    style_function=lambda x: {
        "color": "black",
        "weight": 1.5,
        "fillOpacity": 0,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["GEOID"],
        aliases=["Oxford BG:"]
    )
).add_to(m_out)

# --- Add Centroids for reference ---
for _, row in centroids[centroids["GEOID"].isin(oxford_ids)].iterrows():
    folium.CircleMarker(
        location=[row["centroid"].y, row["centroid"].x],
        radius=3,
        color="black",
        fill=True,
        fill_opacity=1,
        popup=f"BG: {row['GEOID']}"
    ).add_to(m_out)

folium.GeoJson(
    oxford_bgs.to_json(),
    name="Oxford Block Group Tooltips",
    style_function=lambda x: {
        "color": "black",
        "weight": 1.2,
        "fillOpacity": 0,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["GEOID", "inbound_workers", "outbound_workers", "internal_workers"],
        aliases=["Block Group:", "Inbound Workers:", "Outbound Workers:", "Internal Workers:"],
        localize=True,
        sticky=False
    )
).add_to(m_out)


folium.LayerControl().add_to(m_out)
m_out.save("html/employment/oxford_outbound_commuters.html")

#------------------Inbound Worker Map-------------------------
print("Mapping inbound worker commutes...")

m_in = folium.Map(location=[36.31, -78.59], zoom_start=12, tiles="OpenStreetMap")

# --- Identify home block groups (origins) sending >=10 workers into Oxford ---
origin_bgs = (
    od_inbound_gdf.groupby("home_bg", as_index=False)["workers"]
    .sum()
    .query("workers > 1")["home_bg"]
    .tolist()
)

# Filter to only those origin polygons
origin_work_bgs = tracts[tracts["GEOID"].isin(origin_bgs)]

# --- Label number of workers coming into Oxford from each home block group ---

# 1. Aggregate total inbound workers by origin (home_bg)
origin_counts = (
    od_inbound.groupby("home_bg")["workers"]
    .sum()
    .reset_index()
    .rename(columns={"home_bg": "GEOID", "workers": "workers_to_oxford"})
)

# 2. Merge with origin geometries
origin_work_bgs_labeled = origin_work_bgs.merge(origin_counts, on="GEOID", how="left")

# 3. Add numeric labels at centroids
for _, row in origin_work_bgs_labeled.iterrows():
    if pd.notnull(row.geometry):
        centroid = row.geometry.centroid
        folium.map.Marker(
            [centroid.y, centroid.x],
            icon=folium.DivIcon(
                html=f"""
                    <div style='font-size:10pt; color:red; text-align:center;
                                font-weight:bold; text-shadow:1px 1px 2px white;'>
                        {int(row['workers_to_oxford']):,}
                    </div>
                """
            )
        ).add_to(m_in)

# 4. (Optional) Add tooltip layer for home block groups
folium.GeoJson(
    origin_work_bgs_labeled.to_json(),
    name="Origin Block Groups (Home)",
    style_function=lambda x: {
        "color": "red",
        "weight": 1.5,
        "fillColor": "red",
        "fillOpacity": 0.15
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["GEOID", "workers_to_oxford"],
        aliases=["Home Block Group:", "Workers Commuting to Oxford:"],
        localize=True
    )

).add_to(m_in)


for _, row in od_inbound_gdf[od_inbound_gdf["workers"] > 1].iterrows():
    coords = list(row.line_geom.coords)
    folium.PolyLine(
        locations=[[coords[0][1], coords[0][0]], [coords[1][1], coords[1][0]]],
        color="red",
        weight=1 + (row["workers"] / 50),
        opacity=0.6,
        tooltip=f"{row['workers']} workers<br>Home: {row['home_bg']}<br>Work: {row['work_bg']}"
    ).add_to(m_in)

# --- Highlight only origin block groups for inbound workers ---
folium.GeoJson(
    origin_work_bgs.to_json(),
    name="Origin Block Groups (Home)",
    style_function=lambda x: {
        "color": "black",
        "weight": 1.5,
        "fillColor": "red",
        "fillOpacity": 0.15
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["GEOID"],
        aliases=["Home BG:"]
    )
).add_to(m_in)


# --- Add Centroids for reference ---
for _, row in centroids[centroids["GEOID"].isin(oxford_ids)].iterrows():
    folium.CircleMarker(
        location=[row["centroid"].y, row["centroid"].x],
        radius=3,
        color="black",
        fill=True,
        fill_opacity=1,
        popup=f"BG: {row['GEOID']}"
    ).add_to(m_in)

folium.GeoJson(
    oxford_bgs.to_json(),
    name="Oxford Block Group Tooltips",
    style_function=lambda x: {
        "color": "black",
        "weight": 1.2,
        "fillOpacity": 0,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["GEOID", "inbound_workers", "outbound_workers", "internal_workers"],
        aliases=["Block Group:", "Inbound Workers:", "Outbound Workers:", "Internal Workers:"],
        localize=True,
        sticky=False
    )
).add_to(m_in)


folium.LayerControl().add_to(m_in)
m_in.save("html/employment/oxford_inbound_commuters.html")

#----Map internal commuters

print("Mapping Oxford internal worker commutes...")

m_internal = folium.Map(location=[36.31, -78.59], zoom_start=12)
for _, row in od_internal_gdf[od_internal_gdf["workers"] > 10].iterrows():
    coords = list(row.line_geom.coords)
    folium.PolyLine(
        locations=[[coords[0][1], coords[0][0]], [coords[1][1], coords[1][0]]],
        color="green",
        weight=1 + (row["workers"] / 50),
        opacity=0.6,
        tooltip=f"{row['workers']} internal workers<br>{row['home_bg']}"
    ).add_to(m_internal)


# --- Add Oxford block group boundaries ---
folium.GeoJson(
    oxford_bgs.to_json(),
    name="Oxford Block Groups",
    style_function=lambda x: {
        "color": "black",
        "weight": 1.5,
        "fillOpacity": 0,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["GEOID"],
        aliases=["Oxford BG:"]
    )
).add_to(m_internal)

# --- Add Centroids for reference ---
for _, row in centroids[centroids["GEOID"].isin(oxford_ids)].iterrows():
    folium.CircleMarker(
        location=[row["centroid"].y, row["centroid"].x],
        radius=3,
        color="black",
        fill=True,
        fill_opacity=1,
        popup=f"BG: {row['GEOID']}"
    ).add_to(m_internal)

folium.GeoJson(
    oxford_bgs.to_json(),
    name="Oxford Block Group Tooltips",
    style_function=lambda x: {
        "color": "black",
        "weight": 1.2,
        "fillOpacity": 0,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["GEOID", "inbound_workers", "outbound_workers", "internal_workers"],
        aliases=["Block Group:", "Inbound Workers:", "Outbound Workers:", "Internal Workers:"],
        localize=True,
        sticky=False
    )
).add_to(m_internal)


m_internal.save("html/employment/oxford_internal_commuters.html")

# Merge ACS median income
oxford_bgs = oxford_bgs.merge(
    tracts[['GEOID', 'B19013_001E']], 
    on='GEOID', 
    how='left'
)


oxford_bgs.rename(columns={'B19013_001E_y': 'median_income'}, inplace=True)

def predominant_commute(row):
    flows = {
        'Internal': row['internal_workers'],
        'Outbound': row['outbound_workers'],
        'Inbound': row['inbound_workers']
    }
    # return the key with max value
    return max(flows, key=flows.get)

oxford_bgs['predominant_commute'] = oxford_bgs.apply(predominant_commute, axis=1)

import folium
from folium.features import GeoJsonTooltip

# Color map for predominant commute type
commute_colors = {
    'Internal': '#2ca25f',   # green
    'Outbound': '#3182bd',   # blue
    'Inbound': '#de2d26'     # red
}

# Internal
od_internal = od_geo[
    od_geo["home_bg"].isin(oxford_ids) & od_geo["work_bg"].isin(oxford_ids)
]

# Inbound (external → Oxford)
od_inbound = od_geo[
    od_geo["work_bg"].isin(oxford_ids) & ~od_geo["home_bg"].isin(oxford_ids)
]

# Outbound (Oxford → external)
od_outbound = od_geo[
    od_geo["home_bg"].isin(oxford_ids) & ~od_geo["work_bg"].isin(oxford_ids)
]

outbound_counts = od_outbound.groupby("home_bg")["workers"].sum().reset_index().rename(columns={"home_bg": "GEOID", "workers": "outbound_workers"})
inbound_counts = od_inbound.groupby("work_bg")["workers"].sum().reset_index().rename(columns={"work_bg": "GEOID", "workers": "inbound_workers"})
internal_counts = od_internal.groupby("home_bg")["workers"].sum().reset_index().rename(columns={"home_bg": "GEOID", "workers": "internal_workers"})


m = folium.Map(location=[36.31, -78.59], zoom_start=12, tiles="CartoDB positron")

# Add block groups
folium.GeoJson(
    oxford_bgs.to_json(),
    style_function=lambda feature: {
        "fillColor": commute_colors[feature['properties']['predominant_commute']],
        "color": "black",
        "weight": 1,
        "fillOpacity": 0.6
    },
    tooltip=GeoJsonTooltip(
        fields=["GEOID", "median_income", "predominant_commute",
                "internal_workers", "outbound_workers", "inbound_workers"],
        aliases=["Block Group:", "Median Income:", "Predominant Commute:",
                 "Internal Workers:", "Outbound Workers:", "Inbound Workers:"],
        localize=True
    ),
    name="Oxford BGs"
).add_to(m)

folium.LayerControl().add_to(m)
m.save("html/employment/oxford_income_commute.html")

# for vintage in years:
#     print(f"Pulling data for {vintage}")
#     data = ced.download(
#         dataset=ACS5,
#         vintage=vintage,
#         download_variables=bg_vars,
#         state=states.NC,
#         # county=['077'],
#         # tract='*',

#         place="49460",  # Oxford, NC place code (FIPS 3749460)
#         with_geometry=True,
#     )

#     # GEOID + Timestamp
#     # data["GEOID"] = data["STATE"] + data["COUNTY"] + data["TRACT"]
#     data["year"] = vintage
#     data["timestamp"] = pd.to_datetime(f"{vintage}-01-01")

#     data["pct_work_from_home"] = data["B08301_010E"] / data["B08301_001E"]
#     all_years.append(data)

# long_data_geo = pd.concat(all_years, ignore_index=True)
# # long_data_geo = long_data_geo.sort_values(["GEOID", "year"])

# long_data_geo.to_csv("data/wfh.csv", index=False)

total_outbound = oxford_bgs["outbound_workers"].sum()
total_inbound = oxford_bgs["inbound_workers"].sum()
total_internal = oxford_bgs["internal_workers"].sum()
total_residents = oxford_bgs["workers_residing"].sum()
total_jobs = oxford_bgs["workers_working"].sum()

print(f"Total workers living in Oxford: {total_residents:,}")
print(f"  - Internal (live & work in Oxford): {total_internal:,}")
print(f"  - Outbound (commute elsewhere): {total_outbound:,}")
print(f"Total workers working in Oxford: {total_jobs:,}")
print(f"  - Inbound (commute in from elsewhere): {total_inbound:,}")

