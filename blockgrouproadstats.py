import osmnx as ox
import pandas as pd
import geopandas as gpd
import requests
from urllib3.exceptions import InsecureRequestWarning
import urllib3

print("Gathering Road Stats...")
# Load census tracts
from censusdis.data import download
from censusdis.datasets import ACS5
from censusdis import states
import censusdis.data as ced
import censusdis.maps as dem
import geopandas as gpd


all_road_stats= []
# --- Disable SSL warnings globally ---
urllib3.disable_warnings(category=InsecureRequestWarning)
orig_get = requests.get
def unsafe_get(*args, **kwargs):
    kwargs['verify'] = False
    return orig_get(*args, **kwargs)
requests.get = unsafe_get

# Define your target counties
county_fips = ["025", 
               "033", 
               "037", 
               "057", 
               "063", 
               "067", 
               "069", 
               "077",
               "081", 
               "119", 
               "135", 
               "145", 
               "151", 
               "157", 
               "159", 
               "181", 
               "183",
               ]
state_fips = "37"

for fips in county_fips:
    print(f"📍 Processing county {fips}...")


    # Download geometries only for block groups
    block_groups = ced.download(
            dataset=ACS5,
            vintage=2023,
            download_variables=["B17021_001E",],
            state=states.NC,
            county=fips,
            block_group='*',
            with_geometry=True,
    )

    block_groups = gpd.GeoDataFrame(block_groups, geometry="geometry", crs="EPSG:4326")
    block_groups["GEOID"] = block_groups["STATE"] + block_groups["COUNTY"] + block_groups["TRACT"] + block_groups["BLOCK_GROUP"]

    # Download OSM road network and calculate road access
    minx, miny, maxx, maxy = block_groups.total_bounds

    # Step 3: Download road network for that bounding box (corrected for osmnx 2.0+)
    bbox = (minx, miny, maxx, maxy)
    G = ox.graph_from_bbox(bbox, network_type='drive')
    edges = ox.graph_to_gdfs(G, nodes=False)
    edges = edges.to_crs("EPSG:4326")  # Make sure to align CRS if needed

    def estimate_traffic(row):
        highway = row.get('highway', '')
        if isinstance(highway, list):
            highway = highway[0]
        return {'motorway': 5, 'trunk': 4, 'primary': 3, 'secondary': 2, 'tertiary': 1}.get(highway, 0.5)

    edges['traffic_weight'] = edges.apply(estimate_traffic, axis=1)
    edges = edges.to_crs("EPSG:3857")
    edges['length_m'] = edges.geometry.length
    edges['weighted_length'] = edges['length_m'] * edges['traffic_weight']
    edges_with_tracts = gpd.sjoin(edges, block_groups.to_crs("EPSG:3857"), how="inner", predicate="intersects")

    road_stats = edges_with_tracts.groupby("GEOID")["weighted_length"].sum().reset_index()
    road_stats.rename(columns={"weighted_length": "road_access_score"}, inplace=True)
    all_road_stats.append(road_stats)

# Load and filter Oxford ETJ geometry
cities = gpd.read_file("data/NCDOT_City_Boundaries.shp").to_crs("EPSG:4326")
cities["geometry"] = cities["geometry"].buffer(0)
oxford_etj_geom = cities[cities["MunicipalB"].str.upper() == "OXFORD"].copy()

# Add synthetic block group ID
oxford_etj_geom["STATE"] = "37"
oxford_etj_geom["COUNTY"] = "077"
oxford_etj_geom["TRACT"] = "030500"
oxford_etj_geom["BLOCK_GROUP"] = "9"
oxford_etj_geom["GEOID"] = "370770305009"
oxford_etj_geom["NAME"] = "Oxford ETJ"

# Calculate road access for Oxford ETJ polygon
minx, miny, maxx, maxy = oxford_etj_geom.total_bounds
bbox = (minx, miny, maxx, maxy)
G = ox.graph_from_bbox(bbox, network_type='drive')
edges = ox.graph_to_gdfs(G, nodes=False).to_crs("EPSG:4326")

# Estimate traffic
def estimate_traffic(row):
    highway = row.get('highway', '')
    if isinstance(highway, list):
        highway = highway[0]
    return {'motorway': 5, 'trunk': 4, 'primary': 3, 'secondary': 2, 'tertiary': 1}.get(highway, 0.5)

edges['traffic_weight'] = edges.apply(estimate_traffic, axis=1)
edges = edges.to_crs("EPSG:3857")
edges['length_m'] = edges.geometry.length
edges['weighted_length'] = edges['length_m'] * edges['traffic_weight']

# Spatial join with Oxford ETJ
edges_with_etj = gpd.sjoin(edges, oxford_etj_geom.to_crs("EPSG:3857"), how="inner", predicate="intersects")
etj_score = edges_with_etj["weighted_length"].sum()

# Create DataFrame for Oxford ETJ score
oxford_etj_row = pd.DataFrame([{
    "GEOID": "oxford_etj",
    "road_access_score": etj_score
}])

# Final concatenation
final_df = pd.concat(all_road_stats + [oxford_etj_row], ignore_index=True)

# Save
final_df.to_csv("data/block_group_road_access_scores.csv", index=False)
print("✅ Data complete — saved to 'data/block_group_road_access_scores.csv'")

