import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from censusdis.datasets import ACS5
from censusdis import states
from censusdis.data import download
import censusdis.maps as dem
import censusdis.data as ced

vintage = 2022

senior_vars = [
    "B01001_020E", "B01001_021E", "B01001_022E", "B01001_023E", "B01001_024E", "B01001_025E",
    "B01001_044E", "B01001_045E", "B01001_046E", "B01001_047E", "B01001_048E", "B01001_049E",
]

# Step 1: Load your POIs
pois_df = pd.read_csv("data/all_pois.csv")

# Ensure lat/lon columns exist
assert 'lat' in pois_df.columns and 'lng' in pois_df.columns, "Missing latitude or longitude columns"

# Step 2: Convert POIs to GeoDataFrame
geometry = [Point(xy) for xy in zip(pois_df.lng, pois_df.lat)]
pois_gdf = gpd.GeoDataFrame(pois_df, geometry=geometry, crs="EPSG:4326")

# Step 3: Download NC block group geometries using censusdis
# This will include the geometries at block group level (summary_level="150") for all counties in NC
print("Downloading block group geometries for NC...")
bg_gdf = ced.download(
    dataset=ACS5,
    vintage=vintage,
    download_variables= senior_vars,
    state=states.NC,
    county='*',
    tract= '*',
    block_group = '*',
    # place = '*',
    with_geometry=True,
)
bg_gdf["GEOID_POI"] = bg_gdf["STATE"] + bg_gdf["COUNTY"] + bg_gdf["TRACT"] + bg_gdf["BLOCK_GROUP"]
bg_gdf = bg_gdf.to_crs("EPSG:4326")
print(bg_gdf.tail())

joined = gpd.sjoin(
    pois_gdf,             # points GeoDataFrame
    bg_gdf[["GEOID_POI", "geometry"]],  # polygons with GEOID only
    how="left",           # keep all points, even if no polygon matches
    predicate="within"    # points within polygon
)

print(joined.head())

joined.to_csv("data/poisblock.csv", index=False)