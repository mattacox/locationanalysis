import pandas as pd
import requests
import geopandas as gpd
from shapely.geometry import Point
import folium
import json
from folium.plugins import TimeSliderChoropleth

# Load your voter data
people = pd.read_csv("data/votersfull.csv", dtype=str)
people = people[people["municipality_desc"] == "OXFORD"]

# Drop rows missing key address parts
people = people.dropna(subset=["res_street_address", "res_city_desc", "state_cd", "zip_code"])

# Create a full address for geocoding
people["full_address"] = (
    people["res_street_address"].str.strip() + ", " +
    people["res_city_desc"].str.strip() + ", " +
    people["state_cd"].str.strip() + " " +
    people["zip_code"].str.strip()
)

# Deduplicate addresses to reduce geocoding API load
unique_addresses = people[["full_address", "res_street_address", "res_city_desc", "state_cd", "zip_code"]].drop_duplicates().reset_index(drop=True)
unique_addresses["uid"] = unique_addresses.index

# Prepare batch input for Census Geocoder
lines = []
for idx, row in unique_addresses.iterrows():
    line = f"{idx},{row['res_street_address']},{row['res_city_desc']},{row['state_cd']},{row['zip_code']}"
    lines.append(line)

input_file = "temp_addresses.csv"
with open(input_file, "w") as f:
    f.write("\n".join(lines))

# Send to Census Geocoder API
files = {"addressFile": (input_file, open(input_file, "rb"))}
data = {"benchmark": "Public_AR_Current", "vintage": "Current_Current"}

print("Sending batch to Census Geocoder...")
response = requests.post("https://geocoding.geo.census.gov/geocoder/locations/addressbatch", files=files, data=data)
if response.status_code != 200:
    raise Exception("Geocoding request failed:", response.text)

with open("data/census_geocode_response.csv", "w") as f:
    f.write(response.text)

import csv
results = []

with open("data/census_geocode_response.csv", "r") as f:
    reader = csv.reader(f)
    for fields in reader:
        if len(fields) >= 7 and fields[2] == "Match":
            index = int(fields[0])
            full_address = fields[1]
            coords = fields[5].split(",")
            if len(coords) == 2:
                lon = float(coords[0])
                lat = float(coords[1])
            else:
                lat = lon = None
            block_geoid = fields[6]
        else:
            index = int(fields[0])
            full_address = fields[1]
            lat = lon = block_geoid = None

        results.append({
            "index": index,
            "full_address": full_address,
            "lat": lat,
            "lon": lon,
            "block_geoid": block_geoid
        })

geocoded_df = pd.DataFrame(results)

# Merge geocoded results back to unique addresses by uid
geocoded_trimmed = geocoded_df[["index", "lat", "lon", "block_geoid"]]
unique_addresses = unique_addresses.merge(geocoded_trimmed, left_on="uid", right_on="index", how="left")

# Now merge geocoded info back to people on full_address
people_merged = people.merge(
    unique_addresses[["full_address", "lat", "lon", "block_geoid"]],
    on="full_address",
    how="left"
)

# Extract election date from election_desc (format MM/DD/YYYY)
people_merged["election_date"] = pd.to_datetime(
    people_merged["election_desc"].str.extract(r"(\d{2}/\d{2}/\d{4})")[0],
    format="%m/%d/%Y",
    errors="coerce"
)

print("Geocoded data summary:")
print(people_merged[["lat", "lon", "election_date"]].describe())

# Load parcels shapefile and reproject to EPSG:4326 for Folium
bg = gpd.read_file("data/Granville_Tax_Parcels.shp").to_crs("EPSG:4326")

# Drop rows without valid coords
people_points = people_merged.dropna(subset=["lat", "lon"]).copy()

# Create geometry points
people_points["geometry"] = people_points.apply(lambda r: Point(float(r["lon"]), float(r["lat"])), axis=1)

# Create GeoDataFrame of people
people_gdf = gpd.GeoDataFrame(people_points, geometry="geometry", crs="EPSG:4326")

# Reproject to projected CRS for spatial join and buffering (~1 meter buffer)
projected_crs = "EPSG:26917"
bg_projected = bg.to_crs(projected_crs)
people_gdf = people_gdf.to_crs(projected_crs)
people_gdf["geometry"] = people_gdf.geometry.buffer(20)  # 20 meters for example

# Spatial join to assign parcel (MAPN) to each voter point
people_with_bg = gpd.sjoin(people_gdf, bg_projected, how="left", predicate="intersects")

# Reproject back to EPSG:4326 for mapping
people_with_bg = people_with_bg.to_crs("EPSG:4326")
bg = bg_projected.to_crs("EPSG:4326")

print("Points with no parcel match:", people_with_bg["MAPN"].isna().sum())

# Group by parcel and election date to count voters
voter_counts = people_with_bg.groupby(["MAPN", "election_date"]).size().reset_index(name="voter_count")

print("Voter counts summary:")
print(voter_counts.describe())

# Filter out any rows with NaN MAPN or election_date just in case
voter_counts = voter_counts.dropna(subset=["MAPN", "election_date"])

# Convert MAPN to string for matching GeoJSON
voter_counts["MAPN"] = voter_counts["MAPN"].astype(str)

# Build style dictionary for folium TimeSliderChoropleth
styledict = {}
for _, row in voter_counts.iterrows():
    mapn = row["MAPN"]
    date_str = row["election_date"].strftime("%Y-%m-%d")
    fill_opacity = min(row["voter_count"] / 10, 1.0)
    if mapn not in styledict:
        styledict[mapn] = {}
    styledict[mapn][date_str] = {
        "color": "black",
        "opacity": 0.3,
        "weight": 0.3,
        "fillColor": "#08519c",
        "fillOpacity": fill_opacity
    }

# Convert parcels GeoDataFrame to GeoJSON
bg["MAPN"] = bg["MAPN"].astype(str)
geojson_data = json.loads(bg.to_json())

# Filter GeoJSON features for only parcels in the styledict keys
geojson_data["features"] = [f for f in geojson_data["features"] if f["properties"]["MAPN"] in styledict]

# Add the 'times' attribute for each feature, required for TimeSliderChoropleth
for feature in geojson_data["features"]:
    mapn = feature["properties"]["MAPN"]
    feature["times"] = sorted(styledict[mapn].keys())

print("Unique election dates (sample):", sorted(voter_counts["election_date"].dropna().dt.strftime("%Y-%m-%d").unique())[:10])
print("Number of parcels in styledict:", len(styledict))
print("Number of parcels in GeoJSON:", len(geojson_data["features"]))

# Check sample keys and values in styledict:
for k in list(styledict.keys())[:5]:
    print(f"Parcel {k} times: {list(styledict[k].keys())}")

# Check first few MAPNs in GeoJSON:
for f in geojson_data["features"][:5]:
    print(f["properties"]["MAPN"], f.get("times", None))



# Create base map
m = folium.Map(location=[36.31, -78.59], zoom_start=13, tiles="cartodbpositron")

# Add TimeSliderChoropleth to map
TimeSliderChoropleth(
    data=geojson_data,
    styledict=styledict
).add_to(m)

# Save the map
m.save("html/voter_timeslider.html")
print("Map saved to html/voter_timeslider.html")
