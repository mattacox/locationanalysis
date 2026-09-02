import pandas as pd
import requests
import geopandas as gpd
from shapely.geometry import Point

# Load your voter data
people = pd.read_csv("data/votersfull.csv", dtype=str)

# Drop rows missing key address parts
people = people.dropna(subset=["res_street_address", "res_city_desc", "state_cd", "zip_code"])

# Create a full address for geocoding
people["full_address"] = people["res_street_address"].str.strip() + ", " + \
                         people["res_city_desc"].str.strip() + ", " + \
                         people["state_cd"].str.strip() + " " + \
                         people["zip_code"].str.strip()

# Prepare batch input (index is used for later merge)
lines = []
for idx, row in people.iterrows():
    line = f"{idx},{row['res_street_address']},{row['res_city_desc']},{row['state_cd']},{row['zip_code']}"
    lines.append(line)

# Write to a temporary batch input file
input_file = "temp_addresses.csv"
with open(input_file, "w") as f:
    f.write("\n".join(lines))

# Send to Census Geocoder API
files = {
    "addressFile": (input_file, open(input_file, "rb")),
}
data = {
    "benchmark": "Public_AR_Current",
    "vintage": "Current_Current",
}

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
people["index"] = people.index
people_merged = people.merge(geocoded_df, on="index", how="left")




# import geopandas as gpd

# # Example: Load 2020 block groups with ACS data
# bg = gpd.read_file("data/Granville_Tax_Parcels.shp").to_crs("EPSG:4326")
# print(bg.columns.to_list())
# from shapely.geometry import Point

# # Drop rows without valid coordinates
# people_points = people_merged.dropna(subset=["lat", "lon"]).copy()

# # Create Point geometries
# people_points["geometry"] = people_points.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)

# # Convert to GeoDataFrame
# people_gdf = gpd.GeoDataFrame(people_points, geometry="geometry", crs="EPSG:4326")

# # Spatial join: add bg attributes to each person based on location
# people_with_bg = gpd.sjoin(people_gdf, bg, how="left", predicate="within")
# # people_with_bg.to_csv("data/peoplewithbg.csv")

# race_counts = (
#     people_with_bg
#     .groupby(["MAPN", "race_code"])
#     .size()
#     .unstack(fill_value=0)
#     .reset_index()
# )

# race_cols = race_counts.columns.drop("MAPN")
# race_counts["total_voters"] = race_counts[race_cols].sum(axis=1)

# # Compute percentage columns
# for col in race_cols:
#     pct_col = f"pct_{col.lower().replace(' ', '_')}"
#     race_counts[pct_col] = race_counts[col] / race_counts["total_voters"]

# def assign_group(row):
#     if row["total_voters"] < 5:
#         return "too_few_voters"
    
#     max_val = 0
#     max_race = None

#     for col in race_cols:
#         pct = row[f"pct_{col.lower().replace(' ', '_')}"]
#         if pct > max_val:
#             max_val = pct
#             max_race = col
    
#     if max_val >= 0.6:
#         return f"majority_{max_race.lower().replace(' ', '_')}"
#     else:
#         return "mixed"

# race_counts["race_category"] = race_counts.apply(assign_group, axis=1)

# bg_with_race = bg.merge(race_counts[["MAPN", "race_category"]], on="MAPN", how="left")
# # Drop parcels without a race_category assigned
# bg_with_race = bg_with_race.dropna(subset=["race_category"]).copy()

# import folium

# color_dict = {
#     "majority_white": "#fef0d9",
#     "majority_black_or_african_american": "#bd0026",
#     "majority_asian": "#1c9099",
#     "majority_american_indian_or_alaska_native": "#fdae6b",
#     "majority_hispanic_or_latino": "#fa9fb5",
#     "mixed": "#cccccc",
#     "too_few_voters": "#ffffff"
# }

# m = folium.Map(location=[36.3, -78.6], zoom_start=13)

# folium.GeoJson(
#     data=bg_with_race.__geo_interface__,
#     style_function=lambda feature: {
#         "fillColor": color_dict.get(feature["properties"]["race_category"], "#eeeeee"),
#         "color": "black",
#         "weight": 0.5,
#         "fillOpacity": 0.8,
#     },
#     tooltip=folium.GeoJsonTooltip(fields=["race_category"]),
# ).add_to(m)

# m.save("html/map_race_category.html")
