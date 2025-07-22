# ===== Imports ======
import pandas as pd
import geopandas as gpd
import folium


# ===== Import Data =====
oxford_etj_df = pd.read_excel("data/oxfordparcels.xlsx", sheet_name="All Oxford Parcels", dtype={"MAPN": str})
cities = gpd.read_file("data/granville_parcels.shp").to_crs("EPSG:4326")

# Ensure MAPN is string
oxford_etj_df["MAPN"] = oxford_etj_df["MAPN"].astype(str)
cities["MAPN"] = cities["MAPN"].astype(str)

# ===== Merge =====
merged_df = pd.merge(
    oxford_etj_df, cities,  on="MAPN", how="left"
)
print(merged_df.columns)
# Only fill NaNs in non-geometry columns, keep geometry and ownership tag
non_geometry_cols = merged_df.columns.difference(['geometry'])
merged_df[non_geometry_cols] = merged_df[non_geometry_cols].fillna(0)
columns_to_keep = ["MAPN", "locally_owned", "Billing City", "geometry"]
merged_df = merged_df[columns_to_keep]

import geopandas as gpd

# Make sure it's a GeoDataFrame
gdf = gpd.GeoDataFrame(merged_df, geometry="geometry", crs="EPSG:4326")

# Create a base map centered on Oxford, NC
m = folium.Map(location=[36.31, -78.59], zoom_start=13, tiles="cartodbpositron")

# Define style function for binary color coding
def style_function(feature):
    owned_outside = feature["properties"]["locally_owned"]
    color = "#e41a1c" if owned_outside == 0 else "#4daf4a"
    return {
        "fillColor": color,
        "color": "black",
        "weight": 0.5,
        "fillOpacity": 0.6,
    }

# Add parcels to the map with styling
folium.GeoJson(
    gdf,
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(fields=["MAPN", "locally_owned","Billing City"]),
    name="Oxford Parcels",
).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Display the map (in Jupyter) or save to file
m.save("html/oxford_etj_parcels.html")
