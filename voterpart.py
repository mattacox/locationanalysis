import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# -------------------------------
# 1. Load voters and convert to points
# -------------------------------
voters = pd.read_csv("data/1142025votersgeocoded.csv", dtype=str)
voters['latitude'] = voters['ddlat'].astype(float)
voters['longitude'] = voters['ddlong'].astype(float)
voters['geometry'] = voters.apply(lambda r: Point(r['longitude'], r['latitude']), axis=1)

voters_gdf = gpd.GeoDataFrame(voters, geometry='geometry', crs="EPSG:4326")

# -------------------------------
# 2. Load parcels
# -------------------------------
parcels = gpd.read_file("data/Granville_Tax_Parcels.shp").to_crs("EPSG:4326")
parcels = parcels[['MAPN', 'geometry']]
parcels_basic = parcels.copy()

# -------------------------------
# 3. Spatial join: voters → parcels
# -------------------------------
voters_with_parcels = gpd.sjoin(
    voters_gdf,
    parcels,
    how="left",
    predicate="within"
).drop(columns=["index_right"])

# At this stage:
# voters_with_parcels: one row per voter, includes MAPN

print("Voters inside parcels:", voters_with_parcels['MAPN'].notna().sum())
print("Voters outside parcels:", voters_with_parcels['MAPN'].isna().sum())

# -------------------------------
# 4. Aggregate to parcel counts
# -------------------------------
parcel_counts = (
    voters_with_parcels
    .dropna(subset=["MAPN"])   # remove voters not in parcels
    .groupby("MAPN")
    .size()
    .reset_index(name="voter_count")
)

# -------------------------------
# 5. Join counts back onto parcel geometry
# -------------------------------
parcels_with_counts = parcels.merge(parcel_counts, on="MAPN", how="left")
parcels_with_counts["voter_count"] = parcels_with_counts["voter_count"].fillna(0).astype(int)  

print(f"Parcels with Counts Columns: {parcels_with_counts.columns}")

import folium
import branca.colormap as cm

# -------------------------------
# 6. Build Folium Map
# -------------------------------

# Center on Oxford, NC
m = folium.Map(location=[36.310, -78.590], zoom_start=13, tiles="OpenSTreetMap")

# Create a color scale
max_count = parcels_with_counts["voter_count"].max()
colormap = cm.LinearColormap(
    colors=["#f2f0f7", "#cbc9e2", "#9e9ac8", "#6a51a3"],
    vmin=0,
    vmax=50,
    caption="Voters per Parcel"
)

# Add to map
colormap.add_to(m)

# ------------------------------------------
# Add parcels as GeoJson with popup + styling
# ------------------------------------------

def style_function(feature):
    count = feature["properties"]["voter_count"]
    return {
        "fillColor": colormap(count),
        "color": "white",
        "weight": 0.1,
        "fillOpacity": 0.7 if count > 0 else 0.05,
    }

def popup_html(props):
    return f"""
    <b>MAPN:</b> {props.get('MAPN', 'N/A')}<br>
    <b>Voter Count:</b> {props.get('voter_count', 0)}
    """

folium.GeoJson(
    parcels_with_counts.to_json(),
    name="Parcel Voter Counts",
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(
        fields=["MAPN", "voter_count"],
        aliases=["Parcel:", "Voters:"],
        localize=True
    ),
    popup=folium.GeoJsonPopup(
        fields=["MAPN", "voter_count"],
        aliases=["Parcel:", "Voters:"],
    )
).add_to(m)

folium.LayerControl().add_to(m)

# -------------------------------------
# Save map
# -------------------------------------
m.save("html/parcel_voter_counts_map.html")
print("Map saved → data/parcel_voter_counts_map.html")


# -------------------------------
# 7. Load tax data
# -------------------------------
tax = pd.read_excel("data/oxfordparcels25.xlsx", sheet_name="PARCELS_25",
                        dtype={"MAPN": str},  # read MAPN as string
                        usecols=["Column1","MAPN"]

)
tax['local_own'] = tax['Column1']
# -------------------------------
# 8. Merge tax info into voters_with_parcels
# -------------------------------
voters_with_tax = voters_with_parcels.merge(
    tax,
    on="MAPN",
    how="left"  # keep all voters, even if tax data is missing
)

print(f"Voter with Tax: {voters_with_tax.columns}")


# Optional: save
# Keep the point geometry of voters
voters_with_tax_gdf = gpd.GeoDataFrame(voters_with_tax, geometry='geometry', crs="EPSG:4326")
voters_with_tax_gdf = voters_with_tax_gdf[voters_with_tax_gdf['MAPN'].notna()].copy()


# Merge parcel geometry from parcels_basic using MAPN
voters_with_tax_and_parcels = voters_with_tax_gdf.merge(
    parcels_basic[['MAPN', 'geometry']],   # only bring geometry
    on='MAPN',
    how='left',
    suffixes=('', '_parcel')
)
voters_with_tax_and_parcels['mail_city'] = (
    voters_with_tax_and_parcels['mail_city']
    .astype(str)
    .str.strip()
    .str.upper()
)
print(f"Voter with Tax and Parcels: {voters_with_tax_and_parcels.columns}")


import folium

# Center on Oxford, NC
m = folium.Map(location=[36.310, -78.590], zoom_start=13, tiles="OpenStreetMap")

# Split voters by taxpayer status
taxpayers = voters_with_tax_and_parcels[voters_with_tax_and_parcels['local_own'] == '1'
]

non_taxpayers = voters_with_tax_and_parcels[
    (voters_with_tax_and_parcels['local_own'] != '1')
]


print("Taxpayers w/ Oxford mail:", len(taxpayers))
print("Non-taxpayers w/ Oxford mail:", len(non_taxpayers))


# Add taxpayers as green circles
for _, row in taxpayers.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=3,
        color='green',
        fill=True,
        fill_opacity=0.7,
        popup=f"Voter: {row['Name'] if 'Name' in row else 'N/A'}\nMAPN: {row['MAPN']}\nPays Tax: Yes"
    ).add_to(m)

# Add non-taxpayers as red circles
for _, row in non_taxpayers.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=3,
        color='red',
        fill=True,
        fill_opacity=0.7,
        popup=f"Voter: {row['Name'] if 'Name' in row else 'N/A'}\nMAPN: {row['MAPN']}\nPays Tax: No"
    ).add_to(m)

# Optional: add layer control if you want toggles
folium.LayerControl().add_to(m)

# Save map
m.save("html/voters_taxpayer_map.html")
print("Map saved → html/voters_taxpayer_map.html")


# -------------------------------
# 1. Determine parcel taxpayer status
# -------------------------------

# Filter to parcels that actually have voters
voters_valid = voters_with_tax_and_parcels.dropna(subset=["MAPN"]).copy()

parcel_tax_status = (
    voters_valid
    .groupby("MAPN")
    .agg(
        taxpayer_any=("local_own", lambda x: (x == '1').any()),
        taxpayer_all=("local_own", lambda x: (x == '1').all()),
        voter_count=("local_own", "size")
    )
    .reset_index()
)

# Define a single binary parcel status:
#   taxpayer = True  → green
#   taxpayer = False → red
parcel_tax_status["taxpayer"] = parcel_tax_status["taxpayer_all"]

parcel_tax_status.head()

parcels_taxmap = parcels_basic.merge(parcel_tax_status, on="MAPN", how="left")

import folium


m = folium.Map(location=[36.310, -78.590], zoom_start=13, tiles="OpenStreetMap")

def tax_style(feature):
    taxpayer = feature["properties"]["taxpayer"]
    voters = feature["properties"]["voter_count"]

    if voters == 0 or voters is None:
        color = "#cccccc"     # grey: parcel has no voters
        opacity = 0.15
    else:
        color = "#2ca25f" if taxpayer else "#de2d26"
        opacity = 0.7

    return {
        "fillColor": color,
        "color": "white",
        "weight": 0.2,
        "fillOpacity": opacity
    }

folium.GeoJson(
    parcels_taxmap.to_json(),
    name="Taxpayer Parcels",
    style_function=tax_style,
    tooltip=folium.GeoJsonTooltip(
        fields=["MAPN", "voter_count", "taxpayer"],
        aliases=["Parcel:", "Voters:", "Taxpayer Parcel:"],
        localize=True
    ),
    popup=folium.GeoJsonPopup(
        fields=["MAPN", "voter_count", "taxpayer"],
        aliases=["Parcel:", "Voters:", "Local Taxpayer Parcel:"],
    )
).add_to(m)

# -------------------------------
# Add parcel outlines as a separate layer
# -------------------------------
folium.GeoJson(
    parcels_basic.to_json(),
    name="Parcel Outlines",
    style_function=lambda feature: {
        "fillColor": "transparent",
        "color": "black",
        "weight": 0.6,
        "fillOpacity": 0
    },
    highlight_function=lambda feature: {
        "color": "yellow",
        "weight": 2,
        "fillOpacity": 0
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["MAPN"],
        aliases=["Parcel:"],
        localize=True
    ),
).add_to(m)

folium.LayerControl().add_to(m)

m.save("html/parcel_taxpayer_choropleth.html")
print("Saved → html/parcel_taxpayer_choropleth.html")
