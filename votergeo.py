import pandas as pd
import geopandas as gpd
import folium

# --- Load parcels ---
parcels = pd.read_csv("data/ncvoter39/ncvoter39parcels.csv")
parcels = parcels[["MAPN", "voter_reg_num", "first_name", "middle_name", "last_name","locally_owned"]]

# --- Load geometries ---
geoms = gpd.read_file("data/Granville_Tax_Parcels.shp").to_crs("EPSG:4326")
outoftown = geoms.copy()
outoftown['not_oxford'] = (outoftown['City'] != 'OXFORD').astype(int)

geoms = geoms[["MAPN", "geometry"]]
# Clean MAPN
geoms["MAPN"] = geoms["MAPN"].astype(str).str.strip()
parcels["MAPN"] = parcels["MAPN"].astype(str).str.strip()

# Merge geometry into parcels
parcels_gdf = parcels.merge(geoms, on="MAPN", how="left")
parcels_gdf = gpd.GeoDataFrame(parcels_gdf, geometry="geometry", crs="EPSG:4326")

# --- Load voter history ---
history = pd.read_csv("data/ncvhis39/ncvhis39.csv")
history["voter_reg_num"] = history["voter_reg_num"].astype(str).str.strip()
parcels_gdf["voter_reg_num"] = parcels_gdf["voter_reg_num"].astype(str).str.strip()
parcels_gdf = parcels_gdf.merge(history, on="voter_reg_num", how="left")

# --- Keep only a specific election ---
parcels_gdf["election_lbl"] = pd.to_datetime(
    parcels_gdf["election_lbl"], errors="coerce"
).dt.strftime("%m/%d/%Y")
parcels_gdf = parcels_gdf[parcels_gdf["election_lbl"] == "11/04/2025"]
# --- Aggregate voter names per parcel ---
parcels_gdf["full_name"] = parcels_gdf["first_name"].fillna('') + ' ' + \
                            parcels_gdf["middle_name"].fillna('') + ' ' + \
                            parcels_gdf["last_name"].fillna('')
voter_names = parcels_gdf.groupby("MAPN")["full_name"].apply(lambda x: ", ".join(x)).reset_index()
voter_names.rename(columns={"full_name": "voter_names"}, inplace=True)

# --- Count voters per parcel ---
parcel_counts = parcels_gdf.groupby("MAPN").size().reset_index(name="voter_count")

# --- Merge counts, names, and party info ---
choropleth_gdf = geoms.merge(parcel_counts, on="MAPN", how="left")
choropleth_gdf = choropleth_gdf.merge(voter_names, on="MAPN", how="left")

# Keep all voters, including party info
choropleth_gdf = choropleth_gdf.merge(
    parcels_gdf[['MAPN', 'voted_party_desc']],
    on='MAPN',
    how='left'
)

# Replace missing party info
choropleth_gdf['voted_party_desc'] = choropleth_gdf['voted_party_desc'].fillna("Unknown")

# Keep only parcels with voters
choropleth_nonzero = choropleth_gdf[choropleth_gdf["voter_count"] > 0]

# --- Create Folium map ---
m = folium.Map(location=[36.31, -78.59], zoom_start=13, tiles="CartoDB positron")

# Add layers per party
for party, party_gdf in choropleth_nonzero.groupby('voted_party_desc'):
    folium.Choropleth(
        geo_data=party_gdf.to_json(),
        data=party_gdf,
        columns=["MAPN", "voter_count"],
        key_on="feature.properties.MAPN",
        fill_color="YlOrRd",
        fill_opacity=0.6,
        line_opacity=0.2,
        name=f"{party} Voters"
    ).add_to(m)

    # Add GeoJson tooltip layer with proper name to avoid MicroElement
    folium.GeoJson(
        party_gdf,
        style_function=lambda x: {"color": "black", "weight": 0.2, "fillOpacity": 0},
        tooltip=folium.GeoJsonTooltip(
            fields=["MAPN", "voter_count", "voter_names", ],
            aliases=["Parcel:", "Voters:", "Names:", ],
            localize=True
        ),
        name=f"{party} Tooltip"
    ).add_to(m)

# Add clean layer toggle
folium.LayerControl(collapsed=False).add_to(m)

# Save map
print("Saving html/voters_2025_party_toggle.html")
m.save("html/voters_2025_party_toggle.html")


# Assuming outoftown GeoDataFrame already has 'not_oxford' column and geometry
# outoftown['not_oxford'] = (outoftown['City'] != 'Oxford').astype(int)

# Base map
m = folium.Map(location=[36.31, -78.59], zoom_start=13, tiles="CartoDB positron")

# Function to style parcels by not_oxford value
def style_func(feature):
    val = feature['properties']['not_oxford']
    color = 'red' if val == 1 else 'lightblue'
    return {
        'fillColor': color,
        'color': 'black',
        'weight': 0.2,
        'fillOpacity': 0.6
    }

# Add GeoJson layer
folium.GeoJson(
    outoftown.to_json(),
    style_function=style_func,
    tooltip=folium.GeoJsonTooltip(
        fields=["MAPN", "not_oxford"],
        aliases=["Parcel:", "Out of Oxford:"],
        localize=True
    ),
    name="Out-of-Town Parcels"
).add_to(m)

# Add layer control
folium.LayerControl(collapsed=False).add_to(m)

# Save map
m.save("html/outoftown_parcels.html")


# Merge not_oxford info into parcels_gdf
parcels_gdf = parcels_gdf.merge(
    outoftown[['MAPN', 'not_oxford']],
    on='MAPN',
    how='left'
)

# Fill NaN (parcels not found in outoftown) as 0 (assume in Oxford)
parcels_gdf['not_oxford'] = parcels_gdf['not_oxford'].fillna(0).astype(int)

# Keep only voters who voted in 11/04/2025
voted_gdf = parcels_gdf[parcels_gdf['election_lbl'] == "11/04/2025"]

# Count voters by not_oxford
in_town_count = voted_gdf[voted_gdf['not_oxford'] == 0].shape[0]
out_of_town_count = voted_gdf[voted_gdf['not_oxford'] == 1].shape[0]

print(f"Voters who own in Oxford: {in_town_count}")
print(f"Voters who aren't property owners Oxford: {out_of_town_count}")


# Base map
m = folium.Map(location=[36.31, -78.59], zoom_start=13, tiles="OpenStreetMap")

# Merge voter counts into outoftown GeoDataFrame
outoftown_counts = parcels_gdf.groupby('MAPN').size().reset_index(name='voter_count')
outoftown_map = outoftown.merge(outoftown_counts, on='MAPN', how='left')
outoftown_map['voter_count'] = outoftown_map['voter_count'].fillna(0).astype(int)

# Style function: red = out of Oxford, light blue = in Oxford
def style_func(feature):
    val = feature['properties']['not_oxford']
    color = 'red' if val == 1 else 'lightblue'
    return {
        'fillColor': color,
        'color': 'black',
        'weight': 0.2,
        'fillOpacity': 0.6
    }

# Add GeoJson layer
folium.GeoJson(
    outoftown_map.to_json(),
    style_function=style_func,
    tooltip=folium.GeoJsonTooltip(
        fields=['MAPN', 'not_oxford', 'voter_count'],
        aliases=['Parcel:', 'Out of Oxford:', 'Voter Count:'],
        localize=True
    ),
    name='Out-of-Town vs In-Town Parcels'
).add_to(m)

# Add LayerControl
folium.LayerControl(collapsed=False).add_to(m)

# Save map
m.save("html/outoftown_voter_counts.html")
print("Map saved to html/outoftown_voter_counts.html")
