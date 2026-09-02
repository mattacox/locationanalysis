import geopandas as gpd
import pandas as pd


countyparcels = gpd.read_file("data/Granville_Tax_Parcels.shp").to_crs("EPSG:4326")
commercialland = pd.read_csv("data/CommercialandIndustrial.csv", encoding='latin1')


countyparcels['MAPN'] = countyparcels['MAPN'].astype(str)
commercialland['MAPN'] = commercialland['MAPN'].astype(str)
key_geom = 'MAPN'       # <-- replace with your actual parcel ID field
key_comm = 'MAPN' 

commercial_gdf = countyparcels.merge(commercialland, left_on=key_geom, right_on=key_comm, how='inner')
commercial_gdf = gpd.GeoDataFrame(commercial_gdf, geometry='geometry', crs="EPSG:4326")
# Split into commercial and industrial subsets
commercial_only = commercial_gdf[commercial_gdf['TR1Categor'] == 'C']
industrial_only = commercial_gdf[commercial_gdf['TR1Categor'] == 'I']

print(f"✅ {len(commercial_gdf)} commercial/industrial parcels matched")


zoning_lookup = pd.read_csv("data/zoninglookup.csv", encoding='latin1')
# Load each shapefile
oxford = gpd.read_file("data/OxfordZoning2024.shp")
oxford['municipality'] = "Oxford"
oxford['NewZone'] = oxford['Zoning']
butner = gpd.read_file("data/Butner_Zoning_2025.shp")
butner['municipality'] = "Butner"
creedmoor1 = gpd.read_file("data/CreedmoorOverlayZoningDistricts2013_05.shp")
creedmoor1['municipality'] = "Creedmoor"
creedmoor1['NewZone'] = creedmoor1['Zoning_Typ']
creedmoor2 = gpd.read_file("data/CreedmoorZoning2017_10.shp")
creedmoor2['municipality'] = "Creedmoor"
creedmoor2['NewZone'] = creedmoor2['Zoning_Typ']
# stem = gpd.read_file("data/stem_parcels.shp")
county = gpd.read_file("data/GC_ZONING.shp")
county['NewZone'] = county['ZONECODE']
county['municipality'] = "Unincorporated"



# Ensure consistent CRS
target_crs = county.crs
datasets = [
    oxford, 
            butner, 
            creedmoor1,
                        creedmoor2, 

            # stem, 
            county]

for df in datasets:
    if df.crs != target_crs:
        df.to_crs(target_crs, inplace=True)


        # Merge all datasets
all_parcels = pd.concat(datasets, ignore_index=True)
all_parcels = gpd.GeoDataFrame(all_parcels, geometry='geometry', crs=target_crs)


# Calculate parcel area in square meters (projected CRS needed)
all_parcels_projected = all_parcels.to_crs(epsg=3857)  # Web Mercator (meters)
all_parcels_projected['area_m2'] = all_parcels_projected.geometry.area

# Summarize total area by zoning type
zone_area = (
    all_parcels_projected.groupby('NewZone', dropna=True)['area_m2']
    .sum()
    .reset_index()
)
zone_area['p_i'] = zone_area['area_m2'] / zone_area['area_m2'].sum()

# Calculate entropy
import numpy as np
zone_area['p_ln_p'] = zone_area['p_i'] * np.log(zone_area['p_i'])
H = -zone_area['p_ln_p'].sum()
n = len(zone_area)
H_norm = H / np.log(n)

print(f"Raw Entropy (H): {H:.3f}")
print(f"Normalized Entropy (H'): {H_norm:.3f}")


# List of values to exclude
exclude_zones = [
    "CREEDMOOR PLANNING AND ZONING AREA",
    "BUTNER PLANNING AND ZONING AREA",
    "OXFORD PLANNING AND ZONING AREA",
]

# Keep only rows NOT in the list
all_parcels = all_parcels[~all_parcels['ZONECODE'].isin(exclude_zones)]
# Optional: remove duplicates (sometimes municipalities overlap county data)
# all_parcels = all_parcels.dissolve(by='OBJECTID', as_index=False)

# Merge descriptive zoning info
all_parcels = all_parcels.merge(
    zoning_lookup,
    on='NewZone',
    how='left'  # left join keeps all parcels, NaN if no match
)


# print(all_parcels.head())
# all_parcels.to_csv("data/allzoning.csv", index=False)
# all_parcels.to_file("data/combined_zoning.shp")

import folium
from folium.plugins import Fullscreen



# Center the map roughly on Granville County
m = folium.Map(location=[36.3, -78.6], zoom_start=11, tiles='OpenStreetMap')

# Optional: fullscreen control
Fullscreen(position='topright').add_to(m)

# Simplify geometries for performance (optional but recommended for large shapefiles)
all_parcels_simplified = all_parcels.copy()
all_parcels_simplified['geometry'] = all_parcels_simplified['geometry'].simplify(10)

# Extract zoning polygons that are classified as Commercial or Industrial
zoning_commercial = all_parcels[all_parcels['Zoning Type'] == 'Commercial']
zoning_industrial = all_parcels[all_parcels['Zoning Type'] == 'Industrial']


# all_parcels_simplified = all_parcels_simplified[
#     ~all_parcels_simplified['Zoning Type'].isin(['Commercial', 'Industrial'])
# ]

# Ensure both are in WGS84
zoning_commercial = zoning_commercial.to_crs("EPSG:4326")
zoning_industrial = zoning_industrial.to_crs("EPSG:4326")
commercial_only = commercial_only.to_crs("EPSG:4326")
industrial_only = industrial_only.to_crs("EPSG:4326")

# Now safe to concatenate
commercial_combined = pd.concat([commercial_only, zoning_commercial], ignore_index=True)
commercial_combined = gpd.GeoDataFrame(commercial_combined, geometry='geometry', crs="EPSG:4326")

industrial_combined = pd.concat([industrial_only, zoning_industrial], ignore_index=True)
industrial_combined = gpd.GeoDataFrame(industrial_combined, geometry='geometry', crs="EPSG:4326")



import matplotlib as mpl
import matplotlib.colors as mcolors

# Create distinct categorical colors for zoning types
unique_zones = all_parcels_simplified['Zoning Type'].dropna().unique()
n_colors = len(unique_zones)

# ✅ Use the new Matplotlib 3.10 colormap API
cmap = mpl.colormaps['tab20']

# Generate colors
color_list = [mcolors.rgb2hex(cmap(i / n_colors)) for i in range(n_colors)]

# Map zoning codes to colors
zone_colors = dict(zip(unique_zones, color_list))


def style_function(feature):
    zone = feature['properties'].get('Zoning Type')
    color = zone_colors.get(zone, '#cccccc')
    return {
        'fillColor': color,
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.5
    }



def make_html_tooltip(row):
    return f"""
    <b>Municipality:</b> {row['municipality']}<br>
    <b>Zoning:</b> {row['NewZone']}<br>
    <b>Type:</b> {row['Zoning Type']}<br>
    """

all_parcels_simplified['tooltip_html'] = all_parcels_simplified.apply(make_html_tooltip, axis=1)

folium.GeoJson(
    all_parcels_simplified,
    name='Zoning Type',
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(
        fields=['tooltip_html'],
        aliases=[''],
        sticky=True,
        localize=True
    )
).add_to(m)


folium.GeoJson(
    commercial_only,
    name="Commercial Parcels",
    style_function=lambda feature: {
        'fillColor': '#1f77b4',   # blue
        'color': '#1f77b4',
        'weight': 1,
        'fillOpacity': 0.6
    },
    tooltip=folium.GeoJsonTooltip(
        fields=[key_geom, 'OwnerName1_x'],
        aliases=["Parcel ID:", "Owner:"],
        sticky=True,
        labels=True,
        localize=True,
        html="""
            <b>Parcel ID:</b> {""" + key_geom + """}<br>
            <b>Owner:</b> {OwnerName1_x}<br>
                        <b>Type:</b> Commercial<br>

        """
    )
).add_to(m)


# Industrial layer
folium.GeoJson(
    industrial_only,
    name="Industrial Parcels",
    style_function=lambda feature: {
        'fillColor': '#d62728',   # red
        'color': '#d62728',
        'weight': 1,
        'fillOpacity': 0.6
    },
    tooltip=folium.GeoJsonTooltip(
        fields=[key_geom, 'OwnerName1_x'],
        aliases=["Parcel ID:", "Owner:"],
        sticky=True,
        localize=True,

        html="""
            <b>Parcel ID:</b> {""" + key_geom + """}<br>
            <b>Owner:</b> {OwnerName1_x}<br>
                        <b>Type:</b> Commercial<br>

        """,
    )
).add_to(m)


# Update layer control and save
folium.LayerControl().add_to(m)

# Save to HTML
m.save('html/zoning_map.html')
print("✅ Zoning map saved to html/zoning_map.html")
