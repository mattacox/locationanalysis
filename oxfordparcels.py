import pandas as pd
import geopandas as gpd
import folium
import branca.colormap as cm
import json
import numpy as np



# === Load 2023 Displacement Risk ===
drisk = pd.read_csv("data/displacement_risk_2023.csv")
drisk["GEOID"] = drisk["GEOID"].astype(str)

parcels = pd.read_excel("data/oxfordparcels.xlsx", sheet_name="PARCELS_25", dtype={"MAPN": str})
parcels = parcels[["MAPN", "Cal_Acres", "vpa", "vpa25","pct_value_increase","oxtaxpctdelta","oxtaxincrease","OWNERNAME1","currentname","msdflag",]]

# Remove decimal part by converting to float then int, then back to string
# Remove leading apostrophe if present and drop decimal portion
geoms = gpd.read_file("data/Granville_Tax_Parcels.shp").to_crs("EPSG:4326")

geoms["MAPN"] = geoms["MAPN"].astype(str).str.strip()
parcels["MAPN"] = parcels["MAPN"].astype(str).str.strip()

# Check length distributions
print("Shapefile MAPN lengths:\n", geoms["MAPN"].str.len().value_counts())
print("Parcels MAPN lengths:\n", parcels["MAPN"].str.len().value_counts())

# Fix leading zeros: pad parcels MAPN to length 12 (assuming 12 digits)
max_len = geoms["MAPN"].str.len().max()
parcels["MAPN"] = parcels["MAPN"].astype(str).str.strip().str.zfill(12)
geoms["MAPN"] = geoms["MAPN"].astype(str).str.strip().str[:12]  # Truncate any overlong ones

# After fix, print samples again
print("Fixed parcels MAPN samples:\n", parcels["MAPN"].head())

# Merge on RECN = PIN
# Merge while retaining only geometry and MAPN
geoms = geoms[["MAPN", "geometry"]].merge(
    parcels,
    on="MAPN",
    how="left"
)

# print(geoms[["MAPN", "vpa","vpa25","geometry",]].dropna().head(10))
print(f"Merged {geoms['vpa'].notna().sum()} parcels with VPA values.")

# Drop parcels without VPA
geoms_vpa = geoms.dropna(subset=["vpa"]).copy()

clipped_vpa = geoms_vpa["vpa"].clip(upper=geoms_vpa["vpa"].quantile(0.95))
vmin, vmax = clipped_vpa.min(), clipped_vpa.max()
colormap = cm.linear.YlOrRd_09.scale(vmin, vmax)
colormap.caption = "Parcel Value per Acre (Clipped at 98th Percentile)"


# Style function using quantile binning
def style_function(feature):
    v = feature["properties"]["vpa"]
    return {
        "fillColor": colormap(v) if v is not None else "#ccc",
        "color": "black",
        "weight": 0.2,
        "fillOpacity": 0.2,
    }


# Drop parcels without VPA or VPA25
geoms_vpa = geoms.dropna(subset=["vpa"]).copy()
geoms_vpa25 = geoms.dropna(subset=["vpa25"]).copy()

# Clip both metrics at 95th percentile
clipped_vpa = geoms_vpa["vpa"].clip(upper=geoms_vpa["vpa"].quantile(0.95))
vmin_vpa, vmax_vpa = clipped_vpa.min(), clipped_vpa.max()

clipped_vpa25 = geoms_vpa25["vpa25"].clip(upper=geoms_vpa25["vpa25"].quantile(0.95))
vmin_vpa25, vmax_vpa25 = clipped_vpa25.min(), clipped_vpa25.max()

# Create colormaps
colormap_vpa = cm.linear.YlOrRd_09.scale(vmin_vpa, vmax_vpa).to_step(n=10)
colormap_vpa.caption = "Value per Acre (VPA, clipped at 95th percentile)"

colormap_vpa25 = cm.linear.BuPu_09.scale(vmin_vpa25, vmax_vpa25).to_step(n=10)
colormap_vpa25.caption = "Value per Acre (VPA25, clipped at 95th percentile)"

# Create base map
center = geoms_vpa.unary_union.centroid.coords[0][::-1]
m = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")

# VPA Layer
vpa_layer = folium.FeatureGroup(name="2021 Values per Acre").add_to(m)
folium.GeoJson(
    geoms_vpa,
    style_function=lambda feature: {
        "fillColor": colormap_vpa(feature["properties"]["vpa"]),
        "color": "black",
        "weight": 0.2,
        "fillOpacity": 0.7,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["currentname","MAPN", "Cal_Acres", "vpa"],
        aliases=["Owner","Parcel ID", "Acres", "Value per Acre (2021)"],
        localize=True
    ),
).add_to(vpa_layer)
colormap_vpa.add_to(m)

# VPA25 Layer
vpa25_layer = folium.FeatureGroup(name="2025 Values per Acre").add_to(m)
folium.GeoJson(
    geoms_vpa25,
    style_function=lambda feature: {
        "fillColor": colormap_vpa25(feature["properties"]["vpa25"]),
        "color": "black",
        "weight": 0.2,
        "fillOpacity": 0.7,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["currentname","MAPN", "Cal_Acres", "vpa25"],
        aliases=["Owner","Parcel ID", "Acres", "Value per Acre (2025)"],
        localize=True
    ),
).add_to(vpa25_layer)
colormap_vpa25.add_to(m)

# Filter parcels within the Municipal Service District (msdflag == 1)
geoms_msd = geoms[geoms["msdflag"] == 1].copy()

# MSD Layer
msd_layer = folium.FeatureGroup(name="Municipal Service District", show=False).add_to(m)
folium.GeoJson(
    geoms_msd,
    style_function=lambda feature: {
        "fillColor": "#ffcc00",  # Yellow
        "color": "black",
        "weight": 0.5,
        "fillOpacity": 0.5,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["currentname","MAPN", "Cal_Acres"],
        aliases=["Owner","Parcel ID", "Acres"],
        localize=True
    ),
).add_to(msd_layer)


# Add toggle control
folium.LayerControl(collapsed=False).add_to(m)
# Save map
m.save("html/tax/vpa_choropleth.html")

center = geoms_vpa.unary_union.centroid.coords[0][::-1]
m = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")

# PCT VALUE INCREASE LAYER
geoms_pct = geoms.dropna(subset=["pct_value_increase"]).copy()
# Format percent increase as a string (e.g., "12.3%")
geoms_pct["pct_value_increase_fmt"] = (geoms_pct["pct_value_increase"] * 100).map("{:.1f}%".format)

clipped_pct = geoms_pct["pct_value_increase"].clip(lower=geoms_pct["pct_value_increase"].quantile(0.02),
                                                    upper=geoms_pct["pct_value_increase"].quantile(0.95))

vmin_pct, vmax_pct = clipped_pct.min(), clipped_pct.max()

colormap_pct = cm.linear.GnBu_09.scale(vmin_pct, vmax_pct).to_step(n=10)
colormap_pct.caption = "Percent Value Increase (Clipped 2nd–95th Percentile)"

pct_layer = folium.FeatureGroup(name="Percent Value Increase").add_to(m)

folium.GeoJson(
    geoms_pct,
    style_function=lambda feature: {
        "fillColor": colormap_pct(feature["properties"]["pct_value_increase"]),
        "color": "black",
        "weight": 0.2,
        "fillOpacity": 0.7,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["currentname","MAPN", "Cal_Acres", "pct_value_increase_fmt"],
        aliases=["Owner","Parcel ID", "Acres", "% Increase"],
        localize=True,
        labels=True
    ),
).add_to(pct_layer)

colormap_pct.add_to(m)
# Add toggle control
folium.LayerControl(collapsed=True).add_to(m)

# Save map
m.save("html/tax/tax_pct_increase_choropleth.html")


center = geoms_vpa.unary_union.centroid.coords[0][::-1]
m = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")

# Drop missing values
geoms_tax = geoms.dropna(subset=["oxtaxpctdelta"]).copy()

# Format as percent string
geoms_tax["oxtaxpctdelta_fmt"] = (geoms_tax["oxtaxpctdelta"] * 100).map("{:.1f}%".format)

# Clip top 2% to reduce skew
clipped_tax = geoms_tax["oxtaxpctdelta"].clip(upper=geoms_tax["oxtaxpctdelta"].quantile(0.95))
vmin_tax, vmax_tax = clipped_tax.min(), clipped_tax.max()

# Define colormap
colormap_tax = cm.linear.PuBuGn_09.scale(vmin_tax, vmax_tax)
colormap_tax.caption = "Oxford Tax % Change (Clipped at 95th percentile)"

folium.GeoJson(
    geoms_tax,
    style_function=lambda feature: {
        "fillColor": colormap_tax(feature["properties"]["oxtaxpctdelta"]),
        "color": "black",
        "weight": 0.2,
        "fillOpacity": 0.7,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["currentname","MAPN", "Cal_Acres", "oxtaxincrease","oxtaxpctdelta_fmt"],
        aliases=["Owner","Parcel ID", "Acres",  "Tax $ Δ", "Tax % Δ"],
        localize=True
    ),
    name="Tax % Change"
).add_to(m)

colormap.add_to(m)        # VPA colormap
colormap_tax.add_to(m)    # Tax % colormap
# Add toggle control
folium.LayerControl(collapsed=False).add_to(m)

# Save map
m.save("html/tax/tax_pct_delta_choropleth.html")

center = geoms_vpa.unary_union.centroid.coords[0][::-1]
m = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")

geoms_tax = geoms.dropna(subset=["oxtaxpctdelta"]).copy()
geoms_tax["oxtaxpctdelta_fmt"] = (geoms_tax["oxtaxpctdelta"] * 100).map("{:+.1f}%".format)  # shows +/-

def red_blue_style(feature):
    value = feature["properties"]["oxtaxpctdelta"]
    if value is None:
        return {"fillOpacity": 0.1, "color": "gray", "weight": 0.1}
    elif value > 0:
        return {"fillColor": "red", "color": "black", "weight": 0.3, "fillOpacity": 0.7}
    else:
        return {"fillColor": "blue", "color": "black", "weight": 0.3, "fillOpacity": 0.7}


folium.GeoJson(
    geoms_tax,
    style_function=red_blue_style,
    tooltip=folium.GeoJsonTooltip(
        fields=["currentname","MAPN", "Cal_Acres", "oxtaxincrease","oxtaxpctdelta_fmt"],
        aliases=["Owner","Parcel ID", "Acres",  "Tax $ Δ", "Tax % Δ"],
        localize=True
    ),
    name="Tax Change: Red = ↑, Blue = ↓"
).add_to(m)

# Add toggle control
folium.LayerControl(collapsed=False).add_to(m)

# Save map
m.save("html/tax/tax_waterfall_choropleth.html")

# === Load parcels (already includes oxtaxpctdelta) ===
parcels = gpd.read_file("data/Granville_Tax_Parcels.shp").to_crs("EPSG:4326")
parcel_tax = pd.read_excel("data/oxfordparcels.xlsx", sheet_name="PARCELS_25", dtype={"MAPN": str})
parcel_tax["MAPN"] = parcel_tax["MAPN"].str.zfill(12)

parcels["MAPN"] = parcels["MAPN"].astype(str).str.strip().str[:12]
parcels = parcels.merge(parcel_tax, on="MAPN", how="left")
parcels = parcels.dropna(subset=["oxtaxpctdelta"])

# === Load Census Block Groups ===
block_groups = gpd.read_file("data/tl_2024_37_bg.shp").to_crs("EPSG:4326")
block_groups = block_groups[block_groups["COUNTYFP"] == "077"]  # Granville County

# === Spatial Join: parcels → block groups ===
joined = gpd.sjoin(parcels, block_groups[["GEOID", "geometry"]], how="inner", predicate="intersects")

# === Aggregate by block group ===

# Median percent increase
median_pct = joined.groupby("GEOID")["oxtaxpctdelta"].median().reset_index()
median_pct.columns = ["GEOID", "median_tax_pct_increase"]

# Sum of total tax dollar increase
total_increase = joined.groupby("GEOID")["oxtaxincrease"].sum().reset_index()
total_increase.columns = ["GEOID", "total_tax_increase_dollars"]

# Merge both into a single DataFrame
tax_by_bg = pd.merge(median_pct, total_increase, on="GEOID", how="outer")

# Optional: format dollar string for tooltips
tax_by_bg["total_tax_increase_fmt"] = tax_by_bg["total_tax_increase_dollars"].map("${:,.0f}".format)


# === Merge with block group geometry ===
bg_with_tax = block_groups.merge(tax_by_bg, on="GEOID", how="left")
bg_with_tax["median_tax_pct_increase_fmt"] = (bg_with_tax["median_tax_pct_increase"] * 100).map("{:.1f}%".format)
# Merge 2023 displacement risk into block group tax data
bg_with_tax = bg_with_tax.merge(drisk, on="GEOID", how="left")

# Optional: Format for display
bg_with_tax["displacement_risk_fmt"] = bg_with_tax["displacement_risk"].map("{:.2f}".format)
# === Clip and Color ===
clipped = bg_with_tax["median_tax_pct_increase"].clip(upper=bg_with_tax["median_tax_pct_increase"].quantile(0.95))

vmin, vmax = clipped.min(), clipped.max()
colormap = cm.linear.YlOrRd_09.scale(vmin, vmax).to_step(10)
colormap.caption = "Median Parcel Tax % Increase by Block Group"

# === Create Map ===
m = folium.Map(location=[36.31, -78.59], zoom_start=13, tiles="cartodbpositron")

colormap = cm.linear.YlOrRd_09.scale(vmin, .50).to_step(10)  # 1.00 = 100%
colormap.caption = "Median Tax Percent Increase by Census Block Group"

folium.GeoJson(
    bg_with_tax,
    style_function=lambda feature: {
        "fillColor": colormap(feature["properties"]["median_tax_pct_increase"]) if feature["properties"]["median_tax_pct_increase"] is not None else "#ccc",
        "color": "black",
        "weight": 0.5,
        "fillOpacity": 0.25,
    },
    tooltip=folium.GeoJsonTooltip(
    fields=["GEOID", "median_tax_pct_increase_fmt", "total_tax_increase_fmt"],
    aliases=["Block Group", "Median % Tax Increase", "Total Tax Increase ($)"],
    localize=True,
),

    name="Median % Change by Block Group"
).add_to(m)

colormap.add_to(m)

# Drop missing values
geoms_tax = geoms.dropna(subset=["oxtaxpctdelta"]).copy()

# Cap the tax percent increase at 100% (1.0)
geoms_tax["oxtaxpctdelta_capped"] = geoms_tax["oxtaxpctdelta"].clip(upper=.6)

# Format as percent string (still show real value)
geoms_tax["oxtaxpctdelta_fmt"] = (geoms_tax["oxtaxpctdelta"] * 100).map("{:.1f}%".format)
print(geoms_tax.columns)
# Define fixed color scale: 0% to 100%
colormap_tax = cm.linear.PuBuGn_09.scale(0, .6)
colormap_tax.caption = "Oxford Tax % Change (Capped at 60%)"

# Map layer
folium.GeoJson(
    geoms_tax,
    style_function=lambda feature: {
        "fillColor": colormap_tax(feature["properties"]["oxtaxpctdelta_capped"]),
        "color": "black",
        "weight": 0.2,
        "fillOpacity": 0.7,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["MAPN", "Cal_Acres", "oxtaxincrease", "oxtaxpctdelta_fmt"],
        aliases=["Parcel ID", "Acres", "Tax $ Δ", "Tax % Δ"],
        localize=True
    ),
    name="Tax % Change"
).add_to(m)

colormap_tax.add_to(m)



# Add toggle control
folium.LayerControl(collapsed=False).add_to(m)


m.save("html/tax/tax_pct_increase_blockgroup.html")
