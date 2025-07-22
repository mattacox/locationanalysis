import osmnx as ox
import pandas as pd
import geopandas as gpd
print("Gathering Road Stats...")
# Load census tracts
tracts = gpd.read_file("data/tl_2024_37_tract.shp").to_crs("EPSG:4326")
tracts = tracts[tracts["COUNTYFP"].isin([# "001",  # Alamance
        "025",  # Cabarrus
        "033",  # Caswell
        "037",  # Chatham
        "057",  # Davidson
        "063",  # Durham
        "067",  # Forsyth
        "069",  # Franklin
        "077",  # Granville
        "081",  # Guilford
        "119",  # Mecklenburg
        "135",  # Orange
        "145",  # Person
        "151",  # Randolph
        "157",  # Rockingham
        "159",  # Rowan
        "181",  # Vance
        "183"   # Wake
        ])]

 # Load Oxford ETJ geometry from city boundaries
cities = gpd.read_file("data/NCDOT_City_Boundaries.shp").to_crs("EPSG:4326")
cities["geometry"] = cities["geometry"].buffer(0)


oxford_etj_geom = cities[cities["MunicipalB"].str.upper() == "OXFORD"].copy()

# Add synthetic tract row
oxford_etj_geom["GEOID"] = "oxford_etj"  # unique identifier
oxford_etj_geom["COUNTYFP"] = "077"     # example county code (Granville)
oxford_etj_geom["STATEFP"] = "37"     # example county code (Granville)
oxford_etj_geom["TRACTCE"] = "002001"     # example county code (Granville)
oxford_etj_geom["NAME"] = "NAME"     # example county code (Granville)
oxford_etj_geom["NAMELSAD"] = "NAME"     # example county code (Granville)
oxford_etj_geom["MTFCC"] = "NAME"     # example county code (Granville)
oxford_etj_geom["GEOIDFQ"] = "1400000USoxford_etj"
oxford_etj_geom["NAME"] = "Oxford ETJ"
oxford_etj_geom["FUNCSTAT"] = "S"
oxford_etj_geom["AWATER"] = 0
oxford_etj_geom["INTPTLAT"] = "+36.3100"  # rough center of Oxford
oxford_etj_geom["INTPTLON"] = "-078.5900"

# Ensure consistent columns
tracts = pd.concat([tracts, oxford_etj_geom], ignore_index=True)
print(tracts.tail())


# Download OSM road network and calculate road access
minx, miny, maxx, maxy = tracts.total_bounds

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
edges_with_tracts = gpd.sjoin(edges, tracts.to_crs("EPSG:3857"), how="inner", predicate="intersects")

road_stats = edges_with_tracts.groupby("GEOID")["weighted_length"].sum().reset_index()
road_stats.rename(columns={"weighted_length": "road_access_score"}, inplace=True)
print(road_stats[road_stats["GEOID"] == "oxford_etj"])

road_stats.to_csv("data/road_access_scores.csv", index=False)
print("Data complete- saveed to 'data/road_access_scores2.csv'")
