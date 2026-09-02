import pandas as pd
import folium
from geopy.geocoders import Nominatim
from folium import PolyLine
import time

# Load CSV
df = pd.read_csv("data/flows.csv")

# Initialize map centered roughly on NC
m = folium.Map(location=[36.0, -79.0], zoom_start=8, tiles="CartoDB positron")

# Initialize geocoder
geolocator = Nominatim(user_agent="flowmap")

# Helper function to geocode city names
def get_coords(city):
    try:
        location = geolocator.geocode(city)
        if location:
            return (location.latitude, location.longitude)

    except:
        pass
    return (None, None)

# Cache coordinates to avoid re-querying
coords_cache = {}

for idx, row in df.iterrows():
    origin = row["origin_city"]
    dest = row["destination_city"]
    value = row.get("value", 1)

    # Get coords (with caching)
    if origin not in coords_cache:
        coords_cache[origin] = get_coords(origin)
        print(f"Getting origin: {origin}")
        time.sleep(1)
    if dest not in coords_cache:
        coords_cache[dest] = get_coords(dest)
        print(f"Getting origin: {dest}")

        time.sleep(1)

    origin_coords = coords_cache[origin]
    dest_coords = coords_cache[dest]

    if None not in origin_coords + dest_coords:
        # Draw line
        folium.PolyLine(
            [origin_coords, dest_coords],
            color="blue",
            weight=value / 10,
            opacity=0.6,
        ).add_to(m)

        # Add markers
        folium.CircleMarker(origin_coords, radius=4, color="green", fill=True).add_to(m)
        folium.CircleMarker(dest_coords, radius=4, color="red", fill=True).add_to(m)

# Save the map
m.save("html/flow_map.html")
print("✅ Map saved as flow_map.html")
