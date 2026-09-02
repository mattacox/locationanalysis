import geopandas as gpd
import folium
import requests, zipfile, io, tempfile, os

# 1. Download Natural Earth zip
url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
r = requests.get(url)

# 2. Extract to temp directory
tmpdir = tempfile.mkdtemp()
with zipfile.ZipFile(io.BytesIO(r.content)) as z:
    z.extractall(tmpdir)

# 3. Find shapefile
shp = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")][0]

# 4. Load into GeoPandas
world = gpd.read_file(shp)

# 5. Select USA
usa = world[world["NAME"] == "United States of America"]

# Remove Alaska/Hawaii via rough bounding box
conus = usa.cx[-130:-60, 20:50]

# 6. Project to CONUS Albers (meters)
conus_alb = conus.to_crs("EPSG:5070")

# 7. Buffer 100 miles
miles100 = 160934.4
inner = conus_alb.buffer(-miles100)
border_zone = conus_alb.difference(inner)

# 8. Back to WGS84
border_zone_wgs = border_zone.to_crs("EPSG:4326")

# 9. Make folium map
m = folium.Map(location=[39, -98], zoom_start=4)
folium.GeoJson(
    border_zone_wgs,
    name="100-mile zone",
    style_function=lambda x: {
        "fillColor": "#ff0000",
        "color": "#660000",
        "fillOpacity": 0.4,
        "weight": 1,
    }
).add_to(m)

# 10. Save
m.save("html/us_100mile_zone.html")
print("Saved us_100mile_zone.html")
