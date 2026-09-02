import pandas as pd
from geopy.geocoders import Nominatim


# Load data
info = pd.read_csv("data/ncvoter39/ncvoter39.csv")
history = pd.read_csv("data/ncvhis39/ncvhis39.csv")
# Make sure voter IDs are strings
info['voter_id'] = info['voter_reg_num'].astype(str)
info['municipality_desc'] = info['municipality_desc'].str.upper().str.strip()
# Filter to Oxford only
info = info[info['municipality_desc'] == "OXFORD"].copy()
# print(info.head)
# history = history[history['election_lbl'] == "11/04/2025"].copy()

history['voter_id'] = history['voter_reg_num'].astype(str)


# # Join info + voting history
df = info.merge(history, on="voter_id", how="left")


# Keep only the columns you actually need
keep_cols = [
    'election_lbl',
    'voter_id',
    'voter_reg_num_x',
    'first_name',
    'middle_name',
    'last_name',
    'res_street_address',
    'res_city_desc',
    'state_cd',
    'zip_code',
    'municipality_desc',
    'party_cd',
    'race_code',
    'ethnic_code',
    'age_at_year_end',
    'ethnic_code',
    'race_code',
]

df = df[keep_cols]

df['res_street_address'] = (
    df['res_street_address']
    .str.strip()
    .str.replace(r'\s+', ' ', regex=True)
)

df['zip_code'] = (
    df['zip_code']
    .astype(str)
    .str.replace('.0', '', regex=False)
    .str.zfill(5)               # ensure 5 digits
)
df = df[df['election_lbl'] == '11/4/2025']


geolocator = Nominatim(user_agent="voter-geocoder")

import time

def geocode_address(row):
    try:
        # Skip missing or bad addresses
        if pd.isna(row['res_street_address']) or pd.isna(row['res_city_desc']):
            print("Skipping missing address")
            return pd.Series([None, None])

        addr = f"{row['res_street_address']}, {row['res_city_desc']}, {row['state_cd']} {row['zip_code']}"
        print("Trying:", addr)

        # NOTE: add timeout
        loc = geolocator.geocode(addr, timeout=5)

        if loc:
            print("Geocoded:", addr)
            time.sleep(1)  # very important for Nominatim
            return pd.Series([loc.latitude, loc.longitude])
        else:
            print("No match:", addr)
            return pd.Series([None, None])

    except Exception as e:
        print("Error:", e)
        return pd.Series([None, None])



df[['lat','lon']] = df.apply(geocode_address, axis=1)

df.to_csv("data/1142025votersgeocoded.csv", index=False)

import folium

df = pd.read_csv("data/1142025votersgeocoded.csv")


# Create a map centered on Oxford, NC
m = folium.Map(location=[36.310, -78.580], zoom_start=13)  # adjust center & zoom

# Add markers for each voter
for _, row in df.iterrows():
    if pd.notna(row['lat']) and pd.notna(row['lon']):
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=3,
            color='blue',
            fill=True,
            fill_opacity=0.7,
            popup=f"{row['first_name']} {row['last_name']}"
        ).add_to(m)

# Save the map to an HTML file
m.save("html/oxford_voters_map.html")
