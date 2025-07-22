from censusdis.data import download
from censusdis.datasets import ACS5
from censusdis import states
import censusdis.data as ced
import censusdis.maps as dem
from urllib3.exceptions import InsecureRequestWarning
import urllib3
import requests
import pandas as pd
import geopandas as gpd
import numpy as np

road_path="data/block_group_road_access_scores.csv"

ruca_path="data/ruca2010revised.xlsx"

def ruca_to_location_type(ruca_code):
        if ruca_code in [1, 2, 3]: return "urban"
        elif ruca_code in [4, 5, 6]: return "micropolitan"
        elif ruca_code in [7, 8, 9]: return "rural"
        elif ruca_code == 10: return "isolated_rural"
        return "unknown"



# Focus counties by FIPS
focus_counties = [
    # "001",  # Alamance
    # "025",  # Cabarrus
    # "033",  # Caswell
    # "037",  # Chatham
    # "057",  # Davidson
    "063",  # Durham
    # "067",  # Forsyth
    "069",  # Franklin
    "077",  # Granville
    # "081",  # Guilford
    # "119",  # Mecklenburg
    # "135",  # Orange
    # "145",  # Person
    # "151",  # Randolph
    # "157",  # Rockingham
    # "159",  # Rowan
    "181",  # Vance
    "183"   # Wake
]
# --- Disable SSL warnings globally ---
urllib3.disable_warnings(category=InsecureRequestWarning)
orig_get = requests.get
def unsafe_get(*args, **kwargs):
    kwargs['verify'] = False
    return orig_get(*args, **kwargs)
requests.get = unsafe_get

block_group_vars = [
    "B19013_001E",  # Median income
    "B25003_001E", "B25003_002E",  # Housing units and owner-occupied
    "B15003_001E",  # Educational attainment base
    "B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E",  # Bachelor's+
    "B01003_001E",  # Population
]

def compute_engineered_features(df):
    df = df.copy()

    if "median_income" in df.columns:
        df["median_income"] = df["median_income"].fillna(1)
        df["log_median_income"] = np.log(df["median_income"].clip(lower=1))
        df["income_squared"] = df["median_income"] ** 2

    if "pop_density" in df.columns:
        df["pop_density"] = df["pop_density"].fillna(1)
        df["log_density"] = np.log(df["pop_density"].clip(lower=1))

    if "median_income" in df.columns and "log_density" in df.columns:
        df["median_income:log_density"] = df["median_income"] * df["log_density"]

    if "pct_commercial" in df.columns:
        df["pct_commercial"] = df["pct_commercial"].fillna(0)
        df["pct_commercial_sq"] = df["pct_commercial"] ** 2

    if "entropy" in df.columns:
        df["entropy"] = df["entropy"].fillna(0)
        if "log_density" in df.columns:
            df["entropy_log_density"] = df["entropy"] * df["log_density"]

    for col in ["loc_micropolitan", "loc_rural", "loc_isolated_rural"]:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0)

    return df

for county in focus_counties:
    df = ced.download(
        dataset=ACS5,
        vintage=2023,
        download_variables=block_group_vars,
        state=states.NC,
        county=county,
        block_group='*',
        with_geometry=True,
    )

df["pct_owner_occupied"] = df["B25003_002E"] / df["B25003_001E"]
df["pct_with_degree"] = (
    df["B15003_022E"] + df["B15003_023E"] + df["B15003_024E"] + df["B15003_025E"]
) / df["B15003_001E"]
df["median_income"] = df["B19013_001E"]
df["population"] = df["B01003_001E"]
df["area_km2"] = df.geometry.to_crs("EPSG:3857").area / 1e6
df["pop_density"] = df["population"] / df["area_km2"]


df["GEOID"] = df["STATE"] + df["COUNTY"] + df["TRACT"] + df["BLOCK_GROUP"]


ruca_df = pd.read_excel(ruca_path, sheet_name="Data", dtype={"GEOID": str})
ruca_df.rename(columns={
    "State-County-Tract FIPS Code (lookup by address at http://www.ffiec.gov/Geocode/)": "GEOID"
}, inplace=True)
ruca_df["primary_location_type"] = ruca_df["Primary RUCA Code 2010"].apply(ruca_to_location_type)

#get road info
road_df = pd.read_csv(road_path)
road_df["GEOID"] = road_df["GEOID"].astype(str)

#get job info

tract_jobs = pd.read_csv('data/nc_wac_S000_JT00_2022.csv.gz', dtype={'w_geocode': str})
tract_jobs['tract'] = tract_jobs['w_geocode'].str[:12]
tract_jobs = tract_jobs.groupby('tract')[['C000', 'CE01', 'CE02', 'CE03']].sum().reset_index()
tract_jobs.columns = ['GEOID', 'total_jobs', 'jobs_0_1250', 'jobs_1251_3333', 'jobs_3333_up']
tract_jobs["share_low_wage"] = tract_jobs["jobs_0_1250"] / tract_jobs["total_jobs"]
tract_jobs["share_mid_wage"] = tract_jobs["jobs_1251_3333"] / tract_jobs["total_jobs"]
tract_jobs["share_high_wage"] = tract_jobs["jobs_3333_up"] / tract_jobs["total_jobs"]
tract_jobs = tract_jobs.replace([np.inf, -np.inf], np.nan).fillna(0)
etj_row = {"GEOID": "oxford_etj",
               "share_low_wage": 0.217794851813231,
               "share_mid_wage": 0.273534316115669,
               "share_high_wage": 0.273534316115669,
               "total_jobs": 1586,
               }
    
tract_jobs = pd.concat([tract_jobs, pd.DataFrame([etj_row])], ignore_index=True)

stores_df = pd.read_csv("data/poisblock.csv")
stores_gdf = gpd.GeoDataFrame(
        stores_df,
        geometry=gpd.points_from_xy(stores_df.lng, stores_df.lat),
        crs="EPSG:4326"
    )

landusegdf = gpd.read_file("data/tracts_with_landuse_metrics.gpkg")
print(landusegdf.sample())

landuse_gdf = gpd.read_file("data/tracts_with_landuse_metrics.gpkg")[["GEOID", "pct_commercial",  "entropy",]]
landuse_gdf["GEOID"] = landuse_gdf["GEOID"].astype(str)



