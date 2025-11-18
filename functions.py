# --- Imports ---
import pandas as pd
import geopandas as gpd
import requests
import folium
import branca.colormap as cm
from folium import Element
from folium.plugins import TimestampedGeoJson
from sklearn.preprocessing import MinMaxScaler
from censusdis.data import download
from censusdis.datasets import ACS5
from censusdis import states
import censusdis.data as ced
import censusdis.maps as dem
from urllib3.exceptions import InsecureRequestWarning
import urllib3
import numpy as np

def simulate_iz_effect(
    df, 
    new_units=20000, 
    iz_rate=0.15, 
    timeline=10, 
    share_from_renters=0.30,
    pass_through=0.5,
    renter_col="B25003_003E",  # renter households
    rent_burden_col="percent_cost_burdened",
    allocation="proportional"
):
    """
    Adjusts rent burden based on an Inclusionary Zoning (IZ) scenario.
    """
    df = df.copy()
    
    # Total IZ units
    iz_units = new_units * iz_rate
    
    # Renters leaving rental pool
    renters_exiting_total = iz_units * share_from_renters
    
    # Allocate renters exiting
    if allocation == "proportional":
        weights = df[renter_col] / df[renter_col].sum()
    else:  # equal allocation
        weights = 1 / len(df)
    
    df["renters_exiting"] = renters_exiting_total * weights
    df["pct_reduction_renters"] = df["renters_exiting"] / df[renter_col]
    
    # Estimate rent change from reduced demand
    df["rent_change_pct"] = df["pct_reduction_renters"] * pass_through
    
    # Adjust rent burden
    df["iz_rent_burden_change"] = df[rent_burden_col] * df["rent_change_pct"]
    df["percent_cost_burdened_iz"] = df[rent_burden_col] - df["iz_rent_burden_change"]
    
    return df


def recompute_displacement_with_adjusted_burden(df_all, adjusted_bg_gdf):
    """
    Given the entire county-level df_all (GeoDataFrame) and a version that has
    percent_cost_burdened adjusted for one block group (in adjusted_bg_gdf),
    recompute the displacement_risk for all BGs using the same method you use
    in your script (normalize, weighted sum, add binary flags).
    Returns a copy of adjusted_bg_gdf with new displacement_risk column.
    """
    tmp = df_all.copy().reset_index(drop=True)
    # replace row(s) for the BG(s) being adjusted
    tmp = tmp.set_index("GEOID")
    adj = adjusted_bg_gdf.set_index("GEOID")
    for col in adj.columns:
        tmp.loc[adj.index, col] = adj[col]
    tmp = tmp.reset_index()

    # Recompute normalized inputs using your risk_fields (but with adjusted percent_cost_burdened)
    fields_for_norm = [
        "rent_share", "percent_cost_burdened", "poverty_rate", "snap_share",
        "unemployment_rate", "senior_share", "inv_vacancy"
    ]
    # If percent_cost_burdened in tmp may be NaN, fill with 0
    scaler_local = MinMaxScaler()
    normalized_local = pd.DataFrame(
        scaler_local.fit_transform(tmp[fields_for_norm].fillna(0)),
        columns=fields_for_norm,
        index=tmp.index,
    )

    tmp["base_displacement_index"] = (
        sum(normalized_local[f] * weights[f] for f in fields_for_norm) / total_weight
    )

    # add back your binary flags (these are already in tmp if you computed earlier)
    tmp["displacement_risk"] = (
        tmp["base_displacement_index"]
        + tmp.get("black_decline", 0) * 0.5
        + tmp.get("latino_decline", 0) * 0.5
        + tmp.get("rapid_rent_increase", 0) * 0.5
    )

    tmp["displacement_risk"] = MinMaxScaler().fit_transform(tmp[["displacement_risk"]])

    # return updated geometry rows (GeoDataFrame)
    return gpd.GeoDataFrame(tmp, geometry="geometry", crs=tmp.crs)

