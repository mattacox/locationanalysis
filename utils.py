import numpy as np
import pandas as pd
import json
import pandas as pd
import warnings

import geopandas as gpd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from libpysal.weights import Queen, lag_spatial
from esda import Moran
import folium
import statsmodels.api as blookie
from folium.features import GeoJsonTooltip
import branca.colormap as cm
from branca.element import Template, MacroElement
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.optimize import minimize
from decimal import Decimal, getcontext
import math
import statsmodels.formula.api as smf
import statsmodels.discrete.discrete_model as dm

def get_line_weight(jobs, min_weight=1, max_weight=6, scale='linear', max_jobs=1):
    if scale == 'linear':
        return min_weight + (max_weight - min_weight) * min(jobs, max_jobs) / max_jobs
    elif scale == 'log':
        return min_weight + (max_weight - min_weight) * np.log1p(jobs) / np.log1p(max_jobs)
    else:
        return min_weight  # fallback


def get_opacity(value, vmin, vmax, min_opacity=0.1, max_opacity=0.9):
    if vmax == vmin:
        return max_opacity  # avoid divide-by-zero
    norm = (value - vmin) / (vmax - vmin)
    return min_opacity + norm * (max_opacity - min_opacity)


def get_quadratic_bezier(p0, p1, p2, num_points=500):
    """Generate points for a quadratic Bezier curve.
    
    p0, p1, p2 are tuples like (lat, lon)
    num_points controls curve smoothness.
    """
    t_values = np.linspace(0, 1, num_points)
    curve_points = []
    for t in t_values:
        lat = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
        lon = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
        curve_points.append((lat, lon))
    return curve_points

def create_curved_line(origin, dest, offset=0.15):
    """
    origin and dest are (lat, lon) tuples
    offset controls the "height" of the curve.
    """
    # Midpoint
    mid_lat = (origin[0] + dest[0]) / 2
    mid_lon = (origin[1] + dest[1]) / 2

    # Compute vector perpendicular to line
    vec_lat = dest[0] - origin[0]
    vec_lon = dest[1] - origin[1]

    # Perpendicular offset (swap and negate one coordinate)
    control_point = (mid_lat + offset * vec_lon, mid_lon - offset * vec_lat)

    # Generate curve points
    return get_quadratic_bezier(origin, control_point, dest)


def load_cns_to_occ_prefixes(json_path):
    """Load CNS to OCC prefix mappings from JSON."""
    with open(json_path, "r") as f:
        return json.load(f)

def matches_prefix(occ_code, prefixes):
    """Return True if occ_code starts with any of the prefixes (prefix format 'xx-0000')."""
    occ_prefix = str(occ_code).split('-')[0]  # e.g. '29' from '29-1141'
    prefix_prefixes = [p.split('-')[0] for p in prefixes]  # e.g. ['29', '31']
    return occ_prefix in prefix_prefixes

def estimate_wage_bounds_for_sector(sector_name, wac_df, bls_df, cns_to_occ_prefixes, area_title):
    """
    Estimate CE01, CE02, CE03 income boundaries for a CNS sector using prefix-based OCC filtering.

    Parameters:
        sector_name (str): e.g. "Health Care"
        wac_df (pd.DataFrame): WAC data with sector columns and CE01/CE02/CE03
        bls_df (pd.DataFrame): BLS OES data with wage percentiles and TOT_EMP
        cns_to_occ_prefixes (dict): CNS sector name → list of OCC_CODE prefixes (e.g., "29-0000")
        area_title (str): geographic filter for BLS data (e.g., "Eastern North Carolina nonmetropolitan area")

    Returns:
        dict or None: wage boundary estimates or None if no data found
    """
    # 1. Get CE counts for the sector
    sector_jobs = wac_df[wac_df[sector_name] > 0]
    ce01_total = sector_jobs['CE01'].sum()
    ce02_total = sector_jobs['CE02'].sum()
    ce03_total = sector_jobs['CE03'].sum()
    total_jobs = ce01_total + ce02_total + ce03_total

    if total_jobs == 0:
        print(f"No jobs found in WAC data for sector {sector_name}")
        return None

    pct_ce01 = ce01_total / total_jobs
    pct_ce02 = ce02_total / total_jobs
    pct_ce03 = ce03_total / total_jobs

    # 2. Get prefixes from mapping
    prefixes = cns_to_occ_prefixes.get(sector_name, [])
    if not prefixes:
        print(f"No OCC prefixes found for sector {sector_name}")
        return None

    # 3. Filter BLS data by area and OCC prefix (and valid percentiles)
    bls_filtered = bls_df[
        (bls_df['AREA_TITLE'] == area_title) &
        (bls_df['A_PCT10'] != '#')
    ].copy()

    # Apply prefix filter on OCC_CODE column
    bls_filtered = bls_filtered[
        bls_filtered['OCC_CODE'].apply(lambda x: matches_prefix(x, prefixes))
    ]

    if bls_filtered.empty:
        print(f"No BLS OCC data found for {sector_name} in {area_title} with prefixes {prefixes}")
        return None

    # 4. Convert percentiles to numeric
    percentiles = ['A_PCT10', 'A_PCT25', 'A_MEDIAN', 'A_PCT75', 'A_PCT90']
    for col in percentiles:
        bls_filtered[col] = pd.to_numeric(bls_filtered[col], errors='coerce')

    bls_filtered['TOT_EMP'] = pd.to_numeric(bls_filtered['TOT_EMP'], errors='coerce')

    # 5. Weighted average of each percentile by employment
    weights = bls_filtered['TOT_EMP']
    weighted_avg = lambda col: (bls_filtered[col] * weights).sum() / weights.sum()

    p10, p25, p50, p75, p90 = [weighted_avg(col) for col in percentiles]

    # 6. Estimate CE group boundaries
    ce01_upper = p25 + (p50 - p25) * (pct_ce01 / (pct_ce01 + pct_ce02))  # between 25th and median
    ce02_upper = p75
    ce03_lower = p75

    return {
        "CE01_upper_bound": ce01_upper,
        "CE02_upper_bound": ce02_upper,
        "CE03_lower_bound": ce03_lower
    }

import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
from geopy.distance import geodesic
import folium
import branca.colormap as cm

# ---------------------------------------------
# 1. Load OD Data and Truncate to Block Group Level
# ---------------------------------------------
def load_od_data(path):
    od = pd.read_csv(path, dtype={"h_geocode": str, "w_geocode": str})
    od['h_bg'] = od['h_geocode'].str[:12]
    od['w_bg'] = od['w_geocode'].str[:12]
    od = od[od['S000'] > 0]  # Keep only flows with jobs
    return od

# ---------------------------------------------
# 2. Load Block Group Shapefile (NC)
# ---------------------------------------------
def load_block_groups(shapefile_path):
    bg = gpd.read_file(shapefile_path)
    bg['GEOID'] = bg['GEOID'].astype(str)
    return bg[['GEOID', 'geometry', 'COUNTYFP']]

# ---------------------------------------------
# 3. Merge OD Data with Home and Work Geometries
# ---------------------------------------------
def attach_geometries(od_df, bg_gdf):
    merged = od_df.merge(
        bg_gdf.rename(columns={'GEOID': 'h_bg', 'geometry': 'h_geometry'}),
        on='h_bg', how='left'
    ).merge(
        bg_gdf.rename(columns={'GEOID': 'w_bg', 'geometry': 'w_geometry'}),
        on='w_bg', how='left'
    )
    return merged

# ---------------------------------------------
# 4. Create Line Geometries Between Centroids
# ---------------------------------------------
def create_flow_lines(od_df):
    od_df['flow_line'] = od_df.apply(
        lambda row: LineString([
            row['h_geometry'].centroid, row['w_geometry'].centroid
        ]) if pd.notnull(row['h_geometry']) and pd.notnull(row['w_geometry']) else None,
        axis=1
    )
    return od_df[od_df['flow_line'].notnull()].copy()

# ---------------------------------------------
# 5. Flag Commutes Within Radius of a Location
# ---------------------------------------------
def flag_within_radius(od_df, center_point, miles=10):
    def in_radius(geom):
        return geodesic((center_point.y, center_point.x), (geom.centroid.y, geom.centroid.x)).miles <= miles

    od_df['is_home_near'] = od_df['h_geometry'].apply(lambda g: in_radius(g) if g else False)
    od_df['is_work_near'] = od_df['w_geometry'].apply(lambda g: in_radius(g) if g else False)
    return od_df

# ---------------------------------------------
# 6. Filter Flows Involving Region of Interest
# ---------------------------------------------
def filter_flows_by_proximity(od_df):
    return od_df[(od_df['is_home_near']) | (od_df['is_work_near'])].copy()

# ---------------------------------------------
# 7. Create Folium Map of Commuting Flows
# ---------------------------------------------
def visualize_flows(od_df, center_lat, center_lon, value_col='S000', filename="flows_map.html"):
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='cartodbpositron')

    max_val = od_df[value_col].max()
    colormap = cm.LinearColormap(['green', 'blue', 'red'], vmin=0, vmax=max_val)

    for _, row in od_df.iterrows():
        coords = [(pt[1], pt[0]) for pt in row['flow_line'].coords]
        folium.PolyLine(
            locations=coords,
            color=colormap(row[value_col]),
            weight=2 + row[value_col] / max_val * 4,
            opacity=0.6,
            popup=f"Jobs: {row[value_col]}<br>From {row['h_bg']} to {row['w_bg']}"
        ).add_to(m)

    colormap.caption = 'Number of Jobs'
    colormap.add_to(m)
    m.save(filename)
    return m


import math

def style_function(feature):
    val = feature["properties"].get("poverty_rate")
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return {
            "fillColor": "#cccccc",  # gray for missing
            "color": "black",
            "weight": 0.5,
            "fillOpacity": 0.6,
        }
    else:
        return {
            "fillColor": colormap(val),
            "color": "black",
            "weight": 0.5,
            "fillOpacity": 0.6,
        }



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

def get_dynamic_feature_bounds(df, features):
    df = compute_engineered_features(df)  # Ensure engineered features exist
    bounds = {}
    for feature in features:
        if feature in df.columns:
            col_data = df[feature].dropna()
            if col_data.nunique() == 2 and set(col_data.unique()) <= {0, 1}:
                bounds[feature] = (0, 1)
            else:
                bounds[feature] = (col_data.min(), col_data.max())
    return bounds

def find_optimal_inputs_for_prob(model, scaler, desired_prob, features, feature_bounds=None):
    """
    Finds the feature values that produce the desired predicted probability using the trained logistic regression model.

    Parameters:
    - model: Trained LogisticRegression.
    - scaler: Fitted StandardScaler.
    - desired_prob: Target probability (e.g., 0.5).
    - features: List of nonzero feature names.
    - feature_bounds: Dict of bounds {feature: (min, max)}.

    Returns:
    - Dict of optimal raw feature values (only for the optimized features), as Decimal values.
    """
    all_features = scaler.feature_names_in_
    full_index = {f: i for i, f in enumerate(all_features)}

    n_opt = len(features)
    x0 = np.zeros(n_opt)

    bounds = []
    for feat in features:
        i = full_index[feat]
        raw_min, raw_max = feature_bounds.get(feat, (-np.inf, np.inf))
        raw_vec_min = np.zeros(len(all_features))
        raw_vec_max = np.zeros(len(all_features))
        raw_vec_min[i] = raw_min
        raw_vec_max[i] = raw_max
        scaled_min = scaler.transform(pd.DataFrame([raw_vec_min], columns=all_features))[0][i]
        scaled_max = scaler.transform(pd.DataFrame([raw_vec_max], columns=all_features))[0][i]
        bounds.append((scaled_min, scaled_max))

    def objective(x_opt_scaled):
        full_scaled = np.zeros(len(all_features))
        for i, feat in enumerate(features):
            full_scaled[full_index[feat]] = x_opt_scaled[i]
        prob = model.predict_proba([full_scaled])[0, 1]
        return (prob - desired_prob) ** 2

    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

    if result.success:
        full_scaled = np.zeros(len(all_features))
        for i, feat in enumerate(features):
            full_scaled[full_index[feat]] = result.x[i]

        full_raw = scaler.inverse_transform([full_scaled])[0]

        # Return as Decimal values
        values = {feat: Decimal(str(full_raw[full_index[feat]])) for feat in features}
        

        # Optional high-precision transformations
        if "log_median_income" in values:
            values["median_income_from_log"] = Decimal(values["log_median_income"]).exp()
            print(f"Extract median_income_from_log: {values['median_income_from_log']:.15f}")

        if "log_density" in values and "pop_density" not in values:
            values["pop_density_from_log"] = Decimal(values["log_density"]).exp()

        return values

    else:
        raise ValueError("Optimization failed: " + result.message)
    
# Filter helper
def filter_by_focus_counties(gdf, county_field="COUNTYFP", focus_counties=None):
    if focus_counties is None:
        focus_counties = ["077", "183", "063", "181", "067"]
    return gdf[gdf[county_field].isin(focus_counties)]


def get_dynamic_feature_bounds(df, features):
    df = compute_engineered_features(df)  # Ensure engineered features exist
    bounds = {}
    for feature in features:
        if feature in df.columns:
            col_data = df[feature].dropna()
            if col_data.nunique() == 2 and set(col_data.unique()) <= {0, 1}:
                bounds[feature] = (0, 1)
            else:
                bounds[feature] = (col_data.min(), col_data.max())
    return bounds

from folium.plugins import TimeSliderChoropleth
import json
import branca.colormap as cm

def build_time_slider_layer(gdf, indicator, cmap, label):
    gdf = gdf.copy()
    gdf = gdf.dropna(subset=[indicator, "geometry"])
    gdf["time"] = pd.to_datetime(gdf["vintage"].astype(str) + "-01-01")

    # Convert to GeoJSON features
    geojson = gdf.set_index("GEOID").__geo_interface__

    styledict = {}
    for geoid, group in gdf.groupby("GEOID"):
        styledict[geoid] = {}
        for _, row in group.iterrows():
            timestamp = row["time"].strftime("%Y-%m-%dT%H:%M:%S")
            val = row[indicator]
            color = cmap(val) if not pd.isnull(val) else "#ccc"
            styledict[geoid][timestamp] = {
                "color": "black",
                "opacity": 0.5,
                "fillColor": color,
                "fillOpacity": 0.7
            }

    return TimeSliderChoropleth(
        data=geojson,
        styledict=styledict,
        name=label
    )
