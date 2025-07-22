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

# Standard Library
import argparse
import math
import warnings
from decimal import Decimal, getcontext

# Scientific Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Statsmodels
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)  # or a higher value

brand = "Harris Teeter"

road_path="data/block_group_road_access_scores.csv"

ruca_path="data/ruca2010revised.xlsx"

def ruca_to_location_type(ruca_code):
        if ruca_code in [1, 2, 3]: return "urban"
        elif ruca_code in [4, 5, 6]: return "micropolitan"
        elif ruca_code in [7, 8, 9]: return "rural"
        elif ruca_code == 10: return "isolated_rural"
        return "unknown"

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
df["GEOID_T"] = df["STATE"] + df["COUNTY"] + df["TRACT"]


ruca_df = pd.read_excel(ruca_path, sheet_name="Data", dtype={"GEOID": str})
ruca_df.rename(columns={
    "State-County-Tract FIPS Code (lookup by address at http://www.ffiec.gov/Geocode/)": "GEOID"
}, inplace=True)
ruca_df.rename(columns={
    "GEOID": "GEOID_ruca"
}, inplace=True)
ruca_df["primary_location_type"] = ruca_df["Primary RUCA Code 2010"].apply(ruca_to_location_type)
ruca_df["GEOID_ruca"] = ruca_df["GEOID_ruca"].astype(str)
ruca_df = ruca_df[["GEOID_ruca","primary_location_type"]]

#get road info
road_df = pd.read_csv(road_path)
road_df["GEOID_road"] = road_df["GEOID_road"].astype(str)

#get job info

tract_jobs = pd.read_csv('data/nc_wac_S000_JT00_2022.csv.gz', dtype={'w_geocode': str})
tract_jobs['GEOID_jobs'] = tract_jobs['w_geocode'].str[:12]
tract_jobs = tract_jobs.groupby('GEOID_jobs')[['C000', 'CE01', 'CE02', 'CE03']].sum().reset_index()
tract_jobs.columns = ['GEOID_jobs', 'total_jobs', 'jobs_0_1250', 'jobs_1251_3333', 'jobs_3333_up']

tract_jobs["share_low_wage"] = tract_jobs["jobs_0_1250"] / tract_jobs["total_jobs"]
tract_jobs["share_mid_wage"] = tract_jobs["jobs_1251_3333"] / tract_jobs["total_jobs"]
tract_jobs["share_high_wage"] = tract_jobs["jobs_3333_up"] / tract_jobs["total_jobs"]
tract_jobs = tract_jobs.replace([np.inf, -np.inf], np.nan).fillna(0)
etj_row = {"GEOID_jobs": "oxford_etj",
               "share_low_wage": 0.217794851813231,
               "share_mid_wage": 0.273534316115669,
               "share_high_wage": 0.273534316115669,
               "total_jobs": 1586,
               }
    
tract_jobs = pd.concat([tract_jobs, pd.DataFrame([etj_row])], ignore_index=True)

stores_df = pd.read_csv("data/poisblock.csv")
stores_df["GEOID_POI"] = stores_df["GEOID_POI"].astype(str)

store_presence = (
        stores_df.groupby(["GEOID_POI", "name"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
store_presence["GEOID_POI"] = store_presence["GEOID_POI"].astype(str)

landuse_gdf = gpd.read_file("data/tracts_with_landuse_metrics.gpkg")[["GEOID", "pct_commercial",  "entropy",]]
landuse_gdf["GEOID_landuse"] = landuse_gdf["GEOID"].astype(str)
etj_row = {
    "GEOID_landuse" : "oxford_etj",
    "pct_commercial": 34.45990350932659,
        "entropy": .3958957659274787,

}
landuse_gdf = pd.concat([landuse_gdf, pd.DataFrame([etj_row])], ignore_index=True)

#merge in all dataframes

merged_df = pd.merge(df, store_presence, left_on="GEOID", right_on="GEOID_POI", how="left").fillna(0)
print(merged_df.columns)
merged_df = pd.merge(merged_df, road_df, left_on="GEOID", right_on="GEOID_road", how="left").fillna(0)
merged_df = pd.merge(merged_df, ruca_df, left_on="GEOID_T", right_on="GEOID_ruca", how="left").fillna(0)
merged_df = pd.merge(merged_df, tract_jobs, left_on="GEOID", right_on="GEOID_jobs", how="left").fillna(0)
merged_df = pd.merge(merged_df, landuse_gdf, left_on="GEOID", right_on="GEOID_landuse", how="left").fillna(0)

# #why this?

location_dummies = pd.get_dummies(merged_df["primary_location_type"], prefix="loc", drop_first=True)
merged_df = pd.concat([merged_df, location_dummies], axis=1)
# Center median_income by subtracting its mean
merged_df["median_income_centered"] = merged_df["median_income"] - merged_df["median_income"].mean()

merged_df["median_income_centered"] = merged_df["median_income"] - merged_df["median_income"].mean()
merged_df["income_squared"] = merged_df["median_income_centered"] ** 2
merged_df["log_density"] = np.log1p(merged_df["pop_density"])
merged_df["log_median_income"] = np.log(merged_df["median_income"])
merged_df["job_density"] = merged_df["total_jobs"]/merged_df["population"]
merged_df["pct_commercial"] = merged_df["pct_commercial"]/100
merged_df['entropy'] = merged_df['entropy']
merged_df['pct_commercial_sq'] = merged_df['pct_commercial'] ** 2
merged_df['entropy_log_density'] = merged_df['entropy'] * merged_df['log_density']
df = compute_engineered_features(merged_df)
df["has_brand"] = (df[brand] > 0).astype(int)
df = df.replace([np.inf, -np.inf], np.nan).dropna()


print(f"\n=== Logistic Regression Analysis for '{brand}' ===")

if formula is None:
    formula = """
    has_brand ~ median_income
    + log_median_income
    + I(median_income**2)
    + log_density
    + median_income:log_density
    + pop_density
    + road_access_score
    + pct_owner_occupied
    + job_density
    + share_high_wage
    + share_mid_wage
    + share_low_wage
    """

coef_df1 = None

# === Define feature list ===
features = [
    "median_income", "income_squared", "log_median_income", 
    "pop_density", "log_density", "road_access_score", 
    "pct_owner_occupied", "job_density", "share_high_wage", 
    "share_mid_wage", "share_low_wage", "pct_commercial", 
    "pct_commercial_sq", "entropy", "entropy_log_density"
]

X = df[features]
y = df["has_brand"]

# === Clean data ===
if not np.isfinite(X).all().all():
    print("Warning: Detected infinite or NaN values in input features. Cleaning them...")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
# Convert X back to DataFrame with original column names
X = pd.DataFrame(X, columns=features)


# === Standardize features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Fit L1-regularized logistic regression ===
model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_scaled, y)

# === Inspect coefficients ===
coef = model.coef_[0]
coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": coef
})
print("\n=== L1 Regularized Coefficients ===")
print(coef_df.sort_values("Coefficient", key=lambda x: abs(x), ascending=False))

# === Identify features with non-zero coefficients ===
tol = 1e-6
nonzero_idx = np.where(np.abs(coef) > tol)[0]
nonzero_features = [features[i] for i in nonzero_idx]

print("\nFeatures with non-zero coefficients:")
print(nonzero_features)

# === Predict probabilities and store in df ===
df["predicted_prob"] = model.predict_proba(X_scaled)[:, 1]
df["predicted_prob_pct"] = (df["predicted_prob"] * 100).round(1).astype(str) + "%"
merged_df = df.copy()

# === Compute feature bounds dynamically based on nonzero features ===
feature_bounds = get_dynamic_feature_bounds(df, nonzero_features)

# === Optimize for 50% probability ===
optimal_inputs = find_optimal_inputs_for_prob(
    model=model,
    scaler=scaler,
    desired_prob=0.5,
    features=nonzero_features,
    feature_bounds=feature_bounds
)

print("\n=== Optimal Feature Values for 50% Brand Presence ===")
for k, v in optimal_inputs.items():
    print(f"{k}: {v:.2f}")