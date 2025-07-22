# Set brand for analysis
brand = "La Farm Bakery"
vintage = 2023
# Core libraries
# Global config
import warnings
warnings.filterwarnings("ignore")

# Standard libraries
import math
from decimal import Decimal, getcontext
getcontext().prec = 30  # Set global decimal precision

# Core libraries
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import folium
import branca.colormap as cm
from branca.element import Template, MacroElement
from folium.features import GeoJsonTooltip

# Geospatial
import geopandas as gpd

# Modeling & evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as blookie
import statsmodels.formula.api as smf

# Spatial analysis
from libpysal.weights import Queen, lag_spatial
from esda import Moran

# Optimization
from scipy.optimize import minimize


# --- Custom Formula ---


custom_formula = """
has_brand ~ median_income
+ log_median_income
+ I(median_income**2)
+ log_density
+ median_income:log_density
+ pop_density
+ road_access_score
+ pct_with_degree
+ pct_owner_occupied
+ job_density
+ share_high_wage
+ share_mid_wage
+ share_low_wage
+ pct_commercial
+ pct_commercial_sq 
+ entropy 
+ entropy_log_density
# + lag_median_income
# + lag_pop_density
# + lag_log_density

"""

# --- Census Variables ---
acs_vars = [
    "B01003_001E",  # Total population
    "B19013_001E",  # Median household income
    "B25003_001E",  # Total housing units
    "B25003_002E",  # Owner-occupied housing units
    "B15003_001E",  # Total population 25+ (education base)
    "B15003_022E",  # Bachelor's
    "B15003_023E",  # Master's
    "B15003_024E",  # Professional
    "B15003_025E",  # Doctorate
]

# --- Download ACS block group data with geometry ---
bg_df = ced.download(
    dataset=ACS5,
    vintage=vintage,
    download_variables=acs_vars,
    state=states.NC,
    county='*',
    tract='*',
    block_group='*',
    with_geometry=True
)

# --- Compute Area ---
bg_df = bg_df.to_crs(epsg=5070)  # Equal-area projection
bg_df["area_sqkm"] = bg_df.geometry.area / 1e6
bg_df["GEOID"] = bg_df["STATE"] + bg_df["COUNTY"] + bg_df["TRACT"] + bg_df["BLOCK_GROUP"]

# --- Compute Derived Features ---
bg_df["population"] = bg_df["B01003_001E"]
bg_df["median_income"] = bg_df["B19013_001E"]
bg_df["pct_owner_occupied"] = bg_df["B25003_002E"] / bg_df["B25003_001E"]

bg_df["pct_degreed"] = (
    bg_df["B15003_022E"]
    + bg_df["B15003_023E"]
    + bg_df["B15003_024E"]
    + bg_df["B15003_025E"]
) / bg_df["B15003_001E"]

bg_df["pop_density"] = bg_df["population"] / bg_df["area_sqkm"]

# Optional log transforms
bg_df["log_median_income"] = np.log(bg_df["median_income"].replace(0, np.nan))
bg_df["log_density"] = np.log(bg_df["pop_density"].replace(0, np.nan))

# --- Placeholders for Additional Features ---
# These will come from other sources (LODES, parcels, OSM, etc.)
bg_df["share_high_wage"] = np.nan
bg_df["share_mid_wage"] = np.nan
bg_df["share_low_wage"] = np.nan
bg_df["pct_commercial"] = np.nan
bg_df["entropy"] = np.nan
bg_df["road_access_score"] = np.nan

# --- Optional Squared Term for Model ---
bg_df["pct_commercial_sq"] = bg_df["pct_commercial"] ** 2
bg_df["entropy_log_density"] = bg_df["entropy"] * bg_df["log_density"]

# --- Keep only relevant columns for modeling ---
model_vars = [
    "GEOID",
    "area_sqkm",
    "population",
    "pop_density",
    "log_density",
    "median_income",
    "log_median_income",
    "pct_owner_occupied",
    "pct_degreed",
    "share_high_wage",
    "share_mid_wage",
    "share_low_wage",
    "pct_commercial",
    "pct_commercial_sq",
    "entropy",
    "entropy_log_density",
    "road_access_score",
    "geometry",
]

bg_model_df = bg_df[model_vars].copy()

def run_brand_analysis(
    brand,
    store_data_path,
    ruca_path,
    road_path,
    tract_shapefile_path,
    census_data_path,
    tract_job_csv_path,
    output_csv_path=None,
    show_plot=True,
    make_map=True,
    formula=None,
    cv_folds=5,

):
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

    # Load Data
    ruca_df = pd.read_excel(ruca_path, sheet_name="Data", dtype={"GEOID": str})
    ruca_df.rename(columns={
        "State-County-Tract FIPS Code (lookup by address at http://www.ffiec.gov/Geocode/)": "GEOID"
    }, inplace=True)
    ruca_df["primary_location_type"] = ruca_df["Primary RUCA Code 2010"].apply(ruca_to_location_type)

    road_df = pd.read_csv(road_path)
    road_df["GEOID"] = road_df["GEOID"].astype(str)
    
    tract_jobs = pd.read_csv('data/nc_wac_S000_JT00_2022.csv.gz', dtype={'w_geocode': str})

    tract_jobs['tract'] = tract_jobs['w_geocode'].str[:11]

    tract_jobs = tract_jobs.groupby('tract')[['C000', 'CE01', 'CE02', 'CE03']].sum().reset_index()
    tract_jobs.columns = ['GEOID', 'total_jobs', 'jobs_0_1250', 'jobs_1251_3333', 'jobs_3333_up']
    etj_row = {"GEOID": "oxford_etj",
        "jobs_0_1250": 348,
               "jobs_1251_3333": 434,
               "jobs_3333_up" : 804,
               "total_jobs": 1586,
               }
    tract_jobs = pd.concat([tract_jobs, pd.DataFrame([etj_row])], ignore_index=True)
    tract_jobs["share_low_wage"] = tract_jobs["jobs_0_1250"] / tract_jobs["total_jobs"]
    tract_jobs["share_mid_wage"] = tract_jobs["jobs_1251_3333"] / tract_jobs["total_jobs"]
    tract_jobs["share_high_wage"] = tract_jobs["jobs_3333_up"] / tract_jobs["total_jobs"]
    tract_jobs = tract_jobs.replace([np.inf, -np.inf], np.nan).fillna(0)
    tracts = gpd.read_file(tract_shapefile_path).to_crs("EPSG:4326")
    # tracts = tracts[tracts["COUNTYFP"].isin(focus_counties)]
    tracts["geometry"] = tracts["geometry"].buffer(0)
    # Load tracts and reproject

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
    stores_df = pd.read_csv(store_data_path)

    stores_gdf = gpd.GeoDataFrame(
        stores_df,
        geometry=gpd.points_from_xy(stores_df.lng, stores_df.lat),
        crs="EPSG:4326"
    )
# Get Oxford ETJ polygon
    oxford_etj_geom = tracts[tracts['GEOID'] == 'oxford_etj'].geometry.iloc[0]

    stores_with_tracts = gpd.sjoin(stores_gdf, tracts, how="left", predicate="within")

    stores_with_tracts.to_csv("data/storeswithtracts.csv", index=False)

    store_presence = (
        stores_with_tracts.groupby(["GEOID_right", "name"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    store_presence["GEOID_right"] = store_presence["GEOID_right"].astype(str)

    store_presence.to_csv("data/storepresence.csv", index=False)

    census_df = pd.read_csv(census_data_path)
    etj_row = {
    "GEOID": "oxford_etj",
    "population": 8972,
    "ALAND": 8972,
    "median_income": 48135,
    "pct_owner_occupied": .525,
    "pct_with_degree" : .207,
    "area_km2": 16.13563,
    "pop_density" : 556.03654769,
    }
    census_df = pd.concat([census_df, pd.DataFrame([etj_row])], ignore_index=True)

    census_df["GEOID"] = census_df["GEOID"].astype(str)
    landusegdf = gpd.read_file("data/tracts_with_landuse_metrics.gpkg")
    landuse_gdf = gpd.read_file("data/tracts_with_landuse_metrics.gpkg")[["GEOID", "pct_commercial",  "entropy",]]
    landuse_gdf["GEOID"] = landuse_gdf["GEOID"].astype(str)
    etj_row = {
        "GEOID" : "oxford_etj",
        "pct_commercial": 34.4599035093265,
         "entropy": .29,

    }
    landuse_gdf = pd.concat([landuse_gdf, pd.DataFrame([etj_row])], ignore_index=True)

    merged_df = pd.merge(census_df, store_presence, left_on="GEOID", right_on="GEOID_right", how="left").fillna(0)
    merged_df = pd.merge(merged_df, road_df, on="GEOID", how="left").fillna(0)
    merged_df = pd.merge(merged_df, ruca_df, on="GEOID", how="left").fillna(0)
    merged_df = pd.merge(merged_df, tract_jobs, on="GEOID", how="left").fillna(0)
    merged_df = pd.merge(merged_df, landuse_gdf, on="GEOID", how="left").fillna(0)

    location_dummies = pd.get_dummies(merged_df["primary_location_type"], prefix="loc", drop_first=True)
    merged_df = pd.concat([merged_df, location_dummies], axis=1)
    # Center median_income by subtracting its mean
    merged_df["median_income_centered"] = merged_df["median_income"] - merged_df["median_income"].mean()

    # After filtering merged_df to match tracts:
    common_geoids = merged_df.index.intersection(tracts.index)
    merged_df = merged_df.loc[common_geoids]


    # 🔁 Now compute engineered features AFTER Oxford is added
    merged_df["median_income_centered"] = merged_df["median_income"] - merged_df["median_income"].mean()
    merged_df["income_squared"] = merged_df["median_income_centered"] ** 2
    merged_df["log_density"] = np.log1p(merged_df["pop_density"])
    merged_df["log_median_income"] = np.log(merged_df["median_income"])
    merged_df["job_density"] = merged_df["total_jobs"]/merged_df["population"]
    merged_df["pct_commercial"] = merged_df["pct_commercial"]/100
    merged_df['entropy'] = merged_df['entropy']
    merged_df['pct_commercial_sq'] = merged_df['pct_commercial'] ** 2
    merged_df['entropy_log_density'] = merged_df['entropy'] * merged_df['log_density']

    # # Compute spatial lags (will be NaN for oxford_etj, which is OK if you handle it later)
    # w = Queen.from_dataframe(tracts, use_index=True)
    # w.transform = 'r'
    # merged_df["lag_median_income"] = lag_spatial(w, merged_df["median_income"])
    # merged_df["lag_pop_density"] = lag_spatial(w, merged_df["pop_density"])
    # merged_df["lag_log_density"] = lag_spatial(w, merged_df["log_density"])

    
    df = compute_engineered_features(merged_df)
    df["has_brand"] = (df[brand] > 0).astype(int)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(df.tail())

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

    # # === VIF Check ===
    # print("\n=== Variance Inflation Factor (VIF) ===")
    # X_with_const = blookie.add_constant(X)
    # vif_data = pd.DataFrame()
    # vif_data["feature"] = X_with_const.columns
    # vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
    # print(vif_data)

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
    print("oxford_etj" in merged_df["GEOID"].unique())

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

model, coef_df, top10_df, coef_df1 = run_brand_analysis(
    brand= brand,
    store_data_path="data/granville_google_eateries_with_population.csv",
    ruca_path="data/ruca2010revised.xlsx",
    road_path="data/road_access_scores.csv",
    tract_shapefile_path="data/tl_2024_37_tract.shp",
    census_data_path="data/poismerged.csv",
    output_csv_path="top10_suggestions.csv",
    tract_job_csv_path="data/wac.csv",
    show_plot=False,
    make_map=True,
    formula=custom_formula  # Pass your custom predictors here
)
