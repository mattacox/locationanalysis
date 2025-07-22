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

# Spatial Libraries
import geopandas as gpd
import folium
from folium.features import GeoJsonTooltip
from branca.element import Template, MacroElement
import branca.colormap as cm

# Statsmodels
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Set decimal precision
getcontext().prec = 30


oxford_etj_df = pd.read_csv("data/oxford_etj_row.csv")

HUMAN_LABELS = {
    "log_median_income": "Log Median Income",
    "job_density": "Job Density",
    "median_income_from_log": "Median Income (Derived)",
    "median_income": "Median Income",
    "log_density": "Log Population Density",
    "pop_density_from_log": "Population Density (Derived)",
    "pop_density": "Population Density",
    "entropy": "Land Use Entropy",
    "pct_commercial": "Percent Commercial Land",
    "road_access_score" : "Road Access Score",
    "pct_owner_occupied" : "Percent Owner Occupied",
    "share_high_wage" : "Percent High Wage Earners",
    "share_mid_wage" : "Percent Mid Wage Earners",
    "share_low_wage" : "Percent Low Wage Earners",
    "pct_commercial_sq" :"Percent Commercial Land Squared",
    "entropy_log_density" : "Entropy Log Density",

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

# Fixed HTML Legend Class
class FixedHTMLLegend(MacroElement):
    def __init__(self, html_content: str, position: str = "bottom: 50px; left: 10px;"):
        super().__init__()
        self._template = Template(f"""
        {{% macro html(this, kwargs) %}}
        <div id="legend" style="
            position: fixed;
            {position}
            z-index: 9999;
            background-color: white;
            padding: 10px;
            border: 2px solid black;
            border-radius: 5px;
            box-shadow: 3px 3px 6px rgba(0,0,0,0.3);
            font-size: 12px;
            max-width: 250px;
        ">
            {html_content}
        </div>
        {{% endmacro %}}
        """)

# Filter helper
def filter_by_focus_counties(gdf, county_field="COUNTYFP", focus_counties=None):
    if focus_counties is None:
        focus_counties = ["077", "183", "063", "181", "067"]
    return gdf[gdf[county_field].isin(focus_counties)]

def run_brand_analysis(
    brand,
    store_data_path,
    ruca_path,
    road_path,
    tract_shapefile_path,
    census_data_path,
    tract_job_csv_path,
    # output_csv_path=None,
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
        "pct_commercial": 34.45990350932659,
         "entropy": .3958957659274787,

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

    if show_plot:
        plt.hist(model.predict(), bins=30)
        plt.title(f"Predicted probabilities for {brand}")
        plt.show()

    top10_df = df[df[brand] == 0].sort_values("predicted_prob", ascending=False).head(10).copy()
    # if output_csv_path:
    #     top10_df.reset_index().to_csv(output_csv_path, index=False)

    # Optional map
    if make_map:
        tracts["predicted_prob"] = merged_df["predicted_prob"]
        tracts["predicted_prob_pct"] = merged_df["predicted_prob_pct"]
        tracts["median_income"] = merged_df["median_income"]
        tracts["pop_density"] = merged_df["pop_density"]
        tracts["population"] = merged_df["population"]
        tracts["area_km2"] = merged_df["area_km2"]
        tracts["road_access_score"] = merged_df["road_access_score"]
        tracts["GEOIDFQ_split"] = merged_df["GEOID_right"].str.split("US").str[-1]
        tracts["pct_with_degree"] = (merged_df["pct_with_degree"]* 100).round(1).astype(str) + "%"
        tracts["pct_owner_occupied"] = (merged_df["pct_owner_occupied"]* 100).round(1).astype(str) + "%"
        tracts["total_jobs"] = merged_df["total_jobs"]
        tracts["job_density"] = merged_df["job_density"]
        tracts["entropy"] = merged_df["entropy"]

        tracts["share_high_wage"] = merged_df["share_high_wage"]
        tracts["share_mid_wage"] = merged_df["share_mid_wage"]
        tracts["share_low_wage"] = merged_df["share_low_wage"]
        tracts["pct_commercial"] = merged_df["pct_commercial"]

        m = folium.Map(location=[36.1, -78.7], zoom_start=9, tiles="cartodbpositron")

        colormap = cm.linear.OrRd_09.scale(tracts["predicted_prob"].min(), tracts["predicted_prob"].max())
        colormap.caption = f"Predicted Probability of a '{brand}'"
        colormap.add_to(m)

        tracts_clean = tracts.dropna(subset=["predicted_prob"])
        predicted_tracts_focus = tracts_clean[tracts_clean["COUNTYFP"].isin(focus_counties)]

        percent_keys = {
            "pct_owner_occupied",
            "share_high_wage",
            "share_mid_wage",
            "share_low_wage",
            "pct_commercial_sq",
            "pct_commercial",
        }

        for col in percent_keys:
            display_col = f"{col}_display"
            if display_col in predicted_tracts_focus.columns:
                predicted_tracts_focus = predicted_tracts_focus.drop(columns=[display_col])
            if col in predicted_tracts_focus.columns:
                predicted_tracts_focus[display_col] = (
                            pd.to_numeric(predicted_tracts_focus[col], errors="coerce") * 100
                        ).round(1).astype(str) + "%"

        prob_layer = folium.FeatureGroup(name="Tract Data", show=True)

        folium.GeoJson(
            predicted_tracts_focus,
            name="Tract Data",
            style_function=lambda feature: {
                'fillColor': colormap(feature['properties']['predicted_prob']),
                'color': 'black',
                'weight': 0.2,
                'fillOpacity': 0.4,
            },
            tooltip = GeoJsonTooltip(fields = [
                "GEOIDFQ_split",
                "predicted_prob_pct",
                "median_income",
                "population", "pop_density",
                "area_km2",
                "road_access_score",
                "pct_with_degree",
                "pct_owner_occupied",
                "pct_commercial_display",
                "entropy",
                "job_density",
                "share_high_wage_display",
                "share_mid_wage_display",
                "share_low_wage_display",
                ], 
                aliases=[
                    "GeoID",
                    f"Probability of {brand}:",
                    "Median Income:",
                    "Population:", 
                    "Population Density (Km2):",
                    "Area (Km2)",
                    "Road Score:",
                    "Degreed:",
                    "Owner-Occupied: ",
                    "Percent Commercial:",
                    "Entropy:",
                    "Job Density:",
                    "High Wage:",
                    "Mid Wage:",
                    "Low Wage:"], 
                    localize=True)
        ).add_to(prob_layer)
        prob_layer.add_to(m)

                # Add Oxford city boundary

        # 🆕 Filter store points to only those within the focused counties
        stores_focus = stores_gdf[stores_gdf["name"] == brand]
        stores_focus = gpd.sjoin(stores_focus, tracts.reset_index()[["GEOID", "geometry","COUNTYFP"]], how="inner", predicate="within")
        stores_focus = stores_focus[stores_focus["COUNTYFP"].isin(focus_counties)]
        store_layer = folium.FeatureGroup(name=f"{brand}s", show=True)

        for _, row in stores_focus.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=4,
                color='blue',
                fill=True,
                fill_opacity=0.8,
                popup=row.get("address", brand)
            ).add_to(store_layer)
            store_layer.add_to(m)

        legend_html = "<h4>Model Predictors</h4><ul>"

        excluded_keys = {
            "median_income_from_log",
            "income_squared",
            "pct_commercial_sq",

        }

        percent_keys = {
            "pct_owner_occupied",
            "share_high_wage",
            "share_mid_wage",
            "share_low_wage",
            "pct_owner_occupied",
            "pct_commercial_sq",
            "pct_commercial",
        }

        for k, v in optimal_inputs.items():
            if k in excluded_keys:
                continue
            label = HUMAN_LABELS.get(k, k)  # fallback to raw key if not found
            try:
                if k in percent_keys:
                    legend_html += f"<li>{label}: {float(v):.0%}</li>"  # convert 0.43 → 43%
                else:
                    legend_html += f"<li>{label}: {float(v):,.2f}</li>"
            except Exception:
                legend_html += f"<li>{label}: {v}</li>"

        legend_html += "</ul>"

        highlight_html = "<ul>"

        if "median_income_from_log" in optimal_inputs:
            highlight_html += f"<li><h5>Optimal Median Income to predict {brand}</h5> ${float(optimal_inputs['median_income_from_log']):,.2f}</li>"
        elif "median_income" in optimal_inputs:
            highlight_html += f"<li><h5>Optimal Median Income to predict {brand}</h5> ${float(optimal_inputs['median_income']):,.2f}</li>"


        # Add this section for population density
        if "pop_density" in optimal_inputs:
            highlight_html += f"<li><h5>Optimal Population Density to predict {brand}</h5> {float(optimal_inputs['pop_density']):,.2f} people/sq Km</li>"
 

        elif "log_density" in optimal_inputs:
            pop_density = math.exp(float(optimal_inputs["log_density"]))
            highlight_html += f"<li><h5>Optimal Population Density to predict {brand}</h5> {pop_density:,.2f} people/sq Km</li>"

        highlight_html += "</ul>"

        m.get_root().add_child(FixedHTMLLegend(highlight_html + legend_html))
        folium.LayerControl().add_to(m)

        m.save(f"html/brands/{brand.replace(' ', '_')}_map.html")

        print(f"Map saved to html/brands/{brand.replace(' ', '_')}_map.html")
 
 

    return model, coef_df, top10_df, coef_df1


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

def main():
    parser = argparse.ArgumentParser(description="Run brand placement suitability model.")
    parser.add_argument("--brand", type=str, required=True, help="Name of the brand to analyze.")
    args = parser.parse_args()

    brand = args.brand

    # Optional: Allow overrides from command line
    model, coef_df, top10_df, coef_df1 = run_brand_analysis(
        brand=brand,
        store_data_path="data/granville_google_eateries_with_population.csv",
        ruca_path="data/ruca2010revised.xlsx",
        road_path="data/road_access_scores.csv",
        tract_shapefile_path="data/tl_2024_37_tract.shp",
        census_data_path="data/poismerged.csv",
        tract_job_csv_path="data/nc_wac_S000_JT00_2022.csv.gz",
        show_plot=False,
        make_map=True,
        formula=None  # Optional: add CLI option for formula if needed
    )

if __name__ == "__main__":
    main()
