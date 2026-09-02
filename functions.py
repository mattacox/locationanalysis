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
import constants

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
        sum(normalized_local[f] * constants.weights[f] for f in fields_for_norm) / constants.total_weight
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

def simulate_local_infill_effect_for_bg(
    base_df, bg_geoid, total_new_units, iz_rate=0.15, share_from_renters=0.30, pass_through=0.5,
    renter_col="B25003_003E", rent_burden_col="percent_cost_burdened"
):
    """
    Apply an infill scenario where 'total_new_units' are all built in bg_geoid.
    Returns an adjusted GeoDataFrame (copy of base_df) with updated percent_cost_burdened
    for that BG and recomputed displacement_risk across all BGs.
    """
    df = base_df.copy().set_index("GEOID")
    if bg_geoid not in df.index:
        raise KeyError(f"{bg_geoid} not found in base_df GEOID index")

    iz_units = total_new_units * iz_rate
    renters_exiting = iz_units * share_from_renters

    # allocate renters_exiting entirely to the tested BG (local infill)
    renters_in_bg = df.at[bg_geoid, renter_col]
    if renters_in_bg <= 0:
        pct_reduction = 0.0
    else:
        pct_reduction = renters_exiting / renters_in_bg

    rent_change_pct = pct_reduction * pass_through

    # Compute new percent_cost_burdened for that BG (avoid negative)
    baseline_burden = df.at[bg_geoid, rent_burden_col]
    new_burden = baseline_burden * (1 - rent_change_pct)
    new_burden = max(new_burden, 0.0)

    adjusted = df.reset_index()
    adjusted.loc[adjusted["GEOID"] == bg_geoid, "percent_cost_burdened"] = new_burden
    adjusted_gdf = gpd.GeoDataFrame(adjusted, geometry="geometry", crs=adjusted.crs)

    # Recompute full displacement index across the county using the helper
    recomputed = recompute_displacement_with_adjusted_burden(base_df, adjusted_gdf)

    return recomputed

def find_min_units_for_target(
    base_df,
    bg_geoid,
    target_risk=0.4,
    iz_rate=0.10,
    share_from_renters=0.30,
    pass_through=0.5,
    max_units=20000,
    tol=1e-3,
    renter_col="B25003_003E"
):
    """
    Binary search on total_new_units built in bg_geoid (infill) to find the minimum total_new_units
    such that the bg's displacement risk falls to <= target_risk.
    Returns the number of units (int), and the final recomputed row for the BG (Series).
    If not achievable within max_units returns None and a message.
    """
    # baseline risk for the BG
    baseline_row = base_df.set_index("GEOID").loc[bg_geoid]
    baseline_risk = baseline_row["displacement_risk"]

    # quick check: if baseline already below target
    if baseline_risk <= target_risk:
        return 0, baseline_row

    lo, hi = 0, max_units
    found = None
    while lo <= hi:
        mid = (lo + hi) // 2
        recomputed = simulate_local_infill_effect_for_bg(
            base_df, bg_geoid, total_new_units=mid,
            iz_rate=iz_rate, share_from_renters=share_from_renters,
            pass_through=pass_through, renter_col=renter_col
        )
        risk_mid = recomputed.set_index("GEOID").at[bg_geoid, "displacement_risk"]
        # print(bg_geoid, "units", mid, "risk", risk_mid)  # debug if you want
        if risk_mid <= target_risk + tol:
            found = mid
            hi = mid - 1
        else:
            lo = mid + 1

    if found is None:
        return None, None
    recomputed_final = simulate_local_infill_effect_for_bg(
        base_df, bg_geoid, total_new_units=found,
        iz_rate=iz_rate, share_from_renters=share_from_renters,
        pass_through=pass_through, renter_col=renter_col
    )
    return found, recomputed_final.set_index("GEOID").loc[bg_geoid]

# Example batch run for all Oxford BGs
def compute_required_units_for_all_bgs(
    base_df,
    bg_list=None,
    target_risk=0.4,
    iz_rate=0.15,
    share_from_renters=0.30,
    pass_through=0.5,
    max_units=20000
):
    """
    For each BG in bg_list (or all in base_df if None), run binary search and return results DF.
    """
    results = []
    if bg_list is None:
        bg_list = base_df["GEOID"].unique().tolist()

    for geoid in bg_list:
        print("Testing", geoid)
        units_needed, final_row = find_min_units_for_target(
            base_df, geoid, target_risk=target_risk,
            iz_rate=iz_rate, share_from_renters=share_from_renters,
            pass_through=pass_through, max_units=max_units
        )
        if units_needed is None:
            results.append({
                "GEOID": geoid,
                "units_needed": None,
                "message": f"> {max_units} units required (not achieved)"
            })
        else:
            results.append({
                "GEOID": geoid,
                "units_needed": int(units_needed),
                "displacement_risk_final": final_row["displacement_risk"],
                "percent_cost_burdened_final": final_row["percent_cost_burdened"],
                "message": "achieved"
            })
    return pd.DataFrame(results)


def sensitivity_curve_for_bg(
    base_df,
    bg_geoid,
    unit_steps=None,
    iz_rate=0.15,
    share_from_renters=0.30,
    pass_through=0.5,
    renter_col="B25003_003E",
    rent_burden_col="percent_cost_burdened"
):
    """
    Run a sensitivity test: incrementally add housing units within a BG
    and record displacement risk at each step.
    
    Returns a DataFrame with GEOID, units_added, displacement_risk, and percent_cost_burdened.
    """
    if unit_steps is None:
        unit_steps = list(range(0, 2001, 100))  # default: 0 to 2000 in steps of 100

    records = []
    for u in unit_steps:
        recomputed = simulate_local_infill_effect_for_bg(
            base_df, bg_geoid, total_new_units=u,
            iz_rate=iz_rate,
            share_from_renters=share_from_renters,
            pass_through=pass_through,
            renter_col=renter_col,
            rent_burden_col=rent_burden_col
        )
        row = recomputed.set_index("GEOID").loc[bg_geoid]
        records.append({
            "GEOID": bg_geoid,
            "units_added": u,
            "displacement_risk": row["displacement_risk"],
            "percent_cost_burdened": row["percent_cost_burdened"]
        })

    return pd.DataFrame(records)



def batch_sensitivity_curves(
    base_df,
    bg_list=None,
    unit_steps=None,
    iz_rate=0.15,
    share_from_renters=0.30,
    pass_through=0.5,
    renter_col="B25003_003E",
    rent_burden_col="percent_cost_burdened",
    out_csv="data/sensitivity_curves.csv",
    plot_dir="plots/sensitivity_curves"
):
    """
    Run sensitivity curves for all BGs in bg_list.
    Saves combined results to CSV and optionally plots each BG's curve.
    """
    if bg_list is None:
        bg_list = base_df["GEOID"].unique().tolist()
    if unit_steps is None:
        unit_steps = list(range(0, 2001, 100))  # default sweep

    all_records = []

    # Create plot directory
    import os
    os.makedirs(plot_dir, exist_ok=True)

    for geoid in bg_list:
        print(f"Running sensitivity test for {geoid}...")
        curve_df = sensitivity_curve_for_bg(
            base_df, geoid, unit_steps=unit_steps,
            iz_rate=iz_rate, share_from_renters=share_from_renters,
            pass_through=pass_through,
            renter_col=renter_col, rent_burden_col=rent_burden_col
        )
        curve_df.to_csv(f"{plot_dir}/{geoid}_curve.csv", index=False)  # individual CSV
        all_records.append(curve_df)

        # Plot curve
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(curve_df["units_added"], curve_df["displacement_risk"], marker="o")
        plt.axhline(0.4, color="red", linestyle="--", label="Target Risk 0.4")
        plt.title(f"Sensitivity Curve for BG {geoid}")
        plt.xlabel("Units Added (infill)")
        plt.ylabel("Displacement Risk")
        plt.legend()
        plt.savefig(f"{plot_dir}/{geoid}_curve.png")
        plt.close()

    # Concatenate all into one CSV
    results_df = pd.concat(all_records, ignore_index=True)
    results_df.to_csv(out_csv, index=False)
    print(f"✅ Saved combined sensitivity results to {out_csv}")

    return results_df


def simulate_equilibrium_projection(
    base_bg_gdf,
    years=20,
    annual_infill_units_per_bg=0,        # if scalar -> applied to all BGs; if dict/series -> per-BG
    annual_external_units=0,             # units added outside (annexation) - reduces county pressure, not local
    iz_rate=0.15,                        # share of new units that are inclusionary (owner-occupied here)
    share_renter_to_owner_iz=0.30,       # share of IZ units that come from existing renters
    pass_through=0.5,                    # fraction of demand reduction that passes into rent decline
    rent_elasticity=0.8,                 # sensitivity of rents to demand-supply imbalance
    vacancy_dampener=0.5,                # reduces rent growth when vacancy increases
    exogenous_renter_growth_rate=0.00,   # annual % change in rental demand from outside (e.g., population growth)
    displacement_sensitivity=1.0,        # scale factor for how rent burden translates to displacement prob
    displacement_burden_threshold=0.40,  # rent burden (share) above which displacement accelerates (0-1)
    income_growth_rate=0.04,             # annual household income growth (affects cost-burden)
    normalize_index_each_year=True,      # whether to recompute MinMax normalization for displacement index
    stop_tol=1e-4,                       # equilibrium tolerance (mean abs change in displacement risk)
    verbose=False
):
    """
    Run a multi-year projection at block-group level and return a time-series DataFrame.
    base_bg_gdf: GeoDataFrame for baseline year containing necessary fields (see below).
    Returns: DataFrame with index = year and columns per BG (or long format time-series).
    IMPORTANT: This is a simplified structural model for scenario comparison, not a forecast.
    """

    # Make working copy and ensure necessary columns exist
    cols_needed = [
        "GEOID", "B25003_003E", "B25003_001E", "median_rent",
        "percent_cost_burdened", "rental_vacancy_rate",
        "rent_share", "poverty_rate", "snap_share", "unemployment_rate",
        "senior_share", "black_share", "latino_share", "white_share"
    ]
    for c in cols_needed:
        if c not in base_bg_gdf.columns:
            raise KeyError(f"Missing required column in base_bg_gdf: {c}")

    # Prepare per-BG arrays
    bg = base_bg_gdf.copy().set_index("GEOID")
    geoids = bg.index.tolist()

    # Baseline arrays
    renters = bg["B25003_003E"].astype(float).fillna(0)
    total_units = bg["B25003_001E"].astype(float).replace(0, np.nan).fillna(1)
    median_rent = bg["median_rent"].astype(float).fillna(0)
    rent_burden = bg["percent_cost_burdened"].astype(float).fillna(0)
    vacancy = bg["rental_vacancy_rate"].astype(float).fillna(0)
    rent_share = bg["rent_share"].astype(float).fillna(0)

    # Keep other risk fields for recomputing index each year
    static_fields = ["poverty_rate", "snap_share", "unemployment_rate", "senior_share", "black_share", "latino_share", "white_share"]
    static_df = bg[static_fields].fillna(0)

    # Pre-allocate time series store
    records = []

    # Normalize helper (we will reuse your weights and total_weight from the script environment)
    # Ensure 'weights' and 'total_weight' are available in caller namespace; else define defaults:
    try:
        weights_local = constants.weights
        total_weight_local = constants.total_weight
    except NameError:
        # fallback defaults in case not defined in caller namespace
        weights_local = {
            "rent_share": 1,
            "percent_cost_burdened": 2,
            "poverty_rate": 1,
            "snap_share": 1,
            "unemployment_rate": 1,
            "senior_share": 1,
            "inv_vacancy": 2,
        }
        total_weight_local = sum(weights_local.values())

    # For convenience allow annual_infill_units_per_bg to be scalar or Series/dict
    if np.isscalar(annual_infill_units_per_bg):
        annual_infill_array = pd.Series(annual_infill_units_per_bg, index=geoids, dtype=float)
    else:
        # try convertable
        annual_infill_array = pd.Series(annual_infill_units_per_bg).reindex(geoids).fillna(0).astype(float)

    # county-wide totals for normalization if needed
    base_for_normalization = bg.copy()

    # Iterative simulation
    prev_disp_risk = bg.get("displacement_risk", pd.Series(0, index=geoids)).astype(float).fillna(0)

    for year in range(1, years + 1):
        # 1) Supply additions this year (infill in-block + external)
        new_units_bg = annual_infill_array.copy()  # units built inside each BG this year
        # Add external units to county (affects county vacancy/rent dynamics but not local supply)
        county_new_units_external = float(annual_external_units)

        # 2) IZ conversions (owner-occupied affordable units) — remove demand of renters if share_from_renters
        iz_units_bg = new_units_bg * iz_rate
        renters_exiting_to_iz = iz_units_bg * share_renter_to_owner_iz

        # Clamp so renters_exiting cannot exceed renters in BG
        renters_exiting_to_iz = renters_exiting_to_iz.clip(upper=renters)

        # 3) Update renter demand with exogenous growth
        renters = renters * (1 + exogenous_renter_growth_rate) - renters_exiting_to_iz

        # 4) Displacement driven by high rent burden
        # displacement_prob ~ logistic or linear above threshold. Simple linear:
        # when rent_burden > threshold, displacement probability increases proportionally
        over_thresh = (rent_burden - displacement_burden_threshold).clip(lower=0)
        # displacement fraction this year
        disp_frac = (over_thresh / (1.0 - displacement_burden_threshold)) * displacement_sensitivity
        disp_frac = disp_frac.clip(0, 0.5)  # cap annual displacement fraction to 50% to keep stable
        renters_displaced = (renters * disp_frac).fillna(0)

        # Remove displaced renters from renter pool
        renters = (renters - renters_displaced).clip(lower=0)

        # 5) Update housing stock (approx): total_units increases by infill. For IZ owner units, they are still units (but reduce renter demand)
        total_units = total_units + new_units_bg

        # county-level demand/supply changes to estimate rent response
        # demand metric: total renters across county (after adjustments)
        county_renter_now = renters.sum()
        county_units_now = total_units.sum() + county_new_units_external

        # compute percent changes relative to previous step for demand and supply
        # For first iteration, assume previous values from baseline bg
        if year == 1:
            prev_county_renter = (base_bg_gdf.set_index("GEOID")["B25003_003E"].astype(float).fillna(0)).sum()
            prev_county_units = (base_bg_gdf.set_index("GEOID")["B25003_001E"].astype(float).replace(0, np.nan).fillna(1)).sum()
        else:
            prev_county_renter = county_renter_prev
            prev_county_units = county_units_prev

        demand_change_pct = (county_renter_now - prev_county_renter) / max(prev_county_renter, 1)
        supply_change_pct = (county_units_now - prev_county_units) / max(prev_county_units, 1)

        # rent growth rule (simple)
        rent_growth_pct_county = rent_elasticity * (demand_change_pct - supply_change_pct) - vacancy_dampener * ( (vacancy.mean() - vacancy.mean()) if False else 0.0 )
        # note: vacancy_dampener placeholder; we don't compute vacancy mean change here - you can extend

        # distribute county rent growth back to BG-level proportionally to exposure (you could weight by rent_share)
        bg_rent_growth = pd.Series(rent_growth_pct_county, index=geoids)

        # 6) Update median_rent and percent_cost_burdened (approx) per BG
        median_rent = median_rent * (1 + bg_rent_growth)
        # income grows slowly -> reduces burden slightly if positive
        # approximate: percent_cost_burdened scales with rent/income. If income grows by g and rent by r, burden scales by (1+r)/(1+g)
        rent_to_income_scale = (1 + bg_rent_growth) / (1 + income_growth_rate)
        rent_burden = (rent_burden * rent_to_income_scale).clip(0, 1)

        # 7) Recompute rent_share and vacancy approximately
        # rent_share = renters / (total_occupied_units). Here approximate total_occupied = total_units * (1 - vacancy)
        occupied_units = total_units * (1 - vacancy)
        # avoid divide by zero
        rent_share = (renters / occupied_units).fillna(0).clip(0, 1)

        # 8) Recompute inv_vacancy for displacement index and normalized fields
        inv_vac = 1 - vacancy
        # Build tmp DF for normalization
        tmp_df = pd.DataFrame({
            "rent_share": rent_share,
            "percent_cost_burdened": rent_burden,
            "poverty_rate": static_df["poverty_rate"],
            "snap_share": static_df["snap_share"],
            "unemployment_rate": static_df["unemployment_rate"],
            "senior_share": static_df["senior_share"],
            "inv_vacancy": inv_vac
        }, index=geoids).fillna(0)

        # Normalize (MinMax) across county for the year if requested
        if normalize_index_each_year:
            scaler_local = MinMaxScaler()
            normalized_local = pd.DataFrame(
                scaler_local.fit_transform(tmp_df),
                columns=tmp_df.columns,
                index=tmp_df.index
            )
        else:
            # if not normalizing each year, you could reuse a baseline scaler (not implemented here)
            scaler_local = MinMaxScaler()
            normalized_local = pd.DataFrame(
                scaler_local.fit_transform(tmp_df),
                columns=tmp_df.columns,
                index=tmp_df.index
            )

        # compute base_displacement_index per your weights
        base_idx = sum(normalized_local[f] * weights_local.get(f, 1) for f in normalized_local.columns) / total_weight_local

        # add binary flags — we reuse the ones from base_bg_gdf where available
        black_decl = bg.get("black_decline", pd.Series(0, index=geoids)).astype(float).fillna(0)
        lat_decl = bg.get("latino_decline", pd.Series(0, index=geoids)).astype(float).fillna(0)
        rapid_rent_inc = bg.get("rapid_rent_increase", pd.Series(0, index=geoids)).astype(float).fillna(0)

        displacement_risk = base_idx + 0.5 * black_decl + 0.5 * lat_decl + 0.5 * rapid_rent_inc
        # re-normalize displacement_risk across BGs to 0-1 for this year
        displacement_risk = MinMaxScaler().fit_transform(displacement_risk.values.reshape(-1, 1)).flatten()
        displacement_risk = pd.Series(displacement_risk, index=geoids)

        # store year snapshot
        rec = pd.DataFrame({
            "GEOID": geoids,
            "year": year,
            "renters": renters,
            "total_units": total_units,
            "median_rent": median_rent,
            "percent_cost_burdened": rent_burden,
            "rental_vacancy_rate": vacancy,
            "rent_share": rent_share,
            "displacement_risk": displacement_risk
        }).reset_index(drop=True)
        records.append(rec)

        # update prev county totals
        county_renter_prev = county_renter_now
        county_units_prev = county_units_now

        # check equilibrium: mean absolute change in displacement risk vs previous year
        mean_abs_change = np.mean(np.abs(displacement_risk.values - prev_disp_risk.values))
        prev_disp_risk = displacement_risk.copy()

        if verbose:
            print(f"Year {year}: mean abs displacement risk change = {mean_abs_change:.6f}")

        if mean_abs_change < stop_tol:
            if verbose:
                print(f"Equilibrium reached at year {year} (mean abs change {mean_abs_change:.6f} < tol {stop_tol})")
            break

    # concatenate records into a long dataframe
    timeseries_df = pd.concat(records, ignore_index=True)
    # join geometry from base_bg_gdf for mapping convenience
    geom = base_bg_gdf.set_index("GEOID")[["geometry"]]
    timeseries_df = timeseries_df.merge(geom, left_on="GEOID", right_index=True, how="left")
    return timeseries_df


import time

def safe_download_acs(vintage, max_retries=3, delay=3):
    """Download ACS data with retries to avoid hard script failure."""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt}: pulling ACS {vintage}")
            data = ced.download(
                dataset=ACS5,
                vintage=vintage,
                download_variables=constants.bg_vars,
                state=states.NC,
                county=['077'],
                block_group='*',
                with_geometry=True,
            )
            return data
        except Exception as e:
            print(f"⚠️ Error pulling ACS {vintage}: {e}")
            if attempt < max_retries:
                print(f"Retrying in {delay} sec...")
                time.sleep(delay)
            else:
                print(f"❌ Failed after {max_retries} attempts — skipping {vintage}")
                return None


def safe_download_acs_tract(vintage, max_retries=3, delay=3):
    """
    Download ACS tract-level data with retries to avoid hard script failure.
    
    Returns a GeoDataFrame with tract geometries and requested variables.
    """
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt}: pulling ACS {vintage} (tract level)")
            data = ced.download(
                dataset=ACS5,
                vintage=vintage,
                download_variables=constants.bg_vars,  # you can rename to tract_vars if needed
                state=states.NC,
                county=['077'],         # Granville County
                tract='*',              # <- tract-level download
                with_geometry=True,
            )
            return data
        except Exception as e:
            print(f"⚠️ Error pulling ACS {vintage}: {e}")
            if attempt < max_retries:
                print(f"Retrying in {delay} sec...")
                time.sleep(delay)
            else:
                print(f"❌ Failed after {max_retries} attempts — skipping {vintage}")
                return None
