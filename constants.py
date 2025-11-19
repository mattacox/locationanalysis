#years and all years for ACS5 data pull
years = [2017, 
         2018, 
         2019, 
         2021, 
         2022, 
         2023, 
        #  2024,
         ]
all_years = []

# --- ACS Variable List for Census API---
bg_vars = [
    "B19013_001E", 
    "B17021_002E", 
    "B17021_001E", 
    "B23025_005E", 
    "B23025_003E",
    "B15003_001E", 
    "B15003_017E", 
    "B15003_022E", 
    "B25064_001E",
    "B25070_003E", 
    "B25070_004E", 
    "B25070_005E", 
    "B25070_006E",
    "B25070_007E", 
    "B25070_008E", 
    "B25070_009E", 
    "B25070_010E", 
    "B25070_001E",
    "B25002_003E", 
    "B25002_001E", 
    "B25003_003E", 
    "B25003_001E",
    "B01001_001E", 
    "B01001_020E", 
    "B01001_021E", 
    "B01001_022E",
    "B01001_023E", 
    "B01001_024E", 
    "B01001_025E", 
    "B01001_044E",
    "B01001_045E", 
    "B01001_046E", 
    "B01001_047E", 
    "B01001_048E", 
    "B01001_049E",
    "B22010_001E", 
    "B22010_002E",
    "B08201_001E", 
    "B08201_002E",
    "B03002_001E", 
    "B03002_003E", 
    "B03002_004E", 
    "B03002_012E",
    "B25004_002E", # For rent
    "B25004_003E", #Rented, not occupied
    "B25004_004E", #For sale only
    "B08301_001E", #total workers
    "B08301_010E", #workers from home
]

# Senior Share for Displacement Index, etc.
senior_vars = [
    "B01001_020E", 
    "B01001_021E", 
    "B01001_022E", 
    "B01001_023E", 
    "B01001_024E", 
    "B01001_025E",
    "B01001_044E", 
    "B01001_045E", 
    "B01001_046E", 
    "B01001_047E", 
    "B01001_048E", 
    "B01001_049E",
]

#Risk Fields for Displacemenr Indexing

risk_fields = [
    "rent_share", "percent_cost_burdened", "poverty_rate", "snap_share",
    "unemployment_rate", "senior_share", "inv_vacancy"
]

# --- Recompute displacement risk using IZ-adjusted rent burden ---

iz_risk_fields = [
    "rent_share", "poverty_rate", "snap_share", "unemployment_rate",
    "senior_share", "inv_vacancy"
]


# --- Weighted composite index for Displacement Risk---

weights = {
    "rent_share": 1,
    "percent_cost_burdened": 2,
    "poverty_rate": 1,
    "snap_share": 1,
    "unemployment_rate": 1,
    "senior_share": 1,
    "inv_vacancy": 2,
}

total_weight = sum(weights.values())

# --- Indicators for individual dempgraphic Map setup ---
indicators = [
    "poverty_rate", "percent_cost_burdened", "unemployment_rate", "snap_share",
    "rent_share", "senior_share", "displacement_risk", "rental_vacancy_rate",
    "median_income", "median_rent", "black_share", "white_share", "latino_share", "pct_work_from_home"

]

# --- ranges for cloropleth mapping ---

indicator_ranges = {
    "poverty_rate": (0, 0.5),
    "percent_cost_burdened": (0, 0.6),
    "unemployment_rate": (0, 0.25),
    "snap_share": (0, 0.5),
    "rent_share": (0, 0.8),
    "senior_share": (0, 0.5),
    "rental_vacancy_rate": (0, 0.4),
    "displacement_risk": (0, 1.0),
    "median_rent": (300, 1500),
    "black_share": (0, 1),
    "white_share": (0, 1),
    "latino_share": (0, 1),
    "median_income": (30000, 200000),
}