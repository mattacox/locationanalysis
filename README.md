# Oxford, NC Location Analysis Maps

This project visualizes socioeconomic and housing indicators over time for census block groups in and around **Oxford, North Carolina**. It uses American Community Survey (ACS) data (2017–2023) and USDA Food Access data to build interactive time-enabled maps using [Folium](https://python-visualization.github.io/folium/).

## 📊 Indicators Included

Each map is animated with a time slider and covers the following indicators:

- **Poverty Rate**  
  The share of the population living below the federal poverty line.  
  *Typical range:* 5–30%; *High risk:* above 20%.

- **Median Household Income**  
  The median annual income for households, indicating overall economic well-being.  
  *Typical range:* $30,000–$70,000; *Low income:* below $40,000.

- **Unemployment Rate**  
  The proportion of the labor force currently unemployed and seeking work.  
  *Typical range:* 3–12%; *High risk:* above 10%.

- **SNAP Participation Share**  
  The share of households receiving Supplemental Nutrition Assistance Program benefits, reflecting economic hardship.  
  *Typical range:* 5–30%; *High risk:* above 20%.

- **Renter Share**  
  The proportion of occupied housing units that are renter-occupied, indicating housing tenure patterns.  
  *Typical range:* 20–60%; varies widely by neighborhood.

- **Rent Burden (≥30%)**  
  Percentage of renters spending 30% or more of household income on rent, signaling housing affordability stress.  
  *Typical range:* 25–50%; *High risk:* above 30%.

- **Senior Population Share**  
  The share of residents aged 65 and older, indicating age demographics and potential service needs.  
  *Typical range:* 10–25%; *Senior-heavy:* above 20%.

- **Vacancy Rate**  
  The share of housing units that are vacant, which can reflect housing market conditions or disinvestment.  
  *Typical range:* 5–15%; *High vacancy:* above 10%.

- **Displacement Risk Index**  
  A composite indicator combining rent burden, poverty, SNAP participation, unemployment, vacancy, and senior share to identify neighborhoods at higher risk of displacement pressures.  
  *Typical range:* 0 (low risk) to 1 (high risk); *High risk:* above 0.6.

  ℹ️ *About Normalization:*  
  The displacement risk index is normalized using all block groups in Granville County to create a consistent reference frame. This approach allows block groups in Oxford to be evaluated in context, highlighting which areas stand out not just locally, but county-wide.

  An Oxford-only normalization would compress the score range and make internal differences appear larger, but it would limit comparability and may understate broader risk patterns. Oxford’s 12 block groups are small in number but internally diverse—normalizing at the county level supports clearer regional policy targeting without distorting intra-city variation.

- **Median Gross Rent**  
  Median monthly rent paid by renters, in dollars.

- **Food Desert Status**  
  Whether the census tract qualifies as a **low-income, low-access (LILA)** food desert, as defined by the USDA Food Access Research Atlas.  
  *Criteria:* Low-income census tracts where a significant portion of residents live more than 1 mile (urban) or 10 miles (rural) from the nearest supermarket.  
  *Why it matters:* Lack of access to healthy, affordable food intersects with affordability, transportation, and health — and can compound displacement pressure in vulnerable communities.

## 🗺️ Interactive Map Viewer

View the maps [here](https://mattacox.github.io/locationanalysis/):

- Each indicator has its own HTML map file.
- Use the time slider to see changes from 2017 to 2023.

## 🔧 Built With

- [`censusdis`](https://pypi.org/project/censusdis/) – ACS data via the Census API
- [`pandas`](https://pandas.pydata.org/) and [`geopandas`](https://geopandas.org/) – data wrangling and spatial analysis
- [`folium`](https://python-visualization.github.io/folium/) – Leaflet.js-powered interactive maps in Python
- [`Leaflet.TimeDimension`](https://github.com/socib/Leaflet.TimeDimension) – animated time slider for map visualization
- [`scikit-learn`](https://scikit-learn.org/stable/) – logistic regression and predictive modeling for brand suitability

## 📁 Folder Structure

📂html/\
├── poverty_rate_timeslider.html\
├── median_income_timeslider.html\
└── ...\
📂data/\
├── Food Access Research Atlas.csv\
└── tl_2024_37_bg.shp (plus supporting files)\
📄 index.html ← links to all map pages\
📄 README.md

> ⚠️ The `data/` folder is **not included** in this repository. See below.

## 📦 Required Data

To reproduce this project locally, you'll need:

- **USDA Food Access Research Atlas CSV**  
  📥 Download from [https://www.ers.usda.gov/data-products/food-access-research-atlas/](https://www.ers.usda.gov/data-products/food-access-research-atlas/)

- **2024 Census Block Group Shapefile for North Carolina**  
  📥 Download from [TIGER/Line Shapefiles](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)

Place both in a `data/` directory:

📂scripts/\
├── your_script.py\
└── 📂data/\
        ├── Food Access Research Atlas.csv\
        └── tl_2024_37_bg.shp (and related files)

## 🚀 How to Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/mattacox/locationanalysis.git
   cd locationanalysis
   ```

2. Set up your Python environment and install dependencies.

3. Place downloaded data in the data/ folder.

4. Run the Python script to generate maps.

5. Open index.html in a browser to explore the maps.

## 🛒 Brand Placement Suitability Model

This script estimates the likelihood of a specific retail brand locating in Oxford, NC or surrounding areas, based on logistic regression analysis of income, population density, land use, and labor market indicators at the census tract level.

### Features

### Features

- Predictive map showing probability of brand presence by tract
- Optimization to identify optimal conditions for attracting a brand
- Synthetic tract representing Oxford’s ETJ is included in model
- Outputs top 10 tracts for brand expansion (excluding existing locations)

### Example Usage

Run from the command line:

```bash
python brandteststable.py --brand "Dollar General"
```
## 💰 Tax Parcel Value & Change Analysis

This project includes a set of parcel-level **choropleth maps** showing changes in property value and taxation across Oxford, NC. These maps are based on **Granville County parcel shapefiles** joined with custom parcel-level tax data.

### Maps Included:

- **🟠 Value per Acre (VPA)**  
  Visualizes the assessed value of land per acre to highlight parcels that deliver the most tax value to the city. Useful for understanding where public investment yields the greatest return.  
  _Layer:_ `vpa_choropleth.html`

- **🔵 Percent Value Increase**  
  Shows how much each parcel’s value has increased since reassessment. Highlights areas experiencing the fastest growth or speculation.  
  _Layer:_ `tax_pct_increase_choropleth.html`

- **🟢 Tax Change (% Delta)**  
  Compares actual tax bills from prior years to current bills, showing the **percent change in taxes owed** at the parcel level.  
  _Layer:_ `tax_pct_delta_choropleth.html`

- **🔴🔵 Tax Change: Red vs. Blue**  
  A clear red-blue map where **red indicates a tax increase** and **blue indicates a tax decrease**, providing a quick visual of where fiscal burdens are shifting.  
  _Layer:_ `tax_waterfall_choropleth.html`

### Why It Matters:

- Helps residents and policymakers **track which parcels or neighborhoods are facing the steepest increases in tax burden**.
- Supports analysis of whether **tax increases align with rising land value**, or if they disproportionately affect certain property types or neighborhoods.
- Informs equitable reinvestment by showing where **value per acre** is highest — a metric often used in smart growth planning.

All maps are interactive and include **hover tooltips** with parcel IDs, acreage, owner names, and tax/value metrics.

📬 Contact
Maintained by Matthew Cox
GitHub: @mattacox
