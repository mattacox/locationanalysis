# Oxford, NC Location Analysis Maps

This project visualizes socioeconomic and housing indicators over time for census block groups in and around **Oxford, North Carolina**. It uses American Community Survey (ACS) data (2017–2023) and USDA Food Access data to build interactive time-enabled maps using [Folium](https://python-visualization.github.io/folium/).

## 📊 Indicators Included

Each map is animated with a time slider and covers the following indicators:

- **Poverty Rate**
- **Median Household Income**
- **Unemployment Rate**
- **SNAP Participation Share**
- **Renter Share**
- **Rent Burden (≥30%)**
- **Senior Population Share**
- **Displacement Risk Index** (composite of rent burden, poverty, SNAP, and more)

## 🗺️ Interactive Map Viewer

View the maps [here](https://mattacox.github.io/locationanalysis/):

- Each indicator has its own HTML map file.
- Use the time slider to see changes from 2017 to 2023.

## 🔧 Built With

- [`censusdis`](https://pypi.org/project/censusdis/) – ACS data via the Census API
- [`pandas`](https://pandas.pydata.org/) and [`geopandas`](https://geopandas.org/) – data wrangling and spatial analysis
- [`folium`](https://python-visualization.github.io/folium/) – Leaflet.js-powered interactive maps in Python
- [`Leaflet.TimeDimension`](https://github.com/socib/Leaflet.TimeDimension) – animated time slider

## 📁 Folder Structure

📂html/
├── poverty_rate_timeslider.html
├── median_income_timeslider.html
├── ...
📂data/
├── Food Access Research Atlas.csv
├── tl_2024_37_bg.shp (plus supporting files)
📄 index.html ← links to all map pages
📄 README.md


> ⚠️ The `data/` folder is **not included** in this repository. See below.

## 📦 Required Data

To reproduce this project locally, you'll need:

- **USDA Food Access Research Atlas CSV**  
  📥 Download from [https://www.ers.usda.gov/data-products/food-access-research-atlas/](https://www.ers.usda.gov/data-products/food-access-research-atlas/)

- **2024 Census Block Group Shapefile for North Carolina**  
  📥 Download from [TIGER/Line Shapefiles](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)

Place both in a `data/` directory:

scripts/
├── your_script.py
└── data/
├── Food Access Research Atlas.csv
└── tl_2024_37_bg.shp (and related files)

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/mattacox/locationanalysis.git
   cd locationanalysis
2. Set up your Python environment and install dependencies.

3. Place downloaded data in the data/ folder.

4. Run the Python script to generate maps.

5. Open index.html in a browser to explore the maps.


📬 Contact
Maintained by Matthew Cox
GitHub: @mattacox