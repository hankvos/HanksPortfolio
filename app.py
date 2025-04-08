import os
import io
import logging
import zipfile
import re
from datetime import datetime
from collections import Counter
from functools import lru_cache
import sys
import time
import psutil

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from flask import Flask, render_template, request, jsonify, send_from_directory
import folium
from folium.plugins import HeatMap

app = Flask(__name__)

# Configure logging with explicit handlers
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
logger.handlers = [handler]
logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# [REGION_POSTCODE_LIST, ALLOWED_POSTCODES, POSTCODE_COORDS, REGION_CENTERS, SUBURB_COORDS unchanged]

# Global variables
df = None
data_loaded = False
last_health_status = None

@app.template_filter('currency')
def currency_filter(value):
    if value is None or pd.isna(value):
        return "N/A"
    return "${:,.0f}".format(value)

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Memory usage: RSS={mem_info.rss / 1024**2:.2f} MB, VMS={mem_info.vms / 1024**2:.2f} MB")

def load_property_data():
    global df, data_loaded
    if data_loaded:
        logger.debug("Data already loaded, returning cached DataFrame")
        return df
    
    start_time = time.time()
    logger.info("Loading property data...")
    log_memory_usage()
    zip_file = "2025.zip"
    if not os.path.exists(zip_file):
        logger.error(f"ZIP file {zip_file} not found in the directory.")
        df = pd.DataFrame()
        data_loaded = True
        logger.info("Set data_loaded (no data)")
        return df

    result_df = pd.DataFrame(columns=["Postcode", "Price", "Settlement Date", "Suburb", "Property Type", "Street", "StreetOnly", "Block Size", "Unit Number"])
    raw_property_types = Counter()
    allowed_types = {"RESIDENCE", "COMMERCIAL", "FARM", "VACANT LAND"}

    try:
        with zipfile.ZipFile(zip_file, 'r') as outer_zip:
            nested_zips = [f for f in outer_zip.namelist() if f.endswith('.zip')]
            for nested_zip_name in sorted(nested_zips, reverse=True):
                try:
                    with outer_zip.open(nested_zip_name) as nested_zip_file:
                        with zipfile.ZipFile(io.BytesIO(nested_zip_file.read())) as nested_zip:
                            dat_files = [f for f in nested_zip.namelist() if f.endswith('.DAT')]
                            for dat_file in dat_files:
                                try:
                                    with nested_zip.open(dat_file) as f:
                                        lines = f.read().decode('latin1').splitlines()
                                        parsed_rows = []
                                        unit_numbers = {}
                                        for line in lines:
                                            row = line.split(';')
                                            if row[0] == 'C' and len(row) > 5 and row[5]:
                                                unit_numbers[row[2]] = row[5]
                                            if row[0] == 'B' and len(row) > 18:
                                                raw_property_types[row[18]] += 1
                                                if row[18] in allowed_types:
                                                    parsed_rows.append(row)
                                        temp_df = pd.DataFrame(parsed_rows)
                                        if not temp_df.empty:
                                            temp_df = temp_df.rename(columns={
                                                7: "House Number",
                                                8: "StreetOnly",
                                                9: "Suburb",
                                                10: "Postcode",
                                                11: "Block Size",
                                                14: "Settlement Date",
                                                15: "Price",
                                                18: "Property Type",
                                                2: "Property ID"
                                            })
                                            temp_df["Unit Number"] = temp_df["Property ID"].map(unit_numbers).fillna("")
                                            temp_df["Street"] = temp_df["House Number"] + " " + temp_df["StreetOnly"]
                                            temp_df["Property Type"] = temp_df["Property Type"].replace("RESIDENCE", "HOUSE")
                                            temp_df["Property Type"] = temp_df.apply(
                                                lambda row: "UNIT" if (
                                                    row["Property Type"] == "HOUSE" and 
                                                    row["Unit Number"] and 
                                                    re.match(r'^\d+[A-Za-z]?$', row["Unit Number"].strip())
                                                ) else row["Property Type"],
                                                axis=1
                                            )
                                            temp_df = temp_df[["Postcode", "Price", "Settlement Date", "Suburb", "Property Type", "Street", "StreetOnly", "Block Size", "Unit Number"]]
                                            temp_df["Postcode"] = temp_df["Postcode"].astype(str)
                                            temp_df["Price"] = pd.to_numeric(temp_df["Price"], errors='coerce', downcast='float')
                                            temp_df["Block Size"] = pd.to_numeric(temp_df["Block Size"], errors='coerce', downcast='float')
                                            temp_df["Settlement Date"] = pd.to_datetime(temp_df["Settlement Date"], format='%Y%m%d', errors='coerce')
                                            temp_df = temp_df[temp_df["Settlement Date"].dt.year >= 2024]
                                            temp_df["Settlement Date"] = temp_df["Settlement Date"].dt.strftime('%d/%m/%Y')
                                            temp_df = temp_df[temp_df["Postcode"].isin(ALLOWED_POSTCODES)]
                                            if not temp_df.empty and not temp_df.isna().all().all():
                                                result_df = pd.concat([result_df, temp_df], ignore_index=True)
                                except Exception as e:
                                    logger.error(f"Error reading {dat_file} in {nested_zip_name}: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"Error processing nested ZIP {nested_zip_name} in {zip_file}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error opening {zip_file}: {e}", exc_info=True)

    if result_df.empty:
        logger.error("No valid DAT files processed or no data matches filters.")
    else:
        # Log Johns River sales after loading
        johns_river_sales = result_df[result_df["Suburb"] == "JOHNS RIVER"][["Price", "Settlement Date", "Street"]].to_dict('records')
        logger.info(f"Johns River raw sales data: {johns_river_sales}")
        logger.info(f"Processed {len(result_df)} records from 2025.zip")
        logger.info(f"Raw Property Type counts (field 18): {dict(raw_property_types)}")
        logger.info(f"Processed Property Type counts: {result_df['Property Type'].value_counts().to_dict()}")
        logger.info(f"Loaded {len(result_df)} records into DataFrame in {time.time() - start_time:.2f} seconds")

    df = result_df
    data_loaded = True
    logger.info("Set data_loaded (data loaded)")
    logger.info("Data load completed successfully")
    log_memory_usage()
    return df

# [generate_region_median_chart, generate_postcode_median_chart, generate_suburb_median_chart, 
#  generate_heatmap_cached, generate_charts_cached, pre_generate_charts, /health, / unchanged]

@app.route('/', methods=["GET", "POST"])
def index():
    start_time = time.time()
    logger.info("Starting index route")
    log_memory_usage()
    
    if not data_loaded:
        logger.info("Data not yet loaded, showing loading page")
        return render_template("loading.html")
    
    logger.info("Data loaded, proceeding with index route")
    try:
        df = load_property_data()
        logger.info(f"DataFrame loaded with {len(df)} rows")
        # ... (rest of index route unchanged) ...
        logger.info("Rendering index.html")
        response = render_template("index.html",
                                  data_source="NSW Valuer General Data",
                                  regions=sorted(REGION_POSTCODE_LIST.keys()),
                                  postcodes=unique_postcodes,
                                  suburbs=unique_suburbs,
                                  property_types=["ALL", "HOUSE", "UNIT", "COMMERCIAL", "FARM", "VACANT LAND"],
                                  selected_region=selected_region,
                                  selected_postcode=selected_postcode,
                                  selected_suburb=selected_suburb,
                                  selected_property_type=selected_property_type,
                                  sort_by=sort_by,
                                  heatmap_path=heatmap_path,
                                  region_median_chart_path=region_median_chart_path,
                                  postcode_median_chart_path=postcode_median_chart_path,
                                  suburb_median_chart_path=suburb_median_chart_path,
                                  median_chart_path=median_chart_path,
                                  price_hist_path=price_hist_path,
                                  region_timeline_path=region_timeline_path,
                                  postcode_timeline_path=postcode_timeline_path,
                                  suburb_timeline_path=suburb_timeline_path,
                                  properties=properties,
                                  avg_price=avg_price,
                                  stats=stats)
        elapsed_time = time.time() - start_time
        logger.info(f"Index route completed in {elapsed_time:.2f} seconds")
        log_memory_usage()
        return response
    except Exception as e:
        logger.error(f"Error in index route: {e}", exc_info=True)
        return "An error occurred on the server", 500

@app.route('/hot_suburbs')
def hot_suburbs():
    start_time = time.time()
    logger.info("Starting hot_suburbs route")
    log_memory_usage()
    
    if not data_loaded:
        logger.info("Data not yet loaded, showing loading page")
        return render_template("loading.html")
    
    logger.info("Data loaded, proceeding with hot_suburbs route")
    try:
        df = load_property_data()
        # Log Johns River sales before median calculation
        johns_river_sales = df[df["Suburb"] == "JOHNS RIVER"]["Price"].tolist()
        logger.info(f"Johns River sales for median calculation: {johns_river_sales}")
        johns_river_median = df[df["Suburb"] == "JOHNS RIVER"]["Price"].median()
        logger.info(f"Johns River calculated median price: {johns_river_median}")
        
        # Group by suburb and calculate median price
        suburb_medians = df.groupby(['Suburb', 'Postcode'])['Price'].median().reset_index()
        
        # Map postcodes to regions
        postcode_to_region = {}
        for region, postcodes in REGION_POSTCODE_LIST.items():
            for pc in postcodes:
                postcode_to_region[pc] = region
        
        suburb_medians['Region'] = suburb_medians['Postcode'].map(postcode_to_region)
        
        # Filter for median price < 9M
        hot_suburbs_df = suburb_medians[suburb_medians['Price'] < 9000000]
        
        # Prepare data for template
        hot_suburbs = [
            {
                'suburb': row['Suburb'],
                'postcode': row['Postcode'],
                'region': row['Region'],
                'median_price': row['Price']
            }
            for _, row in hot_suburbs_df.iterrows()
        ]
        
        response = render_template("hot_suburbs.html",
                                  data_source="NSW Valuer General Data",
                                  hot_suburbs=hot_suburbs)
        elapsed_time = time.time() - start_time
        logger.info(f"Hot_suburbs route completed in {elapsed_time:.2f} seconds")
        log_memory_usage()
        return response
    except Exception as e:
        logger.error(f"Error in hot_suburbs route: {e}", exc_info=True)
        return "An error occurred on the server", 500

# [/get_postcodes, /get_suburbs, /static/<path:filename>, startup code unchanged]

# Load data synchronously before Gunicorn starts
logger.info("Starting synchronous data load before Gunicorn...")
load_property_data()
logger.info("Synchronous data load completed.")
pre_generate_charts()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
else:
    gunicorn_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting gunicorn on port {port} with 1 worker")
    os.environ["GUNICORN_CMD_ARGS"] = f"--bind 0.0.0.0:{port} --workers 1"