import os
import io
import logging
import zipfile
import re
from datetime import datetime
from collections import Counter
import sys
import time
import psutil
import threading
import multiprocessing
from functools import wraps

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from flask import Flask, render_template, request, jsonify, send_from_directory
import folium
from folium.plugins import MarkerCluster

app = Flask(__name__)

# Configure logging with immediate flushing
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
handler.setStream(sys.stdout)
logger.handlers = [handler]
logger.setLevel(logging.INFO)

# Region and postcode mappings (unchanged, truncated for brevity)
REGION_POSTCODE_LIST = {
    "Central Coast": ["2083", "2250", "2251", "2256", "2257", "2258", "2259", "2260", "2261", "2262", "2263", "2775"],
    # ... other regions ...
}
ALLOWED_POSTCODES = {pc for region in REGION_POSTCODE_LIST.values() for pc in region}
POSTCODE_COORDS = {
    "2250": [-33.28, 151.41], "2251": [-33.31, 151.42], # ... other postcodes ...
}
REGION_CENTERS = {}
for region_name, postcodes in REGION_POSTCODE_LIST.items():
    coords = [POSTCODE_COORDS.get(pc) for pc in postcodes if pc in POSTCODE_COORDS]
    coords = [c for c in coords if c]
    if coords:
        REGION_CENTERS[region_name] = [
            sum(c[0] for c in coords) / len(coords),
            sum(c[1] for c in coords) / len(coords)
        ]
SUBURB_COORDS = {
    "2262": {"BLUE HAVEN": [-33.36, 151.43], "BUDGEWOI": [-33.23, 151.56], "DOYALSON": [-33.20, 151.52], "SAN REMO": [-33.21, 151.51]},
    "2443": {"JOHNS RIVER": [-31.73, 152.70]}
}

# Global variables with process-safe state
df = None
data_loaded = False
NATIONAL_MEDIAN = None
lock = threading.Lock()
startup_complete_file = "/tmp/startup_complete.flag"  # Process-safe flag

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
    global df, data_loaded, NATIONAL_MEDIAN
    with lock:
        if data_loaded:
            return df
    
    start_time = time.time()
    logger.info("Loading property data...")
    log_memory_usage()
    zip_files = ["2024.zip", "2025.zip"]
    result_df = pd.DataFrame(columns=["Postcode", "Price", "Settlement Date", "Suburb", "Property Type", "Street", "StreetOnly", "Block Size", "Unit Number"])
    raw_property_types = Counter()
    allowed_types = {"RESIDENCE", "COMMERCIAL", "FARM", "VACANT LAND"}

    for zip_file in zip_files:
        if not os.path.exists(zip_file):
            logger.error(f"ZIP file {zip_file} not found in the directory.")
            continue
        with zipfile.ZipFile(zip_file, 'r') as outer_zip:
            nested_zips = [f for f in outer_zip.namelist() if f.endswith('.zip')]
            for nested_zip_name in sorted(nested_zips, reverse=True):
                if zip_file == "2024.zip" and not nested_zip_name.startswith("202412"):
                    continue
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
                                                7: "House Number", 8: "StreetOnly", 9: "Suburb", 10: "Postcode",
                                                11: "Block Size", 14: "Settlement Date", 15: "Price", 18: "Property Type",
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
                                            temp_df["Block Size"] = pd.to_numeric(temp_df["Block Size"], errors='coerce', downcast='float').round(0)
                                            temp_df["Settlement Date"] = pd.to_datetime(temp_df["Settlement Date"], format='%Y%m%d', errors='coerce')
                                            temp_df = temp_df[temp_df["Settlement Date"].dt.year >= 2024]
                                            temp_df["Settlement Date"] = temp_df["Settlement Date"].dt.strftime('%d/%m/%Y')
                                            temp_df = temp_df[temp_df["Postcode"].isin(ALLOWED_POSTCODES)]
                                            temp_df["Price"] = temp_df["Price"].clip(lower=10000)
                                            if not temp_df.empty and not temp_df.isna().all().all():
                                                result_df = pd.concat([result_df, temp_df], ignore_index=True)
                                except Exception as e:
                                    logger.error(f"Error reading {dat_file} in {nested_zip_name}: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"Error processing nested ZIP {nested_zip_name} in {zip_file}: {e}", exc_info=True)

    if result_df.empty:
        logger.error("No valid DAT files processed or no data matches filters.")
    else:
        NATIONAL_MEDIAN = result_df["Price"].median()
        logger.info(f"National median price calculated: ${NATIONAL_MEDIAN:,.0f}")
        logger.info(f"Processed {len(result_df)} records from {zip_files}")
        logger.info(f"Raw Property Type counts (field 18): {dict(raw_property_types)}")
        logger.info(f"Processed Property Type counts: {result_df['Property Type'].value_counts().to_dict()}")
        logger.info(f"Loaded {len(result_df)} records into DataFrame in {time.time() - start_time:.2f} seconds")

    with lock:
        df = result_df
        data_loaded = True
    logger.info("Data load completed successfully")
    log_memory_usage()
    return df

def generate_heatmap():
    with lock:
        df_local = df.copy() if df is not None else load_property_data()
    logger.info("Generating heatmap...")
    m = folium.Map(zoom_start=6)
    marker_cluster = MarkerCluster().add_to(m)
    coords = df_local[df_local["Price"] < 9_000_000].dropna(subset=["Price"]).sample(min(1000, len(df_local)))
    all_coords = []
    
    for _, row in coords.iterrows():
        latlon = SUBURB_COORDS.get(row["Postcode"], {}).get(row["Suburb"])
        if not latlon:
            latlon = POSTCODE_COORDS.get(row["Postcode"])
        if not latlon:
            region = next((r for r, pcs in REGION_POSTCODE_LIST.items() if row["Postcode"] in pcs), None)
            latlon = REGION_CENTERS.get(region) if region else [-32.5, 152.5]
        folium.Marker(
            location=latlon,
            popup=f"{row['Street']}, {row['Suburb']} {row['Postcode']} - ${row['Price']:,.0f}"
        ).add_to(marker_cluster)
        all_coords.append(latlon)
    
    if all_coords:
        lats = [c[0] for c in all_coords]
        lons = [c[1] for c in all_coords]
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    
    heatmap_path = "static/heatmap.html"
    os.makedirs("static", exist_ok=True)
    m.save(heatmap_path)
    logger.info(f"Heatmap generated at {heatmap_path}")
    return heatmap_path

def generate_region_median_chart(selected_region=None, selected_postcode=None):
    with lock:
        df_local = df.copy() if df is not None else load_property_data()
    logger.info("Generating region median chart...")
    if selected_postcode:
        median_data = df_local[df_local["Postcode"] == selected_postcode].groupby('Suburb')['Price'].median().sort_values()
        title = f"Median House Prices by Suburb (Postcode {selected_postcode})"
        x_label = "Suburb"
    elif selected_region:
        postcodes = REGION_POSTCODE_LIST.get(selected_region, [])
        median_data = df_local[df_local["Postcode"].isin(postcodes)].groupby('Postcode')['Price'].median().sort_values()
        title = f"Median House Prices by Postcode ({selected_region})"
        x_label = "Postcode"
    else:
        region_medians = df_local.groupby('Postcode')['Price'].median().reset_index()
        postcode_to_region = {pc: region for region, pcs in REGION_POSTCODE_LIST.items() for pc in pcs}
        region_medians['Region'] = region_medians['Postcode'].map(postcode_to_region)
        median_data = region_medians.groupby('Region')['Price'].median().sort_values()
        title = "Median House Prices by Region"
        x_label = "Region"
    
    plt.figure(figsize=(12, 6))
    median_data.plot(kind='bar', color='skyblue')
    plt.title(title)
    plt.ylabel('Median Price ($)')
    plt.xlabel(x_label)
    for i, v in enumerate(median_data):
        plt.text(i, v, f"${v:,.0f}", ha='center', va='bottom')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    chart_path = "static/region_median_chart.png"
    os.makedirs("static", exist_ok=True)
    plt.savefig(chart_path)
    plt.close()
    logger.info(f"Region median chart generated at {chart_path}")
    return chart_path

def pre_generate_charts():
    logger.info("Pre-generating charts in background...")
    if not os.path.exists("static/heatmap.html"):
        generate_heatmap()
    if not os.path.exists("static/region_median_chart.png"):
        generate_region_median_chart()

def startup_tasks():
    load_property_data()
    pre_generate_charts()
    with open(startup_complete_file, 'w') as f:
        f.write("1")  # Write flag file
    logger.info("Startup tasks completed")

def is_startup_complete():
    return os.path.exists(startup_complete_file)

@app.route('/health')
def health_check():
    status = "OK" if is_startup_complete() else "LOADING"
    logger.info(f"Health check: {status}")
    return status, 200

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    sys.stdout.flush()
    return "An error occurred on the server", 500

def timeout_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{func.__name__} completed in {elapsed:.2f} seconds")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.2f} seconds: {str(e)}", exc_info=True)
            sys.stdout.flush()
            raise
    return wrapper

@app.route('/', methods=["GET", "POST"])
@timeout_handler
def index():
    global NATIONAL_MEDIAN
    logger.info("Entering index route")
    sys.stdout.flush()
    
    logger.info(f"Startup complete: {is_startup_complete()}")
    sys.stdout.flush()
    if not is_startup_complete():
        logger.info("Data not yet loaded, showing loading page")
        sys.stdout.flush()
        return render_template("loading.html")
    
    try:
        logger.info("Acquiring lock and copying DataFrame")
        sys.stdout.flush()
        with lock:
            if df is None:
                logger.error("DataFrame is None after startup")
                sys.stdout.flush()
                raise ValueError("DataFrame not initialized")
            df_local = df.copy()
        logger.info(f"DataFrame copied with {len(df_local)} rows")
        sys.stdout.flush()
        
        logger.info("Fetching unique postcodes and suburbs")
        sys.stdout.flush()
        unique_postcodes = sorted(df_local["Postcode"].unique())
        unique_suburbs = sorted(df_local["Suburb"].unique())
        
        logger.info("Processing form data")
        sys.stdout.flush()
        selected_region = request.form.get("region", "")
        selected_postcode = request.form.get("postcode", "")
        selected_suburb = request.form.get("suburb", "")
        selected_property_type = request.form.get("property_type", "HOUSE")
        sort_by = request.form.get("sort_by", "Street")
        
        logger.info(f"Form data: region={selected_region}, postcode={selected_postcode}, suburb={selected_suburb}, property_type={selected_property_type}, sort_by={sort_by}")
        sys.stdout.flush()
        
        display_postcode = selected_postcode if selected_region else ""
        display_suburb = selected_suburb if selected_region and selected_postcode else ""
        
        logger.info("Generating chart")
        sys.stdout.flush()
        chart_path = generate_region_median_chart(selected_region, selected_postcode)
        
        logger.info("Filtering DataFrame")
        sys.stdout.flush()
        filtered_df = df_local.copy()
        properties = []
        median_price = 0
        
        if selected_region and not selected_postcode and not selected_suburb:
            filtered_df = filtered_df[filtered_df["Postcode"].isin(REGION_POSTCODE_LIST.get(selected_region, []))]
        elif selected_postcode and not selected_suburb:
            filtered_df = filtered_df[filtered_df["Postcode"] == selected_postcode]
        elif selected_suburb:
            filtered_df = filtered_df[filtered_df["Suburb"] == selected_suburb]
        
        if selected_property_type != "ALL":
            filtered_df = filtered_df[filtered_df["Property Type"] == selected_property_type]
        
        if not filtered_df.empty:
            if sort_by == "Street":
                properties = filtered_df.sort_values(by="StreetOnly").to_dict('records')
            else:
                properties = filtered_df.sort_values(by=sort_by).to_dict('records')
            median_price = filtered_df["Price"].median()
        
        logger.info(f"Filtered properties: {len(properties)} records")
        sys.stdout.flush()
        
        logger.info("Checking static files")
        sys.stdout.flush()
        heatmap_path = "static/heatmap.html" if os.path.exists("static/heatmap.html") else None
        region_median_chart_path = "static/region_median_chart.png" if os.path.exists("static/region_median_chart.png") else None
        logger.info(f"Heatmap path: {heatmap_path}, Chart path: {region_median_chart_path}")
        sys.stdout.flush()
        
        logger.info("Rendering index.html")
        sys.stdout.flush()
        response = render_template("index.html",
                                  data_source="NSW Valuer General Data",
                                  regions=sorted(REGION_POSTCODE_LIST.keys()),
                                  postcodes=unique_postcodes,
                                  suburbs=unique_suburbs,
                                  property_types=["ALL", "HOUSE", "UNIT", "COMMERCIAL", "FARM", "VACANT LAND"],
                                  selected_region=selected_region,
                                  selected_postcode=display_postcode,
                                  selected_suburb=display_suburb,
                                  selected_property_type=selected_property_type,
                                  sort_by=sort_by,
                                  heatmap_path=heatmap_path,
                                  region_median_chart_path=region_median_chart_path,
                                  properties=properties,
                                  median_price=median_price,
                                  national_median=NATIONAL_MEDIAN,
                                  display_suburb=selected_suburb if selected_suburb else None)
        logger.info("Index route rendering complete")
        sys.stdout.flush()
        log_memory_usage()
        return response
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}", exc_info=True)
        sys.stdout.flush()
        raise

@app.route('/hot_suburbs', methods=["GET", "POST"])
@timeout_handler
def hot_suburbs():
    global NATIONAL_MEDIAN
    logger.info("Starting hot_suburbs route")
    log_memory_usage()
    
    if not is_startup_complete():
        logger.info("Data not yet loaded, showing loading page")
        return render_template("loading.html")
    
    try:
        with lock:
            df_local = df.copy() if df is not None else load_property_data()
        suburb_medians = df_local.groupby(['Suburb', 'Postcode'])['Price'].median().reset_index()
        postcode_to_region = {pc: region for region, postcodes in REGION_POSTCODE_LIST.items() for pc in postcodes}
        suburb_medians['Region'] = suburb_medians['Postcode'].map(postcode_to_region)
        hot_suburbs_df = suburb_medians[suburb_medians['Price'] < NATIONAL_MEDIAN]
        
        sort_by = request.form.get("sort_by", "Suburb")
        if sort_by == "Suburb":
            hot_suburbs_df = hot_suburbs_df.sort_values(by='Suburb')
        elif sort_by == "Postcode":
            hot_suburbs_df = hot_suburbs_df.sort_values(by='Postcode')
        elif sort_by == "Region":
            hot_suburbs_df = hot_suburbs_df.sort_values(by='Region')
        elif sort_by == "Median Price":
            hot_suburbs_df = hot_suburbs_df.sort_values(by='Price')
        
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
                                  hot_suburbs=hot_suburbs,
                                  national_median=NATIONAL_MEDIAN,
                                  sort_by=sort_by)
        logger.info("Hot_suburbs route rendering complete")
        log_memory_usage()
        sys.stdout.flush()
        return response
    except Exception as e:
        logger.error(f"Error in hot_suburbs route: {str(e)}", exc_info=True)
        sys.stdout.flush()
        raise

@app.route('/get_postcodes')
def get_postcodes():
    try:
        if not is_startup_complete():
            return jsonify({"error": "Data still loading"}), 503
        region = request.args.get('region')
        postcodes = REGION_POSTCODE_LIST.get(region, [])
        return jsonify(postcodes)
    except Exception as e:
        logger.error(f"Error in get_postcodes: {str(e)}", exc_info=True)
        sys.stdout.flush()
        raise

@app.route('/get_suburbs')
def get_suburbs():
    try:
        if not is_startup_complete():
            return jsonify({"error": "Data still loading"}), 503
        with lock:
            df_local = df.copy() if df is not None else load_property_data()
        region = request.args.get('region')
        postcode = request.args.get('postcode')
        filtered_df = df_local[df_local["Postcode"].isin(REGION_POSTCODE_LIST.get(region, []))]
        if postcode:
            filtered_df = filtered_df[filtered_df["Postcode"] == postcode]
        suburbs = sorted(filtered_df["Suburb"].unique().tolist())
        return jsonify(suburbs)
    except Exception as e:
        logger.error(f"Error in get_suburbs: {str(e)}", exc_info=True)
        sys.stdout.flush()
        raise

@app.route('/static/<path:filename>')
def static_files(filename):
    try:
        return send_from_directory(app.static_folder, filename)
    except Exception as e:
        logger.error(f"Error serving static file {filename}: {str(e)}", exc_info=True)
        sys.stdout.flush()
        raise

# Start background loading
logger.info("Starting background data load...")
threading.Thread(target=startup_tasks, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
else:
    gunicorn_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting gunicorn on port {port} with 1 worker")
    os.environ["GUNICORN_CMD_ARGS"] = f"--bind 0.0.0.0:{port} --workers 1 --timeout 120"
    sys.stdout.flush()