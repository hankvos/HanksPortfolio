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
from folium.plugins import MarkerCluster

app = Flask(__name__)

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
logger.handlers = [handler]
logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Region and postcode mappings
REGION_POSTCODE_LIST = {
    "Central Coast": ["2083", "2250", "2251", "2256", "2257", "2258", "2259", "2260", "2261", "2262", "2263", "2775"],
    "Coffs Harbour - Grafton": ["2370", "2441", "2448", "2449", "2450", "2452", "2453", "2454", "2455", "2456", "2460", "2462", "2463", "2464", "2465", "2466", "2469"],
    "Hunter Valley excl Newcastle": ["2250", "2311", "2314", "2315", "2316", "2317", "2318", "2319", "2320", "2321", "2322", "2323", "2324", "2325", "2326", "2327", "2328", "2329", "2330", "2331", "2333", "2334", "2335", "2336", "2337", "2338", "2420", "2421", "2850"],
    "Newcastle and Lake Macquarie": ["2259", "2264", "2265", "2267", "2278", "2280", "2281", "2282", "2283", "2284", "2285", "2286", "2287", "2289", "2290", "2291", "2292", "2293", "2294", "2295", "2296", "2297", "2298", "2299", "2300", "2302", "2303", "2304", "2305", "2306", "2307", "2308", "2318", "2322", "2323"],
    "Mid North Coast": ["2312", "2324", "2415", "2420", "2422", "2423", "2424", "2425", "2426", "2427", "2428", "2429", "2430", "2431", "2439", "2440", "2441", "2443", "2444", "2445", "2446", "2447", "2448", "244 Parts", "2898"],
    "Richmond - Tweed": ["2469", "2470", "2471", "2472", "2473", "2474", "2475", "2476", "2477", "2478", "2479", "2480", "2481", "2482", "2483", "2484", "2485", "2486", "2487", "2488", "2489", "2490"]
}

ALLOWED_POSTCODES = {pc for region in REGION_POSTCODE_LIST.values() for pc in region}

POSTCODE_COORDS = {
    "2250": [-33.28, 151.41], "2251": [-33.31, 151.42], "2256": [-33.47, 151.32], "2257": [-33.49, 151.35],
    "2258": [-33.41, 151.37], "2259": [-33.22, 151.42], "2260": [-33.27, 151.46], "2261": [-33.33, 151.47],
    "2262": [-33.36, 151.43], "2263": [-33.39, 151.45], "2450": [-30.30, 153.12], "2452": [-30.36, 153.09],
    "2454": [-30.63, 152.97], "2455": [-30.71, 152.93], "2456": [-30.65, 152.91], "2320": [-32.73, 151.55],
    "2321": [-32.75, 151.61], "2325": [-32.58, 151.33], "2326": [-32.77, 151.48], "2327": [-32.79, 151.50],
    "2280": [-32.91, 151.62], "2281": [-32.88, 151.65], "2282": [-32.93, 151.66], "2283": [-32.86, 151.70],
    "2284": [-32.89, 151.60], "2285": [-32.94, 151.64], "2286": [-32.91, 151.58], "2287": [-32.92, 151.68],
    "2289": [-32.94, 151.73], "2290": [-32.92, 151.70], "2291": [-32.91, 151.75], "2430": [-31.65, 152.78],
    "2440": [-31.43, 152.91], "2441": [-31.48, 152.73], "2443": [-31.59, 152.82], "2444": [-31.36, 152.84],
    "2445": [-31.65, 152.84], "2446": [-31.68, 152.79], "2477": [-28.81, 153.28], "2478": [-28.86, 153.58],
    "2480": [-28.81, 153.44], "2481": [-28.67, 153.58], "2482": [-28.71, 153.52], "2483": [-28.76, 153.47]
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
    "2262": {
        "BLUE HAVEN": [-33.36, 151.43],
        "BUDGEWOI": [-33.23, 151.56],
        "DOYALSON": [-33.20, 151.52],
        "SAN REMO": [-33.21, 151.51]
    },
    "2443": {
        "JOHNS RIVER": [-31.73, 152.70]
    }
}

# Global variables
df = None
data_loaded = False
last_health_status = None
NATIONAL_MEDIAN = None

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
                                            temp_df["Block Size"] = pd.to_numeric(temp_df["Block Size"], errors='coerce', downcast='float').round(0)  # Round to 0 decimal places
                                            temp_df["Settlement Date"] = pd.to_datetime(temp_df["Settlement Date"], format='%Y%m%d', errors='coerce')
                                            temp_df = temp_df[temp_df["Settlement Date"].dt.year >= 2024]
                                            temp_df["Settlement Date"] = temp_df["Settlement Date"].dt.strftime('%d/%m/%Y')
                                            temp_df = temp_df[temp_df["Postcode"].isin(ALLOWED_POSTCODES)]
                                            temp_df = temp_df[temp_df["Price"] >= 10000]
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
        NATIONAL_MEDIAN = result_df["Price"].median()
        logger.info(f"National median price calculated: ${NATIONAL_MEDIAN:,.0f}")
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

def generate_heatmap():
    df = load_property_data()
    m = folium.Map(location=[-32.5, 152.5], zoom_start=8)
    coords = df[df["Price"] < 9_000_000].dropna(subset=["Price"])
    marker_cluster = MarkerCluster().add_to(m)
    for _, row in coords.iterrows():
        lat = SUBURB_COORDS.get(row["Postcode"], {}).get(row["Suburb"], POSTCODE_COORDS.get(row["Postcode"], REGION_CENTERS.get(next(iter(REGION_POSTCODE_LIST)), [-32.5, 152.5])))[0]
        lon = SUBURB_COORDS.get(row["Postcode"], {}).get(row["Suburb"], POSTCODE_COORDS.get(row["Postcode"], REGION_CENTERS.get(next(iter(REGION_POSTCODE_LIST)), [-32.5, 152.5])))[1]
        folium.Marker(
            location=[lat, lon],
            popup=f"{row['Suburb']} ({row['Postcode']}): ${row['Price']:,.0f}",
        ).add_to(marker_cluster)
    heatmap_path = "static/heatmap.html"
    m.save(heatmap_path)
    return heatmap_path

def generate_region_median_chart(selected_region=None, selected_postcode=None):
    df = load_property_data()
    if selected_postcode:
        median_data = df[df["Postcode"] == selected_postcode].groupby('Suburb')['Price'].median().sort_values()
        title = f"Median House Prices by Suburb (Postcode {selected_postcode})"
        x_label = "Suburb"
    elif selected_region:
        postcodes = REGION_POSTCODE_LIST.get(selected_region, [])
        median_data = df[df["Postcode"].isin(postcodes)].groupby('Postcode')['Price'].median().sort_values()
        title = f"Median House Prices by Postcode ({selected_region})"
        x_label = "Postcode"
    else:
        region_medians = df.groupby('Postcode')['Price'].median().reset_index()
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
    plt.savefig(chart_path)
    plt.close()
    return chart_path

def pre_generate_charts():
    logger.info("Pre-generating charts...")
    heatmap_path = generate_heatmap()
    region_median_chart_path = generate_region_median_chart()
    logger.info(f"Generated heatmap at {heatmap_path}")
    logger.info(f"Generated region median chart at {region_median_chart_path}")

@app.route('/health')
def health_check():
    global last_health_status
    status = "OK" if data_loaded else "LOADING"
    logger.debug(f"Health check: data_loaded={data_loaded}, returning '{status}'")
    if status != last_health_status:
        logger.info(f"Health status changed to: {status}")
        last_health_status = status
    return status, 200

@app.route('/', methods=["GET", "POST"])
def index():
    global NATIONAL_MEDIAN
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
        unique_postcodes = sorted(df["Postcode"].unique())
        unique_suburbs = sorted(df["Suburb"].unique())
        selected_region = request.form.get("region", "")
        selected_postcode = request.form.get("postcode", "")
        selected_suburb = request.form.get("suburb", "")
        selected_property_type = request.form.get("property_type", "HOUSE")
        sort_by = request.form.get("sort_by", "Settlement Date")
        filtered_df = df.copy()
        
        chart_path = generate_region_median_chart(selected_region, selected_postcode)
        
        properties = []
        avg_price = 0
        stats = {"mean": 0, "median": 0, "std": 0}
        if selected_region or selected_postcode or selected_suburb or selected_property_type != "HOUSE":
            if selected_region:
                filtered_df = filtered_df[filtered_df["Postcode"].isin(REGION_POSTCODE_LIST.get(selected_region, []))]
            if selected_postcode:
                filtered_df = filtered_df[filtered_df["Postcode"] == selected_postcode]
            if selected_suburb:
                filtered_df = filtered_df[filtered_df["Suburb"] == selected_suburb]
            if selected_property_type != "ALL":
                filtered_df = filtered_df[filtered_df["Property Type"] == selected_property_type]
            properties = filtered_df.sort_values(by=sort_by).to_dict('records')
            avg_price = filtered_df["Price"].mean() if not filtered_df.empty else 0
            stats = {"mean": filtered_df["Price"].mean(), "median": filtered_df["Price"].median(), "std": filtered_df["Price"].std()} if not filtered_df.empty else {"mean": 0, "median": 0, "std": 0}
        
        heatmap_path = "static/heatmap.html" if os.path.exists("static/heatmap.html") else None
        region_median_chart_path = "static/region_median_chart.png" if os.path.exists("static/region_median_chart.png") else None
        
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
                                  properties=properties,
                                  avg_price=avg_price,
                                  stats=stats,
                                  national_median=NATIONAL_MEDIAN)
        elapsed_time = time.time() - start_time
        logger.info(f"Index route completed in {elapsed_time:.2f} seconds")
        log_memory_usage()
        return response
    except Exception as e:
        logger.error(f"Error in index route: {e}", exc_info=True)
        return "An error occurred on the server", 500

@app.route('/hot_suburbs')
def hot_suburbs():
    global NATIONAL_MEDIAN
    start_time = time.time()
    logger.info("Starting hot_suburbs route")
    log_memory_usage()
    
    if not data_loaded:
        logger.info("Data not yet loaded, showing loading page")
        return render_template("loading.html")
    
    logger.info("Data loaded, proceeding with hot_suburbs route")
    try:
        df = load_property_data()
        suburb_medians = df.groupby(['Suburb', 'Postcode'])['Price'].median().reset_index()
        postcode_to_region = {pc: region for region, postcodes in REGION_POSTCODE_LIST.items() for pc in postcodes}
        suburb_medians['Region'] = suburb_medians['Postcode'].map(postcode_to_region)
        hot_suburbs_df = suburb_medians[suburb_medians['Price'] < NATIONAL_MEDIAN].sort_values(by='Price')  # Sort by median price
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
                                  national_median=NATIONAL_MEDIAN)
        elapsed_time = time.time() - start_time
        logger.info(f"Hot_suburbs route completed in {elapsed_time:.2f} seconds")
        log_memory_usage()
        return response
    except Exception as e:
        logger.error(f"Error in hot_suburbs route: {e}", exc_info=True)
        return "An error occurred on the server", 500

@app.route('/get_postcodes')
def get_postcodes():
    try:
        if not data_loaded:
            return jsonify({"error": "Data still loading"}), 503
        region = request.args.get('region')
        postcodes = REGION_POSTCODE_LIST.get(region, [])
        return jsonify(postcodes)
    except Exception as e:
        logger.error(f"Error in get_postcodes: {e}", exc_info=True)
        return jsonify({"error": "Server error"}), 500

@app.route('/get_suburbs')
def get_suburbs():
    try:
        if not data_loaded:
            return jsonify({"error": "Data still loading"}), 503
        df = load_property_data()
        region = request.args.get('region')
        postcode = request.args.get('postcode')
        filtered_df = df[df["Postcode"].isin(REGION_POSTCODE_LIST.get(region, []))]
        if postcode:
            filtered_df = filtered_df[filtered_df["Postcode"] == postcode]
        suburbs = sorted(filtered_df["Suburb"].unique().tolist())
        return jsonify(suburbs)
    except Exception as e:
        logger.error(f"Error in get_suburbs: {e}", exc_info=True)
        return jsonify({"error": "Server error"}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    try:
        return send_from_directory(app.static_folder, filename)
    except Exception as e:
        logger.error(f"Error serving static file {filename}: {e}", exc_info=True)
        return "Static file not found", 404

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