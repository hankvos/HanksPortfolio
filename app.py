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
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*exclude empty or all-NA columns.*")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from flask import Flask, render_template, request, jsonify, send_from_directory, g
import folium
from folium.plugins import HeatMap

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(levelname)s:%(name)s:%(message)s'
)

# [REGION_POSTCODE_LIST, ALLOWED_POSTCODES, POSTCODE_COORDS, REGION_CENTERS, SUBURB_COORDS unchanged]

# Initialize global property data as None; load lazily on first request
g.property_df = None

def load_property_data():
    if hasattr(g, 'property_df') and g.property_df is not None:
        logging.info("Reusing cached property data")
        return g.property_df
    
    start_time = time.time()
    logging.info("Loading property data from scratch...")
    zip_files = [f for f in os.listdir() if f.endswith('.zip') and "2025.zip" in f]
    if not zip_files:
        logging.error("No 2025.zip files found in the directory.")
        g.property_df = pd.DataFrame()
        return g.property_df

    result_df = pd.DataFrame(columns=["Postcode", "Price", "Settlement Date", "Suburb", "Property Type", "Street", "StreetOnly", "Block Size", "Unit Number"])
    raw_property_types = Counter()
    allowed_types = {"RESIDENCE", "COMMERCIAL", "FARM", "VACANT LAND"}

    for zip_file in zip_files:
        logging.info(f"Processing {zip_file}")
        process_start = time.time()
        try:
            with zipfile.ZipFile(zip_file, 'r') as outer_zip:
                nested_zips = [f for f in outer_zip.namelist() if f.endswith('.zip')]
                for nested_zip_name in sorted(nested_zips, reverse=True):
                    logging.info(f"Opening nested ZIP: {nested_zip_name}")
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
                                                if not temp_df.empty:
                                                    result_df = pd.concat([result_df, temp_df], ignore_index=True)
                                    except Exception as e:
                                        logging.error(f"Error reading {dat_file} in {nested_zip_name}: {e}")
                    except Exception as e:
                        logging.error(f"Error processing nested ZIP {nested_zip_name} in {zip_file}: {e}")
            logging.info(f"Finished processing {zip_file} in {time.time() - process_start:.2f} seconds")
            break
        except Exception as e:
            logging.error(f"Error opening {zip_file}: {e}")

    if result_df.empty:
        logging.error("No valid DAT files processed or no data matches filters.")
    else:
        logging.info(f"Processed {len(result_df)} records from {zip_file}")
        logging.info(f"Raw Property Type counts (field 18): {dict(raw_property_types)}")
        logging.info(f"Processed Property Type counts: {result_df['Property Type'].value_counts().to_dict()}")

    g.property_df = result_df
    logging.info(f"Loaded {len(result_df)} records into DataFrame in {time.time() - start_time:.2f} seconds")
    return g.property_df

# [Rest of the functions: get_hot_suburbs, generate_region_median_chart, etc., unchanged]

@app.route('/health')
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route('/', methods=["GET", "POST"])
def index():
    start_time = time.time()
    logging.info("Starting index route")
    
    # Lazy-load data only when needed
    df = load_property_data()
    selected_region = request.form.get("region", None) if request.method == "POST" else None
    selected_postcode = request.form.get("postcode", None) if request.method == "POST" else None
    selected_suburb = request.form.get("suburb", None) if request.method == "POST" else None
    selected_property_type = request.form.get("property_type", "HOUSE") if request.method == "POST" else "HOUSE"
    sort_by = request.form.get("sort_by", "Settlement Date") if request.method == "POST" else "Settlement Date"
    show_hot_suburbs = request.form.get("hot_suburbs", None) == "true" if request.method == "POST" else False

    filtered_df = df.copy()
    hot_suburbs = []
    if show_hot_suburbs:
        hot_suburbs = get_hot_suburbs()
        filtered_df = df[df["Suburb"].isin([hs["Suburb"] for hs in hot_suburbs])]
    else:
        if selected_region:
            filtered_df = filtered_df[filtered_df["Postcode"].isin(REGION_POSTCODE_LIST.get(selected_region, []))]
        if selected_postcode:
            filtered_df = filtered_df[filtered_df["Postcode"] == selected_postcode]
        if selected_suburb:
            filtered_df = filtered_df[filtered_df["Suburb"] == selected_suburb]
        if selected_property_type != "ALL":
            filtered_df = filtered_df[filtered_df["Property Type"] == selected_property_type]
    
    heatmap_path = generate_heatmap_cached(selected_region, selected_postcode, selected_suburb)
    
    median_chart_path = None
    price_hist_path = None
    region_timeline_path = None
    postcode_timeline_path = None
    suburb_timeline_path = None
    region_median_chart_path = None
    postcode_median_chart_path = None
    suburb_median_chart_path = None
    
    if selected_region or selected_postcode or selected_suburb or show_hot_suburbs:
        median_chart_path, price_hist_path, region_timeline_path, postcode_timeline_path, suburb_timeline_path = generate_charts_cached(selected_region, selected_postcode, selected_suburb)
    if not selected_region and not selected_postcode and not selected_suburb and not show_hot_suburbs:
        region_median_chart_path = generate_region_median_chart()
    elif selected_region and not selected_postcode and not selected_suburb:
        postcode_median_chart_path = generate_postcode_median_chart(region=selected_region)
    elif selected_postcode and not selected_suburb:
        suburb_median_chart_path = generate_suburb_median_chart(selected_postcode)
    
    properties = None
    if selected_region or selected_postcode or selected_suburb or show_hot_suburbs:
        filtered_df["Address"] = filtered_df["Street"] + ", " + filtered_df["Suburb"] + " NSW " + filtered_df["Postcode"]
        filtered_df["Map Link"] = filtered_df["Address"].apply(
            lambda addr: f"https://www.google.com/maps/place/{addr.replace(' ', '+')}"
        )
        if sort_by == "Address":
            properties = filtered_df.sort_values(by="StreetOnly")[["Address", "Price", "Settlement Date", "Block Size", "Map Link"]].to_dict(orient="records")
        elif sort_by == "Settlement Date":
            filtered_df["Settlement Date"] = pd.to_datetime(filtered_df["Settlement Date"], format='%d/%m/%Y')
            properties = filtered_df.sort_values(by="Settlement Date")[["Address", "Price", "Settlement Date", "Block Size", "Map Link"]]
            properties = [{k: v.strftime('%d/%m/%Y') if k == "Settlement Date" else v for k, v in prop.items()} for prop in properties.to_dict(orient="records")]
        else:  # Price
            properties = filtered_df.sort_values(by="Price")[["Address", "Price", "Settlement Date", "Block Size", "Map Link"]].to_dict(orient="records")
    
    if filtered_df.empty and (selected_region or selected_postcode or selected_suburb or show_hot_suburbs):
        return render_template("index.html", 
                               regions=REGION_POSTCODE_LIST.keys(), 
                               postcodes=[], 
                               suburbs=[], 
                               property_types=["ALL", "HOUSE", "UNIT", "COMMERCIAL", "FARM", "VACANT LAND"], 
                               heatmap_path=heatmap_path, 
                               median_chart_path=median_chart_path,
                               region_median_chart_path=region_median_chart_path,
                               postcode_median_chart_path=postcode_median_chart_path,
                               suburb_median_chart_path=suburb_median_chart_path,
                               data_source="NSW Valuer General Data", 
                               error="No properties found for the selected filters.")
    
    unique_postcodes = sorted(filtered_df["Postcode"].unique().tolist()) if (selected_region or selected_postcode or show_hot_suburbs) else []
    unique_suburbs = sorted(filtered_df["Suburb"].unique().tolist()) if (selected_region or selected_postcode or selected_suburb or show_hot_suburbs) else []
    avg_price = filtered_df["Price"].mean() if not filtered_df.empty else None
    stats_dict = {
        "mean": filtered_df["Price"].mean(),
        "median": filtered_df["Price"].median(),
        "std": filtered_df["Price"].std()
    } if not filtered_df.empty else {}
    
    elapsed_time = time.time() - start_time
    logging.info(f"Index route completed in {elapsed_time:.2f} seconds")
    
    return render_template("index.html", 
                           regions=REGION_POSTCODE_LIST.keys(),
                           postcodes=unique_postcodes,
                           suburbs=unique_suburbs,
                           property_types=["ALL", "HOUSE", "UNIT", "COMMERCIAL", "FARM", "VACANT LAND"],
                           properties=properties,
                           avg_price=avg_price,
                           stats=stats_dict,
                           selected_region=selected_region or "",
                           selected_postcode=selected_postcode or "",
                           selected_suburb=selected_suburb or "",
                           selected_property_type=selected_property_type,
                           sort_by=sort_by,
                           show_hot_suburbs=show_hot_suburbs,
                           hot_suburbs=hot_suburbs,
                           heatmap_path=heatmap_path,
                           median_chart_path=median_chart_path,
                           price_hist_path=price_hist_path,
                           region_timeline_path=region_timeline_path,
                           postcode_timeline_path=postcode_timeline_path,
                           suburb_timeline_path=suburb_timeline_path,
                           region_median_chart_path=region_median_chart_path,
                           postcode_median_chart_path=postcode_median_chart_path,
                           suburb_median_chart_path=suburb_median_chart_path,
                           data_source="NSW Valuer General Data")

@app.route('/get_postcodes')
def get_postcodes():
    region = request.args.get('region')
    postcodes = REGION_POSTCODE_LIST.get(region, [])
    return jsonify(postcodes)

@app.route('/get_suburbs')
def get_suburbs():
    df = load_property_data()
    region = request.args.get('region')
    postcode = request.args.get('postcode')
    filtered_df = df[df["Postcode"].isin(REGION_POSTCODE_LIST.get(region, []))]
    if postcode:
        filtered_df = filtered_df[filtered_df["Postcode"] == postcode]
    suburbs = sorted(filtered_df["Suburb"].unique().tolist())
    return jsonify(suburbs)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    # Explicitly use Render's PORT environment variable, default to 5000 locally
    port = int(os.environ.get('PORT', 5000))
    logging.info(f"Starting Flask server on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)