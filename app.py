import os
import io
import logging
import zipfile
import re
from datetime import datetime
import sys
import time
import psutil
import threading

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from flask import Flask, render_template, request

app = Flask(__name__)

# Configure logging
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s"))
logger.addHandler(handler)

# Region and postcode mappings (expand as needed)
REGION_POSTCODE_LIST = {
    "Central Coast": ["2083", "2250", "2251", "2256", "2257", "2258", "2259", "2260", "2261", "2262", "2263", "2775"],
}
ALLOWED_POSTCODES = {pc for region in REGION_POSTCODE_LIST.values() for pc in region}
POSTCODE_COORDS = {
    "2250": [-33.28, 151.41], "2251": [-33.31, 151.42], "2621": [-35.42, 149.81], "2217": [-33.97, 151.11],
    "2408": [-29.05, 148.77], "2460": [-29.68, 152.93], "2582": [-34.98, 149.23], "2580": [-34.57, 148.93],
    "2843": [-31.95, 149.43], "2650": [-35.12, 147.35], "2795": [-33.42, 149.58], "2444": [-31.65, 152.79],
    "2000": [-33.87, 151.21]
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
    "2443": {"JOHNS RIVER": [-31.73, 152.70]},
    "2621": {"BUNGENDORE": [-35.25, 149.44]},
    "2217": {"MONTEREY": [-33.97, 151.15]},
    "2408": {"NORTH STAR": [-28.93, 150.39]},
    "2460": {"COOMBADJHA": [-29.03, 152.38]},
    "2582": {"YASS": [-34.84, 148.91]},
    "2580": {"GOULBURN": [-34.75, 149.72]},
    "2843": {"COOLAH": [-31.82, 149.72]},
    "2650": {"WAGGA WAGGA": [-35.12, 147.35]},
    "2795": {"BATHURST": [-33.42, 149.58]},
    "2444": {"THRUMSTER": [-31.47, 152.83]},
    "2000": {"SYDNEY": [-33.87, 151.21]}
}

# Global variables
df = None
data_loaded = False
NATIONAL_MEDIAN = None
lock = threading.Lock()

@app.template_filter('currency')
def currency_filter(value):
    if value is None or pd.isna(value):
        return "N/A"
    return "${:,.0f}".format(value)

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    logger.info(f"Memory usage: RSS={mem.rss / 1024 / 1024:.2f} MB, VMS={mem.vms / 1024 / 1024:.2f} MB")
    sys.stdout.flush()

def load_property_data():
    global df, data_loaded, NATIONAL_MEDIAN
    with lock:
        if data_loaded:
            logger.info("Data already loaded, skipping reload")
            sys.stdout.flush()
            return df
    
    start_time = time.time()
    logger.info("Loading property data from 2025.zip...")
    sys.stdout.flush()
    log_memory_usage()
    
    zip_file = "2025.zip"
    result_df = pd.DataFrame(columns=["Postcode", "Price", "Settlement Date", "Suburb", "Property Type", "Street", "StreetOnly", "Block Size", "Unit Number"])
    allowed_types = {"RESIDENCE", "COMMERCIAL", "FARM", "VACANT LAND"}
    
    try:
        if not os.path.exists(zip_file):
            logger.error(f"ZIP file {zip_file} not found")
            sys.stdout.flush()
            return
        
        with zipfile.ZipFile(zip_file, 'r') as outer_zip:
            nested_zips = [f for f in outer_zip.namelist() if f.endswith('.zip')]
            if not nested_zips:
                logger.warning(f"No nested ZIP files found in {zip_file}")
                sys.stdout.flush()
                return
            
            logger.info(f"Found nested zips: {nested_zips}")
            sys.stdout.flush()
            
            for nested_zip_name in sorted(nested_zips, reverse=True):
                logger.info(f"Processing {nested_zip_name}")
                sys.stdout.flush()
                try:
                    with outer_zip.open(nested_zip_name) as nested_zip_file:
                        with zipfile.ZipFile(io.BytesIO(nested_zip_file.read())) as nested_zip:
                            dat_files = [f for f in nested_zip.namelist() if f.endswith('.DAT')]
                            if not dat_files:
                                logger.warning(f"No .DAT files in {nested_zip_name}")
                                sys.stdout.flush()
                                continue
                            
                            for dat_file in dat_files:
                                logger.info(f"Reading {dat_file}")
                                sys.stdout.flush()
                                try:
                                    with nested_zip.open(dat_file) as f:
                                        lines = f.read().decode('latin1').splitlines()
                                        logger.info(f"First few lines of {dat_file}: {lines[:5]}")
                                        sys.stdout.flush()
                                        parsed_rows = []
                                        unit_numbers = {}
                                        for line in lines:
                                            row = line.split(';')
                                            if row[0] == 'C' and len(row) > 5 and row[5]:
                                                unit_numbers[row[2]] = row[5]
                                            if row[0] == 'B' and len(row) > 18 and row[18] in allowed_types:
                                                parsed_rows.append(row)
                                        
                                        if not parsed_rows:
                                            logger.info(f"No valid B records in {dat_file}")
                                            sys.stdout.flush()
                                            continue
                                        
                                        temp_df = pd.DataFrame(parsed_rows)
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
                                    logger.error(f"Error reading {dat_file}: {str(e)}", exc_info=True)
                                    sys.stdout.flush()
                except Exception as e:
                    logger.error(f"Error processing {nested_zip_name}: {str(e)}", exc_info=True)
                    sys.stdout.flush()
        
        if result_df.empty:
            logger.warning("No valid data processed from 2025.zip")
            sys.stdout.flush()
        else:
            NATIONAL_MEDIAN = result_df["Price"].median()
            logger.info(f"National median price calculated: ${NATIONAL_MEDIAN:,.0f}")
            logger.info(f"Processed {len(result_df)} records from {zip_file}")
            sys.stdout.flush()
        
        with lock:
            df = result_df
            data_loaded = True
        logger.info(f"Data load completed in {time.time() - start_time:.2f} seconds")
        log_memory_usage()
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}", exc_info=True)
        sys.stdout.flush()

def generate_heatmap():
    logger.info("Generating heatmap...")
    sys.stdout.flush()
    with lock:
        df_local = df.copy() if df is not None else pd.DataFrame()
    if df_local.empty:
        logger.warning("No data for heatmap")
        sys.stdout.flush()
        m = folium.Map(location=[-33.8688, 151.2093], zoom_start=10)
    else:
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
    try:
        m.save(heatmap_path)
        logger.info(f"Heatmap generated at {heatmap_path}")
    except Exception as e:
        logger.error(f"Failed to save heatmap: {str(e)}", exc_info=True)
    sys.stdout.flush()
    return heatmap_path

def generate_region_median_chart(selected_region=None, selected_postcode=None):
    logger.info("Generating region median chart...")
    sys.stdout.flush()
    with lock:
        df_local = df.copy() if df is not None else pd.DataFrame()
    if df_local.empty:
        logger.warning("No data for chart")
        sys.stdout.flush()
        plt.figure(figsize=(12, 6))
        plt.title("No Data Available")
        plt.savefig("static/region_median_chart.png")
        plt.close()
        return "static/region_median_chart.png"
    
    try:
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
        plt.savefig(chart_path)
        plt.close()
        logger.info(f"Region median chart generated at {chart_path}")
    except Exception as e:
        logger.error(f"Failed to generate chart: {str(e)}", exc_info=True)
        chart_path = "static/region_median_chart.png"
        plt.figure(figsize=(12, 6))
        plt.title("Chart Generation Failed")
        plt.savefig(chart_path)
        plt.close()
    sys.stdout.flush()
    return chart_path

def startup_tasks():
    logger.info("Starting background data load...")
    sys.stdout.flush()
    load_property_data()
    if df is not None and not df.empty:
        logger.info("Data loaded, generating charts...")
        generate_heatmap()
        generate_region_median_chart()
    else:
        logger.warning("No data loaded, skipping chart generation")
    with open("/tmp/startup_complete.flag", "w") as f:
        f.write("done")
    logger.info("Startup tasks completed")
    sys.stdout.flush()

def is_startup_complete():
    return os.path.exists("/tmp/startup_complete.flag")

@app.route('/health')
def health_check():
    if is_startup_complete():
        logger.info("Health check: OK")
        sys.stdout.flush()
        return "OK", 200
    logger.info("Health check: LOADING")
    sys.stdout.flush()
    return "LOADING", 200

@app.route('/', methods=["GET", "POST"])
def index():
    logger.info("START: Request received for /")
    sys.stdout.flush()
    
    if not is_startup_complete():
        logger.info("STARTUP INCOMPLETE: Returning loading.html")
        sys.stdout.flush()
        return render_template("loading.html")
    
    try:
        with lock:
            df_local = df.copy() if df is not None else pd.DataFrame()
        logger.info(f"DATAFRAME: Copied {len(df_local)} rows")
        sys.stdout.flush()
        
        unique_postcodes = sorted(df_local["Postcode"].unique()) if not df_local.empty else []
        unique_suburbs = sorted(df_local["Suburb"].unique()) if not df_local.empty else []
        
        selected_region = request.form.get("region", "")
        selected_postcode = request.form.get("postcode", "")
        selected_suburb = request.form.get("suburb", "")
        selected_property_type = request.form.get("property_type", "HOUSE")
        sort_by = request.form.get("sort_by", "Street")
        
        logger.info(f"FORM: region={selected_region}, postcode={selected_postcode}, suburb={selected_suburb}, type={selected_property_type}, sort={sort_by}")
        sys.stdout.flush()
        
        display_postcode = selected_postcode if selected_region else ""
        display_suburb = selected_suburb if selected_region and selected_postcode else ""
        
        chart_path = generate_region_median_chart(selected_region, selected_postcode)
        
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
        
        logger.info(f"FILTERED: {len(properties)} properties")
        sys.stdout.flush()
        
        heatmap_path = "static/heatmap.html" if os.path.exists("static/heatmap.html") else None
        region_median_chart_path = "static/region_median_chart.png" if os.path.exists("static/region_median_chart.png") else None
        
        logger.info("RENDERING: Starting index.html render")
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
        logger.info("RENDERING: Completed")
        sys.stdout.flush()
        return response
    except Exception as e:
        logger.error(f"ERROR: {str(e)}", exc_info=True)
        sys.stdout.flush()
        return "Internal Server Error", 500

if __name__ == "__main__":
    if not os.path.exists("static"):
        os.makedirs("static")
    threading.Thread(target=startup_tasks, daemon=True).start()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
else:
    gunicorn_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    if not os.path.exists("static"):
        os.makedirs("static")
    threading.Thread(target=startup_tasks, daemon=True).start()