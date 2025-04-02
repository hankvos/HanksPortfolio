import os
import io
import logging
import zipfile
import re
from datetime import datetime
from collections import Counter
from functools import lru_cache
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from flask import Flask, render_template, request, jsonify, send_from_directory
import folium
from folium.plugins import HeatMap

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(levelname)s:%(name)s:%(message)s'
)

REGION_POSTCODE_LIST = {
    "Central Coast": ["2250", "2251", "2256", "2257", "2258", "2259", "2260", "2261", "2262", "2263"],
    "Coffs Harbour - Grafton": ["2450", "2452", "2454", "2455", "2456"],
    "Hunter Valley excl Newcastle": ["2320", "2321", "2325", "2326", "2327"],
    "Newcastle and Lake Macquarie": ["2280", "2281", "2282", "2283", "2284", "2285", "2286", "2287", "2289", "2290", "2291"],
    "Mid North Coast": ["2430", "2440", "2441", "2443", "2444", "2445", "2446"],
    "Richmond - Tweed": ["2477", "2478", "2480", "2481", "2482", "2483"]
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

def load_property_data():
    zip_files = [f for f in os.listdir() if f.endswith('.zip')]
    if not zip_files:
        logging.error("No ZIP files found in the directory.")
        return pd.DataFrame()

    result_df = pd.DataFrame(columns=["Postcode", "Price", "Settlement Date", "Suburb", "Property Type", "Street", "Block Size", "Unit Number"])
    raw_property_types = Counter()
    allowed_types = {"RESIDENCE", "COMMERCIAL", "FARM", "VACANT LAND"}

    for zip_file in sorted(zip_files, reverse=True):
        if "2025.zip" not in zip_file:
            logging.info(f"Skipping {zip_file} as we're focusing on 2025.zip")
            continue
        
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
                                            df = pd.DataFrame(parsed_rows)
                                            if not df.empty:
                                                df = df.rename(columns={
                                                    8: "Street",
                                                    9: "Suburb",
                                                    10: "Postcode",
                                                    11: "Block Size",
                                                    14: "Settlement Date",
                                                    15: "Price",
                                                    18: "Property Type",
                                                    2: "Property ID"
                                                })
                                                df["Unit Number"] = df["Property ID"].map(unit_numbers).fillna("")
                                                df["Property Type"] = df["Property Type"].replace("RESIDENCE", "HOUSE")
                                                df["Property Type"] = df.apply(
                                                    lambda row: "UNIT" if (
                                                        row["Property Type"] == "HOUSE" and 
                                                        (row["Unit Number"] or re.match(r'^\d+/\d+', row["Street"]))
                                                    ) else row["Property Type"],
                                                    axis=1
                                                )
                                                df = df[["Postcode", "Price", "Settlement Date", "Suburb", "Property Type", "Street", "Block Size", "Unit Number"]]
                                                df["Postcode"] = df["Postcode"].astype(str)
                                                df["Price"] = pd.to_numeric(df["Price"], errors='coerce', downcast='float')
                                                df["Block Size"] = pd.to_numeric(df["Block Size"], errors='coerce', downcast='float')
                                                df["Settlement Date"] = pd.to_datetime(df["Settlement Date"], format='%Y%m%d', errors='coerce')
                                                df = df[df["Postcode"].isin(ALLOWED_POSTCODES)]
                                                result_df = pd.concat([result_df, df], ignore_index=True)
                                    except Exception as e:
                                        logging.error(f"Error reading {dat_file} in {nested_zip_name}: {e}")
                    except Exception as e:
                        logging.error(f"Error processing nested ZIP {nested_zip_name} in {zip_file}: {e}")
        except Exception as e:
            logging.error(f"Error opening {zip_file}: {e}")

    if result_df.empty:
        logging.error("No valid DAT files processed or no data matches filters.")
    else:
        logging.info(f"Processed {len(result_df)} records from 2025.zip")
        logging.info(f"Raw Property Type counts (field 18): {dict(raw_property_types)}")
        logging.info("Street values for first 200 records:")
        for i, street in enumerate(result_df["Street"].head(200)):
            logging.info(f"Record {i}: {street}")
        logging.info(f"Loaded {len(result_df)} records into DataFrame at startup.")

    return result_df

# Cache DataFrame globally
df = load_property_data()

def generate_region_median_chart():
    os.makedirs('static', exist_ok=True)
    median_prices = {}
    for region, postcodes in REGION_POSTCODE_LIST.items():
        region_df = df[df["Postcode"].isin(postcodes)]
        if not region_df.empty:
            median_prices[region] = region_df["Price"].median()
    if not median_prices:
        return None
    regions, prices = zip(*sorted(median_prices.items(), key=lambda x: x[1]))
    plt.figure(figsize=(10, 6))
    plt.barh(regions, prices)  # Reversed: Regions on y-axis, Prices on x-axis
    plt.title("Median Price by Region")
    plt.xlabel("Median Price ($)")  # Swapped to x-axis
    plt.ylabel("Region")  # Swapped to y-axis
    plt.tight_layout()
    chart_path = os.path.join(app.static_folder, "region_median_prices.png")
    plt.savefig(chart_path)
    plt.close()
    return chart_path

@lru_cache(maxsize=32)
def generate_heatmap_cached(region=None, postcode=None, suburb=None):
    filtered_df = df.copy()
    if region:
        filtered_df = filtered_df[filtered_df["Postcode"].isin(REGION_POSTCODE_LIST.get(region, []))]
    if postcode:
        filtered_df = filtered_df[filtered_df["Postcode"] == postcode]
    if suburb:
        filtered_df = filtered_df[filtered_df["Suburb"] == suburb]
    os.makedirs('static', exist_ok=True)
    heatmap_path = os.path.join(app.static_folder, f"heatmap_{region or 'all'}_{postcode or 'all'}_{suburb or 'all'}.html")
    all_coords = [coord for pc in POSTCODE_COORDS for coord in [POSTCODE_COORDS[pc]]]
    center_lat, center_lon = (min(c[0] for c in all_coords) + max(c[0] for c in all_coords)) / 2, (min(c[1] for c in all_coords) + max(c[1] for c in all_coords)) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="CartoDB positron")
    if not filtered_df.empty:
        heatmap_data = filtered_df[filtered_df["Postcode"].isin(POSTCODE_COORDS.keys())].groupby("Postcode").agg({"Price": "median"}).reset_index()
        heat_data = [[POSTCODE_COORDS[row["Postcode"]][0], POSTCODE_COORDS[row["Postcode"]][1], row["Price"] / 1e6]
                     for _, row in heatmap_data.iterrows() if row["Postcode"] in POSTCODE_COORDS]
        if heat_data:
            HeatMap(heat_data, radius=15, blur=20).add_to(m)
    for i, (region_name, postcodes) in enumerate(REGION_POSTCODE_LIST.items()):
        coords = [POSTCODE_COORDS.get(pc) for pc in postcodes if pc in POSTCODE_COORDS]
        coords = [c for c in coords if c]
        if coords:
            lat = sum(c[0] for c in coords) / len(coords) + (i * 0.05)
            lon = sum(c[1] for c in coords) / len(coords)
            popup_html = f'<a href="#" onclick="window.parent.document.getElementById(\'region\').value=\'{region_name}\'; window.parent.updatePostcodes(); window.parent.document.forms[0].submit();">{region_name}</a>'
            folium.Marker([lat, lon], tooltip=region_name, popup=folium.Popup(popup_html, max_width=300), icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)
    m.fit_bounds([[min(c[0] for c in all_coords), min(c[1] for c in all_coords)], [max(c[0] for c in all_coords), max(c[1] for c in all_coords)]])
    m.save(heatmap_path)
    return f"/static/{os.path.basename(heatmap_path)}"

@lru_cache(maxsize=32)
def generate_charts_cached(region=None, postcode=None, suburb=None):
    filtered_df = df.copy()
    if region:
        filtered_df = filtered_df[filtered_df["Postcode"].isin(REGION_POSTCODE_LIST.get(region, []))]
    if postcode:
        filtered_df = filtered_df[filtered_df["Postcode"] == postcode]
    if suburb:
        filtered_df = filtered_df[filtered_df["Suburb"] == suburb]
    os.makedirs('static', exist_ok=True)
    chart_prefix = f"{region or 'all'}_{postcode or 'all'}_{suburb or 'all'}"
    if filtered_df.empty:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No Data Available", fontsize=12, ha='center', va='center')
        plt.title("Median House Price Over Time")
        plt.xlabel("Settlement Date")
        plt.ylabel("Price ($)")
        median_chart_path = os.path.join(app.static_folder, f"median_house_price_{chart_prefix}.png")
        plt.savefig(median_chart_path)
        plt.close()
        return median_chart_path, None, None, None, None
    plt.figure(figsize=(10, 6))
    filtered_df.groupby(filtered_df["Settlement Date"].dt.to_period("M"))["Price"].median().plot()
    plt.title("Median House Price Over Time")
    plt.xlabel("Settlement Date")
    plt.ylabel("Price ($)")
    median_chart_path = os.path.join(app.static_folder, f"median_house_price_{chart_prefix}.png")
    plt.savefig(median_chart_path)
    plt.close()
    plt.figure(figsize=(10, 6))
    filtered_df["Price"].hist(bins=30)
    plt.title("Price Histogram")
    plt.xlabel("Price ($)")
    plt.ylabel("Frequency")
    price_hist_path = os.path.join(app.static_folder, f"price_hist_{chart_prefix}.png")
    plt.savefig(price_hist_path)
    plt.close()
    plt.figure(figsize=(10, 6))
    if region:
        df[df["Postcode"].isin(REGION_POSTCODE_LIST.get(region, []))].groupby("Settlement Date")["Price"].median().plot()
    plt.title(f"Region Price Timeline: {region or 'All'}")
    plt.xlabel("Settlement Date")
    plt.ylabel("Price ($)")
    region_timeline_path = os.path.join(app.static_folder, f"region_timeline_{chart_prefix}.png")
    plt.savefig(region_timeline_path)
    plt.close()
    plt.figure(figsize=(10, 6))
    if postcode:
        df[df["Postcode"] == postcode].groupby("Settlement Date")["Price"].median().plot()
    plt.title(f"Postcode Price Timeline: {postcode or 'All'}")
    plt.xlabel("Settlement Date")
    plt.ylabel("Price ($)")
    postcode_timeline_path = os.path.join(app.static_folder, f"postcode_timeline_{chart_prefix}.png")
    plt.savefig(postcode_timeline_path)
    plt.close()
    suburb_timeline_path = None
    if suburb:  # Only generate if suburb is selected
        plt.figure(figsize=(10, 6))
        df[df["Suburb"] == suburb].groupby("Settlement Date")["Price"].median().plot()
        plt.title(f"Suburb Price Timeline: {suburb}")
        plt.xlabel("Settlement Date")
        plt.ylabel("Price ($)")
        suburb_timeline_path = os.path.join(app.static_folder, f"suburb_timeline_{chart_prefix}.png")
        plt.savefig(suburb_timeline_path)
        plt.close()
    return median_chart_path, price_hist_path, region_timeline_path, postcode_timeline_path, suburb_timeline_path

@app.route('/', methods=["GET", "POST"])
def index():
    selected_region = request.form.get("region", None) if request.method == "POST" else None
    selected_postcode = request.form.get("postcode", None) if request.method == "POST" else None
    selected_suburb = request.form.get("suburb", None) if request.method == "POST" else None
    selected_property_type = request.form.get("property_type", "HOUSE") if request.method == "POST" else "HOUSE"
    sort_by = request.form.get("sort_by", "Settlement Date") if request.method == "POST" else "Settlement Date"
    
    filtered_df = df.copy()
    if selected_region:
        filtered_df = filtered_df[filtered_df["Postcode"].isin(REGION_POSTCODE_LIST.get(selected_region, []))]
    if selected_postcode:
        filtered_df = filtered_df[filtered_df["Postcode"] == selected_postcode]
    if selected_suburb:
        filtered_df = filtered_df[filtered_df["Suburb"] == selected_suburb]
    if selected_property_type != "ALL":
        filtered_df = filtered_df[filtered_df["Property Type"] == selected_property_type]
    
    heatmap_path = generate_heatmap_cached(selected_region, selected_postcode, selected_suburb)
    median_chart_path, price_hist_path, region_timeline_path, postcode_timeline_path, suburb_timeline_path = generate_charts_cached(selected_region, selected_postcode, selected_suburb)
    region_median_chart_path = generate_region_median_chart() if not (selected_region or selected_postcode or selected_suburb) else None
    
    properties = None
    if selected_region or selected_postcode or selected_suburb:
        filtered_df["Address"] = filtered_df["Street"] + ", " + filtered_df["Suburb"] + " NSW " + filtered_df["Postcode"]
        filtered_df["Map Link"] = filtered_df["Address"].apply(
            lambda addr: f"https://www.google.com/maps/place/{addr.replace(' ', '+')}"
        )
        properties = filtered_df.sort_values(by=sort_by)[["Address", "Price", "Settlement Date", "Map Link"]].to_dict(orient="records")
    
    if filtered_df.empty and (selected_region or selected_postcode or selected_suburb):
        return render_template("index.html", 
                               regions=REGION_POSTCODE_LIST.keys(), 
                               postcodes=[], 
                               suburbs=[], 
                               property_types=["ALL", "HOUSE", "UNIT", "COMMERCIAL", "FARM", "VACANT LAND"], 
                               heatmap_path=heatmap_path, 
                               median_chart_path=median_chart_path,
                               region_median_chart_path=region_median_chart_path,
                               data_source="NSW Valuer General Data", 
                               error="No properties found for the selected filters.")
    
    unique_postcodes = sorted(filtered_df["Postcode"].unique().tolist()) if (selected_region or selected_postcode) else []
    unique_suburbs = sorted(filtered_df["Suburb"].unique().tolist()) if (selected_region or selected_postcode or selected_suburb) else []
    avg_price = filtered_df["Price"].mean() if not filtered_df.empty else None
    stats_dict = {
        "mean": filtered_df["Price"].mean(),
        "median": filtered_df["Price"].median(),
        "std": filtered_df["Price"].std()
    } if not filtered_df.empty else {}
    
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
                           heatmap_path=heatmap_path,
                           median_chart_path=median_chart_path,
                           price_hist_path=price_hist_path,
                           region_timeline_path=region_timeline_path,
                           postcode_timeline_path=postcode_timeline_path,
                           suburb_timeline_path=suburb_timeline_path,
                           region_median_chart_path=region_median_chart_path,
                           data_source="NSW Valuer General Data")

@app.route('/get_postcodes')
def get_postcodes():
    region = request.args.get('region')
    postcodes = REGION_POSTCODE_LIST.get(region, [])
    return jsonify(postcodes)

@app.route('/get_suburbs')
def get_suburbs():
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)