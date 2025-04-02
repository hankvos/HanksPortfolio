import os
import io
import logging
import zipfile
from datetime import datetime
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from flask import Flask, render_template, request, jsonify, send_from_directory
import folium
from folium.plugins import HeatMap

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

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

# Define 2024 nested ZIPs we want (Oct 7 - Dec 30)
TARGET_2024_ZIPS = [
    "20241007.zip", "20241014.zip", "20241021.zip", "20241028.zip",
    "20241104.zip", "20241111.zip", "20241118.zip", "20241125.zip",
    "20241202.zip", "20241209.zip", "20241216.zip", "20241223.zip", "20241230.zip"
]

def load_property_data():
    zip_files = [f for f in os.listdir() if f.endswith('.zip')]
    logging.info(f"Found {len(zip_files)} ZIP files in directory")
    
    if not zip_files:
        logging.error("No ZIP files found in the directory.")
        return pd.DataFrame()

    result_df = pd.DataFrame(columns=["Postcode", "Price", "Settlement Date", "Suburb", "Property Type"])
    
    # Process 2025.zip first, all 12 nested ZIPs
    for zip_file in sorted(zip_files, reverse=True):  # 2025.zip first
        try:
            with zipfile.ZipFile(zip_file, 'r') as outer_zip:
                nested_zips = [f for f in outer_zip.namelist() if f.endswith('.zip')]
                logging.info(f"Processing {zip_file} with {len(nested_zips)} nested ZIPs")
                
                if not nested_zips:
                    logging.warning(f"No nested ZIP files found in {zip_file}")
                    continue
                
                # Filter nested ZIPs based on file name
                if "2025.zip" in zip_file:
                    target_zips = nested_zips  # Take all 2025 nested ZIPs
                elif "2024.zip" in zip_file:
                    target_zips = [z for z in nested_zips if z in TARGET_2024_ZIPS]
                else:
                    logging.info(f"Skipping unexpected ZIP file: {zip_file}")
                    continue
                
                # Sort for consistency (latest first)
                target_zips.sort(reverse=True)
                logging.info(f"Target nested ZIPs for {zip_file}: {target_zips}")
                
                for nested_zip_name in target_zips:
                    try:
                        with outer_zip.open(nested_zip_name) as nested_zip_file:
                            with zipfile.ZipFile(io.BytesIO(nested_zip_file.read())) as nested_zip:
                                dat_files = [f for f in nested_zip.namelist() if f.endswith('.DAT')]
                                logging.info(f"Found {len(dat_files)} DAT files in {nested_zip_name}")
                                
                                if not dat_files:
                                    logging.warning(f"No DAT files found in {nested_zip_name}")
                                    continue
                                
                                for dat_file in dat_files:
                                    try:
                                        with nested_zip.open(dat_file) as f:
                                            df = pd.read_csv(
                                                io.BytesIO(f.read()),
                                                encoding='latin1',
                                                on_bad_lines='skip',
                                                dtype={
                                                    "Postcode": "str",
                                                    "Price": "float32",
                                                    "Settlement Date": "str",
                                                    "Suburb": "str",
                                                    "Property Type": "str"
                                                }
                                            )
                                            # Filter early: 6 regions only
                                            df = df[df["Postcode"].isin(ALLOWED_POSTCODES)]
                                            
                                            # Date filter: Oct 2024 - Mar 2025
                                            df['Settlement Date'] = pd.to_datetime(df['Settlement Date'], format='%Y%m%d', errors='coerce')
                                            df = df[
                                                ((df['Settlement Date'].dt.year == 2024) & (df['Settlement Date'].dt.month >= 10)) |
                                                ((df['Settlement Date'].dt.year == 2025) & (df['Settlement Date'].dt.month <= 3))
                                            ]
                                            
                                            result_df = pd.concat([result_df, df], ignore_index=True)
                                    except Exception as e:
                                        logging.error(f"Error reading {dat_file} in {nested_zip_name}: {e}")
                    except Exception as e:
                        logging.error(f"Error processing nested ZIP {nested_zip_name} in {zip_file}: {e}")
        except Exception as e:
            logging.error(f"Error opening {zip_file}: {e}")
    
    if result_df.empty:
        logging.error("No valid DAT files processed or no data matches filters.")
        return pd.DataFrame(columns=["Postcode", "Price", "Settlement Date", "Suburb", "Property Type"])
    
    logging.info(f"Processed {len(result_df)} records from {len(TARGET_2024_ZIPS) + 12} nested ZIPs")
    logging.info(f"Unique postcodes in loaded data: {result_df['Postcode'].unique().tolist()}")
    
    return result_df

def generate_heatmap(df):
    os.makedirs('static', exist_ok=True)
    
    heatmap_path = os.path.join(app.static_folder, "heatmap.html")
    if os.path.exists(heatmap_path):
        os.remove(heatmap_path)
        logging.info(f"Removed existing {heatmap_path} to force regeneration")
    
    all_coords = [coord for pc in POSTCODE_COORDS for coord in [POSTCODE_COORDS[pc]]]
    if not all_coords:
        logging.warning("No coordinates found in POSTCODE_COORDS.")
        center_lat, center_lon = -30.0, 153.0
    else:
        min_lat, max_lat = min(c[0] for c in all_coords), max(c[0] for c in all_coords)
        min_lon, max_lon = min(c[1] for c in all_coords), max(c[1] for c in all_coords)
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="CartoDB positron")
    
    if not df.empty and "Postcode" in df.columns and "Price" in df.columns:
        heatmap_data = df[df["Postcode"].isin(POSTCODE_COORDS.keys())].groupby("Postcode").agg({"Price": "median"}).reset_index()
        heat_data = [[POSTCODE_COORDS[row["Postcode"]][0], POSTCODE_COORDS[row["Postcode"]][1], row["Price"] / 1e6]
                     for _, row in heatmap_data.iterrows() if row["Postcode"] in POSTCODE_COORDS]
        logging.info(f"Heatmap data points: {len(heat_data)}")
        if heat_data:
            HeatMap(heat_data, radius=15, blur=20).add_to(m)
    else:
        logging.info("No valid data for heatmap; skipping heatmap layer")
    
    markers_added = 0
    for i, (region, postcodes) in enumerate(REGION_POSTCODE_LIST.items()):
        coords = [POSTCODE_COORDS.get(pc) for pc in postcodes if pc in POSTCODE_COORDS]
        coords = [c for c in coords if c]
        if coords:
            lat = sum(c[0] for c in coords) / len(coords) + (i * 0.05)
            lon = sum(c[1] for c in coords) / len(coords)
            popup_html = f'<a href="#" onclick="window.parent.document.getElementById(\'region\').value=\'{region}\'; window.parent.updatePostcodes(); window.parent.document.forms[0].submit();">{region}</a>'
            folium.Marker(
                [lat, lon],
                tooltip=region,
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(m)
            markers_added += 1
    
    logging.info(f"Added {markers_added} markers to heatmap")
    
    if all_coords:
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    
    m.save(heatmap_path)
    logging.info(f"Heatmap saved to {heatmap_path}, file exists: {os.path.exists(heatmap_path)}")
    return "/static/heatmap.html"

def generate_charts(df, selected_region=None, selected_postcode=None, selected_suburb=None):
    os.makedirs('static', exist_ok=True)
    
    filtered_df = df.copy()
    if selected_region:
        filtered_df = filtered_df[filtered_df["Postcode"].isin(REGION_POSTCODE_LIST.get(selected_region, []))]
    if selected_postcode:
        filtered_df = filtered_df[filtered_df["Postcode"] == selected_postcode]
    if selected_suburb:
        filtered_df = filtered_df[filtered_df["Suburb"] == selected_suburb]
    
    logging.info(f"Filter applied - Region: {selected_region}, Postcode: {selected_postcode}, Suburb: {selected_suburb}, Records: {len(filtered_df)}")
    
    if filtered_df.empty:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No Data Available", fontsize=12, ha='center', va='center')
        plt.title("Median House Price Over Time")
        plt.xlabel("Settlement Date")
        plt.ylabel("Price ($)")
        median_chart_path = os.path.join(app.static_folder, "median_house_price_chart.png")
        plt.savefig(median_chart_path)
        plt.close()
        logging.info(f"Generated placeholder chart at {median_chart_path}")
        return median_chart_path, None, None, None, None, None
    
    plt.figure(figsize=(10, 6))
    filtered_df.groupby(filtered_df["Settlement Date"].dt.to_period("M"))["Price"].median().plot()
    plt.title("Median House Price Over Time")
    plt.xlabel("Settlement Date")
    plt.ylabel("Price ($)")
    median_chart_path = os.path.join(app.static_folder, "median_house_price_chart.png")
    plt.savefig(median_chart_path)
    plt.close()
    logging.info(f"Generated median chart at {median_chart_path}")
    
    plt.figure(figsize=(10, 6))
    filtered_df["Price"].hist(bins=30)
    plt.title("Price Histogram")
    plt.xlabel("Price ($)")
    plt.ylabel("Frequency")
    price_hist_path = os.path.join(app.static_folder, "price_hist.png")
    plt.savefig(price_hist_path)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_df.get("Block Size", pd.Series()), filtered_df["Price"], alpha=0.5)
    plt.title("Price vs Block Size")
    plt.xlabel("Block Size (sqm)")
    plt.ylabel("Price ($)")
    price_size_scatter_path = os.path.join(app.static_folder, "price_size_scatter.png")
    plt.savefig(price_size_scatter_path)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    if selected_region:
        df[df["Postcode"].isin(REGION_POSTCODE_LIST.get(selected_region, []))].groupby("Settlement Date")["Price"].median().plot()
    plt.title(f"Region Price Timeline: {selected_region}")
    plt.xlabel("Settlement Date")
    plt.ylabel("Price ($)")
    region_timeline_path = os.path.join(app.static_folder, "region_timeline.png")
    plt.savefig(region_timeline_path)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    if selected_postcode:
        df[df["Postcode"] == selected_postcode].groupby("Settlement Date")["Price"].median().plot()
    plt.title(f"Postcode Price Timeline: {selected_postcode}")
    plt.xlabel("Settlement Date")
    plt.ylabel("Price ($)")
    postcode_timeline_path = os.path.join(app.static_folder, "postcode_timeline.png")
    plt.savefig(postcode_timeline_path)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    if selected_suburb:
        df[df["Suburb"] == selected_suburb].groupby("Settlement Date")["Price"].median().plot()
    plt.title(f"Suburb Price Timeline: {selected_suburb}")
    plt.xlabel("Settlement Date")
    plt.ylabel("Price ($)")
    suburb_timeline_path = os.path.join(app.static_folder, "suburb_timeline.png")
    plt.savefig(suburb_timeline_path)
    plt.close()
    
    return median_chart_path, price_hist_path, price_size_scatter_path, region_timeline_path, postcode_timeline_path, suburb_timeline_path

df = load_property_data()
logging.info(f"Loaded {len(df)} records into DataFrame.")

@app.route('/', methods=["GET", "POST"])
def index():
    selected_region = request.form.get("region", "")
    selected_postcode = request.form.get("postcode", "")
    selected_suburb = request.form.get("suburb", "")
    selected_property_type = request.form.get("property_type", "ALL")
    sort_by = request.form.get("sort_by", "Settlement Date")
    
    filtered_df = df.copy()
    if selected_region:
        filtered_df = filtered_df[filtered_df["Postcode"].isin(REGION_POSTCODE_LIST.get(selected_region, []))]
    if selected_postcode:
        filtered_df = filtered_df[filtered_df["Postcode"] == selected_postcode]
    if selected_suburb:
        filtered_df = filtered_df[filtered_df["Suburb"] == selected_suburb]
    if selected_property_type != "ALL":
        filtered_df = filtered_df[filtered_df["Property Type"] == selected_property_type]
    
    logging.info(f"Index filter - Region: {selected_region}, Postcode: {selected_postcode}, Suburb: {selected_suburb}, Type: {selected_property_type}, Records: {len(filtered_df)}")
    
    heatmap_path = generate_heatmap(filtered_df)
    logging.info(f"Passing heatmap_path to template: {heatmap_path}")
    
    median_chart_path, price_hist_path, price_size_scatter_path, region_timeline_path, postcode_timeline_path, suburb_timeline_path = generate_charts(filtered_df, selected_region, selected_postcode, selected_suburb)
    
    if filtered_df.empty:
        return render_template("index.html", 
                               regions=REGION_POSTCODE_LIST.keys(), 
                               postcodes=[], 
                               suburbs=[], 
                               property_types=["ALL"] + sorted(df["Property Type"].unique().tolist()), 
                               heatmap_path=heatmap_path, 
                               median_chart_path=median_chart_path,
                               data_source="NSW Valuer General Data", 
                               error="No properties found for the selected filters.")
    
    filtered_df["Map Link"] = filtered_df.apply(
        lambda row: f"https://www.google.com/maps/search/?api=1&query={row['Latitude']},{row['Longitude']}" 
        if pd.notna(row.get('Latitude')) and pd.notna(row.get('Longitude')) else "#", axis=1
    )
    
    properties = filtered_df.sort_values(by=sort_by)[["Address", "Price", "Size", "Settlement Date", "Map Link"]].to_dict(orient="records")
    avg_price = filtered_df["Price"].mean()
    stats_dict = {
        "mean": filtered_df["Price"].mean(),
        "median": filtered_df["Price"].median(),
        "std": filtered_df["Price"].std()
    }
    
    unique_postcodes = sorted(filtered_df["Postcode"].unique().tolist())
    unique_suburbs = sorted(filtered_df["Suburb"].unique().tolist())
    
    return render_template("index.html", 
                           regions=REGION_POSTCODE_LIST.keys(),
                           postcodes=unique_postcodes,
                           suburbs=unique_suburbs,
                           property_types=["ALL"] + sorted(df["Property Type"].unique().tolist()),
                           properties=properties,
                           avg_price=avg_price,
                           stats=stats_dict,
                           selected_region=selected_region,
                           selected_postcode=selected_postcode,
                           selected_suburb=selected_suburb,
                           selected_property_type=selected_property_type,
                           sort_by=sort_by,
                           heatmap_path=heatmap_path,
                           median_chart_path=median_chart_path,
                           price_hist_path=price_hist_path,
                           price_size_scatter_path=price_size_scatter_path,
                           region_timeline_path=region_timeline_path,
                           postcode_timeline_path=postcode_timeline_path,
                           suburb_timeline_path=suburb_timeline_path,
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
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))