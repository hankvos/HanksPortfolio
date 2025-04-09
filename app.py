import os
import sys
import time
import logging
import zipfile
import threading
from flask import Flask, render_template, request
import pandas as pd
import folium
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for matplotlib
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader

app = Flask(__name__)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Global variables
lock = threading.Lock()
df = None
MEDIAN_ALL_REGIONS = 0
startup_complete = False

# Region to postcode mapping (example, expand as needed)
REGION_POSTCODE_LIST = {
    "Central Coast": ["2250", "2251", "2261"],
    "Sydney": ["2000", "2010", "2020"],
    "Northern Rivers": ["2477", "2478", "2480", "2483", "2484", "2486", "2487", "2488"],
    # Add more regions and postcodes
}

# Postcode coordinates (example, expand as needed)
POSTCODE_COORDS = {
    "2250": [-33.28, 151.41],
    "2000": [-33.87, 151.21],
    "2480": [-28.81, 153.28],  # Example for GOONELLABAH
    # Add more postcodes from your data
}

def is_startup_complete():
    global startup_complete
    return startup_complete

def generate_heatmap():
    logger.info("Generating heatmap...")
    sys.stdout.flush()
    with lock:
        df_local = df.copy() if df is not None else pd.DataFrame()
    
    if df_local.empty:
        logger.warning("No data for heatmap")
        sys.stdout.flush()
        m = folium.Map(location=[-32.5, 152.5], zoom_start=6)
        m.save("static/heatmap.html")
        return "static/heatmap.html"
    
    coordinates = []
    for _, row in df_local.iterrows():
        postcode = row["Postcode"]
        suburb = row["Suburb"]
        coords = POSTCODE_COORDS.get(postcode)
        if coords:
            coordinates.append(coords)
        else:
            logger.warning(f"No coordinates for Postcode: {postcode}, Suburb: {suburb}")
            coordinates.append([-32.5, 152.5])  # Fallback
    
    if not coordinates:
        logger.warning("No valid coordinates found, using default bounds")
        bounds = [[-34.0, 150.0], [-32.0, 154.0]]
    else:
        bounds = [
            [min(c[0] for c in coordinates) - 0.1, min(c[1] for c in coordinates) - 0.1],
            [max(c[0] for c in coordinates) + 0.1, max(c[1] for c in coordinates) + 0.1]
        ]
    
    sample_size = min(len(coordinates), 1000)
    sampled_coords = coordinates[:sample_size] if len(coordinates) <= 1000 else pd.DataFrame(coordinates).sample(sample_size).values.tolist()
    
    m = folium.Map(location=[(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2], zoom_start=6)
    for coord in sampled_coords:
        folium.CircleMarker(location=coord, radius=5, fill=True, fill_opacity=0.7).add_to(m)
    m.fit_bounds(bounds)
    m.save("static/heatmap.html")
    logger.info("Heatmap generated at static/heatmap.html")
    sys.stdout.flush()
    return "static/heatmap.html"

def generate_region_median_chart(selected_region=None, selected_postcode=None):
    logger.info("Generating region median chart...")
    sys.stdout.flush()
    with lock:
        df_local = df.copy() if df is not None else pd.DataFrame()
    chart_path = "static/region_median_chart.png"
    
    if df_local.empty:
        logger.warning("No data for chart")
        sys.stdout.flush()
        plt.figure(figsize=(12, 6))
        plt.title("No Data Available")
        plt.savefig(chart_path, bbox_inches='tight')
        plt.close()
        return chart_path
    
    try:
        logger.info(f"Chart data size: {len(df_local)} rows")
        if selected_postcode:
            median_data = df_local[df_local["Postcode"] == selected_postcode].groupby('Suburb')['Price'].median().sort_values()
            title = f"Median House Prices by Suburb (Postcode {selected_postcode})"
            x_label = "Suburb"
            logger.info(f"Postcode filter: {selected_postcode}, {len(median_data)} suburbs")
        elif selected_region:
            postcodes = REGION_POSTCODE_LIST.get(selected_region, [])
            median_data = df_local[df_local["Postcode"].isin(postcodes)].groupby('Postcode')['Price'].median().sort_values()
            title = f"Median House Prices by Postcode ({selected_region})"
            x_label = "Postcode"
            logger.info(f"Region filter: {selected_region}, {len(median_data)} postcodes")
        else:
            region_medians = df_local.groupby('Postcode')['Price'].median().reset_index()
            postcode_to_region = {pc: region for region, pcs in REGION_POSTCODE_LIST.items() for pc in pcs}
            region_medians['Region'] = region_medians['Postcode'].map(postcode_to_region)
            median_data = region_medians.groupby('Region')['Price'].median().sort_values()
            title = "Median House Prices by Region"
            x_label = "Region"
            logger.info(f"No filter: {len(median_data)} regions")
        
        if median_data.empty:
            logger.warning("No median data to plot")
            plt.figure(figsize=(12, 6))
            plt.title("No Data Matches Filters")
            plt.savefig(chart_path, bbox_inches='tight')
            plt.close()
            return chart_path
        
        plt.figure(figsize=(12, 6))
        median_data.plot(kind='bar', color='skyblue')
        plt.title(title)
        plt.ylabel('Median Price ($)')
        plt.xlabel(x_label)
        for i, v in enumerate(median_data):
            plt.text(i, v, f"${v:,.0f}", ha='center', va='bottom')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(chart_path, bbox_inches='tight', dpi=100)
        plt.close()
        logger.info(f"Region median chart generated at {chart_path}")
    except Exception as e:
        logger.error(f"Failed to generate chart: {str(e)}", exc_info=True)
        plt.figure(figsize=(12, 6))
        plt.title("Chart Generation Failed")
        plt.savefig(chart_path, bbox_inches='tight')
        plt.close()
    sys.stdout.flush()
    return chart_path

def startup_tasks():
    global df, MEDIAN_ALL_REGIONS, startup_complete
    logger.info("Starting startup tasks...")
    sys.stdout.flush()
    
    try:
        with lock:
            with zipfile.ZipFile("static/Property_Sales_Data.zip", 'r') as zip_ref:
                with zip_ref.open(zip_ref.namelist()[0]) as csv_file:
                    df = pd.read_csv(csv_file)
            df["Settlement Date"] = pd.to_datetime(df["Settlement Date"])
            df["Settlement Date Str"] = df["Settlement Date"].dt.strftime('%Y-%m-%d')
            df["StreetOnly"] = df["Street"].str.extract(r'^\d*\s*(.*)$', expand=False)
            MEDIAN_ALL_REGIONS = df["Price"].median()
        logger.info(f"Data loaded: {len(df)} rows")
        sys.stdout.flush()
        
        generate_heatmap()
        generate_region_median_chart()
        
        startup_complete = True
        logger.info("Startup tasks completed")
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}", exc_info=True)
        sys.stdout.flush()

@app.route('/health')
def health_check():
    logger.info("Health check: OK")
    sys.stdout.flush()
    return "OK", 200

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
        
        # Handle both POST (form) and GET (URL) requests
        if request.method == "POST":
            selected_region = request.form.get("region", "")
            selected_postcode = request.form.get("postcode", "")
            selected_suburb = request.form.get("suburb", "")
            if "region" in request.form or "postcode" in request.form:
                selected_suburb = ""  # Reset to "All Suburbs"
            selected_property_type = request.form.get("property_type", "HOUSE")
            sort_by = request.form.get("sort_by", "Street")
        else:
            selected_region = request.args.get("region", "")
            selected_postcode = request.args.get("postcode", "")
            selected_suburb = request.args.get("suburb", "")
            selected_property_type = request.args.get("property_type", "HOUSE")
            sort_by = request.args.get("sort_by", "Street")
        
        unique_postcodes = sorted(df_local["Postcode"].unique()) if not df_local.empty else []
        unique_suburbs = sorted(df_local["Suburb"].unique()) if not df_local.empty else []
        
        if selected_region:
            unique_postcodes = [pc for pc in unique_postcodes if pc in REGION_POSTCODE_LIST.get(selected_region, [])]
        if selected_postcode:
            unique_suburbs = sorted(df_local[df_local["Postcode"] == selected_postcode]["Suburb"].unique())
        elif selected_region:
            postcodes = REGION_POSTCODE_LIST.get(selected_region, [])
            unique_suburbs = sorted(df_local[df_local["Postcode"].isin(postcodes)]["Suburb"].unique())
        
        logger.info(f"REQUEST: method={request.method}, region={selected_region}, postcode={selected_postcode}, suburb={selected_suburb}, type={selected_property_type}, sort={sort_by}")
        sys.stdout.flush()
        
        display_postcode = selected_postcode if selected_region else ""
        display_suburb = selected_suburb if selected_region and selected_postcode else ""
        
        chart_path = generate_region_median_chart(selected_region, selected_postcode)
        
        filtered_df = df_local.copy()
        properties = []
        median_price = 0
        
        # Apply filters if selected, otherwise use all data
        if selected_suburb:
            filtered_df = filtered_df[filtered_df["Suburb"].str.upper() == selected_suburb.upper()]
        if selected_postcode:  # Changed from elif to if to allow stacking filters
            filtered_df = filtered_df[filtered_df["Postcode"] == selected_postcode]
        if selected_region:    # Changed from elif to if
            filtered_df = filtered_df[filtered_df["Postcode"].isin(REGION_POSTCODE_LIST.get(selected_region, []))]
        
        if selected_property_type != "ALL":
            filtered_df = filtered_df[filtered_df["Property Type"] == selected_property_type]
        
        if not filtered_df.empty:
            if sort_by == "Street":
                properties = filtered_df.sort_values(by="StreetOnly").to_dict('records')
            elif sort_by == "Settlement Date":
                properties = filtered_df.sort_values(by="Settlement Date").to_dict('records')
            else:
                properties = filtered_df.sort_values(by=sort_by).to_dict('records')
            median_price = filtered_df["Price"].median()
        else:
            logger.warning("No properties match the filters")
        
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
                                  median_all_regions=MEDIAN_ALL_REGIONS,
                                  display_suburb=selected_suburb if selected_suburb else None,
                                  now_timestamp=int(time.time()))  # Cache-busting timestamp
        logger.info("RENDERING: Completed")
        sys.stdout.flush()
        return response
    except Exception as e:
        logger.error(f"ERROR: {str(e)}", exc_info=True)
        sys.stdout.flush()
        return "Internal Server Error", 500

# Add custom filter for currency formatting
@app.template_filter('currency')
def currency_filter(value):
    return "${:,.2f}".format(value) if value else "N/A"

if __name__ == "__main__":
    if os.getenv("GUNICORN_CMD_ARGS"):
        logger.info("Gunicorn environment detected, running startup tasks")
        startup_tasks()
    else:
        logger.info("Running in development mode, starting startup tasks in thread")
        threading.Thread(target=startup_tasks, daemon=True).start()
    app.run(host="0.0.0.0", port=10000)