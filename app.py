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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader

app = Flask(__name__)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', force=True)
logger = logging.getLogger(__name__)

# Global variables
lock = threading.Lock()
df = None
MEDIAN_ALL_REGIONS = 0
startup_complete = False

# Region and postcode mappings
REGION_POSTCODE_LIST = {
    "Central Coast": ["2083", "2250", "2251", "2256", "2257", "2258", "2259", "2260", "2261", "2262", "2263", "2775"],
    "Coffs Harbour - Grafton": ["2370", "2450", "2456", "2460", "2462", "2463", "2464", "2465", "2466", "2469", "2441", "2448", "2449", "2450", "2452", "2453", "2454", "2455", "2456", "2460"],
    "Hunter Valley excl Newcastle": ["2250", "2311", "2320", "2321", "2322", "2323", "2325", "2326", "2327", "2330", "2331", "2334", "2335", "2420", "2421", "2314", "2315", "2316", "2317", "2318", "2319", "2324", "2328", "2329", "2333", "2336", "2337", "2338", "2850"],
    "Mid North Coast": ["2312", "2324", "2415", "2420", "2422", "2423", "2425", "2428", "2429", "2430", "2431", "2440", "2441", "2447", "2448", "2449", "2898", "2439", "2443", "2444", "2445", "2446", "2424", "2426", "2427"],
    "Newcastle and Lake Macquarie": ["2259", "2264", "2265", "2267", "2278", "2280", "2281", "2282", "2283", "2284", "2285", "2286", "2287", "2289", "2290", "2291", "2292", "2293", "2294", "2295", "2296", "2297", "2298", "2299", "2300", "2302", "2303", "2304", "2305", "2306", "2307", "2308", "2318", "2322", "2323"]
}
ALLOWED_POSTCODES = {pc for region in REGION_POSTCODE_LIST.values() for pc in region}
POSTCODE_COORDS = {
    "2250": [-33.28, 151.41], "2251": [-33.31, 151.42], "2621": [-35.42, 149.81], "2217": [-33.97, 151.11],
    "2408": [-29.05, 148.77], "2460": [-29.68, 152.93], "2582": [-34.98, 149.23], "2580": [-34.57, 148.93],
    "2843": [-31.95, 149.43], "2650": [-35.12, 147.35], "2795": [-33.42, 149.58], "2444": [-31.65, 152.79],
    "2000": [-33.87, 151.21], "2261": [-33.30, 151.50], "2450": [-30.30, 153.12], "2320": [-32.73, 151.55],
    "2325": [-32.58, 151.17], "2430": [-31.90, 152.46], "2300": [-32.9283, 151.7817], "2280": [-33.0333, 151.6333],
    "2285": [-32.9667, 151.6333]  # Added 2325 for Cessnock
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
    "2443": {"JOHNS RIVER": [-31.73, 152.70]}, "2621": {"BUNGENDORE": [-35.25, 149.44]},
    "2217": {"MONTEREY": [-33.97, 151.15]}, "2408": {"NORTH STAR": [-28.93, 150.39]},
    "2460": {"COOMBADJHA": [-29.03, 152.38]}, "2582": {"YASS": [-34.84, 148.91]},
    "2580": {"GOULBURN": [-34.75, 149.72]}, "2843": {"COOLAH": [-31.82, 149.72]},
    "2650": {"WAGGA WAGGA": [-35.12, 147.35]}, "2795": {"BATHURST": [-33.42, 149.58]},
    "2444": {"THRUMSTER": [-31.47, 152.83]}, "2000": {"SYDNEY": [-33.87, 151.21]},
    "2325": {"CESSNOCK": [-32.83, 151.35]}  # Added Cessnock
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

def parse_dat_file(dat_content):
    """Parse .DAT file content into a DataFrame based on B records."""
    try:
        lines = dat_content.read().decode('utf-8').splitlines()
        data = []
        for line in lines:
            if line.startswith('B'):
                fields = line.split(';')
                logger.info(f"Raw DAT line: {line}")  # Log raw line for debugging
                if len(fields) >= 19:
                    try:
                        record = {
                            "Postcode": fields[10].strip(),  # Corrected to field 10
                            "Suburb": fields[9].strip().upper(),  # Corrected to field 9
                            "Price": int(float(fields[15].strip() or 0)),  # Field 15 is correct
                            "Settlement Date": fields[14].strip(),  # Field 14 is correct
                            "Street": f"{fields[7].strip()} {fields[8].strip()}".strip() if fields[7].strip() else fields[8].strip(),  # Fields 7+8 correct
                            "Property Type": "RESIDENCE" if fields[17].strip() == "R" else "VACANT LAND"  # Field 17 correct
                        }
                        # Validate postcode
                        if not (record["Postcode"].isdigit() and len(record["Postcode"]) == 4):
                            logger.warning(f"Invalid postcode: {record['Postcode']} in line: {line}")
                            continue
                        data.append(record)
                    except (ValueError, IndexError) as e:
                        logger.error(f"Error parsing line: {line} - {str(e)}")
                        continue
        if not data:
            logger.warning("No valid B records found in .DAT file")
            return pd.DataFrame(columns=["Postcode", "Suburb", "Price", "Settlement Date", "Street", "Property Type"])
        df = pd.DataFrame(data)
        logger.info(f"Parsed {len(df)} valid records")
        return df
    except Exception as e:
        logger.error(f"Failed to parse .DAT file: {str(e)}", exc_info=True)
        return pd.DataFrame(columns=["Postcode", "Suburb", "Price", "Settlement Date", "Street", "Property Type"])

def startup_tasks():
    global df, MEDIAN_ALL_REGIONS, startup_complete
    logger.info("STARTUP: Entering startup_tasks...")
    sys.stdout.flush()
    
    try:
        with lock:
            zip_path = "2025.zip"
            logger.info(f"STARTUP: Checking directory: {os.getcwd()}")
            logger.info(f"STARTUP: Files present: {os.listdir('.')}")
            logger.info(f"STARTUP: Looking for {zip_path}")
            sys.stdout.flush()
            
            if not os.path.exists(zip_path):
                logger.error(f"STARTUP: File not found: {zip_path}")
                df = pd.DataFrame(columns=["Postcode", "Suburb", "Price", "Settlement Date", "Street", "Property Type"])
                startup_complete = True
                logger.info("STARTUP: Proceeding with empty DataFrame due to missing file")
                sys.stdout.flush()
                return
            
            logger.info("STARTUP: Opening 2025.zip...")
            sys.stdout.flush()
            with zipfile.ZipFile(zip_path, 'r') as outer_zip:
                inner_zips = [f for f in outer_zip.namelist() if f.endswith('.zip')]
                logger.info(f"STARTUP: Found {len(inner_zips)} inner zips: {inner_zips}")
                sys.stdout.flush()
                
                if not inner_zips:
                    logger.error("STARTUP: No inner zip files found in 2025.zip")
                    df = pd.DataFrame(columns=["Postcode", "Suburb", "Price", "Settlement Date", "Street", "Property Type"])
                    startup_complete = True
                    logger.info("STARTUP: Proceeding with empty DataFrame due to no inner zips")
                    sys.stdout.flush()
                    return
                
                all_data = []
                for inner_zip_name in inner_zips:
                    logger.info(f"STARTUP: Processing {inner_zip_name}")
                    sys.stdout.flush()
                    with outer_zip.open(inner_zip_name) as inner_zip_file:
                        with zipfile.ZipFile(inner_zip_file, 'r') as inner_zip:
                            dat_files = [f for f in inner_zip.namelist() if f.endswith('.DAT')]
                            logger.info(f"STARTUP: Found {len(dat_files)} .DAT files: {dat_files}")
                            sys.stdout.flush()
                            
                            for dat_file in dat_files:
                                logger.info(f"STARTUP: Parsing {dat_file}")
                                sys.stdout.flush()
                                with inner_zip.open(dat_file) as dat_content:
                                    df_chunk = parse_dat_file(dat_content)
                                    if not df_chunk.empty:
                                        all_data.append(df_chunk)
                
                if not all_data:
                    logger.warning("STARTUP: No data loaded, using empty DataFrame")
                    df = pd.DataFrame(columns=["Postcode", "Suburb", "Price", "Settlement Date", "Street", "Property Type"])
                else:
                    df = pd.concat(all_data, ignore_index=True)
                    logger.info(f"STARTUP: Combined {len(df)} rows")
                    sys.stdout.flush()
                    
                    df["Settlement Date"] = pd.to_datetime(df["Settlement Date"], format='%Y%m%d')
                    df["Settlement Date Str"] = df["Settlement Date"].dt.strftime('%Y-%m-%d')
                    df["StreetOnly"] = df["Street"].str.extract(r'^\d*\s*(.*)$', expand=False)
                    MEDIAN_ALL_REGIONS = df["Price"].median()
        
        logger.info(f"STARTUP: Data loaded: {len(df)} rows")
        sys.stdout.flush()
        
        generate_heatmap()
        generate_region_median_chart()
        
        startup_complete = True
        logger.info("STARTUP: Startup tasks completed successfully")
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"STARTUP: Failed: {str(e)}", exc_info=True)
        sys.stdout.flush()
        df = pd.DataFrame(columns=["Postcode", "Suburb", "Price", "Settlement Date", "Street", "Property Type"])
        startup_complete = True

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
        
        if request.method == "POST":
            selected_region = request.form.get("region", "")
            selected_postcode = request.form.get("postcode", "")
            selected_suburb = request.form.get("suburb", "")
            if "region" in request.form or "postcode" in request.form:
                selected_suburb = ""
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
        
        if selected_suburb:
            filtered_df = filtered_df[filtered_df["Suburb"].str.upper() == selected_suburb.upper()]
        if selected_postcode:
            filtered_df = filtered_df[filtered_df["Postcode"] == selected_postcode]
        if selected_region:
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
                                  now_timestamp=int(time.time()))
        logger.info("RENDERING: Completed")
        sys.stdout.flush()
        return response
    except Exception as e:
        logger.error(f"ERROR: {str(e)}", exc_info=True)
        sys.stdout.flush()
        return "Internal Server Error", 500

@app.route('/hot_suburbs', methods=["GET", "POST"])
def hot_suburbs():
    logger.info("START: Request received for /hot_suburbs")
    sys.stdout.flush()
    
    if not is_startup_complete():
        logger.info("STARTUP INCOMPLETE: Returning loading.html")
        sys.stdout.flush()
        return render_template("loading.html")
    
    with lock:
        df_local = df.copy() if df is not None else pd.DataFrame()
    
    sort_by = request.form.get("sort_by", "Suburb") if request.method == "POST" else "Suburb"
    
    if df_local.empty:
        logger.warning("No data for hot suburbs")
        return render_template("hot_suburbs.html", data_source="NSW Valuer General Data", hot_suburbs=[], total_suburbs=0, median_all_regions=0, sort_by=sort_by)
    
    # Calculate median prices per suburb
    suburb_medians = df_local[df_local["Property Type"] == "RESIDENCE"].groupby(["Suburb", "Postcode"])["Price"].median().reset_index()
    postcode_to_region = {pc: region for region, pcs in REGION_POSTCODE_LIST.items() for pc in pcs}
    suburb_medians["Region"] = suburb_medians["Postcode"].map(postcode_to_region)
    
    # Filter hot suburbs (below overall median)
    hot_suburbs_df = suburb_medians[suburb_medians["Price"] < MEDIAN_ALL_REGIONS].dropna()
    
    # Sort
    if sort_by == "Median Price (House)":
        hot_suburbs_df = hot_suburbs_df.sort_values("Price")
    else:
        hot_suburbs_df = hot_suburbs_df.sort_values("Suburb")
    
    # Convert to list of dicts
    hot_suburbs = [
        {"suburb": row["Suburb"], "postcode": row["Postcode"], "region": row["Region"], "median_price": row["Price"]}
        for _, row in hot_suburbs_df.iterrows()
    ]
    
    total_suburbs = len(suburb_medians)
    
    logger.info(f"RENDERING: Hot suburbs - {len(hot_suburbs)} found")
    sys.stdout.flush()
    return render_template("hot_suburbs.html", data_source="NSW Valuer General Data", hot_suburbs=hot_suburbs, total_suburbs=total_suburbs, median_all_regions=MEDIAN_ALL_REGIONS, sort_by=sort_by)

@app.template_filter('currency')
def currency_filter(value):
    return "${:,.2f}".format(value) if value else "N/A"

# Start startup tasks in a background thread
logger.info("INIT: Before starting startup thread")
sys.stdout.flush()
startup_thread = threading.Thread(target=startup_tasks, name="StartupThread")
startup_thread.start()
logger.info("INIT: Startup thread started")
sys.stdout.flush()

if __name__ == "__main__":
    logger.info("Starting Flask app...")
    sys.stdout.flush()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))