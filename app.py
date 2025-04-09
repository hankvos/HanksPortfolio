import logging
import sys
import os
import zipfile
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for chart generation
import matplotlib.pyplot as plt
import folium
from flask import Flask, render_template, request
from threading import Thread, Lock
import psutil

app = Flask(__name__)

# Configure logging
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s"))
logger.addHandler(handler)

# Global variables
df = None
lock = Lock()
NATIONAL_MEDIAN = 0
REGION_POSTCODE_LIST = {}  # Populate this if needed, e.g., from a config file

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    logger.info(f"Memory usage: RSS={mem.rss / 1024 / 1024:.2f} MB, VMS={mem.vms / 1024 / 1024:.2f} MB")
    sys.stdout.flush()

def load_property_data():
    global df, NATIONAL_MEDIAN
    logger.info("Loading property data...")
    sys.stdout.flush()
    
    try:
        # Only process 2025.zip
        zip_files = ["2025.zip"]
        logger.info(f"Processing files: {zip_files}")
        sys.stdout.flush()
        
        result_df = pd.DataFrame()
        for zip_file in zip_files:
            if not os.path.exists(zip_file):
                logger.warning(f"File not found: {zip_file}, skipping")
                sys.stdout.flush()
                continue
            with zipfile.ZipFile(zip_file, 'r') as z:
                for file_name in z.namelist():
                    if file_name.endswith('.csv'):
                        logger.info(f"Reading {file_name} from {zip_file}")
                        sys.stdout.flush()
                        with z.open(file_name) as f:
                            temp_df = pd.read_csv(f, delimiter=';', quotechar='"', low_memory=False)
                            result_df = pd.concat([result_df, temp_df], ignore_index=True)
        
        logger.info(f"Processed {len(result_df)} records from {zip_files}")
        sys.stdout.flush()
        
        # Clean and process data (adjust based on your CSV structure)
        result_df['Price'] = pd.to_numeric(result_df['Sale Price'], errors='coerce')
        result_df['Property Type'] = result_df['Property Type'].fillna('UNKNOWN').str.upper()
        simplified_types = {
            'RESIDENCE': 'HOUSE', 'HOME UNIT': 'HOUSE', 'DUPLEX': 'HOUSE',
            'VACANT LAND': 'VACANT LAND', 'FARM LAND': 'FARM', 'FARMLAND': 'FARM', 'FARM': 'FARM',
            'COMMERCIAL': 'COMMERCIAL', 'WAREHOUSE': 'COMMERCIAL', 'SHOP': 'COMMERCIAL',
            'FACTORY': 'COMMERCIAL', 'OFFICE': 'COMMERCIAL', 'RETAIL': 'COMMERCIAL'
        }
        result_df['Property Type'] = result_df['Property Type'].map(lambda x: simplified_types.get(x, 'OTHER'))
        
        with lock:
            df = result_df
            NATIONAL_MEDIAN = df['Price'].median() if not df['Price'].isna().all() else 0
        logger.info(f"National median price calculated: ${NATIONAL_MEDIAN:,.0f}")
        logger.info(f"Loaded {len(df)} records into DataFrame")
        sys.stdout.flush()
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}", exc_info=True)
        sys.stdout.flush()
        raise

def generate_heatmap():
    logger.info("Generating heatmap...")
    sys.stdout.flush()
    m = folium.Map(location=[-33.8688, 151.2093], zoom_start=10)  # Example: Sydney
    m.save("static/heatmap.html")
    logger.info("Heatmap generated at static/heatmap.html")
    sys.stdout.flush()

def generate_region_median_chart(region=None, postcode=None):
    logger.info("Generating region median chart...")
    sys.stdout.flush()
    plt.figure(figsize=(10, 6))
    plt.title("Region Median Chart (Placeholder)")
    plt.savefig("static/region_median_chart.png")
    plt.close()
    logger.info("Region median chart generated at static/region_median_chart.png")
    sys.stdout.flush()
    return "static/region_median_chart.png"

def startup_tasks():
    logger.info("Starting background data load...")
    sys.stdout.flush()
    load_property_data()
    log_memory_usage()
    logger.info("Pre-generating charts in background...")
    sys.stdout.flush()
    generate_heatmap()
    generate_region_median_chart()
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
    
    startup_complete = is_startup_complete()
    logger.info(f"STARTUP STATUS: complete={startup_complete}")
    sys.stdout.flush()
    if not startup_complete:
        logger.info("STARTUP INCOMPLETE: Returning loading.html")
        sys.stdout.flush()
        return render_template("loading.html")
    
    try:
        logger.info("PROCESSING: Entering main logic")
        sys.stdout.flush()
        
        # Log memory before heavy operations
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024
        logger.info(f"MEMORY BEFORE: RSS={mem_before:.2f} MB")
        sys.stdout.flush()
        
        logger.info("DATAFRAME: Attempting to access df")
        sys.stdout.flush()
        if 'df' not in globals() or df is None:
            logger.error("DATAFRAME ERROR: df is None or not initialized")
            sys.stdout.flush()
            raise ValueError("DataFrame not initialized")
        
        logger.info("DATAFRAME: Copying df")
        sys.stdout.flush()
        df_local = df.copy()
        logger.info(f"DATAFRAME: Copied {len(df_local)} rows")
        sys.stdout.flush()
        
        # Log memory after copy
        mem_after = process.memory_info().rss / 1024 / 1024
        logger.info(f"MEMORY AFTER COPY: RSS={mem_after:.2f} MB")
        sys.stdout.flush()
        
        # Minimal response for testing
        logger.info("RENDERING: Starting template render")
        sys.stdout.flush()
        response = render_template("index.html",
                                  properties=df_local.head(10).to_dict('records'),
                                  national_median=NATIONAL_MEDIAN,
                                  heatmap_path="static/heatmap.html" if os.path.exists("static/heatmap.html") else None,
                                  region_median_chart_path="static/region_median_chart.png" if os.path.exists("static/region_median_chart.png") else None)
        logger.info("RENDERING: Template rendered successfully")
        sys.stdout.flush()
        
        return response
    except Exception as e:
        logger.error(f"ERROR: Exception in index: {str(e)}", exc_info=True)
        sys.stdout.flush()
        return "Internal Server Error", 500
    finally:
        logger.info("END: Request processing completed")
        sys.stdout.flush()

if __name__ == "__main__":
    if not os.path.exists("static"):
        os.makedirs("static")
    Thread(target=startup_tasks).start()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
else:
    gunicorn_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    if not os.path.exists("static"):
        os.makedirs("static")
    Thread(target=startup_tasks).start()