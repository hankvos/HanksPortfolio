# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import pandas as pd
import zipfile
from io import BytesIO
import glob
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

app = Flask(__name__)

# Define regions and their postcode ranges
REGION_POSTCODES = {
    "Central Coast": ["2083", "2250", "2251", "2256", "2257", "2258", "2259", "2260", "2261", "2262", "2263", "2775"],
    "Coffs Harbour - Grafton": ["2370", "2450", "2456", "2460", "2462", "2463", "2464", "2465", "2466", "2469", "2441", "2448", "2449", "2452", "2453", "2454", "2455"],
    "Hunter Valley excl Newcastle": ["2250", "2311", "2320", "2321", "2322", "2323", "2325", "2326", "2327", "2330", "2331", "2334", "2335", "2420", "2421", "2314", "2315", "2316", "2317", "2318", "2319", "2324", "2328", "2329", "2333", "2336", "2337", "2338", "2850"],
    "Mid North Coast": ["2312", "2324", "2415", "2420", "2422", "2423", "2425", "2428", "2429", "2430", "2431", "2440", "2441", "2447", "2448", "2449", "2898", "2439", "2443", "2444", "2445", "2446", "2424", "2426", "2427"],
    "Newcastle and Lake Macquarie": ["2259", "2264", "2265", "2267", "2278", "2280", "2281", "2282", "2283", "2284", "2285", "2286", "2287", "2289", "2290", "2291", "2292", "2293", "2294", "2295", "2296", "2297", "2298", "2299", "2300", "2302", "2303", "2304", "2305", "2306", "2307", "2308", "2318", "2322", "2323"]
}

def expand_postcode_ranges(ranges):
    """Expand postcode ranges (e.g., '2250-2252' -> ['2250', '2251', '2252'])."""
    postcodes = []
    for r in ranges:
        if "-" in r:
            start, end = map(int, r.split("-"))
            postcodes.extend(str(i) for i in range(start, end + 1))
        else:
            postcodes.append(r)
    return sorted(postcodes)

REGION_POSTCODE_LIST = {region: expand_postcode_ranges(ranges) for region, ranges in REGION_POSTCODES.items()}
PROPERTY_DF = None
REGION_SUBURBS = {}

def load_property_data(zip_files=None):
    """Load property data: all inner ZIPs from 2025, only 20241230 from 2024."""
    if zip_files is None:
        zip_files = glob.glob("[2][0][2][4-5].zip")
        print(f"ZIP files found: {zip_files}")
    if not zip_files:
        raise ValueError("No ZIP files found for 2024-2025.")

    column_names = [
        "Record Type", "District Code", "Property ID", "Unit Number",
        "Unused1", "Unused2", "Potential Unit", "House Number",
        "Street Name", "Suburb", "Postcode", "Area",
        "Unused3", "Unused4", "Settlement Date", "Sale Price",
        "Unused5", "Unused6", "Property Type"
    ]

    def record_generator():
        date_counts = {}
        for zip_path in sorted(zip_files):
            with zipfile.ZipFile(zip_path, "r") as outer_zip:
                inner_zips = [f for f in outer_zip.namelist() if f.endswith(".zip")]
                dat_files = [f for f in outer_zip.namelist() if f.endswith(".DAT")]
                
                if inner_zips:
                    for inner_zip_name in inner_zips:
                        # Filter: 2024.zip -> only 20241230, 2025.zip -> all
                        if "2024" in zip_path and "20241230" not in inner_zip_name:
                            continue
                        with outer_zip.open(inner_zip_name) as inner_zip_file:
                            with zipfile.ZipFile(BytesIO(inner_zip_file.read())) as inner_zip:
                                for dat_file in inner_zip.namelist():
                                    if dat_file.endswith(".DAT"):
                                        with inner_zip.open(dat_file) as f:
                                            for line in f.read().decode("utf-8").splitlines():
                                                if line.startswith("B;"):
                                                    record = line.split(";")
                                                    if len(record) < 19:
                                                        record.extend([""] * (19 - len(record)))
                                                    elif len(record) > 19:
                                                        record = record[:19]
                                                    settlement_date = record[14]  # 15th field
                                                    date_counts[settlement_date] = date_counts.get(settlement_date, 0) + 1
                                                    yield record
                elif dat_files:
                    for dat_file in dat_files:
                        if "2024" in zip_path and "20241230" not in dat_file:
                            continue
                        with outer_zip.open(dat_file) as f:
                            for line in f.read().decode("utf-8").splitlines():
                                if line.startswith("B;"):
                                    record = line.split(";")
                                    if len(record) < 19:
                                        record.extend([""] * (19 - len(record)))
                                    elif len(record) > 19:
                                        record = record[:19]
                                    settlement_date = record[14]
                                    date_counts[settlement_date] = date_counts.get(settlement_date, 0) + 1
                                    yield record
        print("Unique Settlement Dates and Counts:", date_counts)

    df = pd.DataFrame(record_generator(), columns=column_names)
    df = df.drop(columns=[col for col in df.columns if col.startswith("Unused")])
    
    if df.empty:
        raise ValueError("No B records found in any .DAT files.")
    
    print("Raw Settlement Dates (first 5 records):", df["Settlement Date"].head().tolist())
    
    def determine_property_type(row):
        base_type = row["Property Type"].strip().upper()
        potential_unit = row["Potential Unit"].strip()
        area = row["Area"].strip()
        type_map = {
            "R": "RESIDENCE", "RES": "RESIDENCE", "RESIDENCE": "RESIDENCE",
            "C": "COMMERCIAL", "COM": "COMMERCIAL", "COMMERCIAL": "COMMERCIAL",
            "V": "VACANT LAND", "VAC": "VACANT LAND", "VACANT": "VACANT LAND",
            "S": "SHOP", "SH": "SHOP", "SHOP": "SHOP", "3": "SHOP",
            "": "VACANT LAND"
        }
        mapped_type = type_map.get(base_type, "RESIDENCE")
        if not area and not potential_unit and mapped_type == "RESIDENCE":
            return "VACANT LAND"
        if mapped_type == "RESIDENCE":
            return "UNIT" if potential_unit and potential_unit.isdigit() else "HOUSE"
        return mapped_type

    df["Property Type"] = df.apply(determine_property_type, axis=1)
    df["Sale Price"] = pd.to_numeric(df["Sale Price"], errors="coerce")
    df_filtered = df[df["Sale Price"].notna() & df["Sale Price"].gt(0)].copy()
    
    if df_filtered.empty:
        raise ValueError("No valid Sale Price data after filtering.")
    
    print("Before conversion (first 5 records):", df_filtered["Settlement Date"].head().tolist())
    df_filtered["Settlement Date"] = pd.to_datetime(df_filtered["Settlement Date"], format="%Y%m%d", errors="coerce").dt.strftime("%d/%m/%Y")
    print("After conversion (first 5 records):", df_filtered["Settlement Date"].head().tolist())
    
    for col in ["Unit Number", "House Number", "Street Name", "Suburb", "Postcode"]:
        df_filtered[col] = df_filtered[col].astype(str).replace("", "")
    
    df_filtered["Address"] = df_filtered.apply(
        lambda row: f"{row['Unit Number']} / {row['House Number']} {row['Street Name']}, {row['Suburb']} NSW {row['Postcode']}"
        if row["Property Type"] == "UNIT" and row["Unit Number"]
        else f"{row['House Number']} {row['Street Name']}, {row['Suburb']} NSW {row['Postcode']}", axis=1
    )
    df_filtered = df_filtered.rename(columns={"Sale Price": "Price"})
    df_filtered["Price"] = df_filtered["Price"].astype(float).round(0)
    df_filtered["Size"] = df_filtered["Area"].replace("", "N/A").fillna("N/A").astype(str) + " sqm"
    
    print(f"Loaded {len(df_filtered)} records into DataFrame.")
    return df_filtered

def get_region_data(df, region=None, postcode=None, suburb=None, property_type=None, sort_by="Address"):
    """Filter and sort property data based on user inputs."""
    region_data = df.copy()
    if region and region in REGION_POSTCODE_LIST:
        region_data = region_data[region_data["Postcode"].isin(REGION_POSTCODE_LIST[region])]
    if postcode:
        region_data = region_data[region_data["Postcode"] == postcode]
    if suburb:
        region_data = region_data[region_data["Suburb"] == suburb]
    if property_type:
        region_data = region_data[region_data["Property Type"] == property_type]
    if region_data.empty:
        return pd.DataFrame()

    if sort_by == "Address":
        region_data["HouseNumNumeric"] = pd.to_numeric(region_data["House Number"].str.extract(r'^(\d+)')[0], errors="coerce").fillna(999999)
        region_data = region_data.sort_values(["Street Name", "HouseNumNumeric"])
    elif sort_by == "Price":
        region_data = region_data.sort_values("Price")
    elif sort_by == "Block Size":
        region_data["SizeNumeric"] = pd.to_numeric(region_data["Size"].str.replace(" sqm", ""), errors="coerce").fillna(0)
        region_data = region_data.sort_values("SizeNumeric")
    elif sort_by == "Settlement Date":
        region_data["SaleDateNumeric"] = pd.to_datetime(region_data["Settlement Date"], format="%d/%m/%Y", errors="coerce")
        region_data = region_data.sort_values("SaleDateNumeric")
    return region_data.drop(columns=["HouseNumNumeric", "SizeNumeric", "SaleDateNumeric"], errors="ignore")

def calculate_avg_price(properties):
    """Calculate average price from a DataFrame."""
    return round(properties["Price"].mean()) if not properties["Price"].empty else 0

def calculate_median_house_by_region(df):
    """Calculate median house price by region."""
    median_by_region = {}
    for region, postcodes in REGION_POSTCODE_LIST.items():
        region_data = df[(df["Postcode"].isin(postcodes)) & (df["Property Type"] == "HOUSE")]
        if not region_data.empty:
            median_price = region_data["Price"].median()
            if np.isfinite(median_price):
                median_by_region[region] = {"price": int(median_price)}
    return median_by_region

def calculate_median_house_by_postcode(df, region):
    """Calculate median house price by postcode within a region."""
    median_by_postcode = {}
    region_data = df[(df["Postcode"].isin(REGION_POSTCODE_LIST[region])) & (df["Property Type"] == "HOUSE")]
    for postcode in sorted(region_data["Postcode"].unique()):
        postcode_data = region_data[region_data["Postcode"] == postcode]
        if not postcode_data.empty:
            median_price = postcode_data["Price"].median()
            if np.isfinite(median_price):
                median_by_postcode[postcode] = {"price": int(median_price)}
    return median_by_postcode

def calculate_median_house_by_suburb(df, postcode):
    """Calculate median house price by suburb within a postcode."""
    median_by_suburb = {}
    postcode_data = df[(df["Postcode"] == postcode) & (df["Property Type"] == "HOUSE")]
    for suburb in sorted(postcode_data["Suburb"].unique()):
        suburb_data = postcode_data[postcode_data["Suburb"] == suburb]
        if not suburb_data.empty:
            median_price = suburb_data["Price"].median()
            if np.isfinite(median_price):
                median_by_suburb[suburb] = {"price": int(median_price)}
    return median_by_suburb

def generate_median_house_price_chart(df, data_dict, chart_type="region", selected_region=None, selected_postcode=None):
    """Generate a bar chart for median house prices."""
    if not data_dict:
        return None
    os.makedirs('static', exist_ok=True)

    labels = list(data_dict.keys())
    prices = [data["price"] for data in data_dict.values()]
    title_prefix = {
        "region": "Median House Price by Region",
        "postcode": f"Median House Price by Postcode in {selected_region}",
        "suburb": f"Median House Price by Suburb in Postcode {selected_postcode}"
    }.get(chart_type, "Median House Price")

    house_data = df[df["Property Type"] == "HOUSE"].copy()
    house_data["Settlement Date"] = pd.to_datetime(house_data["Settlement Date"], format="%d/%m/%Y", errors="coerce")
    date_range = f" (Jan 2024 - {datetime.now().strftime('%b %Y')})" if house_data["Settlement Date"].dropna().empty else \
                 f" ({house_data['Settlement Date'].min().strftime('%b %Y')} - {house_data['Settlement Date'].max().strftime('%b %Y')})"

    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, prices, color='#4682B4', edgecolor='black', width=0.6)
    plt.title(f"{title_prefix}{date_range}", fontsize=12, pad=15)
    plt.xlabel("Suburb" if chart_type == "suburb" else "Postcode" if chart_type == "postcode" else "Region", fontsize=10)
    plt.ylabel("Median Price ($)", fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.grid(True, axis='y', alpha=0.2)
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(prices)*0.02,
                 f"${bar.get_height():,}", ha='center', va='bottom', fontsize=8, color='#333')
    
    chart_path = 'static/median_house_price_chart.png'
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    return chart_path

def generate_plots(region_data, selected_region, selected_postcode, selected_suburb):
    """Generate histogram and scatter plot for price distribution and price vs size."""
    os.makedirs('static', exist_ok=True)

    # Price Histogram
    prices = region_data["Price"].dropna()
    plt.figure(figsize=(8, 4))
    plt.hist(prices / 1e6, bins=20, range=(0, 3), color='#87CEEB', edgecolor='black')
    plt.title(f"Price Distribution - {selected_region}{' - ' + selected_postcode if selected_postcode else ''}{' - ' + selected_suburb if selected_suburb else ''}", fontsize=12)
    plt.xlabel("Sale Price ($million)", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.xlim(0, 3)
    plt.grid(True, alpha=0.2)
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
    price_hist_path = 'static/price_histogram.png'
    plt.savefig(price_hist_path)
    plt.close()

    # Price vs Size Scatter (if Size data is available)
    sizes = pd.to_numeric(region_data["Size"].str.replace(" sqm", ""), errors="coerce")
    if sizes.notna().sum() > 0:
        plt.figure(figsize=(8, 4))
        plt.scatter(sizes, prices / 1e6, alpha=0.5, color='#FF6347')
        plt.title(f"Price vs Size - {selected_region}{' - ' + selected_postcode if selected_postcode else ''}{' - ' + selected_suburb if selected_suburb else ''}", fontsize=12)
        plt.xlabel("Size (sqm)", fontsize=10)
        plt.ylabel("Sale Price ($million)", fontsize=10)
        plt.grid(True, alpha=0.2)
        plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
        price_size_scatter_path = 'static/price_size_scatter.png'
        plt.savefig(price_size_scatter_path)
        plt.close()
    else:
        price_size_scatter_path = None

    return price_hist_path, price_size_scatter_path

def calculate_stats(region_data):
    """Calculate mean, median, and standard deviation of prices."""
    prices = region_data["Price"].dropna()
    return {
        "mean": round(np.mean(prices)) if not prices.empty else 0,
        "median": round(np.median(prices)) if not prices.empty else 0,
        "std": round(np.std(prices)) if not prices.empty else 0
    }

@app.route("/", methods=["GET", "POST"])
def index():
    """Main route for rendering the property analyzer interface."""
    try:
        regions = sorted(REGION_POSTCODE_LIST.keys())
        selected_region = selected_postcode = selected_suburb = None
        selected_property_type = "HOUSE"
        properties = []
        avg_price = 0
        sort_by = "Address"
        postcodes = suburbs = []
        price_hist_path = price_size_scatter_path = None
        stats = {"mean": 0, "median": 0, "std": 0}
        data_source = "Data provided by NSW Valuer General Property Sales Information, last updated March 24, 2025"

        median_by_region = calculate_median_house_by_region(PROPERTY_DF)
        median_chart_path = generate_median_house_price_chart(PROPERTY_DF, median_by_region, chart_type="region")

        if request.method == "POST" and request.form.get("region"):
            selected_region = request.form.get("region")
            selected_postcode = request.form.get("postcode", "")
            selected_suburb = request.form.get("suburb", "")
            selected_property_type = request.form.get("property_type", "HOUSE")
            sort_by = request.form.get("sort_by", "Address")

            if selected_region in REGION_POSTCODE_LIST:
                postcodes = sorted(PROPERTY_DF[PROPERTY_DF["Postcode"].isin(REGION_POSTCODE_LIST[selected_region])]["Postcode"].unique())
            if selected_postcode:
                suburbs = sorted(PROPERTY_DF[PROPERTY_DF["Postcode"] == selected_postcode]["Suburb"].unique())

            region_data = get_region_data(PROPERTY_DF, selected_region, selected_postcode, selected_suburb, selected_property_type, sort_by)
            if not region_data.empty:
                properties = region_data[["Address", "Price", "Size", "Settlement Date"]].to_dict("records")
                avg_price = calculate_avg_price(region_data)
                price_hist_path, price_size_scatter_path = generate_plots(region_data, selected_region, selected_postcode, selected_suburb)
                stats = calculate_stats(region_data)

                if selected_region and not selected_postcode:
                    median_by_postcode = calculate_median_house_by_postcode(PROPERTY_DF, selected_region)
                    median_chart_path = generate_median_house_price_chart(PROPERTY_DF, median_by_postcode, chart_type="postcode", selected_region=selected_region)
                elif selected_postcode:
                    median_by_suburb = calculate_median_house_by_suburb(PROPERTY_DF, selected_postcode)
                    median_chart_path = generate_median_house_price_chart(PROPERTY_DF, median_by_suburb, chart_type="suburb", selected_postcode=selected_postcode)

        return render_template(
            "index.html",
            regions=regions,
            postcodes=postcodes,
            suburbs=suburbs,
            selected_region=selected_region,
            selected_postcode=selected_postcode,
            selected_suburb=selected_suburb,
            selected_property_type=selected_property_type,
            properties=properties,
            avg_price=avg_price,
            sort_by=sort_by,
            price_hist_path=price_hist_path,
            stats=stats,
            median_chart_path=median_chart_path,
            price_size_scatter_path=price_size_scatter_path,
            data_source=data_source
        )
    except Exception as e:
        return render_template(
            "index.html",
            error=str(e),
            regions=sorted(REGION_POSTCODE_LIST.keys()),
            data_source=data_source
        )

@app.route("/get_postcodes", methods=["GET"])
def get_postcodes():
    """Return postcodes for a selected region."""
    region = request.args.get("region")
    if region in REGION_POSTCODE_LIST:
        return jsonify(sorted(PROPERTY_DF[PROPERTY_DF["Postcode"].isin(REGION_POSTCODE_LIST[region])]["Postcode"].unique()))
    return jsonify([])

@app.route("/get_suburbs", methods=["GET"])
def get_suburbs():
    """Return suburbs for a selected region and postcode."""
    region = request.args.get("region")
    postcode = request.args.get("postcode")
    if region in REGION_POSTCODE_LIST and postcode:
        region_data = PROPERTY_DF[(PROPERTY_DF["Postcode"] == postcode) & (PROPERTY_DF["Postcode"].isin(REGION_POSTCODE_LIST[region]))]
        return jsonify(sorted(region_data["Suburb"].unique()))
    return jsonify([])

def initialize_data():
    """Initialize global PROPERTY_DF and REGION_SUBURBS."""
    global PROPERTY_DF, REGION_SUBURBS
    PROPERTY_DF = load_property_data()
    REGION_SUBURBS = {region: sorted(PROPERTY_DF[PROPERTY_DF["Postcode"].isin(postcodes)]["Suburb"].unique())
                      for region, postcodes in REGION_POSTCODE_LIST.items()}

# Initialize data at startup
initialize_data()

# Check if running on Render by looking for an environment variable Render sets
if os.environ.get("RENDER"):  # Render sets this
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)  # Production
else:
    app.run(debug=True)  # Local development on 127.0.0.1:5000