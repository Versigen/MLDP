import numpy as np 
import pandas as pd
import joblib
import streamlit as st
import datetime
import geopandas as gpd
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt

#Function to calculate haversine distance
def haversine(coord1, coord2):
    from math import radians, sin, cos, sqrt, atan2
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

# --- Page Configuration ---
st.set_page_config(
    page_title="Flight Price Prediction - Premium Service",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS Styling ---
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles with background image */
    .stApp {
        background: 
            linear-gradient(rgba(15, 20, 25, 0.7), rgba(26, 35, 50, 0.8), rgba(45, 55, 72, 0.9)),
            url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 800"><rect fill="%23f0f9ff" width="1200" height="800"/><path fill="%23e0f2fe" d="M0,400 Q300,200 600,350 T1200,300 L1200,800 L0,800 Z"/><circle cx="200" cy="150" r="80" fill="%23bae6fd" opacity="0.6"/><circle cx="800" cy="250" r="60" fill="%237dd3fc" opacity="0.5"/><circle cx="1000" cy="120" r="90" fill="%2338bdf8" opacity="0.4"/><path fill="%233b82f6" opacity="0.3" d="M100,500 Q200,450 300,480 T500,460 L500,600 L100,600 Z"/><path fill="%231e40af" opacity="0.2" d="M700,520 Q850,480 1000,510 T1200,500 L1200,650 L700,650 Z"/></svg>') center/cover;
        font-family: 'Inter', sans-serif;
        color: #1f2937;
        min-height: 100vh;
    }
    
    /* Text color overrides - only for main container content */
    .main-container *, .form-section *, .info-card * {
        color: #1f2937 !important;
    }
    
    /* Streamlit default text on dark background */
    .stApp > div > div > div > div > div {
        color: #e5e7eb;
    }
    
    /* Tab labels */
    .stTabs [data-baseweb="tab-list"] button div {
        color: #6b7280 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] div {
        color: white !important;
    }
    
    /* Section headers outside containers */
    .element-container h1, .element-container h2, .element-container h3, 
    .element-container h4, .element-container h5, .element-container h6 {
        color: #f9fafb !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container with enhanced styling */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 25px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem 0;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        border-radius: 15px;
        color: white;
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="20" cy="20" r="2" fill="rgba(255,255,255,0.1)"/><circle cx="80" cy="40" r="1.5" fill="rgba(255,255,255,0.1)"/><circle cx="40" cy="80" r="1" fill="rgba(255,255,255,0.1)"/></svg>');
        opacity: 0.3;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
        position: relative;
        z-index: 1;
    }
    
    /* Form styling */
    .form-section {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .form-section h3 {
        color: #1e40af;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 0.5rem;
    }
    
    /* Enhanced button styling to match new theme */
    .stButton > button {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2.5rem;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(245, 158, 11, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(245, 158, 11, 0.6);
        background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
    }
    
    /* Input styling */
    .stSelectbox > div > div {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        transition: border-color 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    .stDateInput > div > div {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
    }
    
    .stNumberInput > div > div {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
    }
    
    /* Success message styling */
    .success-message {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.3);
        margin: 1rem 0;
    }
    
    /* Info cards */
    .info-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    .info-card h4 {
        color: #1e40af !important;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .info-card p, .info-card div {
        color: #374151 !important;
    }
    
    /* Route display */
    .route-display {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 2px solid #0ea5e9;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .route-text {
        font-size: 1.2rem;
        font-weight: 600;
        color: #0c4a6e !important;
    }
    
    /* Form section headings */
    .form-section h3 {
        color: #1e40af !important;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 0.5rem;
    }
    
    /* Form section text colors for new background */
    .form-section p, .form-section div, .form-section span {
        color: #92400e !important;
        position: relative;
        z-index: 1;
    }
    
    /* Form labels in new color scheme */
    .form-section .stSelectbox label, .form-section .stDateInput label, 
    .form-section .stNumberInput label, .form-section .stCheckbox label {
        color: #92400e !important;
        font-weight: 600;
        text-shadow: 0 1px 2px rgba(255,255,255,0.5);
    }
    
    /* Input labels and text - only within containers */
    .form-section .stSelectbox label, .form-section .stDateInput label, 
    .form-section .stNumberInput label, .form-section .stCheckbox label {
        color: #374151 !important;
        font-weight: 500;
    }
    
    /* Labels outside containers (on dark background) */
    .stSelectbox label, .stDateInput label, .stNumberInput label, .stCheckbox label {
        color: #e5e7eb !important;
        font-weight: 500;
    }
    
    /* Select box text - make dropdown text black */
    .stSelectbox > div > div > div {
        color: #000000 !important;
    }
    
    /* Selected value text */
    .stSelectbox [data-baseweb="select"] > div > div {
        color: #000000 !important;
    }
    
    /* Dropdown menu options */
    .stSelectbox [data-baseweb="popover"] li {
        color: #000000 !important;
    }
    
    /* All select box internal text */
    .stSelectbox * {
        color: #000000 !important;
    }
    
    /* Tab text */
    .stTabs [data-baseweb="tab"] {
        color: #9ca3af !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: white !important;
    }
    
    /* Success and info messages */
    .element-container div[data-testid="stMarkdownContainer"] p {
        color: #e5e7eb !important;
    }
    
    /* Text within containers */
    .form-section div[data-testid="stMarkdownContainer"] p,
    .info-card div[data-testid="stMarkdownContainer"] p {
        color: #374151 !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f1f5f9;
        border-radius: 8px;
        color: #475569;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Load trained model ---
try:
    model = joblib.load('flight_price_final_model.pkl')
    model_loaded = True
except:
    model_loaded = False
    st.error("Model file not found. Please ensure 'flight_price_model2.pkl' is in the same directory.")

# --- Hardcoded training feature columns ---
feature_columns = [
 'duration', 'days_left', 'distance',
 'airline_AirAsia', 'airline_Air_India', 'airline_GO_FIRST', 'airline_Indigo',
 'airline_SpiceJet', 'airline_Vistara',
 'source_city_Bangalore', 'source_city_Chennai', 'source_city_Delhi',
 'source_city_Hyderabad', 'source_city_Kolkata', 'source_city_Mumbai',
 'departure_time_Afternoon', 'departure_time_Early_Morning',
 'departure_time_Evening', 'departure_time_Late_Night',
 'departure_time_Morning', 'departure_time_Night',
 'stops_one', 'stops_two_or_more', 'stops_zero',
 'arrival_time_Afternoon', 'arrival_time_Early_Morning',
 'arrival_time_Evening', 'arrival_time_Late_Night',
 'arrival_time_Morning', 'arrival_time_Night',
 'class_Business', 'class_Economy',
 'destination_city_Bangalore', 'destination_city_Chennai',
 'destination_city_Delhi', 'destination_city_Hyderabad',
 'destination_city_Kolkata', 'destination_city_Mumbai',
 'airline_category_budget', 'airline_category_standard'
]

city_distances = {
    ('Delhi', 'Mumbai'): 1450, ('Delhi', 'Bangalore'): 2160, ('Delhi', 'Kolkata'): 1520,
    ('Delhi', 'Hyderabad'): 1750, ('Delhi', 'Chennai'): 2160,
    ('Mumbai', 'Delhi'): 1450, ('Mumbai', 'Bangalore'): 980, ('Mumbai', 'Kolkata'): 1950,
    ('Mumbai', 'Hyderabad'): 710, ('Mumbai', 'Chennai'): 1330,
    ('Bangalore', 'Delhi'): 2160, ('Bangalore', 'Mumbai'): 980, ('Bangalore', 'Kolkata'): 1300,
    ('Bangalore', 'Hyderabad'): 660, ('Bangalore', 'Chennai'): 350,
    ('Kolkata', 'Delhi'): 1520, ('Kolkata', 'Mumbai'): 1950, ('Kolkata', 'Bangalore'): 1300,
    ('Kolkata', 'Hyderabad'): 1200, ('Kolkata', 'Chennai'): 1700,
    ('Hyderabad', 'Delhi'): 1750, ('Hyderabad', 'Mumbai'): 710, ('Hyderabad', 'Bangalore'): 660,
    ('Hyderabad', 'Kolkata'): 1200, ('Hyderabad', 'Chennai'): 660,
    ('Chennai', 'Delhi'): 2160, ('Chennai', 'Mumbai'): 1330, ('Chennai', 'Bangalore'): 350,
    ('Chennai', 'Kolkata'): 1700, ('Chennai', 'Hyderabad'): 660
}

city_coordinates = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639)
}

# --- Preprocessing function ---
def preprocess_input(raw_input):
    df = pd.DataFrame([raw_input])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df

# --- Main App Layout ---
st.markdown("""
<div class="header-container">
    <div class="header-title">✈️ Premium Flight Price Prediction</div>
    <div class="header-subtitle">Discover your perfect journey with intelligent fare forecasting</div>
</div>
""", unsafe_allow_html=True)

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["🎯 Flight Search", "📊 Price Analysis", "ℹ️ Flight Details"])

with tab1:
    # Main booking form
    col1, col2 = st.columns([2, 1])
    
    with col1:

        st.markdown("### ✈️ Flight Information")
        
        # Airline and class selection
        col_a, col_b = st.columns(2)
        with col_a:
            airline = st.selectbox("✈️ Preferred Airline", 
                                 ["AirAsia", "Air India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"])
            airline_category = st.selectbox("🏷️ Service Category", ["budget", "standard"])
        
        with col_b:
            flight_class = st.selectbox("🎭 Travel Class", ["Economy", "Business"])
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Route selection
        st.markdown("### 🗺️ Journey Planning")
        
        cities = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            source_city = st.selectbox("🛫 Departure City", cities, key="source_city")
        with col_r2:
            destination_city = st.selectbox("🛬 Arrival City", cities, key="destination_city")
        
        # Multi-city option
        multi_flight = st.checkbox("🌟 Add Multiple Destinations", help="Create a multi-city journey")
        
        route = [(source_city, destination_city)]
        
        if multi_flight:
            st.markdown("#### Additional Journey Legs")
            num_stops = st.number_input("Number of additional legs", min_value=1, max_value=3, step=1)
            for i in range(num_stops):
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    next_stop = st.selectbox(f"Stop {i+1} From", cities, key=f"stop_{i}")
                with col_s2:
                    final_dest = st.selectbox(f"Stop {i+1} To", cities, key=f"dest_{i}")
                route.append((next_stop, final_dest))
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Time preferences
        st.markdown("### ⏰ Time Preferences")
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            departure_time = st.selectbox("🌅 Departure Time", 
                                        ["Early Morning", "Morning", "Afternoon", "Evening", "Night", "Late Night"])
        with col_t2:
            arrival_time = st.selectbox("🌆 Arrival Time", 
                                      ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"])
        
        flight_date = st.date_input("📅 Travel Date", min_value=datetime.date.today())
        days_left = (flight_date - datetime.date.today()).days
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Flight summary card
        st.markdown("#### 🎯 Journey Summary")
        
        # Calculate totals
        # --- Compute distance ---


        distance = haversine(city_coordinates[source_city], city_coordinates[destination_city])
        total_distance = sum(city_distances.get(seg, 0) for seg in route)
        stops = "zero" if len(route) == 1 else ("one" if len(route) == 2 else "two_or_more")
        duration = round(total_distance / 700, 2)
        
        # Display route
        route_str = " → ".join([f"{seg[0]} to {seg[1]}" for seg in route])
        st.markdown(f'<div class="route-display"><div class="route-text">{route_str}</div></div>', unsafe_allow_html=True)
        
        st.write(f"**📏 Total Distance:** {total_distance:,} km")
        st.write(f"**⏱️ Estimated Duration:** {duration} hours")
        st.write(f"**🛑 Stops:** {stops.replace('_', ' ').title()}")
        st.write(f"**📅 Days Until Flight:** {days_left} days")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Prediction button
        if model_loaded:
            if st.button("🚀 Predict Flight Price", type="primary"):
                # Mapping user inputs
                airline_map = {
                    "AirAsia": "AirAsia", "Air India": "Air_India", "GO_FIRST": "GO_FIRST",
                    "Indigo": "Indigo", "SpiceJet": "SpiceJet", "Vistara": "Vistara"
                }
                source_city_map = {
                    "Bangalore": "Bangalore", "Chennai": "Chennai", "Delhi": "Delhi",
                    "Hyderabad": "Hyderabad", "Kolkata": "Kolkata", "Mumbai": "Mumbai"
                }
                departure_time_map = {
                    "Afternoon": "Afternoon", "Early Morning": "Early_Morning", "Evening": "Evening",
                    "Late Night": "Late_Night", "Morning": "Morning", "Night": "Night"
                }
                
                raw_input = {
                    "airline": airline_map[airline],
                    "source_city": source_city_map[source_city],
                    "departure_time": departure_time_map[departure_time],
                    "stops": stops,
                    "arrival_time": arrival_time,
                    "destination_city": destination_city,
                    "class": flight_class,
                    "airline_category": airline_category,
                    "duration": duration,
                    "days_left": days_left,
                    "distance": total_distance
                }
                
                try:
                    processed = preprocess_input(raw_input)
                    prediction = model.predict(processed)
                    
                    st.markdown(f"""
                    <div class="success-message">
                        💰 Predicted Flight Price: ₹{float(prediction[0]):,.2f}
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")

with tab2:
    st.markdown("### 📊 Flight Price Analysis")
    
    if model_loaded:
        col1, col2 = st.columns(2)
        
        with col1:

            st.markdown("#### 💡 Pricing Insights")
            st.write("• **Best Time to Book:** 21-45 days in advance")
            st.write("• **Cheapest Days:** Tuesday & Wednesday")
            st.write("• **Peak Season:** December & January")
            st.write("• **Budget Airlines:** Generally 20-30% cheaper")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:

            st.markdown("#### 📈 Price Factors")
            st.write("• **Distance:** Major factor in pricing")
            st.write("• **Advance Booking:** Earlier = Better deals")
            st.write("• **Stops:** Direct flights cost more")
            st.write("• **Time of Day:** Morning flights often cheaper")
            st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown("### ℹ️ Flight Details & Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏢 Available Airlines")
        airlines_info = {
            "AirAsia": "Low-cost carrier",
            "Air India": "National carrier",
            "GO_FIRST": "Budget airline",
            "Indigo": "Largest low-cost carrier",
            "SpiceJet": "Budget airline",
            "Vistara": "Premium service"
        }
        for airline, desc in airlines_info.items():
            st.write(f"• **{airline}:** {desc}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🏙️ Covered Cities")
        cities_info = {
            "Delhi": "Capital region",
            "Mumbai": "Financial capital",
            "Bangalore": "Tech hub",
            "Chennai": "South India gateway",
            "Hyderabad": "Cyberabad",
            "Kolkata": "Cultural capital"
        }
        for city, desc in cities_info.items():
            st.write(f"• **{city}:** {desc}")
        st.markdown('</div>', unsafe_allow_html=True)

# Debug section (hidden by default)
if st.checkbox("🔧 Debug Mode (Developer Options)", value=False):
    if model_loaded and 'raw_input' in locals():
        st.markdown("### 🔍 Debug Information")
        processed = preprocess_input(raw_input)
        st.write("**Columns Match:** ", list(processed.columns) == feature_columns)
        st.dataframe(processed)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #64748b; border-top: 1px solid #e2e8f0; margin-top: 2rem;">
    <p>✈️ Premium Flight Price Prediction Service | Powered by Advanced ML Algorithms</p>
    <p><small>Experience the future of travel planning with intelligent price forecasting</small></p>
</div>
""", unsafe_allow_html=True)


    # --- Map Plotting ---
# if st.button("Show Flight Path"):
#     coords = [city_coordinates[source_city], city_coordinates[destination_city]]
#     line = LineString([(lon, lat) for lat, lon in coords])

#     gdf = gpd.GeoDataFrame(geometry=[line])
#     fig, ax = plt.subplots(figsize=(6, 6))
#     world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
#     world.plot(ax=ax, color='lightgray')
#     gdf.plot(ax=ax, color='blue', linewidth=2)
#     for city, (lat, lon) in city_coordinates.items():
#         ax.plot(lon, lat, 'ro')
#         ax.text(lon+0.5, lat+0.5, city, fontsize=8)
#     ax.set_title(f"Flight Path: {source_city} → {destination_city}")
#     st.pyplot(fig)

if st.button("Show Flight Path"):
    # -------- Coordinates Handling --------
    # Extend this coords list later for multi-stop flights
    coords = [city_coordinates[source_city], city_coordinates[destination_city]]

    # Create curved arc points for premium look
    def create_arc(coord1, coord2, n_points=100):
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        lats = np.linspace(lat1, lat2, n_points)
        lons = np.linspace(lon1, lon2, n_points)
        # Apply a slight curve: offset in the middle
        curve_offset = 0.5
        mid = n_points // 2
        lats[mid] += curve_offset
        return lats, lons

    # Load world map from GeoJSON
    url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
    world = gpd.read_file(url)

    # -------- Plotting --------
    fig, ax = plt.subplots(figsize=(8, 6))
    world.plot(ax=ax, color='lightgray', edgecolor='white')

    # Plot each leg (in future, iterate through route legs)
    lats, lons = create_arc(coords[0], coords[1])
    ax.plot(lons, lats, color='blue', linewidth=2, label="Flight Path")

    # Plot source and destination with plane icon at mid
    ax.scatter(coords[0][1], coords[0][0], color='green', s=80, label="Origin")
    ax.scatter(coords[1][1], coords[1][0], color='red', s=80, label="Destination")

    # Add plane icon at midpoint
    plane_lat, plane_lon = lats[len(lats)//2], lons[len(lons)//2]
    ax.text(plane_lon, plane_lat, "✈", fontsize=16, ha='center', va='center', color="darkblue")

    # Annotate city names
    ax.text(coords[0][1] + 0.3, coords[0][0] + 0.3, source_city, fontsize=8, color='green')
    ax.text(coords[1][1] + 0.3, coords[1][0] + 0.3, destination_city, fontsize=8, color='red')

    # Add legend
    ax.legend()

    # -------- Zoom on flight path --------
    lat_vals = [c[0] for c in coords]
    lon_vals = [c[1] for c in coords]
    lat_margin = 3
    lon_margin = 3
    ax.set_xlim(min(lon_vals) - lon_margin, max(lon_vals) + lon_margin)
    ax.set_ylim(min(lat_vals) - lat_margin, max(lat_vals) + lat_margin)

    ax.set_title(f"Flight Path: {source_city} → {destination_city}")
    st.pyplot(fig)