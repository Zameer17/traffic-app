import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
from datetime import datetime
import openrouteservice
from openrouteservice import convert
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Page setup
st.set_page_config(page_title="Traffic Congestion Predictor", layout="wide")
st.title("🚦 Traffic Congestion Predictor: Default vs Optimized Route")

# Load datasets
road_df = pd.read_csv("cleaned_expanded_dataset (1).csv")
traffic_df = pd.read_csv("traffic_dataset.csv")

# Train ML models
@st.cache_resource
def train_models(df):
    df = df.copy()
    le_time, le_day, le_weather = LabelEncoder(), LabelEncoder(), LabelEncoder()
    df["time_of_day"] = le_time.fit_transform(df["time_of_day"])
    df["day_of_week"] = le_day.fit_transform(df["day_of_week"])
    df["weather"] = le_weather.fit_transform(df["weather"])

    # Travel time model
    X_time = df[["distance_km", "congestion_index", "time_of_day", "day_of_week", "weather", "ors_time"]]
    y_time = df["actual_travel_time"]
    time_model = RandomForestRegressor(n_estimators=200, random_state=42)
    time_model.fit(X_time, y_time)

    # Congestion model
    X_cong = df[["distance_km", "time_of_day", "day_of_week", "weather", "ors_time"]]
    y_cong = df["congestion_index"]
    cong_model = RandomForestRegressor(n_estimators=150, random_state=42)
    cong_model.fit(X_cong, y_cong)

    # Accuracy
    time_r2 = r2_score(y_time, time_model.predict(X_time))
    cong_r2 = r2_score(y_cong, cong_model.predict(X_cong))

    return time_model, cong_model, le_time, le_day, le_weather, time_r2, cong_r2

time_model, cong_model, le_time, le_day, le_weather, time_r2, cong_r2 = train_models(traffic_df)

# User inputs
st.sidebar.header("Enter Route Details")
source = st.sidebar.text_input("Source Location", "Andheri")
dest = st.sidebar.text_input("Destination Location", "Bandra")

# Show model accuracy
st.sidebar.subheader("📊 Model Accuracy")
st.sidebar.markdown(f"*Travel Time Model Accuracy (R²):* {time_r2:.3f}")
st.sidebar.markdown(f"*Congestion Model Accuracy (R²):* {cong_r2:.3f}")

# API keys
weather_api_key = "Your_Key"
ors_api_key = "Your_Key"

# Get weather
def get_weather(api_key):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q=Mumbai&appid={api_key}&units=metric"
        r = requests.get(url).json()
        if r.get("cod") != 200:
            return "Clear"
        desc = r["weather"][0]["description"].capitalize()
        if "rain" in desc.lower(): return "Rainy"
        elif "cloud" in desc.lower(): return "Cloudy"
        elif "storm" in desc.lower(): return "Stormy"
        else: return "Clear"
    except:
        return "Clear"

weather_now = get_weather(weather_api_key)
day_now = datetime.now().strftime("%A")
hour = datetime.now().hour
if 7 <= hour < 11: time_now = "Morning Peak"
elif 11 <= hour < 17: time_now = "Midday"
elif 17 <= hour < 21: time_now = "Evening Peak"
else: time_now = "Night"

st.sidebar.markdown(f"📅 Date & Time: *{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
st.sidebar.markdown(f"🌤 Weather: *{weather_now}*")
st.sidebar.markdown(f"🕒 Time of Day: *{time_now}*")

# -----------------------------
# Geocoding
# -----------------------------
def geocode_location(place, key):
    client = openrouteservice.Client(key=key)
    try:
        res = client.pelias_search(place, size=1)
        if res and "features" in res and len(res["features"]) > 0:
            return tuple(res["features"][0]["geometry"]["coordinates"])
    except openrouteservice.exceptions.ApiError:
        return None
    return None

# Get ORS route
def get_ors_route(src, dst, preference="fastest"):
    client = openrouteservice.Client(key=ors_api_key)
    start, end = geocode_location(src, ors_api_key), geocode_location(dst, ors_api_key)
    if not start or not end:
        st.error(f"❌ Could not geocode {src} or {dst}. Try a more specific name.")
        return None, None, None, None, None
    try:
        route = client.directions([start, end], profile="driving-car", preference=preference)
        geom = route['routes'][0]['geometry']
        decoded = convert.decode_polyline(geom)
        duration = route['routes'][0]['summary']['duration'] / 60
        distance = route['routes'][0]['summary']['distance'] / 1000
        return decoded['coordinates'], duration, distance, start, end
    except openrouteservice.exceptions.ApiError as e:
        st.error(f"⚠ Routing failed: {e}")
        return None, None, None, None, None

default_coords, default_time, default_dist, src_coord, dst_coord = get_ors_route(source, dest, "fastest")
opt_coords, opt_time, opt_dist, _, _ = get_ors_route(source, dest, "shortest")

# Predict congestion
def predict_congestion(distance_km, ors_time, time_now, day_now, weather_now, optimized=False):
    if ors_time is None: ors_time = 0
    X_pred = pd.DataFrame([[distance_km,
                            le_time.transform([time_now])[0],
                            le_day.transform([day_now])[0],
                            le_weather.transform([weather_now])[0],
                            ors_time]],
                          columns=["distance_km", "time_of_day", "day_of_week", "weather", "ors_time"])
    cong = cong_model.predict(X_pred)[0]
    if optimized: cong *= 0.7
    return cong

# Predict travel time
def predict_time(distance_km, congestion_index, ors_time, time_now, day_now, weather_now):
    if ors_time is None or ors_time == 0: return 0
    adjusted_ors_time = ors_time * (1 + congestion_index / 100)
    if time_now in ["Morning Peak", "Evening Peak"]:
        adjusted_ors_time *= 1.3
    X_pred = pd.DataFrame([[distance_km,
                            congestion_index,
                            le_time.transform([time_now])[0],
                            le_day.transform([day_now])[0],
                            le_weather.transform([weather_now])[0],
                            adjusted_ors_time]],
                          columns=["distance_km", "congestion_index", "time_of_day", "day_of_week", "weather", "ors_time"])
    return time_model.predict(X_pred)[0]

# Format minutes into h/m
def format_time(minutes):
    if minutes < 60: return f"{minutes:.0f} min"
    else:
        hours, mins = int(minutes // 60), int(minutes % 60)
        return f"{hours}h {mins}m"

# Compute values
default_cong = predict_congestion(default_dist, default_time, time_now, day_now, weather_now) if default_time else 0
opt_cong = predict_congestion(opt_dist, opt_time, time_now, day_now, weather_now, optimized=True) if opt_time else 0
default_pred_time = predict_time(default_dist, default_cong, default_time, time_now, day_now, weather_now) if default_time else 0
opt_pred_time = predict_time(opt_dist, opt_cong, opt_time, time_now, day_now, weather_now) if opt_time else 0

# -----------------------------
# 📈 Visualization Section (Strip Style)
# -----------------------------
st.markdown("---")
st.subheader("📊 Travel Time Comparison")

routes = ["Default Route", "Optimized Route"]
times = [default_pred_time, opt_pred_time]

# Strip-style figure (wide & very short)
fig, ax = plt.subplots(figsize=(6,1.5))  # width=6, height=1.5 inches
ax.bar(routes, times, color=["red", "green"])

# Keep labels readable
ax.set_ylabel("Time (min)", fontsize=8)
ax.set_title("Travel Time: Default vs Optimized Route", fontsize=10)
ax.tick_params(axis="x", labelsize=8)
ax.tick_params(axis="y", labelsize=8)

st.pyplot(fig)


# Show results
col1, col2 = st.columns(2)
with col1:
    st.subheader("📍 Default Route (Fastest)")
    if default_time:
        st.info(f"🕒 Predicted Actual: {format_time(default_pred_time)} | ORS Predicted: {format_time(default_time)} | 📏 {default_dist:.1f} km | 🚦 Congestion: {default_cong:.2f}")
    else:
        st.warning("Could not fetch default route.")

with col2:
    st.subheader("⚡ Optimized Route (Shortest)")
    if opt_time:
        st.info(f"🕒 Predicted Actual: {format_time(opt_pred_time)} | ORS Predicted: {format_time(opt_time)} | 📏 {opt_dist:.1f} km | 🚦 Congestion: {opt_cong:.2f}")
    else:
        st.warning("Could not fetch optimized route.")

# Plot maps
def plot_map(route_coords, color):
    if not route_coords:
        return folium.Map(location=[19.076, 72.8777], zoom_start=12)
    m = folium.Map(location=[route_coords[0][1], route_coords[0][0]], zoom_start=12)
    folium.Marker([route_coords[0][1], route_coords[0][0]], tooltip="Source", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([route_coords[-1][1], route_coords[-1][0]], tooltip="Destination", icon=folium.Icon(color="red")).add_to(m)
    route_latlon = [(lat, lon) for lon, lat in route_coords]
    folium.PolyLine(route_latlon, color=color, weight=5, opacity=0.7).add_to(m)
    return m

map1 = plot_map(default_coords, "blue")
map2 = plot_map(opt_coords, "orange")

col1, col2 = st.columns(2)
with col1:
    st.subheader("🗺 Default Route Map")
    st_folium(map1, width=700, height=500)
with col2:
    st.subheader("🗺 Optimized Route Map")
    st_folium(map2, width=700, height=500)

