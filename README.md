🚦 Traffic Congestion Predictor: Default vs Optimized Route

This project is a Streamlit web app that predicts traffic congestion levels and compares default (fastest) vs optimized (shortest) road routes in Mumbai using:

OpenRouteService API → for routing & distance/time

OpenWeather API → for live weather data

Machine Learning (Random Forest) → for predicting travel time & congestion index

Streamlit + Folium → for interactive maps & dashboards

📌 Features

✅ Compare Default (Fastest) vs Optimized (Shortest) routes
✅ Predict congestion index using ML models
✅ Predict actual travel time considering weather, time of day & congestion
✅ Visualize routes on interactive maps (Folium in Streamlit)
✅ Display congestion-adjusted vs ORS predicted time
✅ Show model accuracy (R² scores) for ML models

🏗️ Tech Stack

Frontend/UI → Streamlit

Maps → Folium
 + streamlit-folium

Routing & Distance → OpenRouteService API

Weather Data → OpenWeather API

Machine Learning → scikit-learn (Random Forest Regressor, Label Encoding)

Visualization → matplotlib, seaborn

📂 Dataset

cleaned_expanded_dataset (1).csv → Road & travel dataset

traffic_dataset.csv → Training dataset for congestion & travel time prediction

⚙️ Installation

Clone the repo:

git clone https://github.com/your-username/traffic-congestion-predictor.git
cd traffic-congestion-predictor


Install dependencies:

pip install -r requirements.txt


Create a .env file and add your API keys:

ORS_API_KEY=your_openrouteservice_api_key
WEATHER_API_KEY=your_openweather_api_key

🚀 Run the App
streamlit run app.py

📊 Example Output

Comparison of Travel Times (Default vs Optimized)

Congestion Index for both routes

Interactive Maps showing suggested paths

(Default Route vs Optimized Route Maps will be shown side by side)

📈 ML Model Performance

Travel Time Model (R²): ~0.80 – 0.90

Congestion Index Model (R²): ~0.70 – 0.85

🤔 FAQ

Q. Why is the "default route" sometimes faster than the "optimized route"?

The default route (fastest) is calculated by ORS purely based on travel time, while the optimized route (shortest) reduces congestion & distance but may not always be quicker in absolute time.

Our ML model adjusts for realistic traffic, congestion, and weather conditions → so optimized route may be more reliable even if slightly slower.

📌 Future Improvements

Support for public transport & walking routes

Integration with live traffic APIs (Google, TomTom, HERE)

Deployment on Streamlit Cloud or Docker
