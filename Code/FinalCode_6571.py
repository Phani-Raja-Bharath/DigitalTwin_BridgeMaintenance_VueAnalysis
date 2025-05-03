# BLOCK 1 Imports
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import cv2
import os
import time
import json
import datetime
import scipy.stats as stats
from PIL import Image
from shapely.geometry import LineString
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Page Setup
st.set_page_config(page_title="Bridge Traffic Simulation", layout="wide")

# Project Title and Intro 

st.title("Bridge Traffic Simulation and Fatigue Monitoring")
st.markdown("""
#### Using a Hybrid Digital Twin Approach.  
This dashboard simulates real-time traffic flow over the Melbourne Causeway Bridge, estimates stress and fatigue buildup, 
and provides predictive maintenance recommendations using Monte Carlo simulations and Machine Learning model-Random Forest.
""")
st.divider()


# Global Variables
live_log = []
training_data = []
fatigue_distribution = []
shockwave_speeds = []
rf_model_trained = False
predicted_fatigue_score = None
safety_message = None
sim = None
trigger_jam = False
guided_snapshots_log = []


# Utility Functions

def load_bridge_geojson(path):
    bridge_gdf, road_length = gpd.GeoDataFrame(), 2000
    try:
        with open(path, "r") as f:
            gdf = gpd.GeoDataFrame.from_features(json.load(f)["features"]).set_crs("EPSG:4326").to_crs(epsg=32617)
            if "name" in gdf.columns:
                gdf = gdf[gdf['name'].str.contains("Melbourne Causeway", case=False, na=False)]
            road_length = gdf.length.sum() if gdf.length.sum() > 100 else 2000
            gdf = gdf.to_crs("EPSG:4326")
            gdf['lon'] = gdf.to_crs(epsg=32617).centroid.to_crs(epsg=4326).x
            gdf['lat'] = gdf.to_crs(epsg=32617).centroid.to_crs(epsg=4326).y
            bridge_gdf = gdf.copy()
    except Exception:
        st.warning("Fallback bridge length 2.0 km.")
    return bridge_gdf, road_length

# Live Traffic Snapshot
def fetch_live_snapshot(is_night_mode=False):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920x1080")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get("https://fl511.com/map")
    time.sleep(12)
    driver.save_screenshot("fl511_map.png")
    driver.quit()
    

    img = cv2.imread("fl511_map.png")
    hsv = cv2.cvtColor(img[350:500, 700:900], cv2.COLOR_BGR2HSV)
    avg_hue = np.mean(hsv[:, :, 0])
    os.remove("fl511_map.png")

    # Dynamic mapping based on day/night mode
    if is_night_mode:
        estimated_density = np.interp(avg_hue, [40, 90], [0.9, 0.0])
    else:
        estimated_density = np.interp(avg_hue, [65, 90], [0.9, 0.2])

    estimated_density = np.clip(estimated_density, 0.0, 1.0)

    # Traffic Status
    if avg_hue < (50 if is_night_mode else 65):
        status = "Heavy Congestion"
    elif avg_hue < (65 if is_night_mode else 80):
        status = "Moderate"
    else:
        status = "Light"

    # Fatigue Estimate
    fatigue = np.interp(avg_hue, [65, 90], [80, 20])

    # Safe Shockwave Estimate
    rho1 = estimated_density
    rho2 = estimated_density + np.random.normal(0, 0.02)
    q1 = rho1 * (60 * (1 - rho1 / 0.3))
    q2 = rho2 * (60 * (1 - rho2 / 0.3))

    delta_rho = rho2 - rho1

    if abs(delta_rho) > 0.01:
        shockwave = (q2 - q1) / delta_rho
    else:
        shockwave = 0.0

    # Record all data
    record = {
        "timestamp": datetime.datetime.now(),
        "Fatigue Score": round(fatigue, 2),
        "Estimated Density": round(estimated_density, 3),
        "Live Shockwave Speed": round(shockwave, 2)
    }

    live_log.append(record)
    time.sleep(2) 
    return status, record

# Class: Traffic Simulation

class BridgeTrafficSimulation:
    def __init__(self, road_length, dx, dt, rho_max, alpha, failure_threshold, sensor_interval, v_max_lane1, v_max_lane2, car_pct, truck_pct, bus_pct):
        self.road_length = road_length
        self.dx = dx
        self.dt = dt
        self.rho_max = rho_max
        self.alpha = alpha
        self.failure_threshold = failure_threshold
        self.sensor_interval = sensor_interval
        self.v_max_lane1 = v_max_lane1
        self.v_max_lane2 = v_max_lane2
        self.car_pct = car_pct
        self.truck_pct = truck_pct
        self.bus_pct = bus_pct
        self.x = np.linspace(0, road_length, int(road_length / dx) + 1)
        self.nx = len(self.x)
        self.rho_lane1 = np.zeros(self.nx)
        self.rho_lane2 = np.zeros(self.nx)
        self.stress = np.zeros(self.nx)
        self.fatigue_score = 0

    def initialize_density(self, lane1_init, lane2_init, jam_lane1=False, jam_lane2=False):
        center = int(self.nx / 2)
        width = int(self.nx / 10)
        self.rho_lane1[:] = lane1_init
        self.rho_lane2[:] = lane2_init
        if jam_lane1:
            self.rho_lane1[center-width:center+width] = self.rho_max * 0.9
        if jam_lane2:
            self.rho_lane2[center-width:center+width] = self.rho_max * 0.9

    def velocity(self, rho, lane=1):
        v_max = self.v_max_lane1 if lane == 1 else self.v_max_lane2
        return v_max * (1 - rho / self.rho_max)

    def traffic_flow(self, rho, lane=1):
        return rho * self.velocity(rho, lane)

    def update_density(self, time_step, freeze_lane1=False, freeze_lane2=False):
        wave_speed = 2 * np.sin(0.05 * time_step)
        if not freeze_lane1:
            self.rho_lane1 = np.roll(self.rho_lane1, int(wave_speed))
            q1 = self.traffic_flow(self.rho_lane1, lane=1)
            self.rho_lane1[1:-1] -= self.dt / self.dx * (q1[2:] - q1[:-2])
            self.rho_lane1 = np.clip(self.rho_lane1, 0, self.rho_max)
        if not freeze_lane2:
            self.rho_lane2 = np.roll(self.rho_lane2, int(wave_speed))
            q2 = self.traffic_flow(self.rho_lane2, lane=2)
            self.rho_lane2[1:-1] -= self.dt / self.dx * (q2[2:] - q2[:-2])
            self.rho_lane2 = np.clip(self.rho_lane2, 0, self.rho_max)

    def update_stress(self):
        base_weight = 1500  # kg
        factor = (self.car_pct / 100) * (1500 / base_weight) + (self.truck_pct / 100) * (30000 / base_weight) + (self.bus_pct / 100) * (18000 / base_weight)
        avg_density = (self.rho_lane1 + self.rho_lane2) / 2
        self.stress = self.alpha * avg_density * factor

    def update_fatigue(self):
        avg_density = np.mean((self.rho_lane1 + self.rho_lane2) / 2)
        self.fatigue_score += (avg_density / self.rho_max) * 2

    def compute_shockwave_speed(self, idx1, idx2):
        rho1 = (self.rho_lane1[idx1] + self.rho_lane2[idx1]) / 2
        rho2 = (self.rho_lane1[idx2] + self.rho_lane2[idx2]) / 2
        q1 = self.traffic_flow(rho1)
        q2 = self.traffic_flow(rho2)
        return (q2 - q1) / (rho2 - rho1) if rho2 != rho1 else 0.0

# Sidebar UI Controls

with st.sidebar:
    st.header("Live Snapshot Settings")
    enable_live_snapshot = st.checkbox("Enable Real-Time Snapshot")
    is_night_mode = st.checkbox("Night Mode (for Dark Theme Maps)")

    if enable_live_snapshot and st.button("Fetch Live Snapshot"):
        status, live_data = fetch_live_snapshot(is_night_mode=is_night_mode)
        live_log.append(live_data)
        st.success(f"Traffic Status: {status}")
        st.metric("Estimated Density", f"{live_data['Estimated Density']:.2f}")
        st.metric("Shockwave Speed", f"{live_data['Live Shockwave Speed']:.2f} m/s")
        st.metric("Fatigue Score", f"{live_data['Fatigue Score']:.2f}")

    st.divider()

    st.header("Lane Traffic Settings")
    lane1_density = st.slider("Lane 1 Initial Density", 0.0, 1.0, 0.10, 0.01)
    lane2_density = st.slider("Lane 2 Initial Density", 0.0, 1.0, 0.10, 0.01)
    v_max_lane1 = st.slider("Lane 1 Max Speed (km/h)", 30, 120, 80, 5)
    v_max_lane2 = st.slider("Lane 2 Max Speed (km/h)", 30, 120, 80, 5)

    jam_lane1 = st.checkbox("Induce Jam in Lane 1")
    jam_lane2 = st.checkbox("Induce Jam in Lane 2")
    freeze_lane1 = st.checkbox("Freeze Lane 1 Traffic")
    freeze_lane2 = st.checkbox("Freeze Lane 2 Traffic")
    trigger_jam = st.checkbox("Enable Jam Detection Monitoring")

    st.divider()

    st.header("Vehicle Composition")
    car_pct = st.slider("Cars (%)", 0, 100, 80, 5)
    truck_pct = st.slider("Trucks (%)", 0, 100, 15, 5)
    bus_pct = st.slider("Buses (%)", 0, 100, 5, 5)

    total_pct = car_pct + truck_pct + bus_pct
    if total_pct != 100:
        st.warning("Adjusting vehicle percentages to 100%")
        car_pct = (car_pct / total_pct) * 100
        truck_pct = (truck_pct / total_pct) * 100
        bus_pct = (bus_pct / total_pct) * 100

    st.divider()

    st.header("Simulation Settings")
    num_runs = st.slider("Monte Carlo Simulation Runs", 10, 500, 100, 10)
    alpha = st.number_input("Structural Alpha", 0.00001, 0.01, 0.00005, step=0.00001, format="%.5f")
    sensor_interval = st.slider("Sensor Spacing (meters)", 50, 500, 200, 50)

    st.divider()

    st.header("Overall Traffic Settings")
    input_density = st.number_input("Overall Initial Traffic Density", min_value=0.0, max_value=1.0, value=0.15, step=0.01)

    st.divider()

    run_simulation = st.button("Run Monte Carlo Simulation")


   
# Banner: Real Image of the Bridge

st.subheader("Melbourne Causeway")

image_path = "___________________________" #enter your local Melbourne Causeway image path, use '/' instead of '\' in the path

if os.path.exists(image_path):
    try:
        bridge_image = Image.open(image_path)
        st.image(bridge_image, caption="Melbourne Causeway (© Wikipedia)", use_column_width=True)
    except Exception as e:
        st.warning(f"Image found but could not be opened. Error: {e}")
else:
    st.warning(f"Real image not found at {image_path}. Please check the file exists.")

# Load Bridge Geometry and Show Maps

geojson_path = "________________________" #enter your local GeoJSON path, use '/' instead of '\' in the path
bridge_gdf, road_length_from_geo = load_bridge_geojson(geojson_path)

col_left, col_right = st.columns(2)

# Left: Bridge Centroid View
with col_left:
    st.subheader("Bridge Centroid View")
    if not bridge_gdf.empty:
        fig_centroid = px.scatter_mapbox(
            bridge_gdf,
            lat="lat",
            lon="lon",
            hover_name="name",
            zoom=12,
            height=500
        )
        fig_centroid.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_centroid, use_container_width=True)
    else:
        st.warning("Bridge centroid data not loaded.")
        
# Right Column - Live Traffic Snapshots
with col_right:
    st.subheader("Live Traffic Density View")

    if live_log:  # if fetch_live_snapshot() was clicked at least once
        latest = live_log[-1]
        density = latest['Estimated Density']

        color = (
            "green" if density < 0.2 else
            "orange" if density < 0.4 else
            "red"
        )

        fig_density = go.Figure()

        for idx, row in bridge_gdf.iterrows():
            if isinstance(row.geometry, LineString):
                x, y = row.geometry.xy
                fig_density.add_trace(go.Scattermapbox(
                    mode="lines",
                    lon=list(x),
                    lat=list(y),
                    line=dict(width=6, color=color),
                    name=f"Live Density: {density:.2f}"
                ))

        fig_density.update_layout(
            mapbox_style="open-street-map",
            mapbox_zoom=12,
            mapbox_center={"lat": bridge_gdf['lat'].mean(), "lon": bridge_gdf['lon'].mean()},
            height=500,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_density, use_container_width=True)
    else:
        st.info("Click **Fetch Live Snapshot** to view live density color map.")
        
# Details (Bridge & Live data)
        
col_bridgedetails, col_live = st.columns(2)

# Left Column - Bridge details
with col_bridgedetails:
    # Bridge Overview Details
    
    st.subheader("Bridge Overview and Sensor Setup Details")
    
    analyzed_length = sensor_interval * (road_length_from_geo // sensor_interval)
    percent_covered = (analyzed_length / road_length_from_geo) * 100
    
    st.markdown(f"""
    **Bridge Length:** {road_length_from_geo:.2f} meters ({road_length_from_geo / 1000:.2f} km)  
    **Simulated Coverage:** {analyzed_length:.2f} meters ({percent_covered:.1f}% coverage)  
    **Sensor Spacing:** {sensor_interval} meters
    """)

# Right Column - live Snapshots
with col_live:
    st.markdown("### Live Traffic Snapshots")

    if live_log:
        df_live = pd.DataFrame(live_log)

        # Check safely if 'timestamp' exists
        if 'timestamp' in df_live.columns:
            df_live['timestamp'] = pd.to_datetime(df_live['timestamp'])
            df_live['S.No'] = np.arange(1, len(df_live) + 1)
            df_live = df_live[['S.No', 'timestamp', 'Fatigue Score', 'Estimated Density', 'Live Shockwave Speed']]
            st.table(df_live.set_index('S.No'))

            st.markdown("### Live Traffic Density Trend")
            fig_live_density = px.line(
                df_live,
                x='timestamp',
                y='Estimated Density',
                markers=True,
                title='Live Traffic Density Over Time'
            )
            st.plotly_chart(fig_live_density, use_container_width=True)
        else:
            # Safe fallback if no timestamp exists
            df_live['S.No'] = np.arange(1, len(df_live) + 1)
            st.dataframe(df_live[['S.No', 'Fatigue Score', 'Estimated Density', 'Live Shockwave Speed']])
            st.warning(" No timestamps available in current live data.")
    else:
        st.info(" No live snapshots captured yet. Please use 'Fetch Live Snapshot'.")

# Guided Live Snapshot Monitoring

st.subheader("Guided Live Snapshot Capture and Monitoring")

# Settings for multiple snapshots
num_snapshots = st.number_input(
    "Number of Snapshots to Take",
    min_value=2, max_value=10, value=2, step=1
)

wait_seconds = st.number_input(
    "Wait Time Between Snapshots (seconds)",
    min_value=5, max_value=60, value=15, step=1
)

# Start Snapshot Process
if st.button("Start Guided Snapshot Run"):
    st.success(f"Starting {num_snapshots} snapshots with {wait_seconds} seconds gap...")

    for snap in range(1, num_snapshots + 1):
        with st.status(f"Taking Snapshot {snap}...", expanded=True) as snap_status:
            status, live_data = fetch_live_snapshot()

            # Append to BOTH logs
            guided_snapshots_log.append(live_data)
            live_log.append(live_data)

            st.write(f"Snapshot {snap}: Traffic Status: {status}")
            st.write(f"Estimated Density: {live_data['Estimated Density']:.2f}")
            st.write(f"Shockwave Speed: {live_data['Live Shockwave Speed']:.2f} m/s")
            st.write(f"Fatigue Score: {live_data['Fatigue Score']:.2f}")

            snap_status.update(label=f"Snapshot {snap} Complete!", state="complete")

        if snap != num_snapshots:
            with st.status(f"Waiting {wait_seconds} seconds before next snapshot...") as wait_status:
                for i in range(wait_seconds, 0, -1):
                    st.write(f"{i} seconds remaining...")
                    time.sleep(1)
                wait_status.update(label="Wait Complete!", state="complete")

    #  Shockwave Speed Trend after all snapshots
    if len(guided_snapshots_log) > 1:
        st.subheader("Updated Guided Shockwave Speed Trend")

        df_guided = pd.DataFrame(guided_snapshots_log)

        if 'timestamp' in df_guided.columns:
            df_guided['timestamp'] = pd.to_datetime(df_guided['timestamp'])

            fig_trend = px.line(
                df_guided,
                x='timestamp',
                y='Live Shockwave Speed',
                markers=True,
                title='Guided Snapshot: Shockwave Speed Over Time',
                labels={'timestamp': 'Snapshot Time', 'Live Shockwave Speed': 'Shockwave Speed (m/s)'}
            )

            fig_trend.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.warning(" No timestamps found in guided snapshots.")
    else:
        st.warning("Not enough guided snapshots to build a trend graph.")

# Monte Carlo Simulation

if run_simulation:
    st.subheader("Monte Carlo Simulation Results")
    
    if enable_live_snapshot:
        status, live_data = fetch_live_snapshot(is_night_mode=False)  # or True if you want night mode
        live_log.append(live_data)
        st.success(f"Snapshot captured: Traffic Status {status}")

    fatigue_distribution.clear()
    shockwave_speeds.clear()
    training_data.clear()

    for _ in range(num_runs):
        v_max_sim = np.random.choice([40, 60, 80, 100])
        alpha_sim = np.random.uniform(0.00005, 0.001)
        sensor_interval_sim = np.random.choice([100, 200, 300])
        initial_density_sim = np.random.uniform(0.2, 0.6)

        sim = BridgeTrafficSimulation(
            road_length=road_length_from_geo,
            dx=10,
            dt=0.05,
            rho_max=0.3,
            alpha=alpha_sim,
            failure_threshold=0.015,
            sensor_interval=sensor_interval_sim,
            v_max_lane1=v_max_sim,
            v_max_lane2=v_max_sim,
            car_pct=car_pct,
            truck_pct=truck_pct,
            bus_pct=bus_pct
        )

        sim.initialize_density(
            lane1_init=lane1_density,
            lane2_init=lane2_density,
            jam_lane1=jam_lane1 or trigger_jam,
            jam_lane2=jam_lane2 or trigger_jam
        )

        for t in range(100):
            # Adding random noise to densities
            noise1 = np.random.normal(0, 0.02, sim.nx)
            noise2 = np.random.normal(0, 0.02, sim.nx)
        
            sim.rho_lane1 += noise1
            sim.rho_lane2 += noise2       
            sim.rho_lane1 = np.clip(sim.rho_lane1, 0, sim.rho_max)
            sim.rho_lane2 = np.clip(sim.rho_lane2, 0, sim.rho_max)
        
            if t == 50:
                jam_width = int(sim.nx / 8)
                center = int(sim.nx / 2)
        
                sim.rho_lane1[center - jam_width:center + jam_width] = sim.rho_max * 0.9
                sim.rho_lane2[center - jam_width:center + jam_width] = sim.rho_max * 0.9
        
            sim.update_density(t, freeze_lane1=freeze_lane1, freeze_lane2=freeze_lane2)
            sim.update_stress()
            sim.update_fatigue()

        fatigue_distribution.append(sim.fatigue_score)
        shockwave_speeds.append(sim.compute_shockwave_speed(int(sim.nx * 0.4), int(sim.nx * 0.6)))

        training_data.append({
            'initial_density': initial_density_sim,
            'alpha': alpha_sim,
            'v_max': v_max_sim,
            'sensor_interval': sensor_interval_sim,
            'fatigue_score': sim.fatigue_score,
            'shockwave_speed': shockwave_speeds[-1]
        })

    st.success(f"Completed {num_runs} simulation runs!")
    
    # Traffic Density Profile

    st.subheader("Final Traffic Density Profiles")
    
    # Combined Lane 1 and Lane 2 in One Plot
    st.markdown("#### Combined Lane Density View")
    
    fig_combined = go.Figure()
    
    fig_combined.add_trace(go.Scatter(
        x=sim.x,
        y=sim.rho_lane1,
        mode='lines',
        name='Lane 1 Density',
        line=dict(color='blue')
    ))
    
    fig_combined.add_trace(go.Scatter(
        x=sim.x,
        y=sim.rho_lane2,
        mode='lines',
        name='Lane 2 Density',
        line=dict(color='red')
    ))
    
    fig_combined.update_layout(
        title="Combined Lane 1 and Lane 2 Traffic Densities",
        xaxis_title="Distance along Bridge (m)",
        yaxis_title="Traffic Density",
        legend_title="Lane",
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig_combined, use_container_width=True)
    
    st.divider()
    
    # Lane 1 Plot
    st.markdown("#### Lane 1 Density View")
    
    fig_lane1 = px.line(
        x=sim.x,
        y=sim.rho_lane1,
        labels={'x': 'Distance along Bridge (m)', 'y': 'Lane 1 Traffic Density'},
        title="Lane 1 Traffic Density Profile"
    )
    st.plotly_chart(fig_lane1, use_container_width=True)
    
    st.divider()
    
    # Separate Lane 2 Plot
    st.markdown("#### Lane 2 Density View")
    
    fig_lane2 = px.line(
        x=sim.x,
        y=sim.rho_lane2,
        labels={'x': 'Distance along Bridge (m)', 'y': 'Lane 2 Traffic Density'},
        title="Lane 2 Traffic Density Profile"
    )
    st.plotly_chart(fig_lane2, use_container_width=True)

    # Jam Detection after Simulation
    st.subheader("Jam Detection Summary")
    jam_threshold = 0.7 * sim.rho_max  # ✅ Set jam threshold to 70% of maximum density
    
    max_density_lane1 = np.max(sim.rho_lane1)
    max_density_lane2 = np.max(sim.rho_lane2)
    
    col_jam1, col_jam2 = st.columns(2)
    with col_jam1:
        st.metric("Lane 1 Max Density", f"{max_density_lane1:.2f}")
    with col_jam2:
        st.metric("Lane 2 Max Density", f"{max_density_lane2:.2f}")
    
    # Scientific Jam Status Detection
    lane1_status = "Full Jam Detected" if max_density_lane1 >= jam_threshold else "No Jam Detected"
    lane2_status = "Full Jam Detected" if max_density_lane2 >= jam_threshold else "No Jam Detected"
    
    st.markdown(f"**Lane 1 Status:** {lane1_status}")
    st.markdown(f"**Lane 2 Status:** {lane2_status}")
    
    if lane1_status == "Full Jam Detected" or lane2_status == "Full Jam Detected":
        st.error("⚠️ Traffic jam detected in one or both lanes! Please monitor bridge stress levels.")
    else:
        st.success("✅ No traffic jam detected. Bridge is operating normally.")



    # Stress Distribution Plot
    st.subheader("Structural Stress Distribution (Last Simulation)")
    fig_stress = px.line(
        x=sim.x,
        y=sim.stress,
        labels={'x': 'Distance along Bridge (m)', 'y': 'Stress (arbitrary units)'},
        title='Stress Distribution along Bridge'
    )
    st.plotly_chart(fig_stress, use_container_width=True)
    
    # Final Density Distribution Plot
    st.subheader("Final Traffic Density Profile along the Bridge")
    
    avg_density_profile = (sim.rho_lane1 + sim.rho_lane2) / 2
    
    fig_density_profile = px.line(
        x=sim.x,
        y=avg_density_profile,
        labels={'x': 'Distance along Bridge (m)', 'y': 'Average Traffic Density'},
        title='Final Traffic Density along Bridge'
    )
    st.plotly_chart(fig_density_profile, use_container_width=True)


    # Fatigue Score Histogram + Normal Fit
    st.subheader("Fatigue Score Distribution and Normal Fit")
    mu, sigma = np.mean(fatigue_distribution), np.std(fatigue_distribution)
    x_vals = np.linspace(min(fatigue_distribution), max(fatigue_distribution), 100)
    y_vals = stats.norm.pdf(x_vals, mu, sigma)
    y_vals_scaled = y_vals * len(fatigue_distribution) * (x_vals[1] - x_vals[0])

    fig_fatigue_distribution = go.Figure()
    fig_fatigue_distribution.add_trace(go.Histogram(
        x=fatigue_distribution,
        nbinsx=20,
        name="Fatigue Scores",
        opacity=0.7
    ))
    fig_fatigue_distribution.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals_scaled,
        mode='lines',
        name="Normal Curve",
        line=dict(color='red', dash='dash')
    ))

    fig_fatigue_distribution.update_layout(
        barmode='overlay',
        title="Fatigue Score Histogram with Normal Distribution Fit",
        xaxis_title="Fatigue Score",
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig_fatigue_distribution, use_container_width=True)

# Random Forest Model Training

if run_simulation and len(training_data) > 0:
    st.subheader("Random Forest Fatigue Prediction Model")

    df_train = pd.DataFrame(training_data)
    X = df_train.drop(columns=['fatigue_score'])
    y = df_train['fatigue_score']

    rf_model = RandomForestRegressor()
    rf_model.fit(X, y)
    rf_model_trained = True

    y_pred = rf_model.predict(X)

    # Predicting fatigue score based on average input conditions
    input_features = pd.DataFrame({
        'initial_density': [X['initial_density'].mean()],
        'alpha': [X['alpha'].mean()],
        'v_max': [X['v_max'].mean()],
        'sensor_interval': [X['sensor_interval'].mean()],
        'shockwave_speed': [np.mean(shockwave_speeds)]
    })

    predicted_fatigue_score = rf_model.predict(input_features)[0]

    if predicted_fatigue_score > np.percentile(fatigue_distribution, 95):
        safety_message = "Immediate Maintenance Required"
    elif predicted_fatigue_score > np.mean(fatigue_distribution) * 1.2:
        safety_message = "Warning: Monitor Fatigue Levels"
    else:
        safety_message = "Safe Operation"

    # Metrics
    r2 = r2_score(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)

    col_rf1, col_rf2, col_rf3 = st.columns(3)
    with col_rf1:
        st.metric("R² Score", f"{r2:.2f}")
    with col_rf2:
        st.metric("RMSE", f"{rmse:.2f}")
    with col_rf3:
        st.metric("MAE", f"{mae:.2f}")

    # Actual vs Predicted Plot
    fig_rf_actual_vs_pred = px.scatter(
        x=y,
        y=y_pred,
        labels={"x": "Actual Fatigue Score", "y": "Predicted Fatigue Score"},
        title="Random Forest: Actual vs Predicted Fatigue Score"
    )
    fig_rf_actual_vs_pred.add_shape(
        type="line", x0=min(y), y0=min(y), x1=max(y), y1=max(y),
        line=dict(color="gray", dash="dash")
    )
    st.plotly_chart(fig_rf_actual_vs_pred, use_container_width=True)

    # Feature Importance
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig_rf_importance = px.bar(
        importance_df,
        x='Feature',
        y='Importance',
        title='Random Forest Feature Importance'
    )
    st.plotly_chart(fig_rf_importance, use_container_width=True)

    # Shockwave Speed vs Fatigue Scatter
    st.subheader("Shockwave Speed vs Fatigue Score")
    fig_shockwave_vs_fatigue = px.scatter(
        x=shockwave_speeds,
        y=fatigue_distribution,
        labels={"x": "Shockwave Speed", "y": "Fatigue Score"},
        trendline='ols',
        title="Shockwave Speed vs Fatigue Score"
    )
    st.plotly_chart(fig_shockwave_vs_fatigue, use_container_width=True)

# Fatigue Health Monitoring Dashboard

if len(live_log) > 1 and run_simulation and rf_model_trained:
    st.subheader("Fatigue Health Monitoring Dashboard")

    avg_sim_fatigue = np.mean(fatigue_distribution)
    avg_live_fatigue = np.mean([entry['Fatigue Score'] for entry in live_log])
    abs_error = abs(avg_sim_fatigue - avg_live_fatigue)
    perc_error = (abs_error / avg_sim_fatigue) * 100

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Average Simulated Fatigue", value=f"{avg_sim_fatigue:.2f}")
        st.metric(label="Average Live Fatigue", value=f"{avg_live_fatigue:.2f}")
        st.metric(label="Error (%)", value=f"{perc_error:.2f}%")

    with col2:
        safe_upper_limit = avg_sim_fatigue * 1.15
        maintenance_limit = avg_sim_fatigue * 1.30

        if avg_live_fatigue > maintenance_limit:
            st.error(f"Immediate Maintenance Needed! ({avg_live_fatigue:.2f} > {maintenance_limit:.2f})")
        elif avg_live_fatigue > safe_upper_limit:
            st.warning(f"Fatigue Rising, Monitor Closely ({avg_live_fatigue:.2f} > {safe_upper_limit:.2f})")
        else:
            st.success("All fatigue scores within safe operating range.")

    # Simulated vs Live Fatigue Score Distribution
    st.subheader("Fatigue Score Comparison: Simulated vs Live Snapshots")
    
    # Safe way to prepare comparison DataFrame
    min_length = min(len(fatigue_distribution), len(live_log))
    
    if min_length > 0:
        df_compare = pd.DataFrame({
            'Live Fatigue Scores': [entry['Fatigue Score'] for entry in live_log[:min_length]],
            'Simulated Fatigue Scores': fatigue_distribution[:min_length]
        })
    
        fig_fatigue_compare = go.Figure()
        fig_fatigue_compare.add_trace(go.Histogram(x=df_compare['Simulated Fatigue Scores'], name="Simulated", opacity=0.7))
        fig_fatigue_compare.add_trace(go.Histogram(x=df_compare['Live Fatigue Scores'], name="Live", opacity=0.6))
    
        fig_fatigue_compare.update_layout(
            barmode='overlay',
            title="Fatigue Score Distribution: Simulated vs Live Snapshots",
            xaxis_title="Fatigue Score",
            yaxis_title="Count"
        )
        st.plotly_chart(fig_fatigue_compare, use_container_width=True)
    else:
        st.warning("Not enough data for comparison plots.")

# HTML Reporting 

def prepare_html_report():

    current_time = datetime.datetime.now().strftime("%B %d, %Y, %I:%M %p")
    live_ready = len(live_log) > 0
    sim_ready = len(training_data) > 0
    rf_ready = rf_model_trained
    warning_section = ""
    if not (live_ready and sim_ready and rf_ready):
        warnings = []
        if not live_ready:
            warnings.append("Live Snapshots Missing")
        if not sim_ready:
            warnings.append("Simulation Not Completed")
        if not rf_ready:
            warnings.append("Random Forest Model Not Trained")
        warning_text = "<br>".join(warnings)
        warning_section = f"""
        <div style='background-color:#ffe6e6;padding:10px;border:1px solid red;border-radius:5px;'>
            <b>Incomplete Data for Full Report:</b><br>{warning_text}
        </div><br>
        """

    # Prepare live table if available
    live_table_html = ""
    if live_ready:
        live_df = pd.DataFrame(live_log)
        live_table_html = live_df.to_html(index=False, border=1)

    #Simulation statistics
    if sim_ready:
        fatigue_array = np.array(fatigue_distribution)
        shockwave_array = np.array(shockwave_speeds)

        fatigue_mean = fatigue_array.mean()
        fatigue_min = fatigue_array.min()
        fatigue_max = fatigue_array.max()
        fatigue_std = fatigue_array.std()

        shockwave_mean = shockwave_array.mean()
        shockwave_min = shockwave_array.min()
        shockwave_max = shockwave_array.max()
    else:
        fatigue_mean = fatigue_min = fatigue_max = fatigue_std = 0
        shockwave_mean = shockwave_min = shockwave_max = 0

    # Random Forest metrics
    if rf_ready:
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = mean_squared_error(y, y_pred, squared=False)

        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        feature_importance_html = feature_importance_df.to_html(index=False, border=1)
    else:
        r2 = mae = rmse = 0
        feature_importance_html = "<p>No RF model trained yet.</p>"

    # Vehicle composition
    vehicle_html = f"""
    <table border="1" style="width:50%; margin-top:10px;">
    <tr><th>Vehicle Type</th><th>Percentage</th></tr>
    <tr><td>Cars</td><td>{car_pct:.1f}%</td></tr>
    <tr><td>Trucks</td><td>{truck_pct:.1f}%</td></tr>
    <tr><td>Buses</td><td>{bus_pct:.1f}%</td></tr>
    </table>
    """

    # Settings
    settings_html = f"""
    <ul>
        <li>Input Density: {input_density:.2f}</li>
        <li>Lane 1 Initial Density: {lane1_density:.2f}</li>
        <li>Lane 2 Initial Density: {lane2_density:.2f}</li>
        <li>Lane 1 Max Speed: {v_max_lane1} km/h</li>
        <li>Lane 2 Max Speed: {v_max_lane2} km/h</li>
        <li>Jam Induced Lane 1: {'Yes' if jam_lane1 else 'No'}</li>
        <li>Jam Induced Lane 2: {'Yes' if jam_lane2 else 'No'}</li>
        <li>Frozen Lane 1: {'Yes' if freeze_lane1 else 'No'}</li>
        <li>Frozen Lane 2: {'Yes' if freeze_lane2 else 'No'}</li>
        <li>Structural Alpha: {alpha:.5f}</li>
    </ul>
    """

    # Predicted Score
    if rf_ready:
        predicted_score_html = f"""
        <p><b>Predicted Fatigue Score:</b> {predicted_fatigue_score:.2f}</p>
        <p><b>Safety Status:</b> {safety_message}</p>
        """
    else:
        predicted_score_html = "<p> No Prediction Available.</p>"

    # Full HTML
    html_report = f"""
    <html>
    <head>
        <title>Bridge Traffic Simulation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            h1, h2, h3 {{ color: #2C3E50; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>

    {warning_section}

    <h1> Bridge Traffic Simulation and Fatigue Monitoring Report</h1>
    <h3> Report Generated on {current_time}</h3>

    <h2> Live Traffic Snapshots</h2>
    {live_table_html}

    <h2> Simulation Summary</h2>
    <ul>
        <li>Fatigue Score Stats → Min: {fatigue_min:.2f} | Max: {fatigue_max:.2f} | Mean: {fatigue_mean:.2f} | Std Dev: {fatigue_std:.2f}</li>
        <li>Shockwave Speed Stats → Min: {shockwave_min:.2f} m/s | Max: {shockwave_max:.2f} m/s | Mean: {shockwave_mean:.2f} m/s</li>
    </ul>

    <h2> Random Forest Model Metrics</h2>
    <ul>
        <li>R² Score: {r2:.3f}</li>
        <li>MAE: {mae:.2f}</li>
        <li>RMSE: {rmse:.2f}</li>
    </ul>

    <h3> Feature Importance</h3>
    {feature_importance_html}

    <h2> Vehicle Composition</h2>
    {vehicle_html}

    <h2> Initial Traffic Simulation Settings</h2>
    {settings_html}

    <h2> Predicted Maintenance Recommendation</h2>
    {predicted_score_html}

    <hr>
    <p><i>Generated by: Bridge Traffic Simulation App (Version 1.0)</i></p>

    </body>
    </html>
    """
    return html_report


#Download Final Report 

st.subheader("Download Final Report")

# Check readiness
live_ready = len(live_log) > 0
sim_ready = len(training_data) > 0
rf_ready = rf_model_trained

if live_ready and sim_ready and rf_ready:
    html_report = prepare_html_report()

    with st.expander("View Full Report (Preview)", expanded=True):
        st.components.v1.html(html_report, height=800, scrolling=True)

    st.download_button(
        label="Download Full Detailed Report (HTML)",
        data=html_report,
        file_name="Bridge_Traffic_Simulation_Full_Report.html",
        mime="text/html"
    )

else:
    st.warning(" Please complete Live Snapshot + Simulation + Random Forest Training to enable report download.")





