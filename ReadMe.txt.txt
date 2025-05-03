README – Bridge Traffic Simulation and Fatigue Monitoring

PROJECT OVERVIEW:
This project simulates realistic bridge traffic flow, monitors traffic congestion, and predicts bridge fatigue using a Hybrid Digital Twin approach.
It integrates live traffic snapshots, Monte Carlo simulation, and machine learning predictions, all visualized through an interactive Streamlit dashboard.

SOFTWARE USED:

1. For Traffic Snapshot Capture and Preprocessing:
   - Selenium and OpenCV
   - GeoJSON file to define bridge geometry and calculate total bridge length.

2. For Simulation and Modeling:
   - IDE: Spyder v6.0 or Anaconda Environment
   - Python Libraries: NumPy, Pandas, SciPy
   - Monte Carlo simulations for uncertainty modeling
   - Modified LWR model for traffic flow simulation.

3. For Machine Learning Prediction:
   - Random Forest Regressor using Scikit-learn
   - Feature Importance Analysis

4. For Visualization and Results:
   - Streamlit (will run in default web browser)
   - Plotly for interactive graphs

HOW TO RUN THE PROJECT:

1. Open Anaconda Prompt.

2. Create and activate the environment (only first time):
	conda create -n bridge_sim python=3.11
	conda activate bridge_sim
	pip install streamlit numpy pandas plotly opencv-python selenium scikit-learn scipy webdriver-manager

3. Navigate to the folder containing the python file (FinalCode_6571.py) 

4. Paste the following command (replace with your correct file path):

   streamlit run "path_to_your/FinalCode_6571.py"

5. Anaconda Prompt will show the following:

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:xxxx
  Network URL: http://your ip address:xxxx

6. Your default browser will open with the Streamlit dashboard.

WHAT YOU WILL SEE:

. Sliders and inputs to adjust simulation settings (e.g., traffic density, vehicle speeds).

. Live Traffic Snapshot capture from FL511 portal.

. Simulated Traffic Density Profiles and Stress Analysis.

. Fatigue Score Distribution (Simulated vs. Live Snapshots).

. Random Forest Prediction Model Evaluation.

. Maintenance Recommendation Output.

. Download HTML Report (will only be available if you either 'Fetch live snap after enabling real-time snapshot'(on the sidebar at the top) or 'Start Guided snap run' (in the main display) then click 'Run Monte Carlo Simulations'

DATA FILES:

- export.geojson – GeoJSON file for bridge geometry extraction.
- MelbourneCauseway.jpg – Image used for bridge visualization.

IMPORTANT:

- Make sure Chrome Browser is installed (Selenium will require it).

- WebDriverManager will automatically download the correct chromedriver.

- Internet connection is required for live traffic snapshot capture.

- Correct file paths must be set inside the Streamlit script if additional resources are moved.

7. Using the dashboard to generate report or using the dashboard

	In your browser, adjust simulation settings:

	Set lane traffic densities and vehicle speeds.

	Choose the car/truck/bus percentage mix.

	Set structural sensitivity (alpha) and sensor spacing.

8. To capture live traffic data:

	Check the "Enable Live Snapshot" option. (Also, see note below)

	Click "Fetch Live Snapshot" to capture FL511 traffic conditions.
	
	Note: In the app, the Fetch Live Snapshot option captures a single real-time traffic snapshot from FL511 for immediate traffic condition analysis.

	      In contrast, the Guided Snapshot Capture feature allows users to schedule and automate the collection of multiple snapshots over time, helping track how traffic congestion patterns and shockwave speeds change dynamically


9. To run simulations:

	Click "Run Monte Carlo Simulation" to generate traffic flow, congestion effects, and fatigue scores.

10. View results:

	See traffic density maps, stress distributions, fatigue scores, and Random Forest predictions.

	Monitor maintenance recommendations (Safe / Warning / Immediate Action).

11. Download report if needed:

	A full HTML report can be generated after simulation.
	Note: HTML Report will only be available if you either 'Fetch live snap after enabling real-time snapshot'(on the sidebar at the top) or 'Start Guided snap run' (in the main display) then click 'Run Monte Carlo Simulations'





END OF ReadMe.txt