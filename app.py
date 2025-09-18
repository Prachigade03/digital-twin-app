# app.py
"""
Digital Twin - Streamlit App with MongoDB
- Simulates or streams sensor data
- Uses trained RandomForest model (food_quality_model.pkl)
- MongoDB Atlas integration for live data
- Role-based login (operator/manager)
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import streamlit as st
from streamlit_autorefresh import st_autorefresh

import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# -------------------------
# --------- CONFIG --------
# -------------------------
USERS = {
    "operator": {"password": "op123", "role": "operator"},
    "manager": {"password": "mg123", "role": "manager"}
}

SIM_BATCH_SIZE = 20
MODEL_FILE = "food_quality_model (2).pkl"
FEATURE_COLS = ["temp", "humidity", "airflow", "vibration"]

# -------------------------
# --- MongoDB Connection ---
# -------------------------
uri = "mongodb+srv://shrinidhi22311609_db_user:shri1511@cluster0.jx3x7n9.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['sensor_db']
collection = db['sensor_data']

def get_latest_data(n=50):
    cursor = collection.find().sort("_id", -1).limit(n)
    return list(cursor)[::-1]

# -------------------------
# ---- Helper functions ---
# -------------------------
def authenticate(username: str, password: str):
    user = USERS.get(username)
    if user and user["password"] == password:
        return {"username": username, "role": user["role"]}
    return None

def init_session():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user = None
    if "data" not in st.session_state:
        st.session_state.data = pd.DataFrame(columns=["timestamp","temp","humidity","airflow","vibration","batch_id","pred_label","pred_prob"])
    if "running" not in st.session_state:
        st.session_state.running = False
    if "sim_index" not in st.session_state:
        st.session_state.sim_index = 0
    if "setpoint_temp" not in st.session_state:
        st.session_state.setpoint_temp = 60.0
    if "setpoint_humidity" not in st.session_state:
        st.session_state.setpoint_humidity = 50.0
    if "setpoint_vibration" not in st.session_state:
        st.session_state.setpoint_vibration = 0.5
    if "setpoint_airflow" not in st.session_state:
        st.session_state.setpoint_airflow = 1.0
    if "model" not in st.session_state:
        st.session_state.model = None
        st.session_state.model_loaded = False
    if "data_source" not in st.session_state:
        st.session_state.data_source = "Simulated Data"  # default

def load_model():
    if os.path.exists(MODEL_FILE):
        st.session_state.model = joblib.load(MODEL_FILE)
        st.session_state.model_loaded = True
        st.success(f"✅ Loaded trained model: {MODEL_FILE}")
    else:
        st.error(f"❌ Model file '{MODEL_FILE}' not found! Place it in the same folder as app.py.")
        st.stop()

def generate_record(i, setpoints):
    temp = np.random.normal(setpoints["temp"], 2.0) + 0.5 * np.sin(i/10)
    humidity = np.random.normal(setpoints["humidity"], 1.5) + 0.2 * np.cos(i/13)
    vibration = max(0.0, np.random.normal(setpoints["vibration"], 0.05) + 0.02 * np.sin(i/7))
    airflow = max(0.0, np.random.normal(setpoints["airflow"], 0.05) + 0.02 * np.cos(i/9))
    batch_id = f"batch_{(i // SIM_BATCH_SIZE) + 1}"
    return {
        "timestamp": pd.Timestamp.now(),
        "temp": round(float(temp), 2),
        "humidity": round(float(humidity), 2),
        "airflow": round(float(airflow), 3),
        "vibration": round(float(vibration), 3),
        "batch_id": batch_id
    }

def append_record(rec):
    df = st.session_state.data
    if st.session_state.model_loaded:
        # Map app names -> model names
        rename_map = {"temp": "temperature", "humidity": "moisture"}
        X = pd.DataFrame([{
            "temperature": rec["temp"],
            "moisture": rec["humidity"],
            "airflow": rec["airflow"],
            "vibration": rec["vibration"]
        }])

        prob = float(st.session_state.model.predict_proba(X)[:,1])
        label = int(st.session_state.model.predict(X)[0])
    else:
        prob, label = None, None
    rec_out = rec.copy()
    rec_out["pred_label"] = label
    rec_out["pred_prob"] = prob
    st.session_state.data = pd.concat([df, pd.DataFrame([rec_out])], ignore_index=True)

# -------------------------
# ------- UI LAYOUT -------
# -------------------------
st.set_page_config(page_title="Digital Twin Dashboard", layout="wide")
init_session()
load_model()

if not st.session_state.logged_in:
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        auth = authenticate(username, password)
        if auth:
            st.session_state.logged_in = True
            st.session_state.user = auth
            st.rerun()
        else:
            st.error("Invalid credentials. Use operator/op123 or manager/mg123")

else:
    user = st.session_state.user
    st.sidebar.title(f"Welcome, {user['username']} ({user['role']})")
    if st.sidebar.button("Logout"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    # Data source selector
    st.sidebar.markdown("### Data Source")
    st.session_state.data_source = st.sidebar.radio("Select source:", ["Simulated Data", "MongoDB Live Data"])

    page = st.sidebar.radio("Page", ["Dashboard", "Control Panel", "Admin" if user["role"]=="manager" else "Info"])
    st.markdown("## 🍪 Digital Twin — Food Process Quality Demo")

    if page == "Dashboard":
        st.subheader("Live Sensor Dashboard")

        if st.session_state.data_source == "Simulated Data":
            left, right = st.columns([3,1])
            with right:
                st.subheader("Simulation Controls")
                if not st.session_state.running:
                    if st.button("Start Simulation"):
                        st.session_state.running = True
                else:
                    if st.button("Stop Simulation"):
                        st.session_state.running = False
                st.markdown("---")
                st.session_state.setpoint_temp = st.slider("Target Temp (°C)", 20.0, 120.0, st.session_state.setpoint_temp)
                st.session_state.setpoint_humidity = st.slider("Target Humidity (%)", 0.0, 100.0, st.session_state.setpoint_humidity)
                st.session_state.setpoint_vibration = st.slider("Target Vibration (g)", 0.0, 5.0, st.session_state.setpoint_vibration)
                st.session_state.setpoint_airflow = st.slider("Target Airflow", 0.0, 5.0, st.session_state.setpoint_airflow)
                if st.button("Clear Data"):
                    st.session_state.data = pd.DataFrame(columns=["timestamp","temp","humidity","airflow","vibration","batch_id","pred_label","pred_prob"])
            with left:
                df = st.session_state.data.copy()
                if st.session_state.running:
                    rec = generate_record(st.session_state.sim_index, {
                        "temp": st.session_state.setpoint_temp,
                        "humidity": st.session_state.setpoint_humidity,
                        "vibration": st.session_state.setpoint_vibration,
                        "airflow": st.session_state.setpoint_airflow
                    })
                    append_record(rec)
                    st.session_state.sim_index += 1
                    time.sleep(0.25)
                if not df.empty:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    st.line_chart(df.set_index("timestamp")[["temp","humidity","airflow","vibration"]].tail(200))
                    display_df = df.tail(20).reset_index(drop=True).copy()
                    display_df["quality"] = display_df["pred_label"].map({1:"DEFECT",0:"GOOD"})
                    st.dataframe(display_df[["timestamp","batch_id","temp","humidity","airflow","vibration","quality","pred_prob"]])
                else:
                    st.info("No simulated data yet. Start the simulation.")

        else:  # MongoDB Live Data
            st.subheader("Fetching from MongoDB Atlas")

            from streamlit.runtime.scriptrunner import add_script_run_ctx

            # Auto-refresh every 5 seconds


            # Auto-refresh every 5 seconds
            count = st_autorefresh(interval=5 * 1000, key="refresh")

            mongo_data = get_latest_data(50)
            if mongo_data:
                df = pd.DataFrame(mongo_data)
                if "_id" in df.columns:
                    df = df.drop(columns=["_id"])
                if st.session_state.model_loaded and all(c in df.columns for c in FEATURE_COLS):
                    X = df[FEATURE_COLS]
                    df["pred_prob"] = st.session_state.model.predict_proba(X)[:,1]
                    df["pred_label"] = st.session_state.model.predict(X)
                    df["quality"] = df["pred_label"].map({1:"DEFECT",0:"GOOD"})
                st.line_chart(df[["temp","humidity","vibration"]])
                st.dataframe(df.tail(20))
            else:
                st.warning("No data found in MongoDB.")

    elif page == "Control Panel":
        st.subheader("Control Panel (Simulated Mode only)")
        st.write(f"Temperature: {st.session_state.setpoint_temp} °C")
        st.write(f"Humidity: {st.session_state.setpoint_humidity} %")
        st.write(f"Airflow: {st.session_state.setpoint_airflow}")
        st.write(f"Vibration: {st.session_state.setpoint_vibration} g")

    elif page == "Admin":
        st.subheader("Manager Analytics")
        df = st.session_state.data.copy()
        if not df.empty:
            st.write(df[["temp","humidity","airflow","vibration"]].describe())
            defect_counts = df.groupby("batch_id")["pred_label"].sum().reset_index().rename(columns={"pred_label":"defect_count"})
            st.table(defect_counts)
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", data=csv, file_name="sensor_data_export.csv", mime="text/csv")
        else:
            st.info("No simulated data available for analytics.")

    else:
        st.subheader("Info")
        st.markdown("This demo app can run in two modes: Simulated Data or MongoDB Live Data.")
