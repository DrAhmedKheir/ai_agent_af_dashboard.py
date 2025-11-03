
# ai_agent_af_dashboard.py
# ------------------------------------------------------------
# Agentic AI Dashboard for Agroforestry Decision Support
# (See comments inside for details)
# ------------------------------------------------------------

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pygam import LinearGAM, s
import matplotlib.pyplot as plt

try:
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

st.set_page_config(page_title="AI-Agent Dashboard (Agroforestry)", layout="wide", page_icon="🌳")

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    required = ["date", "distance_m", "orientation", "air_temp","soil_moist_15cm","solar_rad","wind_speed"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    if "yield_tha" not in df.columns:
        df["yield_tha"] = np.nan
    if "julian_day" not in df.columns:
        df["julian_day"] = df["date"].dt.dayofyear
    return df

def make_sequences(arr, lookback=24, horizon=24):
    X, y = [], []
    for i in range(len(arr) - lookback - horizon + 1):
        X.append(arr[i:i+lookback, :])
        y.append(arr[i+lookback:i+lookback+horizon, :])
    return np.array(X), np.array(y)

def train_lstm_forecaster(df, feature_cols, group_mask, lookback=24, horizon=24):
    dfg = df.loc[group_mask].sort_values("date").copy()
    dfg = dfg.dropna(subset=feature_cols)
    if dfg.empty or len(dfg) < (lookback + horizon + 10):
        raise ValueError("Not enough data for LSTM after filtering.")
    scaler = StandardScaler()
    vals = scaler.fit_transform(dfg[feature_cols].values)
    X, y = make_sequences(vals, lookback=lookback, horizon=horizon)
    n = len(X)
    n_train = int(n * 0.8)
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    model = keras.Sequential([
        layers.Input(shape=(lookback, len(feature_cols))),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dense(horizon * len(feature_cols))
    ])
    model.compile(optimizer="adam", loss="mse")
    es = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(X_train, y_train.reshape((y_train.shape[0], -1)),
              validation_data=(X_val, y_val.reshape((y_val.shape[0], -1))),
              epochs=50, batch_size=32, callbacks=[es], verbose=0)

    y_val_hat = model.predict(X_val, verbose=0).reshape(y_val.shape)
    r2s = {col: r2_score(y_val[:,0,j], y_val_hat[:,0,j]) for j,col in enumerate(feature_cols)}
    return model, scaler, dfg, r2s

def gam_fit(df, predictors, response="yield_tha"):
    d = df.dropna(subset=[response]).copy()
    if d.empty:
        raise ValueError("No yield data for GAM.")
    X = d[predictors].values
    y = d[response].values
    terms = sum([s(i) for i in range(len(predictors))])
    gam = LinearGAM(terms).fit(X, y)
    stats = {"R2_adj": gam.statistics_["pseudo_r2"]["explained_deviance"]}
    return gam, stats

def pareto_recommendation(row):
    score = 0.0
    if 15 <= row.get("air_temp", 0) <= 18: score += 1.0
    sm = row.get("soil_moist_15cm", 0)
    if 0.25 <= sm <= 0.30: score += 1.0
    if row.get("wind_speed", 0) <= 5: score += 0.5
    if (row.get("distance_m",24) in [4,7]) and (str(row.get("orientation","LW")).upper()=="LW"):
        score += 1.0
    return score

# Sidebar
st.sidebar.header("📥 Data")
demo = st.sidebar.checkbox("Use demo synthetic data", value=True)
if demo:
    rng = pd.date_range("2024-03-01", "2024-07-31", freq="H")
    rows = []
    for dist in [1,4,7,24]:
        for ori in ["LW","WW"]:
            air = 12 + 8*np.sin(np.linspace(0, 20*np.pi, len(rng))) + np.random.normal(0,1,len(rng))
            sm = 0.22 + 0.06*np.sin(np.linspace(0, 6*np.pi, len(rng))) + np.random.normal(0,0.02,len(rng))
            sr = np.clip(400*np.maximum(0,np.sin(np.linspace(0, 10*np.pi, len(rng)))),0,400)
            wnd = np.clip(np.random.normal(2.5, 1.0, len(rng)), 0, 7)
            y = 4.2 + (0.3 if dist in [4,7] else 0.0) + (0.1 if ori=="LW" else 0.0)
            for i,t in enumerate(rng):
                rows.append({"date":t,"distance_m":dist,"orientation":ori,
                             "air_temp":air[i],"soil_moist_15cm":sm[i],
                             "solar_rad":sr[i],"wind_speed":wnd[i],
                             "yield_tha": y if (t.hour==12 and t.day%15==0) else np.nan})
    df = pd.DataFrame(rows)
else:
    up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if up is None:
        st.stop()
    df = load_data(up)

st.sidebar.header("🎛️ Controls")
distance = st.sidebar.selectbox("Distance (m)", [1,4,7,24], index=2)
orientation = st.sidebar.selectbox("Orientation", ["LW","WW"], index=0)
horizon_days = st.sidebar.slider("Forecast horizon (days)", 7, 60, 30, step=7)
group_mask = (df["distance_m"]==distance) & (df["orientation"].str.upper()==orientation)

st.markdown("### 🌳 AI-Agent Dashboard")
c1,c2,c3 = st.columns(3)
c1.metric("Distance (m)", distance)
c2.metric("Orientation", orientation)
c3.metric("Forecast Horizon (days)", horizon_days)

feature_cols = ["air_temp","soil_moist_15cm","solar_rad","wind_speed"]
try:
    model, scaler, dfg, r2s = train_lstm_forecaster(df, feature_cols, group_mask, 24, 24)
    future_hours = horizon_days*24
    rolling = dfg[feature_cols].tail(24).values.copy()
    out = []
    for _ in range(future_hours//24):
        seq = scaler.transform(rolling).reshape(1,24,len(feature_cols))
        pred = model.predict(seq, verbose=0).reshape(24, len(feature_cols))
        preds = scaler.inverse_transform(pred)
        out.append(preds)
        rolling = np.vstack([rolling[8:], preds[:8]])
    forecast = np.vstack(out)
    future_index = pd.date_range(dfg["date"].max()+pd.Timedelta(hours=1),
                                 periods=len(forecast), freq="H")
    fc = pd.DataFrame(forecast, columns=feature_cols, index=future_index)
except Exception as e:
    st.warning(f"LSTM fallback: {e}")
    fc = None
    r2s = {}

try:
    gam, gam_stats = gam_fit(df, predictors=feature_cols, response="yield_tha")
except Exception as e:
    gam = None
    gam_stats = {"R2_adj": np.nan}
    st.info(f"GAM skipped: {e}")

def plot_response_panels(dfg_local):
    fig, axs = plt.subplots(2,2, figsize=(10,6))
    ax = axs[0,0]
    ax.scatter(dfg_local["air_temp"], dfg_local["yield_tha"], s=8, alpha=0.4)
    ax.set_xlabel("Air Temperature (°C)"); ax.set_ylabel("Yield (t/ha)")
    ax.axvspan(15,18, color="#2dd4bf", alpha=0.2)
    ax = axs[0,1]
    ax.scatter(dfg_local["soil_moist_15cm"], dfg_local["yield_tha"], s=8, alpha=0.4)
    ax.set_xlabel("Soil Moisture 15 cm (m³/m³)"); ax.set_ylabel("")
    ax.axvspan(0.25,0.30, color="#86efac", alpha=0.3)
    ax = axs[1,0]
    ax.scatter(dfg_local["solar_rad"], dfg_local["yield_tha"], s=8, alpha=0.4)
    ax.set_xlabel("Solar Radiation (W/m²)"); ax.set_ylabel("Yield (t/ha)")
    ax.axvline(200, color="#fbbf24", linestyle="--", alpha=0.8)
    ax = axs[1,1]
    ax.scatter(dfg_local["wind_speed"], dfg_local["yield_tha"], s=8, alpha=0.4)
    ax.set_xlabel("Wind Speed (m/s)"); ax.set_ylabel("")
    ax.invert_xaxis()
    fig.tight_layout()
    return fig

df_show = df.loc[group_mask].copy()
if "yield_tha" in df_show.columns:
    st.pyplot(plot_response_panels(df_show))

if fc is not None:
    row = {
        "air_temp": float(np.median(fc["air_temp"].head(24))),
        "soil_moist_15cm": float(np.median(fc["soil_moist_15cm"].head(24))),
        "solar_rad": float(np.median(fc["solar_rad"].head(24))),
        "wind_speed": float(np.median(fc["wind_speed"].head(24))),
        "distance_m": distance,
        "orientation": orientation
    }
    rec_lines = [
        f"• Distance & orientation: **{distance} m**, **{orientation}**",
        f"• Maintain soil moisture near **0.28 m³/m³** (now {row['soil_moist_15cm']:.2f})",
        f"• Aim for air temp **15–18 °C** (now {row['air_temp']:.1f} °C)",
        f"• Avoid wind > **5 m/s** (now {row['wind_speed']:.1f} m/s)",
        "• Best zones typically **4–7 m Leeward** given thresholds."
    ]
    st.success("**Best Decision Recommendation**  \n" + "\n".join(rec_lines))

st.markdown("---")
st.subheader("🧠 AI Agent (Q&A)")
if LANGCHAIN_AVAILABLE and ("OPENAI_API_KEY" in os.environ):
    system_prompt = (
        "You are an AI Agent for agroforestry microclimate decision support. "
        "Use thresholds: temperature 15–18°C, soil moisture 0.25–0.30 m³/m³, "
        "radiation ~200 W/m² saturation, wind negative >5 m/s. Consider distance/orientation effects."
    )
    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    user_q = st.text_input("Ask the agent about forecasts or decisions:")
    if user_q:
        resp = chat([SystemMessage(content=system_prompt),
                     HumanMessage(content=f"Distance={distance}, Orientation={orientation}. "
                                          f"Question: {user_q}")])
        st.write(resp.content)
else:
    st.info("LangChain/OpenAI not configured. Set OPENAI_API_KEY to enable chat.")

m1, m2, m3, m4 = st.columns(4)
with m1:
    try:
        st.metric("GAM adj. R²", f"{gam_stats.get('R2_adj', float('nan')):.3f}")
    except Exception:
        st.metric("GAM adj. R²", "NA")
with m2:
    st.metric("LSTM R² (AirTemp)", f"{(r2s.get('air_temp', float('nan'))):.2f}" if r2s else "NA")
with m3:
    st.metric("LSTM R² (SoilMoist)", f"{(r2s.get('soil_moist_15cm', float('nan'))):.2f}" if r2s else "NA")
with m4:
    st.metric("LSTM R² (Radiation)", f"{(r2s.get('solar_rad', float('nan'))):.2f}" if r2s else "NA")

st.caption("© 2025 AgML / UFZ style – Agentic AI for Agroforestry DSS")
