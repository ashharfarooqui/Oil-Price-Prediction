# Crude Oil Price Prediction App using Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Crude Oil Price Forecasting", layout="centered")

st.title("📈 Crude Oil Price Forecasting")

st.markdown("""
Welcome to the Oil Price Prediction App!  
You can forecast future crude oil prices using an ARIMA time series model, or try a feature-based prediction using your own inputs.
""")

# --- ARIMA Forecast Section ---
st.header("ARIMA Time Series Forecast")
with open("arima_model.pkl", "rb") as f:
    model = pickle.load(f)

steps = st.slider("Select number of days to forecast", min_value=1, max_value=30, value=7)
forecast = model.forecast(steps=steps)

st.subheader("Forecasted Close/Last Prices")
st.dataframe(pd.DataFrame({'Forecasted Price': forecast}))

fig, ax = plt.subplots(figsize=(10, 5))
pd.Series(model.data.endog).plot(ax=ax, label="Historical Close/Last")
forecast_idx = pd.RangeIndex(start=len(model.data.endog), stop=len(model.data.endog) + steps)
pd.Series(forecast, index=forecast_idx).plot(ax=ax, color='red', label="Forecast")
ax.set_xlabel("Time Index")
ax.set_ylabel("Close/Last Price")
ax.set_title("ARIMA Forecast vs Historical Data")
ax.legend()
st.pyplot(fig)

st.markdown("---")

# --- Feature-based Prediction Section ---
st.header("Feature-based Price Prediction (Demo)")

st.write("Enter feature values to predict the closing price (demo only):")

date = st.date_input("Select a date:", pd.to_datetime("2022-10-28"))
date_timestamp = pd.to_datetime(date).timestamp()

open_ = st.number_input("Open price:", value=70.0)
high = st.number_input("High price:", value=75.0)
low = st.number_input("Low price:", value=68.0)
volume_log = st.number_input("Log(Volume+1):", value=12.0)

if st.button("Predict"):
    # X_new = np.array([[date, open_, high, low, volume_log]])
    # pred = rf.predict(X_new)  # Uncomment and use your trained model
    pred = 70 + np.random.randn()  # Placeholder for demonstration
    st.success(f"Predicted Close/Last Price: {pred:.2f}")

st.markdown("""
---
*This app was built for educational purposes. For best results, use the ARIMA forecast above. Feature-based prediction is a demo placeholder.*
""")