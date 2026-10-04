import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="California House Price Prediction App", layout="centered"
)

st.title("California House Price Prediction App")
st.write(
    "Bu uygulama, Kaliforniya konut veri seti ile eğitilmiş RandomForest modeli kullanarak medyan konut değerini tahmin eder."
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("house_price_at_California_model.pkl")
    scaler = joblib.load("house_price_at_California_scaler.pkl")
    return model, scaler


model, scaler = load_artifacts()

st.subheader("Bölge ve Konut Özelliklerini Giriniz:")

col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input("Medyan Gelir (MedInc)", value=3.5)
    house_age = st.number_input("Bina Yaşı (HouseAge)", value=25.0)
    ave_rooms = st.number_input("Ortalama Oda Sayısı (AveRooms)", value=5.0)
    ave_bedrms = st.number_input(
        "Ortalama Yatak Odası Sayısı (AveBedrms)", value=1.0
    )

with col2:
    population = st.number_input("Nüfus (Population)", value=1400.0)
    ave_occup = st.number_input("Ortalama Hanehalkı Doluluğu (AveOccup)", value=3.0)
    latitude = st.number_input("Enlem (Latitude)", value=34.0)
    longitude = st.number_input("Boylam (Longitude)", value=-118.0)

if st.button("Fiyatı Tahmin Et", type="primary"):
    input_data = np.array(
        [[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]]
    )
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(
        f"Tahmin Edilen Medyan Konut Değeri: **${prediction[0] * 100,000:,.2f}**"
    )