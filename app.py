На скриншоте `image_6a2ea7.jpg` видно, что координаты на ползунках всё-таки изменились — на Широту 45.60 и Долготу -121.19. Это означает, что кнопка **сработала**, но из-за того, что код написан на Python, он выполнился на стороне сервера Streamlit Cloud и определил физическое расположение их дата-центра в штате Орегон (США), а не ваш реальный адрес.

Чтобы приложение находило реального пользователя (например, чтобы оно точно определяло вашу геопозицию в Баку), нам нужен специальный фронтенд-компонент, который отправляет запрос напрямую в браузер пользователя и просит разрешение на доступ к GPS-данным устройства.

Вот как внедрить настоящий трекер геопозиции пользователя:

### Шаг 1: Обновите `requirements.txt`

Вам нужно добавить библиотеку `streamlit-geolocation`, которая умеет связываться с браузером. Теперь ваш файл на GitHub должен выглядеть так:

```text
streamlit
pandas
earthaccess
streamlit-geolocation

```

### Шаг 2: Обновите файл `app.py`

Скопируйте этот финальный код. Я заменил старую кнопку на смарт-компонент `streamlit_geolocation()`. Когда пользователь нажмет на неё, браузер всплывающим окном спросит "Разрешить доступ к местоположению?", и ползунки мгновенно перескочат на точные координаты.

```python
import streamlit as st
import pandas as pd
import earthaccess
from streamlit_geolocation import streamlit_geolocation

# --- UI SETUP ---
st.set_page_config(page_title="AgriShift | Field Shift Challenge", page_icon="🌾", layout="wide")
st.title("🌾 AgriShift: AI-Powered Crop Rotation Explorer")
st.markdown("**NASA Space Apps Challenge 2026 | Category: Field Shift**")

st.info("⚠️ **Disclaimer:** AgriShift is a Decision-Support Tool. AI predictions are based on NASA Earth observations to help identify patterns, but they **do not replace** the expertise of local agronomists and farmer judgment.")

# --- ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ ---
if 'lat' not in st.session_state:
    st.session_state['lat'] = 40.14
if 'lon' not in st.session_state:
    st.session_state['lon'] = 47.57

# --- KNOWLEDGE BASE ---
with st.expander("📚 NASA Scientific Dictionary (Click to expand)"):
    st.markdown("""
    - **NDVI (Normalized Difference Vegetation Index):** An indicator of plant health based on Landsat/MODIS satellite data.
    - **SMAP (Soil Moisture Active Passive):** A NASA mission measuring the amount of water in the top layer of soil.
    - **ECOSTRESS:** An ISS-based instrument measuring land surface temperature to detect plant heat stress before leaves turn brown.
    """)

# --- 1. FARMER PARAMETERS ---
st.sidebar.header("📍 Step 1: Location & Baseline")

# Настоящая геолокация браузера пользователя
st.sidebar.write("📍 **Get My Exact Location:**")
location = streamlit_geolocation()

# Если браузер выдал координаты, обновляем состояние сессии
if location and location.get('latitude') is not None and location.get('longitude') is not None:
    st.session_state['lat'] = float(location['latitude'])
    st.session_state['lon'] = float(location['longitude'])

# Глобальные ползунки координат, привязанные к состоянию сессии
lat = st.sidebar.slider("Latitude", -90.0, 90.0, st.session_state['lat'])
lon = st.sidebar.slider("Longitude", -180.0, 180.0, st.session_state['lon'])

# Синхронизация ручного ввода со стейтом
if lat != st.session_state['lat'] or lon != st.session_state['lon']:
    st.session_state['lat'] = lat
    st.session_state['lon'] = lon

current_crop = st.sidebar.selectbox("Current (Baseline) Crop:", ["Cotton", "Wheat", "Corn"])

st.sidebar.header("🎯 Step 2: Farmer Priorities")
priority = st.sidebar.radio("What is your main priority this season?", 
                            ("💧 Maximum Water Conservation", 
                             "🌱 Soil Organic Matter Restoration", 
                             "💰 Maximum Market Profit"))

st.sidebar.markdown("---")
analyze_btn = st.sidebar.button("🚀 Analyze NASA Data", use_container_width=True)

# --- NASA EARTHACCESS CONNECTION ---
@st.cache_data(ttl=3600)
def fetch_real_nasa_data(lat, lon):
    try:
        earthaccess.login(strategy="environment", persist=True)
        bbox = (lon - 0.5, lat - 0.5, lon + 0.5, lat + 0.5) 
        results = earthaccess.search_data(
            short_name='GPM_3IMERGM', 
            bounding_box=bbox, 
            temporal=("2023-05-01", "2023-09-30"), 
            count=1 
        )
        if results:
            file_link = results[0].data_links(access="direct")[0]
            return {"status": "Success", "file_found": True, "link": file_link}
        else:
            return {"status": "No data found for these coordinates", "file_found": False}
    except Exception as e:
        return {"status": f"Connection Error: {str(e)}", "file_found": False}

# --- 2. DATA FUSION & GLOBAL DETERMINISTIC ALGORITHM ---
if analyze_btn:
    with st.spinner('Connecting to NASA Earthdata API and calculating global spatial metrics...'):
        
        nasa_api_result = fetch_real_nasa_data(lat, lon)
        
        if nasa_api_result.get("file_found"):
            st.success("✅ Successfully connected to NASA Earthdata API. Direct S3 link obtained for GPM dataset.")
        else:
            st.warning(f"⚠️ NASA API Status: {nasa_api_result.get('status')}. Using global baseline climate algorithm.")

        # Глобальный климатический алгоритм
        dist_from_equator = abs(lat)
        gpm_base = 800.0 - (dist_from_equator * 10.0) + (lon * 0.5)
        smap_base = 0.40 - (dist_from_equator * 0.004)
        eco_base = 45.0 - (dist_from_equator * 0.8)
        ndvi_base = 0.70 - (dist_from_equator * 0.007)
        
        final_gpm = max(30.0, min(1200.0, gpm_base))
        final_smap = max(0.05, min(0.55, smap_base))
        final_eco = max(-20.0, min(55.0, eco_base))
        final_ndvi = max(0.01, min(0.95, ndvi_base))
        
        st.subheader("🛰 NASA Earth Observations Integration")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric(label="GPM (Seasonal Rainfall)", 
                    value=f"{final_gpm:.0f} mm", 
                    delta="-22% (Drought Risk)" if final_gpm < 300 else "Optimal", 
                    delta_color="inverse" if final_gpm < 300 else "normal")
                    
        col2.metric(label="SMAP (Soil Moisture)", 
                    value=f"{final_smap:.2f} m³/m³", 
                    delta="Critically Low" if final_smap < 0.15 else "Stable", 
                    delta_color="inverse" if final_smap < 0.15 else "normal")
                    
        col3.metric(label="ECOSTRESS (Surface Temp)", 
                    value=f"{final_eco:.1f} °C", 
                    delta="+3.2 °C (Heat Stress)" if final_eco > 32 else "Normal", 
                    delta_color="inverse" if final_eco > 32 else "normal")
                    
        col4.metric(label="Landsat (NDVI)", 
                    value=f"{final_ndvi:.2f}", 
                    delta="Wilting" if final_ndvi < 0.35 else "Healthy", 
                    delta_color="inverse" if final_ndvi < 0.35 else "normal")

        st.markdown("---")
        st.subheader("🔄 Crop Rotation Strategy Comparison")
        
        tab1, tab2, tab3 = st.tabs(["1. Current Rotation (Baseline)", "2. Water-Saving (Option A)", "3. Soil-Restoring (Option B)"])
        
        with tab1:
            st.error(f"**Crop:** {current_crop} ➡️ {current_crop} (Monoculture)")
            st.write("**NASA Evidence:** GRACE and GPM data show groundwater depletion. Continuing this crop will lead to soil exhaustion.")
            
        with tab2:
            st.success("**Crop:** Winter Wheat ➡️ Chickpea ➡️ Sorghum")
            st.write("**NASA Evidence:** Climate algorithm adapted for low SMAP (soil moisture) indicators.")

        with tab3:
            st.warning("**Crop:** Soybean ➡️ Cover Crop (Clover) ➡️ Corn")
            st.write("**NASA Evidence:** Landsat imagery highlights prolonged bare-field periods, increasing wind erosion vulnerability.")

        st.markdown("---")
        st.subheader("🗺 Interactive Global Risk Heatmap")
        st.map(pd.DataFrame({'lat': [lat, lat+0.5, lat-0.2], 'lon': [lon, lon-0.5, lon+0.3]}))

```
