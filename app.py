import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import datetime

# --- НАСТРОЙКА ИНТЕРФЕЙСА ---
st.set_page_config(page_title="AgriShift | NASA Space Apps", page_icon="🌍", layout="wide")
st.title("🌍 AgriShift: Intelligent Crop Rotation")
st.markdown("**Challenge:** Field Shift: Adapting Farms with NASA Data")
st.markdown("Система поддержки принятия решений для фермеров на базе ИИ и спутниковых данных NASA (GPM & ECOSTRESS).")

# --- 1. ML-МОДЕЛЬ (Искусственный интеллект) ---
# Генерируем обучающий датасет (в реальном проекте это исторические данные NASA)
@st.cache_resource
def train_model():
    np.random.seed(42)
    # Признаки: Осадки (мм), Температура (°C), Текущая культура (0-Хлопок, 1-Пшеница, 2-Кукуруза)
    X = np.random.rand(500, 3) 
    X[:, 0] = X[:, 0] * 800  # Осадки от 0 до 800 мм
    X[:, 1] = X[:, 1] * 20 + 15 # Температура от 15 до 35 C
    X[:, 2] = np.random.randint(0, 3, 500)
    
    # Целевая переменная: Рекомендуемый севооборот (0-Засухоустойчивый, 1-Стандартный, 2-Восстанавливающий)
    y = np.where((X[:, 0] < 300) | (X[:, 1] > 30), 0, np.where(X[:, 0] > 500, 2, 1))
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

ai_model = train_model()

# --- 2. ИНТЕРФЕЙС ПОЛЬЗОВАТЕЛЯ ---
st.sidebar.header("📍 Параметры фермы")
lat = st.sidebar.slider("Широта (Latitude)", 38.0, 42.0, 40.14) # По умолчанию координаты Азербайджана
lon = st.sidebar.slider("Долгота (Longitude)", 44.0, 51.0, 47.57)
current_crop = st.sidebar.selectbox("Текущая культура", ["Хлопок", "Пшеница", "Кукуруза"])

crop_dict = {"Хлопок": 0, "Пшеница": 1, "Кукуруза": 2}

st.sidebar.info("🚀 При нажатии кнопки система 'обращается' к API NASA Earthdata для получения климатических метрик.")

if st.sidebar.button("Анализировать данные NASA"):
    with st.spinner('Подключение к NASA GPM (Осадки) и ECOSTRESS (Температура)...'):
        # Симуляция задержки API
        import time
        time.sleep(2)
        
        # Симуляция полученных данных для выбранных координат
        sim_rain = np.random.uniform(200, 600)
        sim_temp = np.random.uniform(22, 34)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🛰 Осадки (NASA GPM)", f"{sim_rain:.1f} мм", "-15% от нормы", delta_color="inverse")
        col2.metric("🌡 Температура почвы (ECOSTRESS)", f"{sim_temp:.1f} °C", "+2.1 °C", delta_color="inverse")
        col3.metric("🌱 Индекс вегетации (Landsat)", "0.45", "Стресс", delta_color="inverse")

        # Прогноз ИИ
        prediction = ai_model.predict([[sim_rain, sim_temp, crop_dict[current_crop]]])[0]
        
        st.subheader("🤖 ИИ Рекомендация по севообороту")
        
        if prediction == 0:
            st.error("**Стратегия:** Засухоустойчивый севооборот (Высокий риск водного стресса)")
            st.markdown("- **Рекомендуемые культуры:** Нут (Chickpea), Сорго, Озимая пшеница.")
            st.markdown("- **Обоснование ИИ:** Модель зафиксировала критическое снижение осадков. Нут обогатит почву азотом без обильного полива.")
        elif prediction == 1:
            st.warning("**Стратегия:** Стандартный севооборот (Умеренный риск)")
            st.markdown("- **Рекомендуемые культуры:** Подсолнечник, Ячмень.")
            st.markdown("- **Обоснование ИИ:** Климатические условия стабильны, требуется поддержка базового плодородия почвы.")
        else:
            st.success("**Стратегия:** Восстанавливающий севооборот (Благоприятные условия)")
            st.markdown("- **Рекомендуемые культуры:** Соя, Покровный клевер.")
            st.markdown("- **Обоснование ИИ:** Достаточный уровень влаги позволяет высадить культуры для максимального восстановления органики в почве.")
        
        # Карта
        st.subheader("🗺 Карта региона")
        st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}))
