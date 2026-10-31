# 🌾 AgriShift: AI-Powered Crop Rotation Explorer
**NASA Space Apps Challenge 2026 | Category: Field Shift: Adapting Farms with NASA Data**

## 🌍 About the Project
AgriShift is an AI-powered Decision-Support System designed for farmers worldwide. By seamlessly combining live NASA Earth observations — including GPM (Precipitation), SMAP (Soil Moisture), and ECOSTRESS (Land Surface Temperature) — with a localized Machine Learning risk-assessment model, AgriShift dynamically predicts environmental stress and recommends optimal crop rotation strategies. 

The tool empowers farmers to combat climate change, water shortages, and soil degradation by evaluating a comprehensive database of global crops against real-time telemetry and their specific seasonal priorities (e.g., water conservation vs. soil restoration).

## ✨ Key Features
* **Live NASA Data Integration:** Fetches real-time spatial metrics from NASA Earthdata (GPM, SMAP, ECOSTRESS, Landsat NDVI) for any global coordinate.
* **Fully Dynamic AI Analysis:** Evaluates crop compatibility and generates custom, highly adaptive rotation strategies on the fly without static templates.
* **Interactive 3D Risk Mapping:** Utilizes Pydeck to visualize the farmer's exact location alongside regional AI-assessed environmental risk zones.
* **Comprehensive Crop Database:** Features a searchable, auto-completing database of global agricultural crops.
* **Failsafe Geolocation:** Supports Auto-GPS, manual coordinate adjustment, and direct parsing of Google Maps URLs (including shortened mobile links).

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/agrishift.git](https://github.com/yourusername/agrishift.git)
   cd agrishift
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web app:**
   ```bash
   streamlit run app.py
   ```

## 🛠️ Technologies Used
* **Frontend & Framework:** Streamlit
* **AI & Machine Learning:** Scikit-learn (Random Forest Classifier), Python deterministic data fusion
* **Geospatial & Visualization:** Pydeck, Streamlit-Geolocation
* **Data Sources & API:** NASA Earthdata (earthaccess), Requests

## ⚠️ Disclaimer
*AgriShift is a Decision-Support Tool. AI predictions are based on NASA Earth observations to help identify patterns and vulnerabilities, but they **do not replace** the expertise of local agronomists and farmer judgment.*
