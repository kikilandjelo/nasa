# 🌍 AgriShift - NASA Space Apps Challenge 2026
**Challenge:** Field Shift: Adapting Farms with NASA Data

## About the Project
AgriShift is an AI-powered Decision Support System for farmers. By combining **NASA's GPM** (Precipitation), **SMAP** (Soil Moisture), and **ECOSTRESS** (Land Surface Temperature) data with Machine Learning, it predicts and recommends optimal crop rotation strategies to combat climate change, water shortages, and soil degradation. 

The tool empowers farmers to input their specific priorities (e.g., water conservation vs. soil restoration) and compares three distinct rotation strategies, including their baseline monoculture.

## How to Run Locally
1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Run the web app: `streamlit run app.py`

## Technologies Used
- **Frontend:** Streamlit
- **AI/ML:** Scikit-learn (Random Forest Classifier)
- **Data Sources:** NASA Earthdata (GPM, SMAP, ECOSTRESS, Landsat, GRACE)
  
