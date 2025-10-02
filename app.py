import streamlit as st
import pandas as pd
import joblib
import os
from dotenv import load_dotenv
import google.generativeai as genai
from src.data_loader import load_and_clean_data
from src.visualization import (
    plot_market_size_trend,
    plot_regional_growth,
    plot_brand_market_share,
    plot_correlation_heatmap,
    plot_brand_region_heatmap,
    plot_top_company_activity,
)
from huggingface_hub import hf_hub_download

# --- Load Environment Variables ---
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Indian Condom Market Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling ---
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: #e0e0e0;
        }
        .main {
            background-color: #0e1117;
            color: #e0e0e0;
        }
        h1, h2, h3, h4 {
            color: #ff4b4b;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #262730;
            color: white;
            border-radius: 10px 10px 0px 0px;
            padding: 10px 20px;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ff4b4b;
            color: white;
        }
        .stButton button {
            background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton button:hover {
            background: linear-gradient(90deg, #ff6b6b, #ff4b4b);
        }
        .card {
            padding: 15px;
            background-color: #1a1a1a;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Configure Gemini API ---
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
gen_model = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        gen_model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')
    except Exception as e:
        st.error(f"❌ Failed to configure Gemini API: {e}")
else:
    st.warning("⚠️ Gemini API Key not found. AI analysis will be disabled.")

# --- Gemini Insights Function ---
def get_gemini_insights(data_summary: str):
    if not gen_model:
        return "AI analysis is disabled. Please provide a Gemini API key."

    system_prompt = (
        "You are 'Sass,' a witty, sharp, and slightly naughty business strategist specializing in the sexual wellness market. "
        "Your tone is playful but insights are serious. Provide three sections: "
        "1. **🔥 Hot & Heavy Insights**: sharp business observations. "
        "2. **💡 Naughty Marketing Slogans**: edgy, clever slogans. "
        "3. **🤫 Untapped Pleasures**: one untapped opportunity."
    )

    try:
        response = gen_model.generate_content([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the data summary:\n{data_summary}"}
        ])
        return response.text
    except Exception as e:
        st.error(f"Error contacting Gemini API: {e}")
        return "⚠️ AI insights unavailable."

# --- Load Model from Hugging Face ---
@st.cache_resource
def load_model_from_hf():
    try:
        repo_id = "ujan2003/market_size_predictor"
        filename = "market_size_predictor.pkl"
        with st.spinner("📥 Downloading prediction model from Hugging Face..."):
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        model = joblib.load(model_path)
        st.success("✅ Market prediction model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_model_from_hf()

# --- Sidebar ---
with st.sidebar:
    st.image("https://placehold.co/400x150/1a1a1a/ff4b4b?text=Market+Analysis", use_column_width=True)
    st.title("⚙️ Control Panel")
    st.write("Upload data or generate predictions.")
    
    uploaded_file = st.file_uploader("📂 Upload CSV Data", type=["csv"])
    if uploaded_file is not None:
        df = load_and_clean_data(uploaded_file)
    else:
        df = load_and_clean_data("India_Condom_Market_Dataset.csv")

    if model and df is not None and not df.empty:
        with st.expander("🔮 Market Size Predictor", expanded=False):
            with st.form("prediction_form"):
                st.subheader("📝 Enter Market Scenario")
                unique_brands = sorted(df['brand name'].str.title().unique())
                unique_regions = sorted(df['region'].str.title().unique())
                unique_materials = sorted(df['material type'].str.title().unique())
                unique_products = sorted(df['product type'].str.title().unique())
                
                year = st.number_input("📅 Year", min_value=2020, max_value=2040, value=2025)
                brand_name = st.selectbox("🏷️ Brand", options=unique_brands)
                region = st.selectbox("🌍 Region", options=unique_regions)
                material_type = st.selectbox("🔬 Material", options=unique_materials)
                product_type = st.selectbox("📦 Product Type", options=unique_products)
                growth_rate = st.slider("📈 Growth Rate (%)", 0.0, 25.0, 10.5)
                
                submitted = st.form_submit_button("🚀 Predict Market Size")
                if submitted:
                    input_data = pd.DataFrame([{
                        'Year': year,
                        'CAGR (%)': 10.0,
                        'Material Type': material_type.lower(),
                        'Product Type': product_type.lower(),
                        'Distribution Channel': 'e-commerce',
                        'Region': region.lower(),
                        'Market Penetration': 'medium',
                        'Growth Rate (%)': growth_rate,
                        'Brand Name': brand_name.lower(),
                        'Market Share (%)': 20.0,
                        'Revenue Contribution (%)': 5.0,
                        'Innovation Index': 5.0,
                        'Regulatory Impact': 'medium',
                        'Awareness Campaign Impact': 50.0
                    }])
                    with st.spinner("⚡ Running prediction..."):
                        prediction = model.predict(input_data)
                        st.success(f"💰 Predicted Market Size: **${prediction[0]:,.2f} Million**")
    elif not model:
        st.error("⚠️ Prediction model unavailable.")

# --- Main Dashboard ---
if df is not None and not df.empty:
    st.title("📊 Indian Condom Market: Strategic Analysis Dashboard")
    
    # AI Insights
    st.header("🤖 AI Strategic Advisor")
    if st.button("✨ Generate AI Insights"):
        with st.spinner("Consulting Sass, the AI strategist..."):
            top_brands = df['brand name'].value_counts().nlargest(3).index.str.title().tolist()
            fastest_region = df.groupby('region')['growth rate (%)'].mean().idxmax().title()
            data_summary = f"Top 3 brands: {', '.join(top_brands)}. Fastest-growing region: {fastest_region}."
            insights = get_gemini_insights(data_summary)
            st.markdown(insights)
    st.markdown("---")

    # Visualizations
    tab1, tab2, tab3 = st.tabs(["📈 Market Overview", "🏢 Brand Deep Dive", "🗺️ Regional Hotspots"])
    with tab1:
        st.subheader("📊 Market Trends")
        st.pyplot(plot_market_size_trend(df))
        st.pyplot(plot_correlation_heatmap(df))
    with tab2:
        st.subheader("🏢 Brand & Company Analysis")
        st.pyplot(plot_brand_market_share(df))
        st.pyplot(plot_top_company_activity(df))
    with tab3:
        st.subheader("🗺️ Regional Insights")
        st.pyplot(plot_regional_growth(df))
        st.pyplot(plot_brand_region_heatmap(df))
else:
    st.error("⚠️ No data available. Please upload a CSV or ensure the default dataset is present.")
