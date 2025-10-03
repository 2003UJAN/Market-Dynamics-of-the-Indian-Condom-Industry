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

# Load .env file from src directory
dotenv_path = os.path.join(os.path.dirname(__file__), 'src', '.env')
load_dotenv(dotenv_path)

# Page config
st.set_page_config(
    page_title="Indian Condom Market Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom styling omitted for brevity --- Use same style as your original ---

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
gen_model_base = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        gen_model_base = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')
    except Exception as e:
        st.error(f"❌ Failed to configure Gemini API: {e}")
else:
    st.warning("⚠️ Gemini API Key not found. AI analysis will be disabled.")

# Gemini Insights function remains the same

def get_gemini_insights(data_summary: str):
    if not gen_model_base:
        return "AI analysis is disabled. Please provide a Gemini API key."

    system_prompt = (
        "You are 'Sass,' a witty, sharp, and slightly naughty business strategist specializing in the sexual wellness market. "
        "Your tone is playful but your insights are deadly serious and incredibly smart. Provide three sections in your response, using Markdown: "
        "1. **🔥 Hot & Heavy Insights**: Your sharpest, most direct business observations. "
        "2. **💡 Naughty Marketing Slogans**: A few clever, edgy, and memorable marketing slogans. "
        "3. **🤫 Untapped Pleasures**: Identify one key untapped opportunity and explain the strategic move to capture it."
    )

    model_with_prompt = genai.GenerativeModel(
        model_name='gemini-1.5-flash-latest',
        system_instruction=system_prompt
    )

    user_prompt = f"Here is the data summary:\n{data_summary}"

    try:
        response = model_with_prompt.generate_content(user_prompt)
        return response.text
    except Exception as e:
        st.error(f"Error contacting Gemini API: {e}")
        return "⚠️ AI insights unavailable at the moment."

# Load model from Hugging Face
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
        st.error("❌ Failed to load model from Hugging Face.")
        st.error("This is often due to a scikit-learn version mismatch. Ensure your requirements.txt pins the correct version.")
        st.error(f"Specific Error: {e}")
        return None

model = load_model_from_hf()

# Load data function with caching
@st.cache_data

def load_data(uploaded_file, default_data_path):
    if uploaded_file is not None:
        try:
            return load_and_clean_data(uploaded_file)
        except Exception as e:
            st.error(f"Failed to load uploaded data: {e}")
            return pd.DataFrame()
    elif os.path.exists(default_data_path):
        try:
            return load_and_clean_data(default_data_path)
        except Exception as e:
            st.error(f"Failed to load default data: {e}")
            return pd.DataFrame()
    else:
        st.error(f"Default data file not found at {default_data_path}")
        return pd.DataFrame()

# Sidebar and data loading
with st.sidebar:
    st.image("https://placehold.co/400x150/1a1a1a/ff4b4b?text=Market+Analysis", use_column_width=True)
    st.title("⚙️ Control Panel")
    st.write("Upload data or generate predictions.")

    uploaded_file = st.file_uploader("📂 Upload CSV Data", type=["csv"])
    DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "India_Condom_Market_Dataset.csv")

    df = load_data(uploaded_file, DEFAULT_DATA_PATH)

    if model is not None and not df.empty:
        with st.expander("🔮 Market Size Predictor", expanded=False):
            with st.form("prediction_form"):
                st.subheader("📝 Enter Market Scenario")

                # Ensure the data cleaning normalizes column names; use consistent case here
                unique_brands = sorted(df['brand_name'].str.title().unique())
                unique_regions = sorted(df['region'].str.title().unique())
                unique_materials = sorted(df['material_type'].str.title().unique())
                unique_products = sorted(df['product_type'].str.title().unique())

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
    elif model is None:
        st.error("⚠️ Prediction model unavailable.")

# Main dashboard
if df is not None and not df.empty:
    st.title("📊 Indian Condom Market: Strategic Analysis Dashboard")

    # AI Insights
    st.header("🤖 AI Strategic Advisor")
    if st.button("✨ Generate AI Insights"):
        if gen_model_base:
            with st.spinner("Consulting Sass, the AI strategist..."):
                top_brands = df['brand_name'].value_counts().nlargest(3).index.str.title().tolist()
                fastest_region = df.groupby('region')['growth_rate_pct'].mean().idxmax().title()
                data_summary = f"Top 3 brands: {', '.join(top_brands)}. Fastest-growing region: {fastest_region}."
                insights = get_gemini_insights(data_summary)
                st.markdown(f"<div class='card'>{insights}</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ AI analysis is disabled. Please provide a Gemini API key.")
    st.markdown("---")

    # Visualization tabs
    tab1, tab2, tab3 = st.tabs(["📈 Market Overview", "🏢 Brand Deep Dive", "🗺️ Regional Hotspots"])
    with tab1:
        st.subheader("Market Trends")
        st.pyplot(plot_market_size_trend(df))
        st.pyplot(plot_correlation_heatmap(df))
    with tab2:
        st.subheader("Brand & Company Analysis")
        st.pyplot(plot_brand_market_share(df))
        st.pyplot(plot_top_company_activity(df))
    with tab3:
        st.subheader("Regional Insights")
        st.pyplot(plot_regional_growth(df))
        st.pyplot(plot_brand_region_heatmap(df))
else:
    st.error("⚠️ No data available. Please upload a CSV or ensure the default dataset is present in 'data/'.")
