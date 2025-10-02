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

# --- Configure Gemini API ---
# This new section handles the API key and initializes the model
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
gen_model = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        # Using the latest stable Flash model
        gen_model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}")
else:
    st.warning("🔑 Gemini API Key not found. AI analysis will be disabled.", icon="⚠️")


# --- Gemini API Call Function (Updated) ---
def get_gemini_insights(data_summary: str):
    """
    Calls the Gemini Flash model using the google-generativeai library.
    """
    if not gen_model:
        return "AI analysis is disabled. Please provide a Gemini API key."

    # This is the system prompt that defines the AI's persona and task
    system_prompt = (
        "You are 'Sass,' a witty, sharp, and slightly naughty business strategist specializing in the sexual wellness market. "
        "Your tone is playful, using clever puns and innuendos, but your insights are deadly serious and incredibly smart. "
        "Analyze the following data summary of the Indian condom market. Provide three sections in your response, using Markdown formatting: "
        "1. **🔥 Hot & Heavy Insights:** Your sharpest, most direct business observations. "
        "2. **짓 Naughty Marketing Slogans:** A few clever, edgy, and memorable marketing slogans for a top brand based on the data. "
        "3. **🤫 Untapped Pleasures:** Identify a key untapped opportunity (a region, a product type, or a channel) and explain the strategic move to capture it."
    )
    
    # We re-initialize the model here to include the system prompt with the request
    model_with_prompt = genai.GenerativeModel(
        model_name='gemini-1.5-flash-latest',
        system_instruction=system_prompt
    )
    
    user_prompt = f"Here is the data summary:\n{data_summary}"
    
    try:
        response = model_with_prompt.generate_content(user_prompt)
        return response.text
    except Exception as e:
        st.error(f"Error contacting the Gemini API: {e}")
        return "My creative juices just aren't flowing right now..."


# --- Load Model from Hugging Face ---
@st.cache_resource
def load_model_from_hf():
    """
    Downloads and loads the model from the specified Hugging Face Hub repository.
    """
    try:
        repo_id = "ujan2003/market_size_predictor"
        filename = "market_size_predictor.pkl"
        with st.spinner(f"Downloading prediction model from Hugging Face... This might take a moment."):
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        model = joblib.load(model_path)
        st.success("Market prediction model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model from Hugging Face Hub: {e}")
        return None

model = load_model_from_hf()


# --- UI Styling ---
st.markdown("""
    <style>
        .main {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #333333;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ff4b4b;
        }
        .stButton button {
            background-color: #ff4b4b;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
        }
        .stButton button:hover {
            background-color: #ff6b6b;
        }
        .stExpander {
            border: 1px solid #444;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.image("https://placehold.co/400x150/1a1a1a/ff4b4b?text=Market+Analysis&font=playfair-display", use_column_width=True)
    st.title("📊 Control Panel")
    st.write("Upload data or generate a market prediction.")

    uploaded_file = st.file_uploader("Upload your own CSV", type=["csv"])

    if uploaded_file is not None:
        df = load_and_clean_data(uploaded_file)
    else:
        df = load_and_clean_data("India_Condom_Market_Dataset.csv")

    if model and df is not None and not df.empty:
        with st.expander("🔮 Market Size Predictor", expanded=False):
            with st.form("prediction_form"):
                st.subheader("Enter Your Market Scenario")
                unique_brands = sorted(df['brand name'].str.title().unique())
                unique_regions = sorted(df['region'].str.title().unique())
                unique_materials = sorted(df['material type'].str.title().unique())
                unique_products = sorted(df['product type'].str.title().unique())
                
                year = st.number_input("Year", min_value=2020, max_value=2040, value=2025)
                brand_name = st.selectbox("Brand Name", options=unique_brands)
                region = st.selectbox("Region", options=unique_regions)
                material_type = st.selectbox("Material Type", options=unique_materials)
                product_type = st.selectbox("Product Type", options=unique_products)
                growth_rate = st.slider("Anticipated Growth Rate (%)", 0.0, 25.0, 10.5)
                
                submitted = st.form_submit_button("Predict Market Size")
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
                    
                    with st.spinner("The model is calculating..."):
                        prediction = model.predict(input_data)
                        st.success(f"Predicted Market Size: **${prediction[0]:,.2f} Million**")
    elif not model:
        st.error("Model could not be loaded. Prediction is disabled.")


# --- Main Dashboard ---
if df is not None and not df.empty:
    st.title("Indian Condom Market: Strategic Analysis Dashboard")

    # AI Insights Section
    st.markdown("---")
    st.header("🤖 AI Strategic Advisor (Powered by Gemini)")
    if st.button("Generate Strategic Insights 🤫"):
        with st.spinner("Consulting the AI strategist... this might get spicy..."):
            top_brands = df['brand name'].value_counts().nlargest(3).index.str.title().tolist()
            fastest_region = df.groupby('region')['growth rate (%)'].mean().idxmax().title()
            data_summary = f"Top 3 brands are: {', '.join(top_brands)}. The fastest-growing region is {fastest_region}."
            insights = get_gemini_insights(data_summary)
            st.markdown(insights)
    st.markdown("---")

    # Visualizations
    tab1, tab2, tab3 = st.tabs(["📈 Market Overview", "🏢 Brand & Company Deep Dive", "🗺️ Regional Hotspots"])
    with tab1:
        st.pyplot(plot_market_size_trend(df))
        st.pyplot(plot_correlation_heatmap(df))
    with tab2:
        st.pyplot(plot_brand_market_share(df))
        st.pyplot(plot_top_company_activity(df))
    with tab3:
        st.pyplot(plot_regional_growth(df))
        st.pyplot(plot_brand_region_heatmap(df))
else:
    st.error("No data to display. Please upload a CSV file or ensure the default 'India_Condom_Market_Dataset.csv' is present.")

