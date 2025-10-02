import streamlit as st
import pandas as pd
import joblib
import requests
import time
import json
from src.data_loader import load_and_clean_data
from src.visualization import (
    plot_market_size_trend,
    plot_regional_growth,
    plot_brand_market_share,
    plot_correlation_heatmap,
    plot_brand_region_heatmap,
    plot_top_company_activity,
)

# --- Page Configuration ---
st.set_page_config(
    page_title="Indian Condom Market Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Gemini API Call Function ---
def get_gemini_insights(data_summary: str):
    """
    Calls the Gemini 2.5 Flash model to get AI-powered insights.
    Includes exponential backoff for retries.
    """
    api_key = "" # Canvas will supply the key
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    
    system_prompt = (
        "You are 'Sass,' a witty, sharp, and slightly naughty business strategist specializing in the sexual wellness market. "
        "Your tone is playful, using clever puns and innuendos, but your insights are deadly serious and incredibly smart. "
        "Analyze the following data summary of the Indian condom market. Provide three sections in your response, using Markdown formatting: "
        "1. **🔥 Hot & Heavy Insights:** Your sharpest, most direct business observations. "
        "2. **짓 Naughty Marketing Slogans:** A few clever, edgy, and memorable marketing slogans for a top brand based on the data. "
        "3. **🤫 Untapped Pleasures:** Identify a key untapped opportunity (a region, a product type, or a channel) and explain the strategic move to capture it."
    )
    
    user_prompt = f"Here is the data summary:\n{data_summary}"

    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    max_retries = 3
    backoff_factor = 2
    for i in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
            response.raise_for_status() # Raise an exception for bad status codes
            result = response.json()
            
            # Extract text from the correct part of the response structure
            candidate = result.get("candidates", [{}])[0]
            content_part = candidate.get("content", {}).get("parts", [{}])[0]
            return content_part.get("text", "Sorry, I'm feeling a bit shy... Couldn't generate insights right now.")

        except requests.exceptions.RequestException as e:
            if i < max_retries - 1:
                wait_time = backoff_factor ** i
                st.warning(f"API call failed. Retrying in {wait_time} second(s)...")
                time.sleep(wait_time)
            else:
                st.error(f"Failed to get insights after several retries: {e}")
                return "My creative juices just aren't flowing. The server seems to be down."
        except (KeyError, IndexError) as e:
             st.error(f"Unexpected API response structure: {result}")
             return "I got a response, but it's not what I expected. Can't share the secrets right now."


# --- Load Model ---
@st.cache_resource
def load_model(path="models/market_size_predictor.pkl"):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        return None

model = load_model()


# --- UI Styling ---
st.markdown("""
    <style>
        .main {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .st-sidebar {
            background-color: #111111;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 12px;
            border: 1px solid #ff4b4b;
            padding: 10px 20px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #e03c3c;
            border: 1px solid #e03c3c;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #333333;
            border-radius: 8px 8px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ff4b4b;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #ff4b4b;
        }
    </style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.image("https://placehold.co/400x150/1a1a1a/ff4b4b?text=Condom+Chronicles&font=playfair-display", use_column_width=True)
    st.title("🌶️ The Pleasure Panel")
    st.write("Control your experience from here. Upload your data or make a prediction.")

    uploaded_file = st.file_uploader(
        "Upload your own CSV (or use the default)", type=["csv"]
    )

    if uploaded_file is not None:
        # When a file is uploaded, process it
        df = load_and_clean_data(uploaded_file)
    else:
        # Load the default dataset if no file is uploaded
        df = load_and_clean_data("India_Condom_Market_Dataset.csv")

    # --- Manual Prediction Form ---
    if model:
        with st.expander("🔮 Predict Your Fortune", expanded=False):
            with st.form("prediction_form"):
                st.subheader("Enter Your Market Scenario")
                
                # Get unique sorted values from the dataframe for selectboxes
                unique_brands = sorted(df['Brand Name'].unique())
                unique_regions = sorted(df['Region'].unique())
                unique_materials = sorted(df['Material Type'].unique())
                unique_products = sorted(df['Product Type'].unique())
                
                # Input fields
                year = st.number_input("Year", min_value=2020, max_value=2040, value=2025)
                brand_name = st.selectbox("Brand Name", options=unique_brands)
                region = st.selectbox("Region", options=unique_regions)
                material_type = st.selectbox("Material Type", options=unique_materials)
                product_type = st.selectbox("Product Type", options=unique_products)
                growth_rate = st.slider("Anticipated Growth Rate (%)", 0.0, 25.0, 10.5)
                
                submitted = st.form_submit_button("Get Down to Business")

                if submitted:
                    # Create a dataframe from the user's input
                    # Note: We need to fill in all columns the model was trained on
                    input_data = pd.DataFrame([{
                        'Year': year,
                        'CAGR (%)': 10.0, # Example value
                        'Material Type': material_type.lower(),
                        'Product Type': product_type.lower(),
                        'Distribution Channel': 'e-commerce', # Example value
                        'Region': region.lower(),
                        'Market Penetration': 'medium', # Example value
                        'Growth Rate (%)': growth_rate,
                        'Brand Name': brand_name.lower(),
                        'Market Share (%)': 20.0, # Example value
                        'Revenue Contribution (%)': 5.0, # Example value
                        'Innovation Index': 5.0, # Example value
                        'Regulatory Impact': 'medium', # Example value
                        'Awareness Campaign Impact': 50.0 # Example value
                    }])
                    
                    prediction = model.predict(input_data)
                    st.success(f"Predicted Market Size: **${prediction[0]:,.2f} Million**")
    else:
        st.error("Model file not found. Prediction feature is disabled.")


# --- Main Dashboard ---
if df is not None and not df.empty:
    st.title("India's Condom Chronicles")
    st.markdown("### A *very* deep dive into the market's biggest secrets.")

    # --- AI Insights Section ---
    st.markdown("---")
    st.header("🤖 AI Pleasure Advisor (Powered by Gemini)")
    if st.button("Reveal the Secrets 🤫"):
        with st.spinner("Consulting the oracle... this might get spicy..."):
            # Create a concise summary of the data for the AI
            top_brands = df['Brand Name'].value_counts().nlargest(3).index.tolist()
            fastest_region = df.groupby('Region')['Growth Rate (%)'].mean().idxmax()
            
            data_summary = (
                f"The top 3 most mentioned brands are {', '.join(top_brands)}. "
                f"The fastest-growing region on average is {fastest_region}."
            )
            
            insights = get_gemini_insights(data_summary)
            st.markdown(insights)
    st.markdown("---")


    # --- Tabbed Visualizations ---
    tab1, tab2, tab3 = st.tabs(["📈 Market Overview", "🏢 Brand & Company Deep Dive", "🗺️ Regional Hotspots"])

    with tab1:
        st.header("The Big Picture")
        st.pyplot(plot_market_size_trend(df))
        st.pyplot(plot_correlation_heatmap(df))

    with tab2:
        st.header("The Players of the Game")
        st.pyplot(plot_brand_market_share(df))
        st.pyplot(plot_top_company_activity(df))

    with tab3:
        st.header("Where the Action Is")
        st.pyplot(plot_regional_growth(df))
        st.pyplot(plot_brand_region_heatmap(df))

else:
    st.error("No data to display. Please upload a CSV file using the sidebar.")
