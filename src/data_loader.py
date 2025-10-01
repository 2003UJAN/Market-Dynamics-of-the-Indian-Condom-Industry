import pandas as pd
import streamlit as st

@st.cache_data # Use Streamlit's cache to avoid reloading data on every interaction
def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the condom market dataset from a specified CSV file path.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the dataset.
                       Returns an empty DataFrame if the file is not found.
    """
    try:
        df = pd.read_csv(file_path)
        # Basic data cleaning (can be expanded)
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df.dropna(subset=['Year'], inplace=True)
        df['Year'] = df['Year'].astype(int)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file was not found at {file_path}")
        return pd.DataFrame() # Return an empty dataframe on error
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return pd.DataFrame()
