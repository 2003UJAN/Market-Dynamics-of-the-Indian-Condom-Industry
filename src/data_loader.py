import pandas as pd
import streamlit as st

@st.cache_data # Use Streamlit's cache to avoid reloading data on every interaction
def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file, cleans the column names,
    and corrects data types.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A cleaned and preprocessed pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)

        # ---> CRITICAL FIX: Strip whitespace from all column names <---
        df.columns = df.columns.str.strip()
        
        # Correct data types
        df['Event Date'] = pd.to_datetime(df['Event Date'])
        df['Year'] = pd.to_datetime(df['Year'], format='%Y').dt.year
        
        # Standardize categorical variables for consistency
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip().str.lower()
            
        return df
        
    except FileNotFoundError:
        st.error(f"Error: The file was not found at {file_path}")
        return pd.DataFrame() # Return an empty dataframe on error
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return pd.DataFrame()

