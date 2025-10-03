import pandas as pd
import streamlit as st
from typing import Union

@st.cache_data
def load_and_clean_data(file: Union[str, object]) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file or file-like object, cleans the column names,
    and corrects data types.

    Args:
        file (str or file-like): The path to the CSV file or an uploaded file object.

    Returns:
        pd.DataFrame: A cleaned and preprocessed pandas DataFrame.
    """
    try:
        df = pd.read_csv(file)

        # Strip whitespace from all column names
        df.columns = df.columns.str.strip()

        # Convert date columns if they exist
        if 'Event Date' in df.columns:
            df['Event Date'] = pd.to_datetime(df['Event Date'], errors='coerce')
        if 'Year' in df.columns:
            # Try to convert to int if not already
            try:
                df['Year'] = pd.to_datetime(df['Year'], format='%Y', errors='coerce').dt.year
            except Exception:
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')

        # Standardize categorical variables for consistency
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

        return df

    except FileNotFoundError:
        st.error(f"Error: The file was not found at {file}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return pd.DataFrame()
