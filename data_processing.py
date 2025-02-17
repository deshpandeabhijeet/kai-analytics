import pandas as pd
from typing import Dict, Any
import streamlit as st

# def analyze_data_structure(df: pd.DataFrame) -> Dict[str, Any]:
#     """
#     Analyze the structure of the dataframe to understand its contents
#     """
#     structure = {
#         'total_rows': len(df),
#         'total_columns': len(df.columns),
#         'numeric_columns': list(df.select_dtypes(include=['float64', 'int64']).columns),
#         'categorical_columns': list(df.select_dtypes(include=['object']).columns),
#         'datetime_columns': list(df.select_dtypes(include=['datetime64']).columns),
#         'boolean_columns': list(df.select_dtypes(include=['bool']).columns),
#         'missing_values': df.isnull().sum().to_dict()
#     }
    
#     return structure

def clean_and_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and parse date columns in the dataframe
    """
    df_cleaned = df.copy()
    
    # Identify potential date columns
    date_patterns = ['dt', 'date', 'time', 'year', 'month', 'day']
    potential_date_cols = [col for col in df.columns if any(pattern in col.lower() for pattern in date_patterns)]
    
    for col in potential_date_cols:
        try:
            # Convert to datetime, coerce errors to NaT
            df_cleaned[col] = pd.to_datetime(df[col], errors='coerce')
            
            # If more than 50% conversion failed, revert to original
            if df_cleaned[col].isna().sum() > len(df) * 0.5:
                df_cleaned[col] = df[col]
        except Exception:
            # Keep original if conversion fails
            continue
    
    return df_cleaned

def safe_convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely convert numeric columns while handling mixed types
    """
    df_converted = df.copy()
    
    for col in df.columns:
        try:
            # Try to convert to numeric, coerce errors to NaN
            numeric_converted = pd.to_numeric(df[col], errors='coerce')
            
            # If conversion was mostly successful (less than 20% NaN), keep it
            if numeric_converted.isna().sum() <= len(df) * 0.2:
                df_converted[col] = numeric_converted
        except Exception:
            continue
    
    return df_converted

def prepare_data(df):
    """
    Clean and prepare the data.
    """
    with st.spinner("Preparing data..."):
        df = clean_and_parse_dates(df)
        df = safe_convert_numeric(df)
    return df
    
import pandas as pd

def load_data(uploaded_file):
    """
    Load data from the uploaded file.
    """
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None
    return df