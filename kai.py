import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import google.generativeai as genai
from dotenv import load_dotenv
import os
import numpy as np
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Configure Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print(f'GOOGLE_API_KEY: {GOOGLE_API_KEY}')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-pro", 
                            google_api_key=GOOGLE_API_KEY,
                            temperature=0.3)

def analyze_data_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the structure of the dataframe to understand its contents
    """
    structure = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': list(df.select_dtypes(include=['float64', 'int64']).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns),
        'datetime_columns': list(df.select_dtypes(include=['datetime64']).columns),
        'boolean_columns': list(df.select_dtypes(include=['bool']).columns),
        'missing_values': df.isnull().sum().to_dict()
    }
    
    return structure

def suggest_visualizations(df: pd.DataFrame, question: str) -> List[Dict[str, Any]]:
    """
    Suggest appropriate visualizations based on the question and data types.
    """
    print('suggest_visualizations')
    question = question.lower()
    structure = analyze_data_structure(df)
    print(f'Structure: {structure}')
    suggestions = []
    
    # Keywords that suggest certain visualization types
    time_keywords = ['trend', 'over time', 'historical', 'growth', 'changes']
    comparison_keywords = ['compare', 'difference', 'versus', 'vs', 'against']
    distribution_keywords = ['distribution', 'spread', 'range', 'histogram']
    relationship_keywords = ['correlation', 'relationship', 'between', 'affect']
    categorical_keywords = ['category', 'group', 'segment', 'proportion', 'percent']

    # Check for time-based analysis
    if any(keyword in question for keyword in time_keywords) and structure['datetime_columns']:
        suggestions.append({
            'type': 'time_series',
            'columns': structure['numeric_columns'],
            'x_axis': structure['datetime_columns'][0]
        })
    
    # Check for categorical distributions
    if any(keyword in question for keyword in categorical_keywords) and structure['categorical_columns']:
        suggestions.append({
            'type': 'pie_chart',
            'columns': structure['categorical_columns']
        })
    
    # Check for comparisons
    if any(keyword in question for keyword in comparison_keywords):
        if structure['categorical_columns'] and structure['numeric_columns']:
            suggestions.append({
                'type': 'bar_chart',
                'possible_x': structure['categorical_columns'],
                'possible_y': structure['numeric_columns']
            })
    
    # Check for distributions
    if any(keyword in question for keyword in distribution_keywords):
        if structure['numeric_columns']:
            suggestions.append({
                'type': 'histogram',
                'columns': structure['numeric_columns']
            })
            suggestions.append({
                'type': 'box_plot',
                'columns': structure['numeric_columns']
            })
    
    # Check for relationships
    if any(keyword in question for keyword in relationship_keywords):
        if len(structure['numeric_columns']) >= 2:
            suggestions.append({
                'type': 'scatter_plot',
                'possible_x': structure['numeric_columns'],
                'possible_y': structure['numeric_columns']
            })
    
    print(f'suggestions: {suggestions}')
    return suggestions

def create_visualization(df: pd.DataFrame, viz_type: str, columns: Dict[str, str]) -> go.Figure:
    """
    Create a visualization based on the type and selected columns
    """
    print(f'create_visualization, viz_type: {viz_type}')
    print(f'{columns}')
    try:
        if viz_type == 'time_series':
            fig = px.line(df, x=columns['x'], y=columns['y'],
                         title=f'{columns["y"]} Over Time')
            
        elif viz_type == 'bar_chart':
            fig = px.bar(df, x=columns['x'], y=columns['y'],
                        title=f'{columns["y"]} by {columns["x"]}')
            
        elif viz_type == 'histogram':
            fig = px.histogram(df, x=columns['x'],
                             title=f'Distribution of {columns["x"]}')
            
        elif viz_type == 'box_plot':
            if 'group' in columns:
                fig = px.box(df, x=columns['group'], y=columns['y'],
                           title=f'Distribution of {columns["y"]} by {columns["group"]}')
            else:
                fig = px.box(df, y=columns['y'],
                           title=f'Distribution of {columns["y"]}')
                
        elif viz_type == 'scatter_plot':
            fig = px.scatter(df, x=columns['x'], y=columns['y'],
                           title=f'Relationship between {columns["x"]} and {columns["y"]}')
        
        elif viz_type == 'pie_chart':
            fig = px.pie(df, names=columns['x'], title=f'Proportion of {columns["x"]}')
            
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")
            
        return fig, None
        
    except Exception as e:
        return None, str(e)

def analyze_with_llm(df: pd.DataFrame, question: str) -> str:
    """
    Get insights from LLM based on the question and data
    """
    # Prepare data context
    print("LLM Call")
    structure = analyze_data_structure(df)
    print(f'structure: {structure}')
    
    context = f"""
    Analyze this dataset based on the question: {question}
    
    Dataset Information:
    - Total rows: {structure['total_rows']}
    - Total columns: {structure['total_columns']}
    - Numeric columns: {', '.join(structure['numeric_columns'])}
    - Categorical columns: {', '.join(structure['categorical_columns'])}
    
    Sample data:
    {df.head().to_string()}
    
    Basic statistics:
    {df[structure['numeric_columns']].describe().to_string() if structure['numeric_columns'] else 'No numeric columns'}
    
    Please provide insights that are:
    1. Specific to the question asked
    2. Based on actual data patterns
    3. Include relevant numbers and statistics
    4. Suggest potential business implications if applicable
    """
    print(f'final prompt: {context}')
    response = llm.invoke([HumanMessage(content=context)])
    print(f'llm response: {response}')
    return response.content

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

def main():
    st.set_page_config(layout="wide")  # Set page layout to wide

    st.title("Data Insights Explorer")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your data file", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Clean and prepare data
            with st.spinner("Preparing data..."):
                df = clean_and_parse_dates(df)
                df = safe_convert_numeric(df)
            
            # Display data sample
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Display basic data info
            st.subheader("Dataset Information")
            structure = analyze_data_structure(df)
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Total Records: {structure['total_rows']:,}")
                st.write(f"Total Features: {structure['total_columns']}")
            with col2:
                st.write(f"Numeric Columns: {len(structure['numeric_columns'])}")
                st.write(f"Categorical Columns: {len(structure['categorical_columns'])}")
            
            # Question input
            st.subheader("Ask Questions")
            if "user_question" not in st.session_state:
                st.session_state.user_question = ""
            
            st.session_state.user_question = st.text_input("Enter your question:", value=st.session_state.user_question)
            question = st.session_state.user_question
            
            if "stored_insights" not in st.session_state:
                st.session_state["stored_insights"] = ""

            # Analyze data when question changes
            if question and question != st.session_state.get("last_question", ""):
                with st.spinner("Analyzing data..."):
                    insights = analyze_with_llm(df, question)
                    st.session_state["stored_insights"] = insights
                    st.session_state["last_question"] = question  # Store last question
            else:
                insights = st.session_state["stored_insights"]
                    
            # Get visualization suggestions
            viz_suggestions = suggest_visualizations(df, question)
            print(f'viz_suggestions: {viz_suggestions}')
        
            # Create two columns side by side
            insights_col, viz_col = st.columns([0.4, 0.6]) 

            # Insights Column
            with insights_col:
                st.subheader("Analysis Results")
                st.write(insights)
                print('insights Done!')
            
            # Visualizations Column
            with viz_col:
                st.subheader("Visualizations")
                if 'selected_columns' not in st.session_state:
                    st.session_state['selected_columns'] = {}

                # Initialize col_x_key and col_y_key before using them
                col_x_key, col_y_key = None, None  # Add this line
                if not viz_suggestions:
                    st.write('Unfortunately we are unable to build visualizations for your query! Please search with different keywords.')
                else:
                    for suggestion in viz_suggestions:
                        col_x_key = f"{suggestion['type']}_x"
                        col_y_key = f"{suggestion['type']}_y"
                        
                        # Ensure default values if keys are missing
                        st.session_state.selected_columns.setdefault(col_x_key, None)
                        st.session_state.selected_columns.setdefault(col_y_key, None)
                        with st.expander(f"Create {suggestion['type'].replace('_', ' ').title()} Chart"):
                            cols = {}
                            #expander_key = f"{suggestion['type']}_{'_'.join(suggestion.get('columns', ['no_cols']))}"  # Include columns in key

                            if suggestion['type'] == 'time_series':
                                if suggestion.get('x_axis'):
                                    st.session_state.selected_columns[col_x_key] = st.selectbox(
                                        "Select time column",
                                        suggestion['x_axis'],
                                        key=col_x_key
                                    )
                                if suggestion.get('columns'):
                                    st.session_state.selected_columns[col_y_key] = st.selectbox(
                                        "Select value column",
                                        suggestion['columns'],
                                        key=col_y_key
                                    )

                            elif suggestion['type'] in ['bar_chart', 'scatter_plot']:
                                print('scatter_plot')
                                if suggestion.get('possible_x') and suggestion.get('possible_y'):
                                    st.session_state.selected_columns[col_x_key] = st.selectbox(
                                        "Select X-axis",
                                        suggestion['possible_x'],
                                        key=col_x_key
                                    )
                                    st.session_state.selected_columns[col_y_key] = st.selectbox(
                                        "Select Y-axis",
                                        suggestion['possible_y'],
                                        key=col_y_key
                                    )
                                
                            elif suggestion['type'] in ['histogram']:
                                print('inside historgram suggestion')
                                col_key = f"{suggestion['type']}_col_{suggestion['columns'][0]}"
                                print(f'col_key: {col_key}')
                                if col_key not in st.session_state:
                                    st.session_state[col_key] = suggestion['columns'][0]

                                selected_column = st.selectbox(
                                    "Select column",
                                    suggestion['columns'],
                                    index=suggestion['columns'].index(st.session_state[col_key]) if col_key in st.session_state and st.session_state[col_key] in suggestion['columns'] else 0,
                                    key=col_key
                                )
                                print(f'selected_column: {selected_column}')

                                cols['x'] = selected_column

                            elif suggestion['type'] == 'box_plot':
                                col_key = f"{suggestion['type']}_col_{suggestion['columns'][0]}"
                                if col_key not in st.session_state:
                                    st.session_state[col_key] = suggestion['columns'][0]

                                # Ensure unique key by appending the column name or visualization type
                                selected_column = st.selectbox(
                                    "Select column",
                                    suggestion['columns'],
                                    index=suggestion['columns'].index(st.session_state[col_key]) if col_key in st.session_state and st.session_state[col_key] in suggestion['columns'] else 0,
                                    key=f"{col_key}_{suggestion['type']}"  # Make key unique by appending visualization type
                                )

                                cols['y'] = selected_column  # Ensure 'y' is assigned to the selected column

                                # Handle grouping (optional)
                                group_col_key = f"{suggestion['type']}_group_col_{suggestion['columns'][0]}"
                                group_col = st.selectbox(
                                    "Group by (optional)",
                                    ['None'] + structure['categorical_columns'],
                                    key=f"{group_col_key}_{suggestion['type']}"  # Make group column key unique
                                )
                                if group_col != 'None':
                                    cols['group'] = group_col
                                else:
                                    cols['group'] = None  # Ensure 'group' is None if not selected
                            
                            elif suggestion['type'] == 'pie_chart':
                                print("Inside pie chart suggestion")  # Debugging step

                                # Ensure we select a categorical column
                                col_key = f"{suggestion['type']}_col_{suggestion['columns'][0]}"
                                
                                if col_key not in st.session_state:
                                    st.session_state[col_key] = suggestion['columns'][0]

                                selected_column = st.selectbox(
                                    "Select categorical column",
                                    suggestion['columns'],
                                    index=suggestion['columns'].index(st.session_state[col_key]) if col_key in st.session_state and st.session_state[col_key] in suggestion['columns'] else 0,
                                    key=col_key
                                )

                                # Dictionary to store selected column for pie chart
                                cols['x'] = selected_column  

                                # Pass the updated cols dictionary to create_visualization
                            if st.session_state.selected_columns[col_x_key] and st.session_state.selected_columns[col_y_key]:
                                print("Generating visualization for non histogram")  # Debugging step
                                fig, error = create_visualization(df, suggestion['type'], {
                                    'x': st.session_state.selected_columns[col_x_key],
                                    'y': st.session_state.selected_columns[col_y_key]
                                })
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                elif error:
                                    st.error(f"Error creating visualization: {error}")
                            elif suggestion['type'] in ['box_plot', 'histogram', 'pie_chart'] and cols:
                                print("Generating visualization for histogram")  # Debugging step
                                print(f'{cols}')
                                fig, error = create_visualization(df, suggestion['type'], cols)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                elif error:
                                    st.error(f"Error creating visualization: {error}")
                    
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

if __name__ == "__main__":
    main()

#What is the relationship between age and income category
#Show me the age distribution by gender
#Compare 
#What is the proportion of position to company