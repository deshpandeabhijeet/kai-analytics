import numpy as np
import pandas as pd
import streamlit as st
from data_processing import clean_and_parse_dates, load_data, prepare_data, safe_convert_numeric
from utils import select_columns
from visualization import create_visualization, analyze_data_structure
from llm_analysis import analyze_with_llm
from config import GOOGLE_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI
import plotly.express as px

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-pro", 
                            google_api_key=GOOGLE_API_KEY,
                            temperature=0.3)

def display_data_info(df):
    """
    Display data preview and basic dataset information.
    """
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    st.subheader("Dataset Information")
    structure = analyze_data_structure(df)
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Total Records: {structure['total_rows']:,}")
        st.write(f"Total Features: {structure['total_columns']}")
    with col2:
        st.write(f"Numeric Columns: {len(structure['numeric_columns'])}")
        st.write(f"Categorical Columns: {len(structure['categorical_columns'])}")
    return structure
    

def get_insights_and_visualization_suggestions(df, question, llm):
    """Gets insights and visualizations, now handles JSON parsing."""
    if question and question != st.session_state.get("last_question", ""):
        with st.spinner("Analyzing..."):
            response = analyze_with_llm(df, question, llm)
            st.session_state["last_question"] = question
            st.session_state["last_response"] = response # Store the full, parsed JSON response
    else:
        response = st.session_state.get("last_response", {}) # Access potentially cached response

    insights = response.get("insights", "") # Access elements safely
    viz_suggestions = response.get("visualizations", [])
    return insights, viz_suggestions

def handle_visualizations(df, viz_suggestions, structure):
    """Handles visualization rendering. Uses JSON data now."""
    if not viz_suggestions:
        st.write("No visualization suggestions.")
        return

    for i, viz in enumerate(viz_suggestions):
        with st.expander(f"Create {viz['type'].replace('_', ' ').title()}"):
            cols = {}  # Initialize cols for each visualization

            try:
                if viz["type"] == "time_series":
                    # Time series specific handling (ensure correct column types)
                    cols['x'] = st.selectbox("Select time column", structure['datetime_columns'], key=f"time_series_x_{i}")
                    if cols['x'] is not None:  # Check if a time column was selected
                        try: # Try converting the selected column to datetime
                            df[cols['x']] = pd.to_datetime(df[cols['x']]) # Ensure correct type for time series
                        except (ValueError, TypeError): # Informative error message if not time column
                            st.error(f"The selected column '{cols['x']}' cannot be interpreted as a datetime. Please select a valid datetime column.")
                            continue
                    else:
                      st.warning("Please select a time column.")
                      continue
                    cols['y'] = st.selectbox("Select value column", structure['numeric_columns'], key=f"time_series_y_{i}")
                elif viz["type"] in ["histogram", "box_plot"]:
                    print('pending implementaiton for histogram and box plot')
                elif viz["type"].lower() in ['pie_chart', 'pie chart']:
                    print('visulization type is pie chart')
                    try:
                        if "data" in viz and "labels" in viz["data"] and "values" in viz["data"]:
                            print('matched data, labels, values')
                            labels = viz["data"]["labels"]
                            values = viz["data"]["values"]
                            print(f'labels: {labels}, values: {values}')
                            if labels is None or values is None or not all(isinstance(label, str) for label in labels):
                                raise TypeError("Labels must be a list of strings")
                            df_for_pie = pd.DataFrame({"labels": labels, "values": values})
                            print(f'df_for_pie: {df_for_pie}')
                            fig = px.pie(df_for_pie, names='labels', values='values', title=viz.get("options", {}).get("title"))
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{i}")
                        else:  # Handle cases where the LLM *doesn't* provide "data"
                            print('visualization pie chart else block')
                            cols = {} # Very Important: Initialize cols
                            if "x" in viz and viz["x"] in df.columns: # Use 'x' if available and valid
                                print('viz has x column')
                                cols['x'] = viz['x']
                            elif "y" in viz and viz["y"] in df.columns: # Fallback to 'y' if 'x' is not valid.
                                print('viz has y column')
                                cols['x'] = viz['y'] 
                            else:
                                st.warning(f"Skipping pie chart: No valid 'x' or 'y' column provided by LLM or found in the DataFrame.")
                                continue
                            if cols:
                                    try:  # Narrow down the try-except block
                                        print(f"Creating pie chart with names={cols['x']}")

                                        # Ensure correct type for category_orders
                                        # unique_x = df[cols['x']].dropna().unique() # Drop NaNs first
                                        # if isinstance(unique_x, np.ndarray):
                                        #     unique_x = sorted([str(val) for val in unique_x]) # Convert each value to native python str and then sort. Important!
                                        # elif isinstance(unique_x, pd.Series):
                                        #     unique_x = sorted([str(val) for val in unique_x.tolist()]) # Convert each value to native python str after calling tolist, then sort. Important!

                                        # category_orders = {cols['x']: unique_x}  # Values should now be simple lists of Python strings!
                                        # print(f"category_orders: {category_orders}")
                                        
                                        unique_x = df[cols['x']].dropna().unique() #Handle missing values and convert to numpy array

                                        if not all(isinstance(val, str) for val in unique_x): # Check if all are strings
                                            #If mixed data types, convert all to string
                                            unique_x = np.array(unique_x).astype(str) # Convert each element to string if not already

                                        # Sort unique_x correctly (if sorting is desired):
                                        try: # Try sorting normally, will work if all strings.
                                            unique_x_sorted = sorted(unique_x) # Simple sort

                                        except TypeError:  # If sorting fails (mixed types even after conversion)
                                            print("Sorting failed. Using original order")  # Informative message
                                            unique_x_sorted = unique_x #Preserve original order if sorting fails


                                        category_orders = {cols['x']: unique_x_sorted}
                                        print(f"category_orders: {category_orders}")

                                        fig = px.pie(df, names=cols['x'], title=f'Proportion of {cols["x"]}', category_orders=category_orders)
                                        if fig:
                                            st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{i}")
                                            print('fig plotted')

                                    except (TypeError, ValueError, Exception) as e:  # Catch and print detailed exception
                                        print(f"Error creating pie chart for column '{cols['x']}': {type(e).__name__}: {e}")  # Detailed error message
                                        st.error(f"Error creating pie chart for column '{cols['x']}': {e}") # Display the error to the user
                                        continue 
                    except (TypeError, ValueError) as e:
                        print(f"in except block of pie chart. Error: {e}, Column: {cols.get('x', 'Unknown')}") # Print the exception AND the column!
                        st.error(f"Error creating pie chart for column '{cols.get('x', 'Unknown')}': {e}")  # Display error with column name
                        continue  # Move to next chart
                elif "x" in viz and "y" in viz:  # Handle bar chart and scatter plot
                    cols = {"x": viz["x"], "y": viz["y"]}
                    if "color" in viz:
                        cols["color"] = viz["color"]
                else:
                    st.warning(f"Visualization {viz.get('type')} missing required columns ('x' and 'y').")
                    continue
                print(f'cols before null check: {cols}')
                if cols: #Check if cols dictionary is populated before trying to create visualization
                    try:
                        print(f'calling create_visualization, cols: {cols}')
                        missing_cols = [col for col in cols.values() if col not in df.columns and col.lower() != 'count'] # 'count' is a special case
                        if missing_cols:
                            st.warning(f"Skipping visualization: Columns {', '.join(missing_cols)} not found in the DataFrame.")
                            continue
                        print(f'cols lower case: {cols['y'].lower()}')
                        if viz['type'] in ['bar_chart', 'bar chart'] and 'y' in cols and cols['y'].lower() == 'count' and 'count' not in df.columns:
                            # Special handling for bar chart with implied count
                            fig = px.bar(df, x=cols['x'], color=cols.get('color'),  # Use only x and color
                                         title=f'Distribution of {cols["x"]}') # Suitable title
                        else:
                            print('else section of create_visualization')
                            fig, error = create_visualization(df, viz["type"], cols)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{i}")
                        elif error:  # Handle errors from create_visualization
                            st.error(f"Error creating visualization: {error}")
                    except Exception as e:  # Handle any other unexpected exceptions
                        st.error(f"An unexpected error occurred during visualization creation: {e}")
            except Exception as e: #Error during column selection
                st.error(f"An error occurred during column selection: {e}")


def main():
    st.set_page_config(layout="wide")  # Set page layout to wide
    st.title("Data Insights Explorer")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your data file", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)
        if df is not None:
            # Clean and prepare data
            df = prepare_data(df)
            
            # Display data preview and information
            structure = display_data_info(df)
            
            # Question input
            st.subheader("Ask Questions")
            if "user_question" not in st.session_state:
                st.session_state.user_question = ""
            
            st.session_state.user_question = st.text_input("Enter your question:", value=st.session_state.user_question)
            question = st.session_state.user_question.strip()
            
            # # Handle the question and generate insights
            # insights = handle_question_and_get_insights(df, question, llm)
            
            # # Get visualization suggestions only if a question is provided
            # #viz_suggestions = suggest_visualizations(df, question, model) if question and model else []
            # viz_suggestions = suggest_visualizations(df, question)
        
            # Get insights and visualization suggestions
            insights, viz_suggestions = get_insights_and_visualization_suggestions(df, question, llm)
            # Create two columns side by side
            insights_col, viz_col = st.columns([0.4, 0.6]) 

            # Insights Column
            with insights_col:
                st.subheader("Analysis Results")
                st.write(insights)
            
            # Visualizations Column
            with viz_col:
                st.subheader("Visualizations")
                handle_visualizations(df, viz_suggestions, structure)
                
if __name__ == "__main__":
    main()
