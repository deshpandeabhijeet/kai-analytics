import pandas as pd
import streamlit as st
from data_processing import clean_and_parse_dates, load_data, prepare_data, safe_convert_numeric
from utils import select_columns
from visualization import suggest_visualizations, create_visualization, analyze_data_structure
from llm_analysis import analyze_with_llm
from config import GOOGLE_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-pro", 
                            google_api_key=GOOGLE_API_KEY,
                            temperature=0.3)

# Load the trained visualization suggestion model
# try:
#     model = load_model()
# except FileNotFoundError as e:
#     st.error(f"Visualization model not found. Please train the model first: {e}")
#     model = None

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
    
def handle_question(df, question, llm):
    """
    Handle the user's question and generate insights.
    """
    if "stored_insights" not in st.session_state:
        st.session_state["stored_insights"] = ""

    if question and question != st.session_state.get("last_question", ""):
        with st.spinner("Analyzing data..."):
            insights = analyze_with_llm(df, question, llm)
            st.session_state["stored_insights"] = insights
            st.session_state["last_question"] = question  # Store last question
    else:
        insights = st.session_state["stored_insights"]
    return insights

def handle_visualizations(df, viz_suggestions, structure):
    """
    Handle visualization suggestions and rendering.
    """
    if 'selected_columns' not in st.session_state:
        st.session_state['selected_columns'] = {}

    if not viz_suggestions:
        st.write('Unfortunately we are unable to build visualizations for your query! Please search with different keywords.')
    else:
        valid_suggestions = [s for s in viz_suggestions if s]  # Remove empty suggestions
        for suggestion_index, suggestion in enumerate(valid_suggestions):
            with st.expander(f"Create {suggestion['type'].replace('_', ' ').title()} Chart"):
                # Use the helper function to select columns
                cols = select_columns(suggestion, structure, st.session_state.selected_columns, suggestion_index)
                if cols:
                    fig, error = create_visualization(df, suggestion['type'], cols)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{suggestion_index}")
                    elif error:
                        st.error(f"Error creating visualization: {error}")
                        
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
            
            # Handle the question and generate insights
            insights = handle_question(df, question, llm)
            
            # Get visualization suggestions only if a question is provided
            #viz_suggestions = suggest_visualizations(df, question, model) if question and model else []
            viz_suggestions = suggest_visualizations(df, question)
        
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
