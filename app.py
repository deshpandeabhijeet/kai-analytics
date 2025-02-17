import streamlit as st
from data_processing import load_data, prepare_data
from visualization import display_data_info, handle_visualizations
from llm_analysis import get_insights_and_visualization_suggestions
from config import GOOGLE_API_KEY, OPENAI_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
import os

# Configure LLMs based on available API keys
if OPENAI_API_KEY:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, verbose=True, openai_api_key=OPENAI_API_KEY)
    print("Using OpenAI language model.") # Inform user which model is selected
elif GOOGLE_API_KEY:
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.3)
    print("Using Google Gemini language model.")
else:
    llm = None # Or raise an exception if no LLM is available
    st.error("No language model configured. Please set either GOOGLE_API_KEY or OPENAI_API_KEY in your environment or .env file.")


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
            
            # Get insights and visualization suggestions
            insights, viz_suggestions = get_insights_and_visualization_suggestions(df, question, llm)
            
            # Create two columns side by side
            insights_col, viz_col = st.columns([0.4, 0.6]) 

            # Insights Column
            with insights_col:
                st.subheader("Analysis Results")
                if isinstance(insights, list):
                    for insight in insights:
                        for key, value in insight.items():
                            st.write(f"{key.replace('_', ' ').title()}: {value}")
                        st.markdown("---")
                elif isinstance(insights, dict):
                    for key, value in insights.items(): 
                        st.write(f"{key.replace('_', ' ').title()}: {value}")
                else:
                    st.write(insights)
            
            # Visualizations Column
            with viz_col:
                st.subheader("Visualizations")
                handle_visualizations(df, viz_suggestions, structure)
                
if __name__ == "__main__":
    main()